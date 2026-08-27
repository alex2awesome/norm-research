#!/usr/bin/env python3
"""Validate or execute a frozen two-seed Nemotron CE production queue.

Scoring resumes only at immutable shard boundaries.  Existing valid shards
are reused, missing shards are scheduled in deterministic GPU waves, and any
partial or invalid artifact fails closed without deletion.  Merged scores and
the exact-byte content-addressed two-seed manifest are then passed to the
fixed consensus aggregator; no threshold argument exists in this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aggregate_nemotron_ce_seed_consensus import (
    CONSENSUS_REPORT_SCHEMA,
    CONSENSUS_SCHEMA,
    SEED_MANIFEST_SCHEMA,
    load_seed_manifest,
)
from .common import read_jsonl, sha256_file
from .freeze_nemotron_ce_production_queue import (
    QUEUE_SCHEMA,
    QUEUE_STATUS,
    _score_command,
    _validate_task_local_training_report,
    validate_production_pair_report,
)
from .gpu_host_policy import (
    is_sk3_host,
    validate_gpu_indices_for_host,
    validate_launch_gpus,
)
from .run_nemotron_ce import (
    CLASS_NAMES,
    SCORE_SCHEMA,
    _score_meta,
    verify_base_manifest,
    verify_checkpoint_contract,
)


RUN_SCHEMA = "silver-match-v3-nemotron-ce-two-seed-production-run-v1"


def _host_key(value: str) -> str:
    short = value.split(".", 1)[0].lower()
    if short.startswith("skampere"):
        return "sk" + short.removeprefix("skampere")
    return short


def _verify_artifact(value: Mapping[str, Any], label: str) -> Path:
    path = Path(str(value.get("path") or "")).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(value.get("size_bytes", -1))
        or sha256_file(path) != value.get("sha256")
    ):
        raise ValueError(f"frozen {label} artifact changed: {path}")
    return path


def _validate_static_queue(
    plan: Mapping[str, Any], *, hostname: str | None = None, deep_inputs: bool = True
) -> dict[str, Any]:
    safety = plan.get("safety") or {}
    if (
        plan.get("schema_version") != QUEUE_SCHEMA
        or plan.get("status") != QUEUE_STATUS
        or safety.get("production_labels_present") is not False
        or safety.get("threshold_retuning_permitted") is not False
        or safety.get("external_outcomes_opened") is not False
        or safety.get("release_ready") is not False
        or safety.get("thresholds_reused_only_from_each_checkpoint_dev") is not True
        or safety.get("consensus_requires_two_seed_same_leaf_and_both_frozen_gates") is not True
    ):
        raise ValueError("queue schema/status/safety contract failed")
    execution = plan.get("execution") or {}
    actual_host = hostname or platform.node()
    target_host = str(execution.get("target_host") or "")
    if _host_key(actual_host) != _host_key(target_host):
        raise ValueError(f"queue target host differs: {actual_host}/{target_host}")
    gpus = validate_gpu_indices_for_host(
        execution.get("physical_gpus") or [], hostname=actual_host
    )
    if is_sk3_host(actual_host) and len(gpus) > 4:
        raise ValueError("sk3 queue exceeds the four-device allowlist")
    python = Path(str(execution.get("python") or ""))
    if not python.is_absolute() or not python.is_file():
        raise ValueError("frozen Python executable is absent or not absolute")
    if platform.python_version() != execution.get("python_version"):
        raise ValueError("runtime Python version differs from frozen queue")
    for package, expected in (execution.get("packages") or {}).items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            raise ValueError(f"runtime package differs: {package} {actual} != {expected}")
    for name, value in (plan.get("implementations") or {}).items():
        _verify_artifact(value, f"implementation {name}")

    base = plan.get("base_model") or {}
    manifest_path = _verify_artifact(base.get("manifest") or {}, "base manifest")
    model = Path(str(base.get("path") or "")).resolve()
    base_contract = verify_base_manifest(
        model, manifest_path, (base.get("manifest") or {}).get("sha256")
    )
    if base_contract != base.get("verified_contract"):
        raise ValueError("base-model verified contract differs from frozen queue")
    production = plan.get("production_pairs") or {}
    _verify_artifact(production.get("report") or {}, "production pair report")
    _verify_artifact(production.get("pairs") or {}, "production pairs")
    _verify_artifact(production.get("norm_universe") or {}, "production norm universe")
    num_shards = int(execution.get("num_shards_per_seed", -1))
    if num_shards < 2:
        raise ValueError("production score queue must have at least two shards")
    if deep_inputs:
        observed = validate_production_pair_report(
            Path(production["report"]["path"]),
            expected_task=str(plan.get("task") or ""),
            num_shards=num_shards,
        )
        for key in (
            "task",
            "corpus_order",
            "norm_count",
            "candidate_depth",
            "pair_count",
            "shard_pair_counts",
            "shard_norm_counts",
            "labels_present",
            "all_rows_production_split",
        ):
            if observed.get(key) != production.get(key):
                raise ValueError(f"deep production pair audit differs: {key}")
        for key in ("report", "pairs", "norm_universe"):
            if observed[key] != production[key]:
                raise ValueError(f"deep production artifact identity differs: {key}")

    seeds = plan.get("seeds") or []
    if not isinstance(seeds, list) or len(seeds) != 2:
        raise ValueError("queue must contain exactly two seeds")
    seed_ids: set[str] = set()
    fingerprints: set[tuple[str, str, str]] = set()
    run_configs: list[dict[str, Any]] = []
    outputs = plan.get("outputs") or {}
    output_root = Path(str(outputs.get("root") or "")).resolve()
    seed_manifest_directory = Path(
        str(outputs.get("seed_manifest_directory") or "")
    ).resolve()
    top_level_targets = [
        Path(str(outputs.get(name) or "")).resolve()
        for name in ("consensus", "consensus_report", "consensus_log", "run_record")
    ]
    if (
        outputs.get("seed_manifest_filename_rule")
        != "<sha256-of-exact-json-bytes>.json"
        or len(set(top_level_targets)) != len(top_level_targets)
        or not all(
            path == output_root or output_root in path.parents
            for path in (seed_manifest_directory, *top_level_targets)
        )
        or Path(outputs["consensus_report"] + ".log").resolve()
        != Path(outputs["consensus_log"]).resolve()
    ):
        raise ValueError("production output-root/content-addressing contract differs")
    scoring = plan.get("scoring") or {}
    if (
        int(scoring.get("batch_size", -1)) < 1
        or scoring.get("attention") not in {"auto", "eager", "sdpa"}
        or int(scoring.get("logical_cuda_device", -1)) != 0
        or scoring.get("deterministic_norm_sharding") is not True
    ):
        raise ValueError("frozen score runtime contract is invalid")
    output_paths: set[Path] = set()
    for seed in seeds:
        seed_id = str(seed.get("seed_id") or "")
        if not seed_id or seed_id in seed_ids:
            raise ValueError("queue seed IDs are missing or duplicate")
        seed_ids.add(seed_id)
        report_path = _verify_artifact(seed.get("training_report") or {}, f"seed {seed_id} report")
        _verify_artifact(seed.get("run_config") or {}, f"seed {seed_id} run config")
        checkpoint = Path(str(seed.get("checkpoint") or "")).resolve()
        contract = verify_checkpoint_contract(
            checkpoint,
            report_path,
            (seed.get("training_report") or {}).get("sha256"),
            model=model,
        )
        if contract != seed.get("checkpoint_contract"):
            raise ValueError(f"seed checkpoint contract differs: {seed_id}")
        if deep_inputs:
            deep_contract, _, run_config = _validate_task_local_training_report(
                report_path,
                checkpoint,
                task=str(plan.get("task") or ""),
                model=model,
                expected_seed_id=seed_id,
            )
            if deep_contract != contract:
                raise ValueError(f"deep task-local checkpoint contract differs: {seed_id}")
            run_configs.append(run_config)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = report.get("selected_checkpoint") or {}
        if Path(str(selected.get("path") or "")).resolve() != checkpoint:
            raise ValueError(f"seed report no longer selects passed checkpoint: {seed_id}")
        fingerprints.add(
            (
                contract["checkpoint_metadata_sha256"],
                contract["head_sha256"],
                contract["adapter_tree_sha256"],
            )
        )
        shards = seed.get("shards") or []
        if len(shards) != num_shards or [int(row.get("shard_id", -1)) for row in shards] != list(range(num_shards)):
            raise ValueError(f"seed shard coordinates are incomplete: {seed_id}")
        for row in shards:
            shard_id = int(row["shard_id"])
            output = Path(str(row.get("output") or "")).resolve()
            meta = Path(str(row.get("meta") or "")).resolve()
            log = Path(str(row.get("log") or "")).resolve()
            if not all(path == output_root or output_root in path.parents for path in (output, meta, log)):
                raise ValueError(f"score shard target escapes output root: {seed_id}/{shard_id}")
            if output.with_suffix(output.suffix + ".meta.json") != meta:
                raise ValueError(f"score shard metadata path is not canonical: {seed_id}/{shard_id}")
            if output_paths.intersection((output, meta, log)):
                raise ValueError(f"duplicate score runtime target: {seed_id}/{shard_id}")
            output_paths.update((output, meta, log))
            expected_command = _score_command(
                python=python,
                pairs=Path(production["pairs"]["path"]),
                output=output,
                model=model,
                base_manifest=manifest_path,
                base_manifest_sha=(base.get("manifest") or {})["sha256"],
                checkpoint=checkpoint,
                training_report=report_path,
                training_report_sha=(seed.get("training_report") or {})["sha256"],
                batch_size=int(scoring["batch_size"]),
                max_length=int(contract["max_sequence_length"]),
                shard_id=shard_id,
                num_shards=num_shards,
                attention=str(scoring["attention"]),
            )
            if (
                int(row.get("num_shards", -1)) != num_shards
                or int(row.get("physical_gpu", -1)) not in gpus
                or int(row.get("expected_pair_count", -1)) != production["shard_pair_counts"][shard_id]
                or int(row.get("expected_norm_count", -1)) != production["shard_norm_counts"][shard_id]
                or row.get("command") != expected_command
            ):
                raise ValueError(f"seed shard command/count/GPU contract differs: {seed_id}/{shard_id}")
        merged = seed.get("merged") or {}
        merged_output = Path(str(merged.get("scores") or "")).resolve()
        merged_meta = Path(str(merged.get("meta") or "")).resolve()
        merged_log = Path(str(merged.get("log") or "")).resolve()
        if (
            merged_output.with_suffix(merged_output.suffix + ".meta.json") != merged_meta
            or not all(path == output_root or output_root in path.parents for path in (merged_output, merged_meta, merged_log))
            or output_paths.intersection((merged_output, merged_meta, merged_log))
        ):
            raise ValueError(f"merged score target contract differs: {seed_id}")
        output_paths.update((merged_output, merged_meta, merged_log))
        expected_merge = [
            str(python),
            "-u",
            "-m",
            "scripts.tools.silver_match_v3.run_nemotron_ce",
            "merge",
            "--inputs",
            *[str(Path(row["output"]).resolve()) for row in shards],
            "--output",
            str(merged_output),
        ]
        if merged.get("command") != expected_merge:
            raise ValueError(f"merged score command differs: {seed_id}")
    if len(fingerprints) != 2:
        raise ValueError("two seeds resolve to identical checkpoint content")
    if deep_inputs and (
        run_configs[0].get("train_pairs") != run_configs[1].get("train_pairs")
        or run_configs[0].get("dev_pairs") != run_configs[1].get("dev_pairs")
    ):
        raise ValueError("two seed task-local train/dev bindings differ")
    return {"hostname": actual_host, "gpus": gpus, "base_contract": base_contract}


def validate_queue(
    plan: Mapping[str, Any], *, hostname: str | None = None, deep_inputs: bool = True
) -> dict[str, Any]:
    """Public no-GPU queue validator used by tests and validation-only CLI."""

    return _validate_static_queue(plan, hostname=hostname, deep_inputs=deep_inputs)


def _scan_score_rows(path: Path, expected_rows: int) -> None:
    count = 0
    for line_no, row in enumerate(read_jsonl(path), 1):
        if row.get("schema_version") != SCORE_SCHEMA or "gold_relation" in row:
            raise ValueError(f"score row schema/label leakage differs: {path}:{line_no}")
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(CLASS_NAMES):
            raise ValueError(f"score probability schema differs: {path}:{line_no}")
        count += 1
    if count != expected_rows:
        raise ValueError(f"score row count differs: {path}/{count}/{expected_rows}")


def validate_score_shard(plan: Mapping[str, Any], seed: Mapping[str, Any], job: Mapping[str, Any]) -> bool:
    output = Path(job["output"])
    meta_path = Path(job["meta"])
    if not output.exists() and not meta_path.exists():
        return False
    if not output.is_file() or not meta_path.is_file():
        raise ValueError(f"partial score shard exists and is not resume-eligible: {output}")
    meta = _score_meta(output)
    production = plan["production_pairs"]
    expected = {
        "input_pairs": str(Path(production["pairs"]["path"]).resolve()),
        "input_pairs_sha256": production["pairs"]["sha256"],
        "output": str(output.resolve()),
        "row_count": int(job["expected_pair_count"]),
        "norm_group_count": int(job["expected_norm_count"]),
        "shard_id": int(job["shard_id"]),
        "num_shards": int(job["num_shards"]),
        "base_contract": plan["base_model"]["verified_contract"],
        "checkpoint_contract": seed["checkpoint_contract"],
        "labels": list(CLASS_NAMES),
        "bidirectional_concatenation": True,
        "pooling": "native_attention_mask_mean",
        "max_length": int(seed["checkpoint_contract"]["max_sequence_length"]),
        "cuda_bf16": True,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"score shard metadata differs: {output}/{key}")
    _scan_score_rows(output, int(job["expected_pair_count"]))
    return True


def validate_merged_scores(plan: Mapping[str, Any], seed: Mapping[str, Any]) -> bool:
    merged = seed["merged"]
    output = Path(merged["scores"])
    meta_path = Path(merged["meta"])
    if not output.exists() and not meta_path.exists():
        return False
    if not output.is_file() or not meta_path.is_file():
        raise ValueError(f"partial merged score artifact is not resume-eligible: {output}")
    meta = _score_meta(output)
    production = plan["production_pairs"]
    expected = {
        "input_pairs": str(Path(production["pairs"]["path"]).resolve()),
        "input_pairs_sha256": production["pairs"]["sha256"],
        "output": str(output.resolve()),
        "row_count": int(production["pair_count"]),
        "norm_group_count": int(production["norm_count"]),
        "shard_id": 0,
        "num_shards": 1,
        "combined_from_num_shards": int(plan["execution"]["num_shards_per_seed"]),
        "base_contract": plan["base_model"]["verified_contract"],
        "checkpoint_contract": seed["checkpoint_contract"],
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"merged score metadata differs: {output}/{key}")
    if set((meta.get("combined_shards") or {})) != {
        str(index) for index in range(int(plan["execution"]["num_shards_per_seed"]))
    }:
        raise ValueError(f"merged score shard provenance differs: {output}")
    _scan_score_rows(output, int(production["pair_count"]))
    return True


def _ensure_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(message + "\n", encoding="utf-8")


def _run_missing_shards(
    plan: Mapping[str, Any], validation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    pending: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for seed in plan["seeds"]:
        for job in seed["shards"]:
            if validate_score_shard(plan, seed, job):
                _ensure_log(Path(job["log"]), "valid immutable score shard already existed; resumed past scoring")
            else:
                pending.append((seed, job))
    guards: list[dict[str, Any]] = []
    environment_base = os.environ.copy()
    environment_base.update({str(k): str(v) for k, v in plan["environment"].items()})
    while pending:
        used: set[int] = set()
        wave: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        remaining = []
        for seed, job in pending:
            gpu = int(job["physical_gpu"])
            if gpu not in used:
                used.add(gpu)
                wave.append((seed, job))
            else:
                remaining.append((seed, job))
        guard = validate_launch_gpus(sorted(used), hostname=validation["hostname"])
        guards.append(guard)
        processes = []
        for seed, job in wave:
            output = Path(job["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab")
            log_handle.write(
                (f"\n=== immutable shard attempt {datetime.now(timezone.utc).isoformat()} ===\n").encode()
            )
            log_handle.flush()
            env = environment_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job["physical_gpu"])
            process = subprocess.Popen(
                job["command"],
                cwd=plan["execution"]["repo_root"],
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((seed, job, process, log_handle))
        failures = []
        for seed, job, process, log_handle in processes:
            returncode = process.wait()
            log_handle.close()
            if returncode:
                failures.append((seed["seed_id"], job["shard_id"], returncode))
            else:
                validate_score_shard(plan, seed, job)
        if failures:
            raise RuntimeError(
                "score shard wave failed closed; completed shards remain reusable and "
                f"partial shards require quarantine: {failures}"
            )
        pending = remaining
    return guards


def _run_cpu_command(command: Sequence[str], *, plan: Mapping[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in plan["environment"].items()})
    with log_path.open("ab") as handle:
        result = subprocess.run(
            list(command),
            cwd=plan["execution"]["repo_root"],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"frozen CPU stage failed closed with return code {result.returncode}: {command}")


def _merge_seeds(plan: Mapping[str, Any]) -> None:
    for seed in plan["seeds"]:
        if validate_merged_scores(plan, seed):
            _ensure_log(Path(seed["merged"]["log"]), "valid immutable merged scores already existed; resumed past merge")
            continue
        for job in seed["shards"]:
            if not validate_score_shard(plan, seed, job):
                raise ValueError("cannot merge before all frozen score shards exist")
        _run_cpu_command(
            seed["merged"]["command"],
            plan=plan,
            log_path=Path(seed["merged"]["log"]),
        )
        validate_merged_scores(plan, seed)


def seed_manifest_bytes(plan: Mapping[str, Any]) -> bytes:
    payload = {
        "schema_version": SEED_MANIFEST_SCHEMA,
        "task": plan["task"],
        "seeds": [
            {
                "seed_id": seed["seed_id"],
                "scores": str(Path(seed["merged"]["scores"]).resolve()),
                "scores_sha256": sha256_file(Path(seed["merged"]["scores"])),
                "scores_meta": str(Path(seed["merged"]["meta"]).resolve()),
                "scores_meta_sha256": sha256_file(Path(seed["merged"]["meta"])),
                "checkpoint": str(Path(seed["checkpoint"]).resolve()),
                "training_report": seed["training_report"]["path"],
                "training_report_sha256": seed["training_report"]["sha256"],
            }
            for seed in plan["seeds"]
        ],
        "norm_universe": {
            "path": plan["production_pairs"]["norm_universe"]["path"],
            "sha256": plan["production_pairs"]["norm_universe"]["sha256"],
        },
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def freeze_seed_manifest(plan: Mapping[str, Any]) -> tuple[Path, str]:
    raw = seed_manifest_bytes(plan)
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path(plan["outputs"]["seed_manifest_directory"])
    path = directory / f"{digest}.json"
    directory.mkdir(parents=True, exist_ok=True)
    extras = [candidate for candidate in directory.glob("*.json") if candidate != path]
    if extras:
        raise ValueError(f"unexpected competing content-addressed seed manifests: {extras[:3]}")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("content-addressed seed manifest bytes differ")
    else:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    seeds, universe, provenance = load_seed_manifest(path, digest)
    if len(seeds) != 2 or universe is None or provenance["sha256"] != digest:
        raise ValueError("content-addressed seed manifest did not round-trip")
    return path, digest


def build_consensus_command(
    plan: Mapping[str, Any], seed_manifest: Path, seed_manifest_sha256: str
) -> list[str]:
    """Construct the only permitted post-score consensus command."""

    return [
        plan["execution"]["python"],
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.aggregate_nemotron_ce_seed_consensus",
        "--seed-manifest",
        str(seed_manifest.resolve()),
        "--seed-manifest-sha256",
        seed_manifest_sha256,
        "--output",
        plan["outputs"]["consensus"],
        "--report-output",
        plan["outputs"]["consensus_report"],
    ]


def validate_consensus(
    plan: Mapping[str, Any], seed_manifest: Path, seed_manifest_sha256: str
) -> bool:
    output = Path(plan["outputs"]["consensus"])
    report_path = Path(plan["outputs"]["consensus_report"])
    if not output.exists() and not report_path.exists():
        return False
    if not output.is_file() or not report_path.is_file():
        raise ValueError("partial consensus output/report is not resume-eligible")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    production = plan["production_pairs"]
    if (
        report.get("schema_version") != CONSENSUS_REPORT_SCHEMA
        or report.get("status") != "COMPLETE"
        or Path(str(report.get("output") or "")).resolve() != output.resolve()
        or report.get("output_sha256") != sha256_file(output)
        or int(report.get("norm_count", -1)) != int(production["norm_count"])
        or int(report.get("candidate_pair_count", -1)) != int(production["pair_count"])
        or (report.get("manifest_provenance") or {}).get("sha256") != seed_manifest_sha256
        or Path(str((report.get("manifest_provenance") or {}).get("path") or "")).resolve()
        != seed_manifest.resolve()
        or ((report.get("external_norm_universe") or {}).get("sha256"))
        != production["norm_universe"]["sha256"]
        or (report.get("validation") or {}).get("all_thresholds_from_checkpoint_dev") is not True
        or (report.get("validation") or {}).get("test_threshold_tuning_performed") is not False
        or (report.get("validation") or {}).get("all_norms_preserved") is not True
    ):
        raise ValueError("consensus report contract differs from frozen queue")
    corpora = set(production["corpus_order"])
    count = 0
    for line_no, row in enumerate(read_jsonl(output), 1):
        if (
            row.get("schema_version") != CONSENSUS_SCHEMA
            or row.get("task") != plan["task"]
            or row.get("corpus") not in corpora
            or row.get("split") != "production"
            or row.get("human_abstention_subtype_assigned") is not False
        ):
            raise ValueError(f"consensus row routing/task contract differs: {output}:{line_no}")
        count += 1
    if count != int(production["norm_count"]):
        raise ValueError("consensus output count differs from exact norm universe")
    return True


def _validate_complete_run_record(
    plan: Mapping[str, Any], queue_path: Path, seed_manifest: Path, manifest_sha: str
) -> dict[str, Any] | None:
    path = Path(plan["outputs"]["run_record"])
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != RUN_SCHEMA
        or payload.get("status") != "COMPLETE_PROVISIONAL_TWO_SEED_CONSENSUS"
        or payload.get("queue_sha256") != sha256_file(queue_path)
        or payload.get("seed_manifest_sha256") != manifest_sha
        or payload.get("consensus_sha256") != sha256_file(Path(plan["outputs"]["consensus"]))
        or payload.get("consensus_report_sha256") != sha256_file(Path(plan["outputs"]["consensus_report"]))
    ):
        raise ValueError("sealed production run record differs")
    validate_consensus(plan, seed_manifest, manifest_sha)
    return payload


def run(plan: Mapping[str, Any], queue_path: Path) -> dict[str, Any]:
    validation = validate_queue(plan, deep_inputs=True)
    started = datetime.now(timezone.utc)
    guards = _run_missing_shards(plan, validation)
    _merge_seeds(plan)
    seed_manifest, manifest_sha = freeze_seed_manifest(plan)
    existing = _validate_complete_run_record(plan, queue_path, seed_manifest, manifest_sha)
    if existing is not None:
        return existing
    if not validate_consensus(plan, seed_manifest, manifest_sha):
        consensus_log = Path(plan["outputs"]["consensus_log"])
        _run_cpu_command(
            build_consensus_command(plan, seed_manifest, manifest_sha),
            plan=plan,
            log_path=consensus_log,
        )
    validate_consensus(plan, seed_manifest, manifest_sha)
    completed = datetime.now(timezone.utc)
    shard_artifacts = []
    for seed in plan["seeds"]:
        shard_artifacts.append(
            {
                "seed_id": seed["seed_id"],
                "merged_scores_sha256": sha256_file(Path(seed["merged"]["scores"])),
                "merged_scores_meta_sha256": sha256_file(Path(seed["merged"]["meta"])),
                "shards": [
                    {
                        "shard_id": job["shard_id"],
                        "physical_gpu": job["physical_gpu"],
                        "scores_sha256": sha256_file(Path(job["output"])),
                        "meta_sha256": sha256_file(Path(job["meta"])),
                        "log_sha256": sha256_file(Path(job["log"])),
                    }
                    for job in seed["shards"]
                ],
            }
        )
    record = {
        "schema_version": RUN_SCHEMA,
        "status": "COMPLETE_PROVISIONAL_TWO_SEED_CONSENSUS",
        "task": plan["task"],
        "host": platform.node(),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "queue": str(queue_path.resolve()),
        "queue_sha256": sha256_file(queue_path),
        "gpu_launch_guards": guards,
        "seed_manifest": str(seed_manifest),
        "seed_manifest_sha256": manifest_sha,
        "consensus": plan["outputs"]["consensus"],
        "consensus_sha256": sha256_file(Path(plan["outputs"]["consensus"])),
        "consensus_report": plan["outputs"]["consensus_report"],
        "consensus_report_sha256": sha256_file(Path(plan["outputs"]["consensus_report"])),
        "seed_artifacts": shard_artifacts,
        "norm_count": plan["production_pairs"]["norm_count"],
        "candidate_pair_count": plan["production_pairs"]["pair_count"],
        "threshold_retuning_performed": False,
        "external_outcomes_opened": False,
        "release_ready": False,
    }
    run_record = Path(plan["outputs"]["run_record"])
    run_record.parent.mkdir(parents=True, exist_ok=True)
    with run_record.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--run", action="store_true", help="explicitly execute; default validates only")
    parser.add_argument("--shallow-input-validation", action="store_true", help="validation-only diagnostic; execution always repeats the full streamed input audit")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    queue_path = Path(args.queue).resolve()
    plan = json.loads(queue_path.read_text(encoding="utf-8"))
    if not args.run:
        validate_queue(plan, deep_inputs=not args.shallow_input_validation)
        print(json.dumps({"status": "VALIDATED_NOT_LAUNCHED", "queue": str(queue_path), "queue_sha256": sha256_file(queue_path), "task": plan["task"], "norm_count": plan["production_pairs"]["norm_count"], "pair_count": plan["production_pairs"]["pair_count"]}, sort_keys=True))
        return
    record = run(plan, queue_path)
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
