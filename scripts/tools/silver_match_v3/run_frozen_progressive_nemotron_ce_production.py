#!/usr/bin/env python3
"""Validate or execute a frozen progressive two-seed CE production queue.

Norms begin in the first disjoint candidate tier.  After each tier, both seed
scores are accumulated and the ordinary conservative consensus gate is
applied.  A norm exits only at a development-authorized tier when both seeds
pass their immutable checkpoint.dev gates and choose the same leaf.  Every
other norm continues, including single-seed matches, disagreements, rejects,
family signals, below-gate rows, and matches at unauthorized tiers.  The last
tier contributes every bank metric not already scored.

Resume is allowed only across verified immutable score shards and sealed trial
records.  Partial downstream artifacts fail closed and are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .aggregate_nemotron_ce_seed_consensus import (
    CONSENSUS_REPORT_SCHEMA,
    CONSENSUS_SCHEMA,
    SEED_MANIFEST_SCHEMA,
    NormUniverse,
    SeedArtifact,
    aggregate_seed_consensus,
)
from .common import normalize_space, read_jsonl, sha256_file
from .freeze_nemotron_ce_production_queue import (
    _score_command,
    _validate_task_local_training_report,
)
from .freeze_progressive_nemotron_ce_production_queue import (
    QUEUE_SCHEMA,
    QUEUE_STATUS,
    validate_progressive_inputs,
)
from .gpu_host_policy import is_sk3_host, validate_gpu_indices_for_host, validate_launch_gpus
from .materialize_nemotron_ce_production_pairs import UNIVERSE_SCHEMA
from .run_nemotron_ce import (
    CLASS_NAMES,
    SCORE_META_SCHEMA,
    SCORE_SCHEMA,
    _score_meta,
    merge_score_shards,
    pair_shard,
    verify_base_manifest,
    verify_checkpoint_contract,
)


RUN_SCHEMA = "silver-match-v3-progressive-nemotron-ce-production-run-v1"
STAGE_SCHEMA = "silver-match-v3-progressive-nemotron-ce-stage-v1"
ACTIVE_PAIR_META_SCHEMA = "silver-match-v3-progressive-nemotron-ce-active-pairs-v1"


def _host_key(value: str) -> str:
    short = value.split(".", 1)[0].lower()
    return "sk" + short.removeprefix("skampere") if short.startswith("skampere") else short


def _artifact(path: Path, *, count: int | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if count is not None:
        value["count"] = count
    return value


def _verify_artifact(ref: Mapping[str, Any], label: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    if (
        not path.is_file()
        or sha256_file(path) != ref.get("sha256")
        or (
            ref.get("size_bytes") is not None
            and path.stat().st_size != int(ref.get("size_bytes", -1))
        )
    ):
        raise ValueError(f"{label} artifact changed: {path}")
    return path


def validate_queue(
    plan: Mapping[str, Any], *, hostname: str | None = None, deep: bool = True
) -> dict[str, Any]:
    safety = plan.get("safety") or {}
    if (
        plan.get("schema_version") != QUEUE_SCHEMA
        or plan.get("status") != QUEUE_STATUS
        or safety.get("production_labels_present") is not False
        or safety.get("threshold_retuning_permitted") is not False
        or safety.get("test_or_blind_outcomes_read") is not False
        or safety.get("early_exit_requires_two_seed_same_leaf_and_both_checkpoint_dev_gates")
        is not True
        or safety.get("early_exit_requires_dev_policy_authorization") is not True
        or safety.get("every_disagreement_abstention_or_unauthorized_match_continues")
        is not True
        or safety.get("fullbank_terminal_rescue_mandatory") is not True
    ):
        raise ValueError("progressive queue safety contract failed")
    execution = plan.get("execution") or {}
    actual_host = hostname or platform.node()
    if _host_key(actual_host) != _host_key(str(execution.get("target_host") or "")):
        raise ValueError("progressive queue target host differs")
    gpus = list(
        validate_gpu_indices_for_host(
            execution.get("physical_gpus") or [], hostname=actual_host
        )
    )
    if is_sk3_host(actual_host) and len(gpus) > 4:
        raise ValueError("sk3 progressive queue exceeds four permitted GPUs")
    python = Path(str(execution.get("python") or ""))
    if not python.is_absolute() or not python.is_file():
        raise ValueError("frozen Python executable is missing/not absolute")
    if platform.python_version() != execution.get("python_version"):
        raise ValueError("Python version differs from frozen queue")
    for package, expected in (execution.get("packages") or {}).items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            raise ValueError(f"runtime package differs: {package} {actual}/{expected}")
    for name, ref in (plan.get("implementations") or {}).items():
        _verify_artifact(ref, f"implementation {name}")
    manifest_path = _verify_artifact(
        plan.get("progressive_pairs_manifest") or {}, "progressive pair manifest"
    )
    policy_path = _verify_artifact(plan.get("dev_stop_policy") or {}, "dev stop policy")
    manifest, policy = validate_progressive_inputs(
        manifest_path, policy_path, task=str(plan.get("task") or "")
    )
    if [row.get("trial_id") for row in plan.get("trials") or []] != [
        row.get("trial_id") for row in manifest.get("trials") or []
    ]:
        raise ValueError("queue trial order differs from progressive manifest")
    _verify_artifact(plan.get("norm_universe") or {}, "norm universe")
    _verify_artifact(plan.get("bank") or {}, "bank")
    if int(plan.get("norm_count", -1)) != int(manifest.get("norm_count", -2)):
        raise ValueError("queue norm count differs")
    base = plan.get("base_model") or {}
    base_manifest = _verify_artifact(base.get("manifest") or {}, "base manifest")
    model = Path(str(base.get("path") or "")).resolve()
    base_contract = verify_base_manifest(
        model, base_manifest, (base.get("manifest") or {})["sha256"]
    )
    if base_contract != base.get("verified_contract"):
        raise ValueError("base model contract differs")
    run_configs = []
    fingerprints = set()
    if len(plan.get("seeds") or []) != 2:
        raise ValueError("progressive queue requires exactly two seeds")
    for seed in plan["seeds"]:
        report = _verify_artifact(seed.get("training_report") or {}, "training report")
        _verify_artifact(seed.get("run_config") or {}, "run config")
        checkpoint = Path(str(seed.get("checkpoint") or "")).resolve()
        contract = verify_checkpoint_contract(
            checkpoint,
            report,
            (seed.get("training_report") or {})["sha256"],
            model=model,
        )
        if contract != seed.get("checkpoint_contract"):
            raise ValueError("seed checkpoint contract differs")
        fingerprints.add(
            (
                contract["checkpoint_metadata_sha256"],
                contract["head_sha256"],
                contract["adapter_tree_sha256"],
            )
        )
        if deep:
            deep_contract, _, run_config = _validate_task_local_training_report(
                report,
                checkpoint,
                task=plan["task"],
                model=model,
                expected_seed_id=seed["seed_id"],
            )
            if deep_contract != contract:
                raise ValueError("deep task-local seed contract differs")
            run_configs.append(run_config)
    if len(fingerprints) != 2:
        raise ValueError("two seed checkpoints have identical content")
    if deep and (
        run_configs[0].get("train_pairs") != run_configs[1].get("train_pairs")
        or run_configs[0].get("dev_pairs") != run_configs[1].get("dev_pairs")
    ):
        raise ValueError("seed training inputs differ")
    authorized = set(policy["authorized_early_stop_trials"])
    for queue_trial, manifest_trial in zip(
        plan["trials"], manifest["trials"], strict=True
    ):
        for key in (
            "trial_id",
            "ordinal",
            "kind",
            "component_depth",
            "pairs",
            "terminal",
        ):
            if queue_trial.get(key) != manifest_trial.get(key):
                raise ValueError(f"queue/manifest trial differs: {key}")
        if (queue_trial["trial_id"] in authorized) != (
            queue_trial.get("early_stop_authorized") is True
        ):
            raise ValueError("queue early-stop authorization differs from dev policy")
    return {
        "hostname": actual_host,
        "gpus": gpus,
        "model": model,
        "base_manifest": base_manifest,
        "policy": policy,
        "manifest": manifest,
    }


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _active_uids(path: Path, *, original_universe: bool) -> set[str]:
    values = set()
    for line_no, row in enumerate(read_jsonl(path), 1):
        uid = normalize_space(row.get("norm_uid"))
        if not uid or uid in values:
            raise ValueError(f"missing/duplicate active UID: {path}:{line_no}")
        if original_universe and row.get("schema_version") != UNIVERSE_SCHEMA:
            raise ValueError(f"invalid original universe row: {path}:{line_no}")
        if not original_universe and row.get("schema_version") != CONSENSUS_SCHEMA:
            raise ValueError(f"invalid continuation ledger row: {path}:{line_no}")
        values.add(uid)
    return values


def _materialize_active_pairs(
    plan: Mapping[str, Any],
    trial: Mapping[str, Any],
    active: set[str],
    active_source: Path,
) -> tuple[Path, dict[str, Any]]:
    root = Path(trial["runtime_root"])
    output = root / "active.pairs.jsonl"
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    source_pairs = Path(trial["pairs"]["path"])
    if sha256_file(source_pairs) != trial["pairs"]["sha256"]:
        raise ValueError(f"frozen trial pair artifact changed: {source_pairs}")
    if output.exists() or meta_path.exists():
        if not output.is_file() or not meta_path.is_file():
            raise ValueError("partial active pair artifact is not resume eligible")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("schema_version") != ACTIVE_PAIR_META_SCHEMA
            or meta.get("trial_id") != trial["trial_id"]
            or meta.get("source_pairs_sha256") != trial["pairs"]["sha256"]
            or meta.get("active_source_sha256") != sha256_file(active_source)
            or int(meta.get("active_norm_count", -1)) != len(active)
            or meta.get("output_sha256") != sha256_file(output)
        ):
            raise ValueError("existing active pair metadata differs")
        return output, meta
    output.parent.mkdir(parents=True, exist_ok=True)
    num_shards = int(plan["execution"]["num_shards_per_seed"])
    pair_counts = [0] * num_shards
    norm_sets = [set() for _ in range(num_shards)]
    emitted_uids = set()
    with output.open("x", encoding="utf-8") as handle:
        for line_no, row in enumerate(read_jsonl(source_pairs), 1):
            uid = normalize_space(row.get("norm_uid"))
            if uid not in active:
                continue
            if (
                row.get("progressive_trial_id") != trial["trial_id"]
                or row.get("task") != plan["task"]
                or row.get("split") != "production"
            ):
                raise ValueError(f"active tier pair routing differs: {line_no}")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            shard = pair_shard(uid, num_shards)
            pair_counts[shard] += 1
            norm_sets[shard].add(uid)
            emitted_uids.add(uid)
        handle.flush()
        os.fsync(handle.fileno())
    meta = {
        "schema_version": ACTIVE_PAIR_META_SCHEMA,
        "trial_id": trial["trial_id"],
        "source_pairs": trial["pairs"]["path"],
        "source_pairs_sha256": trial["pairs"]["sha256"],
        "active_source": str(active_source.resolve()),
        "active_source_sha256": sha256_file(active_source),
        "active_norm_count": len(active),
        "norms_with_new_candidates": len(emitted_uids),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "pair_count": sum(pair_counts),
        "num_shards": num_shards,
        "shard_pair_counts": pair_counts,
        "shard_norm_counts": [len(values) for values in norm_sets],
    }
    _write_json_new(meta_path, meta)
    return output, meta


def _scan_score(path: Path, *, expected_rows: int) -> None:
    count = 0
    for line_no, row in enumerate(read_jsonl(path), 1):
        probabilities = row.get("probabilities") or {}
        if (
            row.get("schema_version") != SCORE_SCHEMA
            or "gold_relation" in row
            or set(probabilities) != set(CLASS_NAMES)
        ):
            raise ValueError(f"invalid/leaky score row: {path}:{line_no}")
        count += 1
    if count != expected_rows:
        raise ValueError(f"score row count differs: {path}/{count}/{expected_rows}")


def _validate_score_shard(
    plan: Mapping[str, Any],
    seed: Mapping[str, Any],
    *,
    input_pairs: Path,
    input_sha: str,
    output: Path,
    shard_id: int,
    expected_rows: int,
    expected_norms: int,
) -> bool:
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if not output.exists() and not meta_path.exists():
        return False
    if not output.is_file() or not meta_path.is_file():
        raise ValueError(f"partial immutable score shard: {output}")
    meta = _score_meta(output)
    expected = {
        "input_pairs": str(input_pairs.resolve()),
        "input_pairs_sha256": input_sha,
        "output": str(output.resolve()),
        "row_count": expected_rows,
        "norm_group_count": expected_norms,
        "shard_id": shard_id,
        "num_shards": int(plan["execution"]["num_shards_per_seed"]),
        "base_contract": plan["base_model"]["verified_contract"],
        "checkpoint_contract": seed["checkpoint_contract"],
        "labels": list(CLASS_NAMES),
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"immutable score shard metadata differs: {output}")
    _scan_score(output, expected_rows=expected_rows)
    return True


def _score_trial(
    plan: Mapping[str, Any],
    validation: Mapping[str, Any],
    trial: Mapping[str, Any],
    active_pairs: Path,
    active_meta: Mapping[str, Any],
) -> dict[str, Path | None]:
    if int(active_meta["pair_count"]) == 0:
        return {seed["seed_id"]: None for seed in plan["seeds"]}
    num_shards = int(plan["execution"]["num_shards_per_seed"])
    gpus = validation["gpus"]
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in plan["environment"].items()})
    pending = []
    outputs: dict[str, Path] = {}
    for seed_index, seed in enumerate(plan["seeds"]):
        seed_root = Path(trial["runtime_root"]) / f"seed-{seed['seed_id']}"
        shard_paths = []
        for shard_id in range(num_shards):
            output = seed_root / "shards" / f"part-{shard_id:05d}-of-{num_shards:05d}.scores.jsonl"
            shard_paths.append(output)
            if not _validate_score_shard(
                plan,
                seed,
                input_pairs=active_pairs,
                input_sha=active_meta["output_sha256"],
                output=output,
                shard_id=shard_id,
                expected_rows=int(active_meta["shard_pair_counts"][shard_id]),
                expected_norms=int(active_meta["shard_norm_counts"][shard_id]),
            ):
                gpu = gpus[(seed_index * num_shards + shard_id) % len(gpus)]
                command = _score_command(
                    python=Path(plan["execution"]["python"]),
                    pairs=active_pairs,
                    output=output,
                    model=validation["model"],
                    base_manifest=validation["base_manifest"],
                    base_manifest_sha=plan["base_model"]["manifest"]["sha256"],
                    checkpoint=Path(seed["checkpoint"]),
                    training_report=Path(seed["training_report"]["path"]),
                    training_report_sha=seed["training_report"]["sha256"],
                    batch_size=int(plan["execution"]["batch_size"]),
                    max_length=int(seed["checkpoint_contract"]["max_sequence_length"]),
                    shard_id=shard_id,
                    num_shards=num_shards,
                    attention=plan["execution"]["attention"],
                )
                pending.append((seed, shard_id, gpu, output, command))
        merged = seed_root / "merged.scores.jsonl"
        outputs[seed["seed_id"]] = merged
    while pending:
        wave = []
        rest = []
        used = set()
        for job in pending:
            if job[2] in used:
                rest.append(job)
            else:
                used.add(job[2])
                wave.append(job)
        validate_launch_gpus(sorted(used), hostname=validation["hostname"])
        processes = []
        for seed, shard_id, gpu, output, command in wave:
            output.parent.mkdir(parents=True, exist_ok=True)
            log = output.with_suffix(output.suffix + ".log")
            handle = log.open("ab")
            env = environment.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command,
                cwd=plan["execution"]["repo_root"],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((seed, shard_id, output, process, handle))
        failures = []
        for seed, shard_id, output, process, handle in processes:
            returncode = process.wait()
            handle.close()
            if returncode:
                failures.append((seed["seed_id"], shard_id, returncode))
                continue
            _validate_score_shard(
                plan,
                seed,
                input_pairs=active_pairs,
                input_sha=active_meta["output_sha256"],
                output=output,
                shard_id=shard_id,
                expected_rows=int(active_meta["shard_pair_counts"][shard_id]),
                expected_norms=int(active_meta["shard_norm_counts"][shard_id]),
            )
        if failures:
            raise RuntimeError(f"progressive score wave failed closed: {failures}")
        pending = rest
    for seed in plan["seeds"]:
        seed_root = Path(trial["runtime_root"]) / f"seed-{seed['seed_id']}"
        merged = outputs[seed["seed_id"]]
        merged_meta = merged.with_suffix(merged.suffix + ".meta.json")
        if merged.exists() or merged_meta.exists():
            if not merged.is_file() or not merged_meta.is_file():
                raise ValueError(f"partial merged trial score: {merged}")
            meta = _score_meta(merged)
            if (
                int(meta.get("row_count", -1)) != int(active_meta["pair_count"])
                or meta.get("input_pairs_sha256") != active_meta["output_sha256"]
                or meta.get("checkpoint_contract") != seed["checkpoint_contract"]
            ):
                raise ValueError(f"merged trial score metadata differs: {merged}")
            _scan_score(merged, expected_rows=int(active_meta["pair_count"]))
        else:
            shard_paths = [
                seed_root / "shards" / f"part-{index:05d}-of-{num_shards:05d}.scores.jsonl"
                for index in range(num_shards)
            ]
            merge_score_shards(shard_paths, merged)
            _scan_score(merged, expected_rows=int(active_meta["pair_count"]))
    return outputs


def _build_active_universe(
    plan: Mapping[str, Any], trial: Mapping[str, Any], active: set[str]
) -> Path:
    output = Path(trial["runtime_root"]) / "active.universe.jsonl"
    if output.exists():
        observed = _active_uids(output, original_universe=True)
        if observed != active:
            raise ValueError("existing active universe differs")
        return output
    source = Path(plan["norm_universe"]["path"])
    count = _write_jsonl_new(
        output,
        (row for row in read_jsonl(source) if normalize_space(row.get("norm_uid")) in active),
    )
    if count != len(active):
        raise ValueError("active universe filtering dropped norms")
    return output


def _build_cumulative_score(
    plan: Mapping[str, Any],
    trial: Mapping[str, Any],
    seed: Mapping[str, Any],
    active: set[str],
    sources: Sequence[Path],
) -> Path:
    output = Path(trial["runtime_root"]) / f"seed-{seed['seed_id']}" / "cumulative.scores.jsonl"
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    source_refs = [_artifact(path) for path in sources]
    if output.exists() or meta_path.exists():
        if not output.is_file() or not meta_path.is_file():
            raise ValueError("partial cumulative score artifact")
        meta = _score_meta(output)
        if (
            meta.get("progressive_source_scores") != source_refs
            or meta.get("checkpoint_contract") != seed["checkpoint_contract"]
            or int(meta.get("norm_group_count", -1)) != len(active)
        ):
            raise ValueError("existing cumulative score artifact differs")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_norms = set()
    with output.open("x", encoding="utf-8") as handle:
        for source in sources:
            source_meta = _score_meta(source)
            if source_meta.get("checkpoint_contract") != seed["checkpoint_contract"]:
                raise ValueError("cumulative score source checkpoint differs")
            for row in read_jsonl(source):
                uid = normalize_space(row.get("norm_uid"))
                if uid not in active:
                    continue
                if row.get("schema_version") != SCORE_SCHEMA or "gold_relation" in row:
                    raise ValueError("invalid cumulative score source row")
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
                seen_norms.add(uid)
        handle.flush()
        os.fsync(handle.fileno())
    if seen_norms != active:
        raise ValueError(
            f"cumulative score candidates absent for active norms: {len(active-seen_norms)}"
        )
    meta = {
        "schema_version": SCORE_META_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_pairs": "progressive-disjoint-tier-union",
        "input_pairs_sha256": hashlib.sha256(
            "".join(ref["sha256"] for ref in source_refs).encode()
        ).hexdigest(),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "row_count": count,
        "norm_group_count": len(seen_norms),
        "shard_id": 0,
        "num_shards": 1,
        "base_contract": plan["base_model"]["verified_contract"],
        "checkpoint_contract": seed["checkpoint_contract"],
        "labels": list(CLASS_NAMES),
        "bidirectional_concatenation": True,
        "pooling": "native_attention_mask_mean",
        "max_length": seed["checkpoint_contract"]["max_sequence_length"],
        "cuda_bf16": True,
        "progressive_source_scores": source_refs,
    }
    _write_json_new(meta_path, meta)
    return output


def _trial_consensus(
    plan: Mapping[str, Any],
    trial: Mapping[str, Any],
    active_universe: Path,
    cumulative: Mapping[str, Path],
) -> tuple[Path, Path]:
    root = Path(trial["runtime_root"])
    output = root / "consensus.jsonl"
    report = root / "consensus.report.json"
    if output.exists() or report.exists():
        if not output.is_file() or not report.is_file():
            raise ValueError("partial trial consensus")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("output_sha256") != sha256_file(output):
            raise ValueError("existing trial consensus hash differs")
        return output, report
    seeds = []
    for seed in plan["seeds"]:
        score = cumulative[seed["seed_id"]]
        seeds.append(
            SeedArtifact(
                seed_id=seed["seed_id"],
                scores=score,
                scores_sha256=sha256_file(score),
                scores_meta=score.with_suffix(score.suffix + ".meta.json"),
                scores_meta_sha256=sha256_file(
                    score.with_suffix(score.suffix + ".meta.json")
                ),
                checkpoint=Path(seed["checkpoint"]),
                training_report=Path(seed["training_report"]["path"]),
                training_report_sha256=seed["training_report"]["sha256"],
            )
        )
    aggregate_seed_consensus(
        seeds[0],
        seeds[1],
        output=output,
        report_output=report,
        norm_universe=NormUniverse(active_universe, sha256_file(active_universe)),
        manifest_provenance={
            "schema_version": SEED_MANIFEST_SCHEMA,
            "progressive_queue": True,
            "trial_id": trial["trial_id"],
        },
    )
    return output, report


def _partition_trial(
    trial: Mapping[str, Any], consensus: Path, *, terminal: bool
) -> tuple[Path, Path, int, int]:
    root = Path(trial["runtime_root"])
    accepted = root / "accepted.jsonl"
    continued = root / "continue.jsonl"
    if accepted.exists() or continued.exists():
        raise ValueError(
            "unsealed trial partition exists; quarantine it before resuming from "
            "the last immutable score shard"
        )
    allow_exit = trial.get("early_stop_authorized") is True or terminal
    accepted_rows = []
    continued_rows = []
    for row in read_jsonl(consensus):
        annotated = _annotate_progressive(row, trial, allow_exit=allow_exit)
        if allow_exit and row.get("automatic_match") is True:
            accepted_rows.append(annotated)
        else:
            continued_rows.append(annotated)
    _write_jsonl_new(accepted, accepted_rows)
    _write_jsonl_new(continued, continued_rows)
    if terminal and len(accepted_rows) + len(continued_rows) == 0:
        raise ValueError("terminal trial received no active norms")
    return accepted, continued, len(accepted_rows), len(continued_rows)


def _annotate_progressive(
    row: Mapping[str, Any],
    trial: Mapping[str, Any],
    *,
    allow_exit: bool,
) -> dict[str, Any]:
    automatic_exit = allow_exit and row.get("automatic_match") is True
    return {
        **row,
        "progressive": {
            "exit_trial_id": trial["trial_id"] if automatic_exit else None,
            "exit_trial_ordinal": trial["ordinal"] if automatic_exit else None,
            "dev_stop_authorized": trial.get("early_stop_authorized") is True,
            "terminal_complete_bank_trial": trial.get("terminal") is True,
            "all_prior_candidate_tiers_scored": True,
        },
    }


def _validate_stage_record(trial: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(trial["stage_record"])
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema_version") != STAGE_SCHEMA
        or record.get("status") != "COMPLETE_IMMUTABLE_TRIAL"
        or record.get("trial_id") != trial["trial_id"]
        or int(record.get("ordinal", -1)) != int(trial["ordinal"])
        or record.get("terminal") is not (trial.get("terminal") is True)
        or record.get("early_stop_authorized")
        is not (trial.get("early_stop_authorized") is True)
        or record.get("trial_pairs_sha256") != trial["pairs"]["sha256"]
    ):
        raise ValueError("sealed progressive trial record differs")
    verified_paths = {}
    for name in (
        "active_pairs",
        "active_pairs_meta",
        "active_universe",
        "consensus",
        "consensus_report",
        "accepted",
        "continued",
    ):
        verified_paths[name] = _verify_artifact(
            record.get(name) or {}, f"stage {name}"
        )
    active_meta = json.loads(verified_paths["active_pairs_meta"].read_text(encoding="utf-8"))
    if (
        active_meta.get("schema_version") != ACTIVE_PAIR_META_SCHEMA
        or active_meta.get("trial_id") != trial["trial_id"]
        or active_meta.get("source_pairs_sha256") != trial["pairs"]["sha256"]
        or active_meta.get("output_sha256") != sha256_file(verified_paths["active_pairs"])
    ):
        raise ValueError("sealed active-pair metadata differs from trial")
    active_uids = _active_uids(
        verified_paths["active_universe"], original_universe=True
    )
    accepted_uids = _active_uids(
        verified_paths["accepted"], original_universe=False
    )
    continued_uids = _active_uids(
        verified_paths["continued"], original_universe=False
    )
    consensus_uids = _active_uids(
        verified_paths["consensus"], original_universe=False
    )
    if (
        accepted_uids & continued_uids
        or accepted_uids | continued_uids != active_uids
        or consensus_uids != active_uids
        or int(record.get("active_norm_count", -1)) != len(active_uids)
        or int(record.get("accepted_count", -1)) != len(accepted_uids)
        or int(record.get("continued_count", -1)) != len(continued_uids)
        or int(active_meta.get("active_norm_count", -1)) != len(active_uids)
    ):
        raise ValueError("sealed trial accepted/continued partition differs")
    allow_exit = (
        trial.get("early_stop_authorized") is True
        or trial.get("terminal") is True
    )
    expected_rows = {
        normalize_space(row.get("norm_uid")): hashlib.sha256(
            json.dumps(
                _annotate_progressive(row, trial, allow_exit=allow_exit),
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        for row in read_jsonl(verified_paths["consensus"])
    }
    for partition in (verified_paths["accepted"], verified_paths["continued"]):
        for row in read_jsonl(partition):
            uid = normalize_space(row.get("norm_uid"))
            observed = hashlib.sha256(
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if expected_rows.get(uid) != observed:
                raise ValueError(f"sealed trial partition row differs from consensus: {uid}")
    source_artifacts = record.get("cumulative_source_artifacts") or {}
    if not isinstance(source_artifacts, Mapping) or len(source_artifacts) != 2:
        raise ValueError("sealed trial lacks two-seed cumulative source artifacts")
    for seed_id, refs in source_artifacts.items():
        if not isinstance(refs, list):
            raise ValueError(f"sealed cumulative sources are invalid: {seed_id}")
        for ref in refs:
            _verify_artifact((ref or {}).get("scores") or {}, f"{seed_id} tier score")
            _verify_artifact(
                (ref or {}).get("scores_meta") or {}, f"{seed_id} tier score metadata"
            )
    return record


def _run_trial(
    plan: Mapping[str, Any],
    validation: Mapping[str, Any],
    trial: Mapping[str, Any],
    *,
    active_source: Path,
    active_original: bool,
    prior_merged: Mapping[str, Sequence[Path]],
) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    existing = _validate_stage_record(trial)
    if existing is not None:
        merged = {
            seed["seed_id"]: [
                Path(value["scores"]["path"])
                for value in existing["cumulative_source_artifacts"][seed["seed_id"]]
            ]
            for seed in plan["seeds"]
        }
        return existing, merged
    active = _active_uids(active_source, original_universe=active_original)
    if not active:
        root = Path(trial["runtime_root"])
        root.mkdir(parents=True, exist_ok=True)
        active_pairs = root / "active.pairs.jsonl"
        active_meta = active_pairs.with_suffix(active_pairs.suffix + ".meta.json")
        active_universe = root / "active.universe.jsonl"
        consensus = root / "consensus.jsonl"
        consensus_report = root / "consensus.report.json"
        accepted = root / "accepted.jsonl"
        continued = root / "continue.jsonl"
        for path in (active_pairs, active_universe, consensus, accepted, continued):
            if path.exists():
                if path.read_bytes():
                    raise ValueError(f"exhausted-stage artifact is unexpectedly nonempty: {path}")
            else:
                _write_jsonl_new(path, [])
        empty_pair_meta = {
            "schema_version": ACTIVE_PAIR_META_SCHEMA,
            "trial_id": trial["trial_id"],
            "trial_pairs_sha256": trial["pairs"]["sha256"],
            "source_pairs_sha256": trial["pairs"]["sha256"],
            "active_source_sha256": sha256_file(active_source),
            "active_norm_count": 0,
            "output_sha256": sha256_file(active_pairs),
            "pair_count": 0,
            "num_shards": int(plan["execution"]["num_shards_per_seed"]),
            "shard_pair_counts": [0]
            * int(plan["execution"]["num_shards_per_seed"]),
            "shard_norm_counts": [0]
            * int(plan["execution"]["num_shards_per_seed"]),
        }
        if active_meta.exists():
            if json.loads(active_meta.read_text(encoding="utf-8")) != empty_pair_meta:
                raise ValueError("exhausted-stage active-pair metadata differs")
        else:
            _write_json_new(active_meta, empty_pair_meta)
        empty_report = {
            "schema_version": CONSENSUS_REPORT_SCHEMA,
            "status": "COMPLETE",
            "output": str(consensus),
            "output_sha256": sha256_file(consensus),
            "norm_count": 0,
            "candidate_pair_count": 0,
            "validation": {
                "all_thresholds_from_checkpoint_dev": True,
                "test_threshold_tuning_performed": False,
                "all_norms_preserved": True,
                "seed_norm_candidate_source_split_universes_identical": True,
            },
            "vacuous_no_survivors": True,
        }
        if consensus_report.exists():
            if json.loads(consensus_report.read_text(encoding="utf-8")) != empty_report:
                raise ValueError("exhausted-stage consensus report differs")
        else:
            _write_json_new(consensus_report, empty_report)
        record = {
            "schema_version": STAGE_SCHEMA,
            "status": "COMPLETE_IMMUTABLE_TRIAL",
            "task": plan["task"],
            "trial_id": trial["trial_id"],
            "ordinal": trial["ordinal"],
            "terminal": trial.get("terminal") is True,
            "early_stop_authorized": trial.get("early_stop_authorized") is True,
            "active_norm_count": 0,
            "new_pair_count_one_seed": 0,
            "new_pair_count_two_seeds": 0,
            "cumulative_candidate_pair_count": 0,
            "accepted_count": 0,
            "continued_count": 0,
            "active_pairs": _artifact(active_pairs, count=0),
            "active_pairs_meta": _artifact(active_meta),
            "active_universe": _artifact(active_universe, count=0),
            "consensus": _artifact(consensus, count=0),
            "consensus_report": _artifact(consensus_report),
            "accepted": _artifact(accepted, count=0),
            "continued": _artifact(continued, count=0),
            "cumulative_sources": {
                seed_id: [str(path) for path in paths]
                for seed_id, paths in prior_merged.items()
            },
            "cumulative_source_artifacts": {
                seed_id: [
                    {
                        "scores": _artifact(path),
                        "scores_meta": _artifact(
                            path.with_suffix(path.suffix + ".meta.json")
                        ),
                    }
                    for path in paths
                ]
                for seed_id, paths in prior_merged.items()
            },
            "routing_contract": {
                "vacuous_no_survivors": True,
                "complete_bank_rescue_required_for_zero_survivors": False,
                "coverage_preserved": True,
            },
        }
        _write_json_new(Path(trial["stage_record"]), record)
        return record, {seed_id: list(paths) for seed_id, paths in prior_merged.items()}
    active_pairs, active_meta = _materialize_active_pairs(
        plan, trial, active, active_source
    )
    current_scores = _score_trial(
        plan, validation, trial, active_pairs, active_meta
    )
    cumulative_sources: dict[str, list[Path]] = {}
    cumulative: dict[str, Path] = {}
    for seed in plan["seeds"]:
        seed_id = seed["seed_id"]
        sources = list(prior_merged.get(seed_id) or [])
        if current_scores[seed_id] is not None:
            sources.append(Path(current_scores[seed_id]))
        if not sources:
            raise ValueError("first progressive tier produced no candidates")
        cumulative_sources[seed_id] = sources
        cumulative[seed_id] = _build_cumulative_score(
            plan, trial, seed, active, sources
        )
    active_universe = _build_active_universe(plan, trial, active)
    consensus, consensus_report = _trial_consensus(
        plan, trial, active_universe, cumulative
    )
    accepted, continued, accepted_count, continued_count = _partition_trial(
        trial, consensus, terminal=trial.get("terminal") is True
    )
    consensus_payload = json.loads(consensus_report.read_text(encoding="utf-8"))
    record = {
        "schema_version": STAGE_SCHEMA,
        "status": "COMPLETE_IMMUTABLE_TRIAL",
        "task": plan["task"],
        "trial_id": trial["trial_id"],
        "trial_pairs_sha256": trial["pairs"]["sha256"],
        "ordinal": trial["ordinal"],
        "terminal": trial.get("terminal") is True,
        "early_stop_authorized": trial.get("early_stop_authorized") is True,
        "active_norm_count": len(active),
        "new_pair_count_one_seed": int(active_meta["pair_count"]),
        "new_pair_count_two_seeds": 2 * int(active_meta["pair_count"]),
        "cumulative_candidate_pair_count": int(consensus_payload["candidate_pair_count"]),
        "accepted_count": accepted_count,
        "continued_count": continued_count,
        "active_pairs": _artifact(active_pairs, count=int(active_meta["pair_count"])),
        "active_pairs_meta": _artifact(active_pairs.with_suffix(active_pairs.suffix + ".meta.json")),
        "active_universe": _artifact(active_universe, count=len(active)),
        "consensus": _artifact(consensus, count=len(active)),
        "consensus_report": _artifact(consensus_report),
        "accepted": _artifact(accepted, count=accepted_count),
        "continued": _artifact(continued, count=continued_count),
        "cumulative_sources": {
            seed_id: [str(path) for path in paths]
            for seed_id, paths in cumulative_sources.items()
        },
        "cumulative_source_artifacts": {
            seed_id: [
                {
                    "scores": _artifact(path),
                    "scores_meta": _artifact(
                        path.with_suffix(path.suffix + ".meta.json")
                    ),
                }
                for path in paths
            ]
            for seed_id, paths in cumulative_sources.items()
        },
        "routing_contract": {
            "two_seed_same_leaf_gate_required": True,
            "unauthorized_matches_continued": True,
            "all_nonmatches_continued": not trial.get("terminal"),
            "candidate_tiers_scored_are_disjoint": True,
        },
    }
    _write_json_new(Path(trial["stage_record"]), record)
    return record, cumulative_sources


def _merge_final(
    plan: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> tuple[Path, Path, dict[str, Any]]:
    output = Path(plan["outputs"]["progressive_consensus"])
    report_path = Path(plan["outputs"]["progressive_consensus_report"])
    if output.exists() or report_path.exists():
        if not output.is_file() or not report_path.is_file():
            raise ValueError("partial progressive final consensus")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("output_sha256") != sha256_file(output):
            raise ValueError("existing progressive final consensus differs")
        return output, report_path, report
    terminal = records[-1]
    if terminal.get("terminal") is not True:
        raise ValueError("progressive run lacks terminal fullbank record")
    handle = tempfile.NamedTemporaryFile(prefix="progressive-final-", suffix=".sqlite3", delete=False)
    handle.close()
    db_path = Path(handle.name)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE rows (norm_uid TEXT PRIMARY KEY, raw TEXT NOT NULL) WITHOUT ROWID")
    try:
        sources = [Path(record["accepted"]["path"]) for record in records]
        sources.append(Path(terminal["continued"]["path"]))
        with db:
            for source in sources:
                for row in read_jsonl(source):
                    uid = normalize_space(row.get("norm_uid"))
                    try:
                        db.execute(
                            "INSERT INTO rows VALUES (?, ?)",
                            (uid, json.dumps(row, ensure_ascii=False, sort_keys=True)),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"norm received multiple terminal CE decisions: {uid}") from exc
        expected = int(plan["norm_count"])
        if int(db.execute("SELECT COUNT(*) FROM rows").fetchone()[0]) != expected:
            raise ValueError("progressive terminal decision count differs from norm universe")
        output.parent.mkdir(parents=True, exist_ok=True)
        routing: Counter[str] = Counter()
        candidate_pairs = 0
        match_count = 0
        with output.open("x", encoding="utf-8") as out:
            for universe in read_jsonl(Path(plan["norm_universe"]["path"])):
                uid = normalize_space(universe.get("norm_uid"))
                value = db.execute("SELECT raw FROM rows WHERE norm_uid=?", (uid,)).fetchone()
                if value is None:
                    raise ValueError(f"norm lacks terminal CE decision: {uid}")
                row = json.loads(value[0])
                routing[str(row["routing_category"])] += 1
                candidate_pairs += int(row["candidate_count"])
                match_count += int(row.get("automatic_match") is True)
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
        actual_two_seed = 2 * sum(int(record["new_pair_count_one_seed"]) for record in records)
        exhaustive_two_seed = int(
            (json.loads(Path(plan["progressive_pairs_manifest"]["path"]).read_text())[
                "coverage_contract"
            ]["worst_case_two_seed_pair_evaluations"])
        )
        report = {
            "schema_version": CONSENSUS_REPORT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETE",
            "output": str(output),
            "output_sha256": sha256_file(output),
            "norm_count": expected,
            "candidate_pair_count": candidate_pairs,
            "source_group_count": None,
            "external_norm_universe": plan["norm_universe"],
            "progressive_queue": True,
            "progressive_pairs_manifest": plan["progressive_pairs_manifest"],
            "dev_stop_policy": plan["dev_stop_policy"],
            "stage_records": [_artifact(Path(record["stage_record"])) if "stage_record" in record else _artifact(Path(plan["trials"][index]["stage_record"])) for index, record in enumerate(records)],
            "seeds": [
                {
                    "seed_id": seed["seed_id"],
                    "checkpoint": seed["checkpoint"],
                    "training_report": seed["training_report"]["path"],
                    "training_report_sha256": seed["training_report"]["sha256"],
                    "checkpoint_contract": seed["checkpoint_contract"],
                    "frozen_gate": {
                        "score_threshold": seed["checkpoint_contract"]["score_threshold"],
                        "top_margin_threshold": seed["checkpoint_contract"]["top_margin_threshold"],
                        "provenance": "checkpoint.dev",
                    },
                }
                for seed in plan["seeds"]
            ],
            "consensus_policy": {
                "automatic_match": "both seeds pass checkpoint.dev gates, select same leaf, and trial is dev-authorized or terminal",
                "nonmatch_decision": "ROUTE_TO_ADJUDICATION",
                "human_abstention_subtypes_created": False,
                "complete_bank_rescue": "mandatory_for_every_surviving_norm",
            },
            "metrics": {
                "overall": {
                    "norm_count": expected,
                    "automatic_match_count": match_count,
                    "automatic_match_rate": match_count / expected,
                    "provisional_abstention_count": expected - match_count,
                    "provisional_abstention_rate": (expected - match_count) / expected,
                    "routing_category_counts": dict(sorted(routing.items())),
                }
            },
            "compute": {
                "actual_two_seed_pair_evaluations": actual_two_seed,
                "exhaustive_two_seed_pair_evaluations": exhaustive_two_seed,
                "realized_pair_evaluation_reduction_rate": 1.0
                - actual_two_seed / exhaustive_two_seed,
                "coverage_or_recall_sacrificed": False,
            },
            "validation": {
                "score_probability_schema": list(CLASS_NAMES),
                "seed_norm_candidate_source_split_universes_identical": True,
                "all_score_and_metadata_hashes_verified": True,
                "all_checkpoint_artifact_hashes_verified_against_training_reports": True,
                "all_thresholds_from_checkpoint_dev": True,
                "test_threshold_tuning_performed": False,
                "all_norms_preserved": True,
                "one_terminal_ce_decision_per_norm": True,
                "all_early_exits_dev_policy_authorized": True,
                "every_survivor_reached_complete_bank": True,
            },
        }
        _write_json_new(report_path, report)
        return output, report_path, report
    finally:
        db.close()
        db_path.unlink(missing_ok=True)


def run(plan: Mapping[str, Any], queue_path: Path) -> dict[str, Any]:
    validation = validate_queue(plan, deep=True)
    output_root = Path(plan["execution"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    active_source = Path(plan["norm_universe"]["path"])
    active_original = True
    prior_merged: dict[str, Sequence[Path]] = {
        seed["seed_id"]: [] for seed in plan["seeds"]
    }
    records = []
    started = datetime.now(timezone.utc)
    for trial in plan["trials"]:
        record, prior_merged = _run_trial(
            plan,
            validation,
            trial,
            active_source=active_source,
            active_original=active_original,
            prior_merged=prior_merged,
        )
        records.append(record)
        active_source = Path(record["continued"]["path"])
        active_original = False
    final, final_report, report = _merge_final(plan, records)
    run_record = Path(plan["outputs"]["run_record"])
    if run_record.exists():
        payload = json.loads(run_record.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != RUN_SCHEMA
            or payload.get("queue_sha256") != sha256_file(queue_path)
            or payload.get("progressive_consensus_sha256") != sha256_file(final)
        ):
            raise ValueError("sealed progressive run record differs")
        return payload
    completed = datetime.now(timezone.utc)
    payload = {
        "schema_version": RUN_SCHEMA,
        "status": "COMPLETE_PROGRESSIVE_TWO_SEED_CONSENSUS",
        "task": plan["task"],
        "queue": str(queue_path.resolve()),
        "queue_sha256": sha256_file(queue_path),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "progressive_consensus": str(final),
        "progressive_consensus_sha256": sha256_file(final),
        "progressive_consensus_report": str(final_report),
        "progressive_consensus_report_sha256": sha256_file(final_report),
        "norm_count": plan["norm_count"],
        "realized_pair_evaluation_reduction_rate": report["compute"][
            "realized_pair_evaluation_reduction_rate"
        ],
        "fullbank_terminal_rescue_completed": True,
        "release_ready": False,
    }
    _write_json_new(run_record, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--shallow", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    queue_path = Path(args.queue).resolve()
    plan = json.loads(queue_path.read_text(encoding="utf-8"))
    if not args.run:
        validate_queue(plan, deep=not args.shallow)
        print(
            json.dumps(
                {
                    "status": "VALIDATED_PROGRESSIVE_NOT_LAUNCHED",
                    "queue": str(queue_path),
                    "queue_sha256": sha256_file(queue_path),
                    "task": plan["task"],
                    "norm_count": plan["norm_count"],
                },
                sort_keys=True,
            )
        )
        return
    print(json.dumps(run(plan, queue_path), sort_keys=True))


if __name__ == "__main__":
    main()
