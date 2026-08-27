#!/usr/bin/env python3
"""Freeze a task-wide, two-seed Nemotron CE production scoring queue.

The freezer is deliberately expensive and fail closed.  It streams the exact
unlabelled production pair/universe artifacts, verifies the complete task and
metric-bank contract, verifies both selected checkpoints against completed
task-local training reports, and writes commands for immutable deterministic
score shards.  It never reads an outcome label or selects/tunes a threshold.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import normalize_space, read_jsonl, sha256_file
from .gpu_host_policy import validate_gpu_indices_for_host
from .materialize_nemotron_ce_production_pairs import (
    META_SCHEMA,
    PAIR_SCHEMA,
    UNIVERSE_SCHEMA,
)
from .run_nemotron_ce import (
    pair_shard,
    verify_base_manifest,
    verify_checkpoint_contract,
)


QUEUE_SCHEMA = "silver-match-v3-frozen-nemotron-ce-two-seed-production-queue-v1"
QUEUE_STATUS = "FROZEN_READY_NOT_LAUNCHED"
FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "relation",
        "ce_label",
        "target",
        "class_label",
        "label",
        "gold_relation",
        "decision",
        "acceptable_metric_ids",
        "equivalent_metric_ids",
    }
)
IMPLEMENTATION_NAMES = (
    "aggregate_nemotron_ce_seed_consensus.py",
    "common.py",
    "freeze_nemotron_ce_production_queue.py",
    "gpu_host_policy.py",
    "materialize_nemotron_ce_production_pairs.py",
    "run_frozen_nemotron_ce_production.py",
    "run_nemotron_ce.py",
    "train_nemotron_cross_encoder.py",
)


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_ref(raw: Any, anchor: Path) -> Path:
    value = Path(str(raw or ""))
    if not str(value):
        raise ValueError("artifact reference has an empty path")
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def _verify_ref(value: Mapping[str, Any], anchor: Path, label: str) -> Path:
    path = _resolve_ref(value.get("path"), anchor)
    expected = normalize_space(value.get("sha256")).lower()
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"{label} artifact hash differs: {path}")
    return path


def _manifest_task_scope(
    report: Mapping[str, Any], report_path: Path, task: str
) -> tuple[list[str], set[str]]:
    manifest_ref = report.get("manifest")
    bank_ref = report.get("bank")
    if not isinstance(manifest_ref, Mapping) or not isinstance(bank_ref, Mapping):
        raise ValueError("production-pair report lacks manifest/bank references")
    manifest_path = _verify_ref(manifest_ref, report_path, "canonical manifest")
    bank_path = _verify_ref(bank_ref, report_path, "metric bank")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    manifest_bank = (manifest.get("banks") or {}).get(task)
    if not isinstance(manifest_bank, Mapping):
        raise ValueError(f"task is absent from canonical manifest: {task}")
    routed_bank = _resolve_ref(manifest_bank.get("path"), manifest_path)
    bank_source = normalize_space(bank_ref.get("source_sha256"))
    metrics = bank.get("metrics") or []
    metric_ids = [normalize_space(row.get("metric_id")) for row in metrics]
    if (
        routed_bank != bank_path
        or normalize_space(manifest_bank.get("source_sha256")) != bank_source
        or normalize_space(bank.get("source_sha256")) != bank_source
        or int(manifest_bank.get("count", -1)) != len(metrics)
        or int(bank_ref.get("metric_count", -1)) != len(metrics)
        or not metric_ids
        or "" in metric_ids
        or len(metric_ids) != len(set(metric_ids))
    ):
        raise ValueError("metric-bank task/count/provenance contract differs")
    corpus_order = [
        corpus
        for corpus, meta in (manifest.get("corpora") or {}).items()
        if isinstance(meta, Mapping) and meta.get("task") == task
    ]
    if not corpus_order or list(report.get("corpus_order") or []) != corpus_order:
        raise ValueError("production report does not cover the manifest task corpus order")
    report_corpora = report.get("corpora") or {}
    if set(report_corpora) != set(corpus_order):
        raise ValueError("production report corpus set differs from canonical task scope")
    expected_norms = 0
    for corpus in corpus_order:
        canonical_meta = manifest["corpora"][corpus]
        canonical_report = (report_corpora[corpus] or {}).get("canonical") or {}
        canonical_path = _verify_ref(canonical_report, report_path, f"canonical {corpus}")
        expected_path = _resolve_ref(canonical_meta.get("path"), manifest_path)
        expected_count = int(canonical_meta.get("count", -1))
        if canonical_path != expected_path or int(canonical_report.get("count", -1)) != expected_count:
            raise ValueError(f"canonical corpus routing/count differs: {corpus}")
        candidate = (report_corpora[corpus] or {}).get("candidate_union") or {}
        candidate_path = _verify_ref(candidate, report_path, f"candidate union {corpus}")
        candidate_meta_path = _verify_ref(
            {"path": candidate.get("meta"), "sha256": candidate.get("meta_sha256")},
            report_path,
            f"candidate union metadata {corpus}",
        )
        candidate_meta = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
        lanes = ((candidate_meta.get("union") or {}).get("lanes") or [])
        lane_names = [normalize_space(row.get("name")) for row in lanes]
        if (
            candidate_meta.get("output_sha256") != sha256_file(candidate_path)
            or candidate_meta.get("manifest_sha256") != sha256_file(manifest_path)
            or normalize_space(candidate_meta.get("task")) != task
            or normalize_space(candidate_meta.get("corpus")) != corpus
            or normalize_space(candidate_meta.get("bank_source_sha256")) != bank_source
            or int(candidate_meta.get("input_count", -1)) != expected_count
            or int(candidate_meta.get("output_k", -1)) != int(candidate.get("output_k", -2))
            or len(lanes) < 2
            or lane_names != list(candidate.get("lane_names") or [])
        ):
            raise ValueError(f"candidate union is not a complete multi-lane task artifact: {corpus}")
        expected_norms += expected_count
    if int(report.get("norm_count", -1)) != expected_norms:
        raise ValueError("production report norm_count differs from canonical task scope")
    return corpus_order, set(metric_ids)


def validate_production_pair_report(
    report_path: Path, *, expected_task: str, num_shards: int
) -> dict[str, Any]:
    """Deeply validate the exact unlabeled scoring universe and shard counts."""

    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != META_SCHEMA
        or report.get("status")
        != "FROZEN_COMPLETE_UNLABELED_PRODUCTION_PAIR_UNIVERSE"
        or normalize_space(report.get("task")) != expected_task
        or report.get("labels_present") is not False
        or report.get("single_lane_candidates_accepted") is not False
        or report.get("diagnostic_subset_accepted") is not False
        or report.get("release_ready") is not False
    ):
        raise ValueError("production-pair report schema/status/safety contract failed")
    corpus_order, bank_ids = _manifest_task_scope(report, report_path, expected_task)
    pairs_ref = report.get("pairs")
    universe_ref = report.get("norm_universe")
    if not isinstance(pairs_ref, Mapping) or not isinstance(universe_ref, Mapping):
        raise ValueError("production report lacks pair/universe artifacts")
    pair_path = _verify_ref(pairs_ref, report_path, "production pairs")
    universe_path = _verify_ref(universe_ref, report_path, "production norm universe")
    depth = int(report.get("candidate_depth", -1))
    norm_count = int(report.get("norm_count", -1))
    pair_count = int(report.get("pair_count", -1))
    if depth < 1 or norm_count < 1 or pair_count != norm_count * depth:
        raise ValueError("production pair count/depth arithmetic differs")
    if int(universe_ref.get("count", -1)) != norm_count:
        raise ValueError("norm-universe count differs from production report")

    universe_iter = iter(read_jsonl(universe_path))
    pair_iter = iter(read_jsonl(pair_path))
    seen_uids: set[str] = set()
    corpus_counts: Counter[str] = Counter()
    shard_pair_counts = [0] * num_shards
    shard_norm_counts = [0] * num_shards
    for position in range(norm_count):
        try:
            universe = next(universe_iter)
        except StopIteration as exc:
            raise ValueError("norm universe ends before its frozen count") from exc
        uid = normalize_space(universe.get("norm_uid"))
        corpus = normalize_space(universe.get("corpus"))
        source_group = normalize_space(universe.get("source_group"))
        if (
            universe.get("schema_version") != UNIVERSE_SCHEMA
            or universe.get("task") != expected_task
            or corpus not in corpus_order
            or universe.get("split") != "production"
            or not uid
            or uid in seen_uids
            or not source_group
        ):
            raise ValueError(f"invalid/duplicate production universe row {position}: {uid}")
        seen_uids.add(uid)
        corpus_counts[corpus] += 1
        shard_id = pair_shard(uid, num_shards)
        shard_norm_counts[shard_id] += 1
        metric_ids: set[str] = set()
        for rank in range(1, depth + 1):
            try:
                pair = next(pair_iter)
            except StopIteration as exc:
                raise ValueError("production pairs end before frozen pair_count") from exc
            metric_id = normalize_space(pair.get("metric_id"))
            forbidden = FORBIDDEN_LABEL_FIELDS.intersection(pair)
            if (
                forbidden
                or pair.get("schema_version") != PAIR_SCHEMA
                or pair.get("task") != expected_task
                or pair.get("corpus") != corpus
                or normalize_space(pair.get("norm_uid")) != uid
                or normalize_space(pair.get("source_group")) != source_group
                or pair.get("split") != "production"
                or int(pair.get("candidate_rank", -1)) != rank
                or metric_id not in bank_ids
                or metric_id in metric_ids
                or not normalize_space(pair.get("query"))
                or not normalize_space(pair.get("metric_card"))
                or normalize_space(pair.get("current_bank_source_sha256"))
                != normalize_space((report.get("bank") or {}).get("source_sha256"))
            ):
                raise ValueError(
                    f"production pair contract/leakage failed: {uid}/{rank}/{sorted(forbidden)}"
                )
            metric_ids.add(metric_id)
            shard_pair_counts[shard_id] += 1
    try:
        next(universe_iter)
        raise ValueError("norm universe has rows beyond frozen count")
    except StopIteration:
        pass
    try:
        next(pair_iter)
        raise ValueError("production pairs have rows beyond frozen pair_count")
    except StopIteration:
        pass
    expected_by_corpus = {
        corpus: int(((report.get("corpora") or {})[corpus]["canonical"])["count"])
        for corpus in corpus_order
    }
    if dict(corpus_counts) != expected_by_corpus or sum(shard_pair_counts) != pair_count:
        raise ValueError("production universe corpus/shard counts differ")
    return {
        "report": _artifact(report_path),
        "pairs": _artifact(pair_path),
        "norm_universe": _artifact(universe_path),
        "task": expected_task,
        "corpus_order": corpus_order,
        "norm_count": norm_count,
        "candidate_depth": depth,
        "pair_count": pair_count,
        "shard_pair_counts": shard_pair_counts,
        "shard_norm_counts": shard_norm_counts,
        "labels_present": False,
        "all_rows_production_split": True,
    }


def _validate_task_local_training_report(
    report_path: Path,
    checkpoint: Path,
    *,
    task: str,
    model: Path,
    expected_seed_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report_path = report_path.resolve()
    checkpoint = checkpoint.resolve()
    report_sha = sha256_file(report_path)
    contract = verify_checkpoint_contract(
        checkpoint, report_path, report_sha, model=model
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report.get("selected_checkpoint") or {}
    if Path(str(selected.get("path") or "")).resolve() != checkpoint:
        raise ValueError("passed checkpoint is not the report-selected checkpoint path")
    run_config_path = report_path.parent / "run_config.json"
    run_config_ref = ((report.get("input_sha256") or {}).get("run_config"))
    if (
        not run_config_path.is_file()
        or sha256_file(run_config_path) != run_config_ref
    ):
        raise ValueError("training report does not bind its adjacent run_config.json")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    seed_value = str(run_config.get("seed"))
    if (
        run_config.get("schema_version") != report.get("schema_version")
        or Path(str(run_config.get("model") or "")).resolve() != model.resolve()
        or int(run_config.get("max_length", -1)) != int(report.get("max_sequence_length", -2))
        or seed_value != expected_seed_id
    ):
        raise ValueError(f"declared seed ID differs from task training run: {expected_seed_id}/{seed_value}")
    input_hashes = report.get("input_sha256") or {}
    for split_key in ("train_pairs", "dev_pairs"):
        report_inputs = input_hashes.get(split_key)
        config_inputs = run_config.get(split_key)
        if not isinstance(report_inputs, Mapping) or not report_inputs:
            raise ValueError(f"training report lacks explicit task-local {split_key}")
        if report_inputs != config_inputs:
            raise ValueError(f"training report/run config {split_key} bindings differ")
        for raw_path, expected_hash in report_inputs.items():
            path = Path(raw_path).resolve()
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"task-local training pair artifact changed: {path}")
            count = 0
            for line_no, row in enumerate(read_jsonl(path), 1):
                if normalize_space(row.get("task")) != task:
                    raise ValueError(f"foreign/missing task in {path}:{line_no}")
                count += 1
            if not count:
                raise ValueError(f"empty task-local training pair artifact: {path}")
    return contract, report, run_config


def _safe_seed_id(value: str) -> str:
    value = normalize_space(value)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(f"seed ID is empty or unsafe for an artifact path: {value!r}")
    return value


def _score_command(
    *,
    python: Path,
    pairs: Path,
    output: Path,
    model: Path,
    base_manifest: Path,
    base_manifest_sha: str,
    checkpoint: Path,
    training_report: Path,
    training_report_sha: str,
    batch_size: int,
    max_length: int,
    shard_id: int,
    num_shards: int,
    attention: str,
) -> list[str]:
    return [
        str(python), "-u", "-m", "scripts.tools.silver_match_v3.run_nemotron_ce",
        "score", "--input-pairs", str(pairs), "--output", str(output),
        "--model", str(model), "--base-manifest", str(base_manifest),
        "--base-manifest-sha256", base_manifest_sha, "--checkpoint", str(checkpoint),
        "--training-report", str(training_report), "--training-report-sha256",
        training_report_sha, "--batch-size", str(batch_size), "--max-length",
        str(max_length), "--device", "0", "--shard-id", str(shard_id),
        "--num-shards", str(num_shards), "--attention", attention,
    ]


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    task = normalize_space(args.task)
    if not task:
        raise ValueError("task must not be empty")
    num_shards = int(args.num_shards)
    batch_size = int(args.batch_size)
    if num_shards < 2 or batch_size < 1:
        raise ValueError("num_shards must be >=2 and batch_size must be positive")
    seed_ids = [_safe_seed_id(value) for value in args.seed_id]
    reports = [Path(value).resolve() for value in args.training_report]
    checkpoints = [Path(value).resolve() for value in args.checkpoint]
    if len(seed_ids) != 2 or len(reports) != 2 or len(checkpoints) != 2:
        raise ValueError("exactly two --seed-id/--training-report/--checkpoint values are required")
    if len(set(seed_ids)) != 2 or len(set(reports)) != 2 or len(set(checkpoints)) != 2:
        raise ValueError("the two seed/report/checkpoint identities must be distinct")
    target_host = normalize_space(args.target_host).split(".", 1)[0]
    gpus = list(validate_gpu_indices_for_host(args.gpu_index, hostname=target_host))
    repo_root = Path(args.repo_root).resolve()
    python = Path(args.python)
    if not python.is_absolute():
        python = python.absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    model = Path(args.model).resolve()
    base_manifest = Path(args.base_manifest).resolve()
    base_manifest_sha = sha256_file(base_manifest)
    base_contract = verify_base_manifest(model, base_manifest, base_manifest_sha)
    production = validate_production_pair_report(
        Path(args.pair_report), expected_task=task, num_shards=num_shards
    )
    pairs = Path(production["pairs"]["path"])
    seed_contracts: list[dict[str, Any]] = []
    run_configs: list[dict[str, Any]] = []
    for seed_id, report_path, checkpoint in zip(seed_ids, reports, checkpoints, strict=True):
        contract, _, run_config = _validate_task_local_training_report(
            report_path,
            checkpoint,
            task=task,
            model=model,
            expected_seed_id=seed_id,
        )
        seed_contracts.append(contract)
        run_configs.append(run_config)
    if run_configs[0].get("train_pairs") != run_configs[1].get("train_pairs") or run_configs[0].get("dev_pairs") != run_configs[1].get("dev_pairs"):
        raise ValueError("two seeds were not trained on identical task-local train/dev inputs")
    fingerprints = {
        (
            value["checkpoint_metadata_sha256"],
            value["head_sha256"],
            value["adapter_tree_sha256"],
        )
        for value in seed_contracts
    }
    if len(fingerprints) != 2:
        raise ValueError("two seed reports select identical checkpoint content")

    output_root = Path(args.output_root).resolve()
    run_record = output_root / "production_run.json"
    consensus = output_root / "consensus.jsonl"
    consensus_report = output_root / "consensus.report.json"
    consensus_log = output_root / "consensus.report.json.log"
    seed_manifest_dir = output_root / "seed-manifests"
    planned_seeds: list[dict[str, Any]] = []
    all_targets: list[Path] = [run_record, consensus, consensus_report, consensus_log]
    job_ordinal = 0
    for seed_id, report_path, checkpoint, contract in zip(
        seed_ids, reports, checkpoints, seed_contracts, strict=True
    ):
        seed_root = output_root / f"seed-{seed_id}"
        shards = []
        for shard_id in range(num_shards):
            shard_output = seed_root / "shards" / f"part-{shard_id:05d}-of-{num_shards:05d}.scores.jsonl"
            meta = shard_output.with_suffix(shard_output.suffix + ".meta.json")
            log = shard_output.with_suffix(shard_output.suffix + ".log")
            gpu = gpus[job_ordinal % len(gpus)]
            job_ordinal += 1
            command = _score_command(
                python=python,
                pairs=pairs,
                output=shard_output,
                model=model,
                base_manifest=base_manifest,
                base_manifest_sha=base_manifest_sha,
                checkpoint=checkpoint,
                training_report=report_path,
                training_report_sha=sha256_file(report_path),
                batch_size=batch_size,
                max_length=int(contract["max_sequence_length"]),
                shard_id=shard_id,
                num_shards=num_shards,
                attention=args.attention,
            )
            shards.append(
                {
                    "shard_id": shard_id,
                    "num_shards": num_shards,
                    "physical_gpu": gpu,
                    "expected_pair_count": production["shard_pair_counts"][shard_id],
                    "expected_norm_count": production["shard_norm_counts"][shard_id],
                    "output": str(shard_output),
                    "meta": str(meta),
                    "log": str(log),
                    "command": command,
                }
            )
            all_targets.extend((shard_output, meta, log))
        merged = seed_root / "merged.scores.jsonl"
        merged_meta = merged.with_suffix(merged.suffix + ".meta.json")
        merge_log = merged.with_suffix(merged.suffix + ".log")
        merge_command = [
            str(python), "-u", "-m", "scripts.tools.silver_match_v3.run_nemotron_ce",
            "merge", "--inputs", *[row["output"] for row in shards], "--output", str(merged),
        ]
        all_targets.extend((merged, merged_meta, merge_log))
        planned_seeds.append(
            {
                "seed_id": seed_id,
                "training_report": _artifact(report_path),
                "run_config": _artifact(report_path.parent / "run_config.json"),
                "checkpoint": str(checkpoint),
                "checkpoint_contract": contract,
                "shards": shards,
                "merged": {"scores": str(merged), "meta": str(merged_meta), "log": str(merge_log), "command": merge_command},
            }
        )
    existing = [str(path) for path in all_targets if path.exists()]
    if seed_manifest_dir.exists():
        existing.extend(str(path) for path in seed_manifest_dir.glob("*.json"))
    if existing:
        raise FileExistsError(f"refusing to freeze over production runtime artifacts: {existing[:5]}")

    implementations = {
        name: _artifact(repo_root / "scripts" / "tools" / "silver_match_v3" / name)
        for name in IMPLEMENTATION_NAMES
    }
    packages = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "peft", "safetensors", "torch", "transformers")
    }
    return {
        "schema_version": QUEUE_SCHEMA,
        "status": QUEUE_STATUS,
        "task": task,
        "production_pairs": production,
        "base_model": {
            "path": str(model),
            "manifest": _artifact(base_manifest),
            "verified_contract": base_contract,
        },
        "seeds": planned_seeds,
        "execution": {
            "target_host": target_host,
            "physical_gpus": gpus,
            "repo_root": str(repo_root),
            "python": str(python),
            "python_version": platform.python_version(),
            "packages": packages,
            "score_parallelism": "at_most_one_shard_process_per_physical_gpu_per_wave",
            "num_shards_per_seed": num_shards,
        },
        "scoring": {
            "batch_size": batch_size,
            "attention": args.attention,
            "logical_cuda_device": 0,
            "deterministic_norm_sharding": True,
        },
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": f"{repo_root / 'vendor'}:{repo_root}",
        },
        "implementations": implementations,
        "outputs": {
            "root": str(output_root),
            "seed_manifest_directory": str(seed_manifest_dir),
            "seed_manifest_filename_rule": "<sha256-of-exact-json-bytes>.json",
            "consensus": str(consensus),
            "consensus_report": str(consensus_report),
            "consensus_log": str(consensus_log),
            "run_record": str(run_record),
        },
        "safety": {
            "production_labels_present": False,
            "all_norms_and_pairs_task_local": True,
            "training_reports_complete_task_local": True,
            "thresholds_reused_only_from_each_checkpoint_dev": True,
            "threshold_retuning_permitted": False,
            "external_outcomes_opened": False,
            "consensus_requires_two_seed_same_leaf_and_both_frozen_gates": True,
            "nonmatches_route_to_adjudication": True,
            "human_abstention_subtypes_created_by_ce": False,
            "release_ready": False,
        },
    }


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pair-report", required=True)
    parser.add_argument("--seed-id", action="append", required=True)
    parser.add_argument("--training-report", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--gpu-index", action="append", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attention", choices=("auto", "eager", "sdpa"), default="eager")
    parser.add_argument("--output", required=True, help="create-only frozen queue JSON")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    plan = freeze(args)
    output = Path(args.output).resolve()
    _exclusive_json(output, plan)
    print(json.dumps({"status": QUEUE_STATUS, "queue": str(output), "queue_sha256": sha256_file(output), "task": plan["task"], "norm_count": plan["production_pairs"]["norm_count"], "pair_count": plan["production_pairs"]["pair_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
