#!/usr/bin/env python3
"""Freeze a fail-closed two-seed progressive CE production queue.

The queue binds disjoint complete-bank pair tiers, two distinct task-local CE
checkpoints, and an untouched-development early-stop policy.  It contains no
outcome labels and cannot authorize an exit that the policy did not freeze.
Runtime artifacts are create-only and resume only at verified shard/trial
boundaries.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit_progressive_nemotron_ce_dev_policy import SCHEMA as POLICY_SCHEMA
from .audit_progressive_nemotron_ce_dev_policy import STATUS as POLICY_STATUS
from .common import normalize_space, sha256_file
from .freeze_nemotron_ce_production_queue import (
    _artifact,
    _safe_seed_id,
    _validate_task_local_training_report,
)
from .gpu_host_policy import validate_gpu_indices_for_host
from .materialize_progressive_nemotron_ce_pairs import (
    MANIFEST_SCHEMA,
    STATUS as PAIR_STATUS,
)
from .run_nemotron_ce import verify_base_manifest


QUEUE_SCHEMA = "silver-match-v3-progressive-nemotron-ce-production-queue-v1"
QUEUE_STATUS = "FROZEN_PROGRESSIVE_READY_NOT_LAUNCHED"
IMPLEMENTATIONS = (
    "aggregate_nemotron_ce_seed_consensus.py",
    "audit_progressive_nemotron_ce_dev_policy.py",
    "freeze_progressive_nemotron_ce_production_queue.py",
    "materialize_progressive_nemotron_ce_pairs.py",
    "run_frozen_progressive_nemotron_ce_production.py",
    "run_nemotron_ce.py",
)


def _verify_ref(ref: Mapping[str, Any], label: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(ref.get("size_bytes", path.stat().st_size if path.is_file() else -1))
        or sha256_file(path) != ref.get("sha256")
    ):
        raise ValueError(f"{label} artifact changed: {path}")
    return path


def validate_progressive_inputs(
    manifest_path: Path,
    policy_path: Path,
    *,
    task: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    policy_path = policy_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    trials = manifest.get("trials") or []
    trial_ids = [normalize_space(row.get("trial_id")) for row in trials]
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != PAIR_STATUS
        or manifest.get("task") != task
        or not trials
        or "" in trial_ids
        or len(trial_ids) != len(set(trial_ids))
        or [int(row.get("ordinal", -1)) for row in trials]
        != list(range(1, len(trials) + 1))
        or sum(int((row.get("pairs") or {}).get("count", -1)) for row in trials)
        != int((manifest.get("coverage_contract") or {}).get("total_pair_count", -2))
        or (manifest.get("coverage_contract") or {}).get(
            "union_equals_complete_bank_for_every_norm"
        )
        is not True
        or (manifest.get("coverage_contract") or {}).get(
            "candidate_omission_count_after_terminal_trial"
        )
        != 0
        or sum(row.get("terminal") is True for row in trials) != 1
        or trials[-1].get("terminal") is not True
    ):
        raise ValueError("progressive pair manifest coverage contract failed")
    for trial in trials:
        _verify_ref(trial.get("pairs") or {}, f"trial {trial.get('trial_id')}")
    source = manifest.get("source") or {}
    for name in (
        "primary_report",
        "primary_pairs",
        "primary_candidates",
        "primary_candidates_meta",
        "fullbank_report",
        "fullbank_pairs",
        "norm_universe",
        "bank",
    ):
        _verify_ref(source.get(name) or {}, f"progressive source {name}")

    authorized = list(policy.get("authorized_early_stop_trials") or [])
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or policy.get("status") != POLICY_STATUS
        or policy.get("task") != task
        or policy.get("selection_split") != "dev"
        or list(policy.get("trial_order") or []) != trial_ids
        or policy.get("terminal_trial_id") != trial_ids[-1]
        or int(policy.get("terminal_complete_bank_depth", -1))
        != int(manifest.get("fullbank_depth", -2))
        or not set(authorized) <= set(trial_ids[:-1])
        or len(authorized) != len(set(authorized))
        or (policy.get("safety") or {}).get("test_or_blind_labels_read") is not False
        or (policy.get("safety") or {}).get("training_labels_used_for_stop_selection")
        is not False
        or (policy.get("safety") or {}).get(
            "all_thresholds_from_training_reports_checkpoint_dev"
        )
        is not True
        or (policy.get("safety") or {}).get("complete_bank_terminal_coverage") is not True
    ):
        raise ValueError("progressive early-stop policy contract failed")
    audits = {row.get("trial_id"): row for row in policy.get("trial_audits") or []}
    if set(audits) != set(trial_ids) or any(
        (trial_id in authorized) != (audits[trial_id].get("authorized_for_early_stop") is True)
        for trial_id in trial_ids[:-1]
    ):
        raise ValueError("authorized early-stop trials differ from dev audit rows")
    return manifest, policy


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    task = normalize_space(args.task)
    if not task:
        raise ValueError("task must not be empty")
    manifest_path = Path(args.progressive_pairs_manifest).resolve()
    policy_path = Path(args.dev_stop_policy).resolve()
    manifest, policy = validate_progressive_inputs(
        manifest_path, policy_path, task=task
    )
    seed_ids = [_safe_seed_id(value) for value in args.seed_id]
    reports = [Path(value).resolve() for value in args.training_report]
    checkpoints = [Path(value).resolve() for value in args.checkpoint]
    if (
        len(seed_ids) != 2
        or len(set(seed_ids)) != 2
        or len(reports) != 2
        or len(set(reports)) != 2
        or len(checkpoints) != 2
        or len(set(checkpoints)) != 2
    ):
        raise ValueError("exactly two distinct seeds, reports, and checkpoints are required")
    repo_root = Path(args.repo_root).resolve()
    python = Path(args.python).resolve()
    model = Path(args.model).resolve()
    base_manifest = Path(args.base_manifest).resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    base_sha = sha256_file(base_manifest)
    base_contract = verify_base_manifest(model, base_manifest, base_sha)
    seed_rows = []
    run_configs = []
    fingerprints = set()
    for seed_id, report, checkpoint in zip(seed_ids, reports, checkpoints, strict=True):
        contract, _, run_config = _validate_task_local_training_report(
            report,
            checkpoint,
            task=task,
            model=model,
            expected_seed_id=seed_id,
        )
        seed_rows.append(
            {
                "seed_id": seed_id,
                "training_report": _artifact(report),
                "run_config": _artifact(report.parent / "run_config.json"),
                "checkpoint": str(checkpoint),
                "checkpoint_contract": contract,
            }
        )
        run_configs.append(run_config)
        fingerprints.add(
            (
                contract["checkpoint_metadata_sha256"],
                contract["head_sha256"],
                contract["adapter_tree_sha256"],
            )
        )
    if (
        len(fingerprints) != 2
        or run_configs[0].get("train_pairs") != run_configs[1].get("train_pairs")
        or run_configs[0].get("dev_pairs") != run_configs[1].get("dev_pairs")
    ):
        raise ValueError("two seeds do not represent distinct fits of identical task-local data")
    target_host = normalize_space(args.target_host).split(".", 1)[0]
    gpus = list(validate_gpu_indices_for_host(args.gpu_index, hostname=target_host))
    if int(args.num_shards) < 2 or int(args.batch_size) < 1:
        raise ValueError("num_shards must be >=2 and batch_size positive")
    output_root = Path(args.output_root).resolve()
    queue_output = Path(args.output).resolve()
    if output_root.exists() or queue_output.exists():
        raise FileExistsError("refusing to freeze over progressive runtime/queue artifacts")
    implementations = {
        name: _artifact(repo_root / "scripts" / "tools" / "silver_match_v3" / name)
        for name in IMPLEMENTATIONS
    }
    packages = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "peft", "safetensors", "torch", "transformers")
    }
    trials = []
    for trial in manifest["trials"]:
        trial_id = trial["trial_id"]
        trial_root = output_root / f"trial-{int(trial['ordinal']):02d}-{trial_id}"
        trials.append(
            {
                **trial,
                "early_stop_authorized": trial_id
                in set(policy["authorized_early_stop_trials"]),
                "runtime_root": str(trial_root),
                "stage_record": str(trial_root / "STAGE_COMPLETE.json"),
            }
        )
    plan = {
        "schema_version": QUEUE_SCHEMA,
        "status": QUEUE_STATUS,
        "task": task,
        "progressive_pairs_manifest": _artifact(manifest_path),
        "dev_stop_policy": _artifact(policy_path),
        "norm_universe": manifest["source"]["norm_universe"],
        "bank": {
            **manifest["source"]["bank"],
            "source_sha256": manifest["source"]["bank_source_sha256"],
        },
        "norm_count": manifest["norm_count"],
        "trials": trials,
        "seeds": seed_rows,
        "base_model": {
            "path": str(model),
            "manifest": _artifact(base_manifest),
            "verified_contract": base_contract,
        },
        "execution": {
            "target_host": target_host,
            "physical_gpus": gpus,
            "repo_root": str(repo_root),
            "python": str(python),
            "python_version": platform.python_version(),
            "packages": packages,
            "num_shards_per_seed": int(args.num_shards),
            "batch_size": int(args.batch_size),
            "attention": args.attention,
            "output_root": str(output_root),
        },
        "implementations": implementations,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": f"{repo_root / 'vendor'}:{repo_root}",
        },
        "outputs": {
            "progressive_consensus": str(output_root / "progressive-consensus.jsonl"),
            "progressive_consensus_report": str(
                output_root / "progressive-consensus.report.json"
            ),
            "run_record": str(output_root / "RUN_COMPLETE.json"),
        },
        "compute_contract": {
            "estimated_pair_evaluation_reduction_rate": (
                policy.get("estimated_compute") or {}
            ).get("estimated_pair_evaluation_reduction_rate"),
            "estimate_basis": (policy.get("estimated_compute") or {}).get("basis"),
            "worst_case_two_seed_pair_evaluations": 2
            * int((manifest.get("coverage_contract") or {}).get("total_pair_count", -1)),
            "worst_case_reduction_rate": 0.0,
            "coverage_or_recall_sacrificed_for_compute": False,
        },
        "safety": {
            "production_labels_present": False,
            "threshold_retuning_permitted": False,
            "test_or_blind_outcomes_read": False,
            "early_exit_requires_two_seed_same_leaf_and_both_checkpoint_dev_gates": True,
            "early_exit_requires_dev_policy_authorization": True,
            "every_disagreement_abstention_or_unauthorized_match_continues": True,
            "fullbank_terminal_rescue_mandatory": True,
            "one_terminal_ce_decision_per_norm": True,
            "human_abstention_subtypes_created_by_ce": False,
            "release_ready": False,
        },
    }
    queue_output.parent.mkdir(parents=True, exist_ok=True)
    with queue_output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return plan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--progressive-pairs-manifest", required=True)
    parser.add_argument("--dev-stop-policy", required=True)
    parser.add_argument("--seed-id", action="append", required=True)
    parser.add_argument("--training-report", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--gpu-index", action="append", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--attention", choices=("auto", "eager", "sdpa"), default="eager")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    plan = freeze(args)
    output = Path(args.output).resolve()
    print(
        json.dumps(
            {
                "status": plan["status"],
                "queue": str(output),
                "queue_sha256": sha256_file(output),
                "task": plan["task"],
                "norm_count": plan["norm_count"],
                "estimated_pair_evaluation_reduction_rate": plan["compute_contract"]["estimated_pair_evaluation_reduction_rate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
