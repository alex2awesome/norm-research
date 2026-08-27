#!/usr/bin/env python3
"""Freeze an append-only task policy for the metric-balanced CE objective."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import sha256_file
from .train_cross_encoder_balanced import (
    BALANCED_OBJECTIVE_REVISION,
    BALANCED_SCHEMA,
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


def _task_seed(task: str) -> int:
    return 61000 + int.from_bytes(hashlib.sha256(task.encode()).digest()[:2], "big") % 20000


def freeze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = Path(args.source_policy).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if args.task not in (source.get("scope") or []):
        raise ValueError("task is outside source policy scope")
    if source.get("schema_version") not in {
        "silver-match-v3-cross-encoder-alltask-policy-v1",
        "silver-match-v3-cross-encoder-press-releases-policy-v2",
    }:
        raise ValueError("unsupported source policy schema")
    source_eligibility_path = source_path.with_suffix(".ELIGIBILITY.json")
    if not source_eligibility_path.is_file():
        raise FileNotFoundError(source_eligibility_path)
    source_eligibility = json.loads(
        source_eligibility_path.read_text(encoding="utf-8")
    )
    if (
        source_eligibility.get("policy_sha256") != sha256_file(source_path)
        or args.task not in source_eligibility.get("eligible_primary_tasks", [])
    ):
        raise ValueError("source policy eligibility does not authorize task")

    repo_root = Path(args.repo_root).resolve()
    base_trainer = repo_root / "scripts/tools/silver_match_v3/train_cross_encoder.py"
    balanced_trainer = (
        repo_root / "scripts/tools/silver_match_v3/train_cross_encoder_balanced.py"
    )
    selector = repo_root / "scripts/tools/silver_match_v3/select_cross_encoder_variants.py"
    freezer = Path(__file__).resolve()
    for path in (base_trainer, balanced_trainer, selector, freezer):
        if not path.is_file():
            raise FileNotFoundError(path)
    failures = [_artifact(Path(value)) for value in args.failure_report]
    if not failures:
        raise ValueError("at least one pointwise-collapse report is required")
    for artifact in failures:
        report = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
        if report.get("status") not in {"REJECTED_DEV_GATE", "REJECTED_VALIDATION_GATE"}:
            raise ValueError("failure report is not a rejected CE run")
        if report.get("frozen_test_consumed") is not False:
            raise ValueError("failure report consumed a sealed test/blind role")

    policy = copy.deepcopy(source)
    base_seed = _task_seed(args.task)
    prefix = args.task.replace("-", "_") + "-balanced-v4"
    policy.update(
        {
            "scope": [args.task],
            "policy_revision": f"{args.task}-metric-balanced-ce-v4",
            "balanced_objective_revision": BALANCED_OBJECTIVE_REVISION,
            "status": "FROZEN_AFTER_POINTWISE_COLLAPSE_DIAGNOSTIC_BEFORE_BALANCED_TRAINING",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "development_evidence_status": "ADAPTIVE_AFTER_POINTWISE_CE_COLLAPSE_DIAGNOSTIC",
            "blind_status": "SEALED_UNCONSUMED",
            "source_policy": _artifact(source_path),
            "source_policy_eligibility": _artifact(source_eligibility_path),
            "pointwise_collapse_evidence": failures,
            "predeclared_variants": [
                {
                    "name": f"{prefix}-seed{base_seed}-lr5e6",
                    "seed": base_seed,
                    "learning_rate": 5e-6,
                },
                {
                    "name": f"{prefix}-seed{base_seed + 18}-lr1e5",
                    "seed": base_seed + 18,
                    "learning_rate": 1e-5,
                },
                {
                    "name": f"{prefix}-seed{base_seed + 36}-lr15e6",
                    "seed": base_seed + 36,
                    "learning_rate": 1.5e-5,
                },
            ],
            "fixed_training": {
                **policy["fixed_training"],
                "task_specific_model": True,
                "epochs": 2,
                "batch_size": 32,
                "eval_batch_size": 512,
                "max_length": 512,
                "warmup_ratio": 0.1,
                "negatives_per_positive": 8,
                "negatives_per_abstain": 8,
                "strong_positive_repeats": 2,
                "candidate_depth": 50,
            },
            "balanced_training": {
                "schema_version": BALANCED_SCHEMA,
                "sampling_seed": base_seed + 101,
                "max_unique_positive_uids_per_metric": 64,
                "hard_negatives_per_match": 2,
                "global_balanced_negatives_per_match": 6,
                "hard_negatives_per_abstain": 2,
                "global_balanced_negatives_per_abstain": 6,
                "min_negative_exposure_per_bank_metric": 64,
                "target_negative_to_positive_pair_ratio": 2.0,
                "minimum_negative_to_positive_pair_ratio": 2.0,
                "maximum_positive_pair_fraction_per_metric": 0.3333333333333333,
                "positive_cap_selection": "seeded_sha256_without_replacement",
                "global_negative_selection": "largest_normalized_exposure_deficit_then_seeded_sha256",
                "exposure_gate_timing": "before_model_initialization",
            },
            "role_contract": {
                **policy["role_contract"],
                "dev": "adaptive development only after pointwise-collapse diagnosis",
                "test": "unavailable to training and model selection",
                "blind": "sealed uniform final-production match and false-abstention audits only",
                "mi_or_outcome_fields_allowed": False,
            },
            "evaluation_contract": {
                "unit": "canonical norm_uid",
                "candidate_universe": "complete frozen task bank",
                "primary_ranking": "full-bank top-1 exact metric ID",
                "abstention": "joint top score and top1-top2 margin gate",
                "macro_metric_recall_at": [1, 5, 10, 16, 30, 50],
                "prediction_concentration_audit": True,
                "pair_level_random_split_metrics_are_primary": False,
            },
            "implementation": {
                **(policy.get("implementation") or {}),
                "train_cross_encoder_path": str(base_trainer.relative_to(repo_root)),
                "train_cross_encoder_sha256": sha256_file(base_trainer),
                "balanced_train_cross_encoder_path": str(
                    balanced_trainer.relative_to(repo_root)
                ),
                "balanced_train_cross_encoder_sha256": sha256_file(
                    balanced_trainer
                ),
                "base_train_cross_encoder_sha256": sha256_file(base_trainer),
                "select_cross_encoder_variants_path": str(selector.relative_to(repo_root)),
                "select_cross_encoder_variants_sha256": sha256_file(selector),
                "freeze_metric_balanced_ce_policy_path": str(
                    freezer.relative_to(repo_root)
                ),
                "freeze_metric_balanced_ce_policy_sha256": sha256_file(freezer),
            },
        }
    )
    production = policy.setdefault("production_contract", {})
    production.update(
        {
            "cross_encoder_output_is_final_by_itself": False,
            "typed_abstention_cannot_be_inferred_from_ce_score_alone": True,
            "automatic_ce_proposal_requires_same_exact_metric_id": True,
            "automatic_ce_proposal_requires_two_independently_trained_eligible_variants": True,
            "all_other_rows_flow_to_llm_or_full_bank_rescue": True,
            "uniform_blind_false_abstention_audit_required": True,
            "uniform_blind_final_match_precision_audit_required": True,
        }
    )
    policy.pop("policy_sha256", None)
    eligibility = {
        "schema_version": "silver-match-v4-metric-balanced-policy-eligibility-v1",
        "status": "FROZEN_TASK_ELIGIBILITY",
        "eligible_primary_tasks": [args.task],
        "ineligible_tasks": [],
        "basis": "task inherited source-policy eligibility; objective remediation frozen before balanced training",
        "blind_status": "SEALED_UNCONSUMED",
    }
    return policy, eligibility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--failure-report", action="append", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    eligibility_path = output.with_suffix(".ELIGIBILITY.json")
    if output.exists() or eligibility_path.exists():
        raise FileExistsError(output if output.exists() else eligibility_path)
    policy, eligibility = freeze(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    eligibility["policy_sha256"] = sha256_file(output)
    eligibility["policy_path"] = str(output)
    eligibility_path.write_text(
        json.dumps(eligibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "policy": _artifact(output),
                "eligibility": _artifact(eligibility_path),
                "task": args.task,
                "balanced_objective_revision": BALANCED_OBJECTIVE_REVISION,
                "blind_status": "SEALED_UNCONSUMED",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
