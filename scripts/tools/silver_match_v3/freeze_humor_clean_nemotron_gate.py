#!/usr/bin/env python3
"""Predeclare the Humor clean-Nemotron promotion gate before select truth opens."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-report", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidates-meta", required=True)
    parser.add_argument("--select-identities", required=True)
    parser.add_argument("--select-freeze", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--prior-contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        name: Path(value).resolve()
        for name, value in vars(args).items()
        if name != "output"
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)

    teacher_report = json.loads(paths["teacher_report"].read_text(encoding="utf-8"))
    candidate_meta = json.loads(paths["candidates_meta"].read_text(encoding="utf-8"))
    select_freeze = json.loads(paths["select_freeze"].read_text(encoding="utf-8"))
    prior_contract = json.loads(paths["prior_contract"].read_text(encoding="utf-8"))
    teacher_rows = list(read_jsonl(paths["teacher"]))
    select_rows = list(read_jsonl(paths["select_identities"]))
    if (
        teacher_report.get("status") != "FROZEN_CLEAN_TRUTH_MATCH_ONLY_READY"
        or int(teacher_report.get("teacher_rows", -1)) != 388
        or (teacher_report.get("output") or {}).get("sha256")
        != sha256_file(paths["teacher"])
        or len(teacher_rows) != 388
    ):
        raise ValueError("clean teacher/report binding failure")
    if (
        int((candidate_meta.get("output") or {}).get("count", -1)) != 896
        or (candidate_meta.get("output") or {}).get("sha256")
        != sha256_file(paths["candidates"])
    ):
        raise ValueError("clean K50 candidate binding failure")
    if (
        select_freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or int(select_freeze.get("selected_count", -1)) != 300
        or len(select_rows) != 300
        or ((select_freeze.get("outputs") or {}).get("identities") or {}).get("sha256")
        != sha256_file(paths["select_identities"])
    ):
        raise ValueError("select identities were not frozen truth-hidden")
    if prior_contract.get("status") != "FROZEN_BEFORE_TRAINING_RESULT":
        raise ValueError("prior retriever contract is not a valid frozen threshold source")

    teacher_uids = {str(row["norm_uid"]) for row in teacher_rows}
    teacher_groups = {str(row["source_group"]) for row in teacher_rows}
    select_uids = {str(row["norm_uid"]) for row in select_rows}
    select_groups = {str(row["source_group"]) for row in select_rows}
    if teacher_uids & select_uids or teacher_groups & select_groups:
        raise ValueError("select evaluation firewall overlaps training supervision")

    gate = {
        "schema_version": "silver-match-v3-humor-clean-nemotron-promotion-gate-v1",
        "status": "FROZEN_BEFORE_SELECT_TRUTH_OR_TRAINING_RESULT",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": "humor",
        "model_role": "candidate_retriever_only",
        "selection_firewall": {
            "fresh_select_rows": 300,
            "teacher_uid_overlap": 0,
            "teacher_source_group_overlap": 0,
            "may_train_or_select_epoch": False,
            "may_be_opened_only_after_model_config_seed_and_gate_are_frozen": True,
        },
        "frozen_training_recipe": {
            "base_model": "nvidia/llama-embed-nemotron-8b-aa3b43-projection-v2",
            "seed": 94131,
            "split_seed": 874192,
            "epochs": 5,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "triplet_margin": 0.15,
            "hard_negative_pool": 32,
            "negatives_per_positive": 6,
            "lora": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.05,
                "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "epoch_selection": "source-disjoint internal dev; depth_lexicographic_at_50",
            "seed_or_hyperparameter_search": False,
        },
        "promotion_gate": {
            "exact_match_capture": {
                "population": "fresh-select rows whose independent exact truth is MATCH",
                "primary": "exact recall@50 gain over the identically evaluated frozen base",
                "minimum_primary_gain": 0.03,
                "secondary": "exact recall@80 may not decrease",
                "report_depths": [1, 3, 5, 10, 16, 30, 50, 80],
                "paired_rank_changes_required": True,
            },
            "abstention_safety": {
                "retriever_may_emit_typed_abstention": False,
                "gold_match_absent_from_k50_label": "RETRIEVER_MISS_REQUIRES_FULL_BANK_RESCUE",
                "candidate_miss_may_be_called_bank_gap": False,
                "automatic_no_candidate_fits_from_k50_miss": False,
                "report_miss_rate_and_one_sided_95pct_exact_upper_bound": True,
                "claim_under_5pct_only_if_exact_upper_bound_below": 0.05,
                "capture_recapture_may_substitute_for_accuracy_bound": False,
                "all_nonmatch_truth_decisions_reported_by_type": True,
            },
            "order_stability": {
                "bank_orders": ["canonical_metric_id", "reverse", "sha256_seed_20260713"],
                "score_once_per_order": True,
                "tie_break": "metric_id_ascending_after_score_descending",
                "required_top50_metric_set_equality_rate": 1.0,
                "required_exact_match_capture_invariance_rate": 1.0,
                "required_invalid_or_duplicate_candidate_rows": 0,
            },
            "integrity": {
                "adapter_reload_required": True,
                "lora_only_trainable_targets_required": True,
                "source_disjoint_train_dev_test_required": True,
                "zero_weak_or_forced_positive_rows_required": True,
                "all_bound_artifact_hashes_recomputed": True,
            },
        },
        "release_semantics": {
            "internal_dev_gate_alone_sufficient": False,
            "fresh_select_gate_alone_sufficient_for_final_silver_release": False,
            "retriever_promotion_changes_final_labels": False,
            "downstream_adjudicator_and_exhaustive_rescue_still_required": True,
            "failed_gate_action": "retain frozen base retriever; quarantine adapter",
        },
        "bindings": {
            name: _ref(path)
            for name, path in sorted(paths.items())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
