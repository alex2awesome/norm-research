#!/usr/bin/env python3
"""Independently validate a frozen all-task scale-out handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file
from .freeze_alltask_scaleout_handoff import (
    EXPECTED_CORPORA,
    EXPECTED_NORMS,
    EXPECTED_TASKS,
    SCHEMA,
    TASK_ORDER,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == SCHEMA, "schema mismatch")
    _require(payload.get("task_order") == list(TASK_ORDER), "task order mismatch")
    _require(
        payload.get("release_ready") is False
        and payload.get("launch_authorized") is False
        and payload.get("mi_correlation_authorized") is False,
        "handoff claims forbidden readiness",
    )
    scope = payload.get("scope") or {}
    _require(
        scope.get("tasks") == EXPECTED_TASKS
        and scope.get("corpora") == EXPECTED_CORPORA
        and scope.get("norms") == EXPECTED_NORMS
        and scope.get("canonical_finals") == 0
        and scope.get("existing_extractions_reused") is True
        and scope.get("reextraction_required") is False,
        "canonical scope contract failed",
    )
    for name, binding in (payload.get("inputs") or {}).items():
        input_path = Path(str(binding.get("path") or "")).resolve()
        _require(input_path.is_file(), f"missing input: {name}")
        _require(
            sha256_file(input_path) == binding.get("sha256"),
            f"input hash drift: {name}",
        )

    recipe = payload.get("recipe_seed") or {}
    _require(
        recipe.get("reuse_humor_weights_across_tasks") is False
        and recipe.get("fresh_task_local_lora_required") is True
        and recipe.get("fresh_task_local_split_required") is True
        and recipe.get("fresh_task_local_threshold_selection_required") is True,
        "task-local recipe contract failed",
    )
    contracts = payload.get("global_contracts") or {}
    _require(
        contracts.get("humor_first") is True
        and contracts.get("notice_and_comment_last") is True
        and contracts.get("minimum_complete_bank_retrieval_lanes") == 2
        and contracts.get("separate_full_bank_rescue") is True
        and contracts.get("legacy_k50_is_diagnostic_only") is True
        and contracts.get("no_gpu_host_or_device_authorized_by_this_artifact")
        is True
        and contracts.get("sk3_gpu_indices_forbidden") == [1, 2, 3, 4],
        "global scale-out contract failed",
    )

    tasks = payload.get("tasks") or {}
    _require(set(tasks) == set(TASK_ORDER), "task record set mismatch")
    for priority, task in enumerate(TASK_ORDER, start=1):
        row = tasks[task]
        bank_count = int(row.get("bank_metric_count", -1))
        retrieval = row.get("retrieval") or {}
        extraction = row.get("extraction") or {}
        typed_gemma = row.get("typed_gemma") or {}
        final = row.get("canonical_final") or {}
        _require(row.get("priority") == priority, f"priority mismatch: {task}")
        _require(
            row.get("launch_authorized") is False
            and row.get("release_ready") is False,
            f"task claims forbidden readiness: {task}",
        )
        _require(
            extraction.get("status") == "CANONICAL_EXTRACTION_REUSED_COMPLETE"
            and extraction.get("reextraction_required") is False
            and extraction.get("reextraction_authorized") is False,
            f"extraction reuse contract failed: {task}",
        )
        _require(
            retrieval.get("minimum_independent_complete_bank_lanes") == 2
            and retrieval.get("legacy_k50_production_eligible") is False
            and retrieval.get("full_bank_rescue_required") is True
            and retrieval.get("required_primary_k") == min(200, bank_count)
            and retrieval.get("required_full_bank_rescue_k") == bank_count
            and retrieval.get("production_ready") is False,
            f"retrieval contract failed: {task}",
        )
        _require(
            typed_gemma.get("task_local_lora_required") is True
            and typed_gemma.get("shared_cross_task_weights_allowed") is False
            and typed_gemma.get("production_ready") is False,
            f"typed Gemma isolation failed: {task}",
        )
        _require(
            final.get("status") == "NOT_MATERIALIZED"
            and final.get("required_decisions_per_norm") == 1
            and final.get("typed_abstention_noise_required") is True
            and final.get("independent_blind_audit_required") is True
            and final.get("mi_join_authorized") is False,
            f"final-output contract failed: {task}",
        )
    _require(
        tasks["notice-and-comment"]["truth"].get("role_map_rows") == 0
        and tasks["notice-and-comment"]["truth"].get("status")
        == "NO_NORM_TRUTH_ROLE_MAP_NC_LAST",
        "N&C last-stage blocker is missing",
    )
    humor = tasks["humor"]
    humor_retrieval = humor["retrieval"]
    humor_ce = humor["ce_training"]
    evidence = payload.get("validated_humor_evidence") or {}
    _require(
        humor_retrieval.get("status")
        == "PRIMARY_K200_AND_FULLBANK_RESCUE_STRUCTURALLY_COMPLETE"
        and humor_retrieval.get("primary_structurally_complete") is True
        and humor_retrieval.get("full_bank_rescue_structurally_complete") is True
        and humor_retrieval.get("primary_pair_universe_structurally_complete")
        is True
        and humor_retrieval.get("rescue_adjudication_pending") is True,
        "Humor staged retrieval contract failed",
    )
    _require(
        humor_ce.get("status")
        == "K200_UNLABELED_PAIR_UNIVERSE_STAGED_TRUTH_BLOCKED"
        and humor_ce.get("candidate_depth") == 200
        and humor_ce.get("staged_pair_count") == 15_475_600
        and humor_ce.get("pair_universe_structurally_complete") is True
        and humor_ce.get("adjudication_authorized") is False,
        "Humor staged pair-universe contract failed",
    )
    _require(
        evidence.get("primary_k") == 200
        and evidence.get("primary_capture_rate", 0.0) > 0.97
        and evidence.get("primary_miss_upper_bound", 1.0) < 0.05
        and evidence.get("primary_pair_count") == 15_475_600
        and evidence.get("rescue_k") == 285
        and evidence.get("primary_structurally_complete") is True
        and evidence.get("full_bank_rescue_structurally_complete") is True
        and evidence.get("retrieval_and_pair_stage_ready") is True
        and evidence.get("production_model_promoted") is False,
        "Humor bound evidence contract failed",
    )
    return {
        "schema_version": "silver-match-v3-alltask-scaleout-handoff-validation-v2",
        "status": "PASS",
        "release_ready": False,
        "handoff": {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        },
        "validated": {
            "tasks": EXPECTED_TASKS,
            "corpora": EXPECTED_CORPORA,
            "norms": EXPECTED_NORMS,
            "task_local_loras": True,
            "coverage_preserving_retrieval_and_full_bank_rescue": True,
            "humor_k200_and_k285_structurally_staged": True,
            "notice_and_comment_last": True,
            "launch_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(Path(args.handoff))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
