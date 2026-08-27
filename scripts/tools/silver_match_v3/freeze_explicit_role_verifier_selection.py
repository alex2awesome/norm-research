#!/usr/bin/env python3
"""Freeze verifier policy candidates before explicit-role verifier inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


PLAN_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-plan-v1"
FREEZE_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-freeze-v1"


def _output_arg(command: dict[str, Any]) -> Path:
    argv = list(command.get("argv") or [])
    if argv.count("--output") != 1:
        raise ValueError("planned command must contain exactly one --output")
    return Path(argv[argv.index("--output") + 1]).resolve()


def build_freeze(plan_path: Path, role_freeze_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    role_freeze_path = role_freeze_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    role_freeze = json.loads(role_freeze_path.read_text(encoding="utf-8"))
    plan_sha256 = sha256_file(plan_path)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE"
        or role_freeze.get("schema_version") != FREEZE_SCHEMA
        or role_freeze.get("status") != "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE"
        or role_freeze.get("task") != plan.get("task")
        or (role_freeze.get("command_plan") or {}).get("sha256") != plan_sha256
    ):
        raise ValueError("plan/FREEZE contract is missing or hash-drifted")

    thresholds = plan.get("thresholds") or {}
    required = {
        "minimum_point_precision",
        "minimum_wilson_95_lower",
        "minimum_retained",
    }
    if not required <= set(thresholds):
        raise ValueError("plan does not contain the complete verifier gate")

    output_paths = []
    verifier_cells = []
    allowed_stages = {
        "verifier_subset_truth",
        "verifier_subset_candidates",
        "verifier",
        "verifier_score_two_order",
        "verifier_score_three_order",
    }
    for cell in plan.get("commands") or []:
        if cell.get("stage") not in allowed_stages:
            continue
        command = (
            cell.get("direct_batch_command")
            if cell.get("stage") == "verifier"
            else cell.get("command")
        )
        output = _output_arg(command or {})
        output_paths.append(output)
        verifier_cells.append(
            {
                "stage": cell["stage"],
                "adjudicator_variant": cell.get("adjudicator_variant"),
                "verifier_variant": cell.get("verifier_variant"),
                "role": cell.get("role"),
                "order": cell.get("order"),
                "output": str(output),
            }
        )
    if not output_paths or len(output_paths) != len(set(output_paths)):
        raise ValueError("plan has no verifier cells or reuses a verifier output path")
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise ValueError(
            "verifier selection must be frozen before any verifier output exists: "
            f"{existing[:3]}"
        )

    adjudicators = [
        {
            "name": str(row["name"]),
            "combined_prompt_sha256": str(row["combined_prompt_sha256"]),
        }
        for row in plan.get("adjudicator_variants") or []
    ]
    verifiers = [
        {
            "name": str(row["name"]),
            "combined_prompt_sha256": str(row["combined_prompt_sha256"]),
        }
        for row in plan.get("verifier_variants") or []
    ]
    if (
        not adjudicators
        or not verifiers
        or len({row["name"] for row in adjudicators}) != len(adjudicators)
        or len({row["name"] for row in verifiers}) != len(verifiers)
    ):
        raise ValueError("plan has missing or duplicate adjudicator/verifier variants")

    return {
        "schema_version": "silver-match-v3-explicit-role-verifier-selection-freeze-v1",
        "status": "FROZEN_BEFORE_ANY_VERIFIER_INFERENCE_OR_SCORE",
        "task": plan["task"],
        "thresholds": {key: thresholds[key] for key in sorted(required)},
        "selection_rule": thresholds.get("selection_rule"),
        "candidate_policies": [
            {
                "name": "two_order_exact_high",
                "score_stage": "verifier_score_two_order",
                "score_field": "policies.high_only",
                "production_eligible": True,
            },
            {
                "name": "all_three_order_exact_high",
                "score_stage": "verifier_score_three_order",
                "score_field": "policy",
                "production_eligible": True,
            },
        ],
        "diagnostic_policies": [
            {
                "name": "two_order_medium_or_high",
                "score_stage": "verifier_score_two_order",
                "score_field": "policies.medium_or_high",
                "production_eligible": False,
            }
        ],
        "tie_policy": "fail_closed_if_wilson_support_point_key_is_not_unique",
        "adjudicator_variants": adjudicators,
        "verifier_variants": verifiers,
        "planned_verifier_cells": verifier_cells,
        "preinference_audit": {
            "checked_output_path_count": len(output_paths),
            "existing_output_path_count": 0,
            "all_verifier_outputs_absent": True,
        },
        "inputs": {
            "command_plan": {"path": str(plan_path), "sha256": plan_sha256},
            "role_freeze": {
                "path": str(role_freeze_path),
                "sha256": sha256_file(role_freeze_path),
            },
        },
        "test_or_blind_audit_consumed": False,
        "production_consumed": False,
        "outcomes_or_mi_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--role-freeze")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    role_freeze_path = (
        Path(args.role_freeze).resolve()
        if args.role_freeze
        else plan_path.with_name("FREEZE.json")
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = build_freeze(plan_path, role_freeze_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
