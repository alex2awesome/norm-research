#!/usr/bin/env python3
"""Select the adjudicator branch from a frozen explicit-role GEPA plan.

The command plan freezes both the prompt variants and the selection rule before
inference.  This module only replays that rule over the frozen prompt-dev score
files.  It deliberately fails closed when no variant passes or when the
predeclared ranking fields do not identify a unique winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file
from .score_verifier_calibration import wilson_interval


PLAN_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-plan-v1"
FREEZE_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-freeze-v1"
SCORE_SCHEMA = "silver-match-v3-two-order-gepa-score-v1"


def _output_arg(command: dict[str, Any]) -> Path:
    argv = list(command.get("argv") or [])
    if argv.count("--output") != 1:
        raise ValueError("score command must contain exactly one --output")
    return Path(argv[argv.index("--output") + 1]).resolve()


def _load_score(
    path: Path, *, role: str, expected_prompt_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    score = json.loads(path.read_text(encoding="utf-8"))
    expected_panel = "prompt_train" if role == "optimize" else "prompt_dev"
    if (
        score.get("schema_version") != SCORE_SCHEMA
        or score.get("selection_universe") != "predeclared_train_only"
        or score.get("explicit_role") != role
        or score.get("panel_role") != expected_panel
        or score.get("prompt_sha256") != expected_prompt_sha256
    ):
        raise ValueError(f"score role/prompt contract differs from plan: {path}")
    for label, ref in (score.get("inputs") or {}).items():
        ref_path = Path(str(ref.get("path") or "")).resolve()
        if not ref_path.is_file() or sha256_file(ref_path) != str(
            ref.get("sha256") or ""
        ):
            raise ValueError(f"score input is missing or hash-drifted: {label}/{ref_path}")
    strict = (score.get("metrics") or {}).get("strict_consensus") or {}
    support = int(strict.get("confirmed_match_count") or 0)
    correct = int(strict.get("correct_exact_id_count") or 0)
    precision = strict.get("exact_id_precision")
    if correct < 0 or support < correct or (
        precision is not None and abs(float(precision) - correct / support) > 1e-12
    ):
        raise ValueError(f"strict consensus counts/precision are inconsistent: {path}")
    return score, {
        "confirmed_match_count": support,
        "correct_exact_id_count": correct,
        "exact_id_precision": precision,
        "exact_id_precision_wilson_95": wilson_interval(correct, support),
    }


def build_selection(plan_path: Path, freeze_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    freeze_path = freeze_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    plan_sha256 = sha256_file(plan_path)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE"
        or freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("status") != "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE"
        or freeze.get("task") != plan.get("task")
        or (freeze.get("command_plan") or {}).get("sha256") != plan_sha256
    ):
        raise ValueError("plan/FREEZE contract is missing or hash-drifted")

    thresholds = plan.get("thresholds") or {}
    required_thresholds = {
        "minimum_point_precision",
        "minimum_wilson_95_lower",
        "minimum_retained",
    }
    if not required_thresholds <= set(thresholds):
        raise ValueError("plan does not contain the complete selection gate")

    score_paths: dict[str, dict[str, Path]] = {}
    for cell in plan.get("commands") or []:
        if cell.get("stage") != "adjudicator_score":
            continue
        variant = str(cell.get("variant") or "")
        panel_role = str(cell.get("role") or "")
        role = {"prompt_train": "optimize", "prompt_dev": "select"}.get(panel_role)
        if not variant or role is None or role in score_paths.setdefault(variant, {}):
            raise ValueError("duplicate or malformed adjudicator score cell")
        score_paths[variant][role] = _output_arg(cell["command"])

    variants = []
    declared_names = [str(row.get("name") or "") for row in plan.get("adjudicator_variants") or []]
    if not declared_names or len(declared_names) != len(set(declared_names)):
        raise ValueError("plan has missing or duplicate adjudicator variants")
    for declared in plan["adjudicator_variants"]:
        name = str(declared["name"])
        if set(score_paths.get(name) or {}) != {"optimize", "select"}:
            raise ValueError(f"variant lacks one frozen optimize/select score: {name}")
        expected_prompt = str(declared.get("combined_prompt_sha256") or "")
        train, train_strict = _load_score(
            score_paths[name]["optimize"],
            role="optimize",
            expected_prompt_sha256=expected_prompt,
        )
        dev, dev_strict = _load_score(
            score_paths[name]["select"],
            role="select",
            expected_prompt_sha256=expected_prompt,
        )
        if train.get("prompt_sha256") != dev.get("prompt_sha256"):
            raise ValueError(f"variant train/dev prompts differ: {name}")
        interval = dev_strict["exact_id_precision_wilson_95"]
        point = dev_strict["exact_id_precision"]
        support = dev_strict["confirmed_match_count"]
        eligible = (
            point is not None
            and interval is not None
            and float(point) >= float(thresholds["minimum_point_precision"])
            and float(interval[0]) >= float(thresholds["minimum_wilson_95_lower"])
            and support >= int(thresholds["minimum_retained"])
        )
        variants.append(
            {
                "name": name,
                "prompt_sha256": expected_prompt,
                "eligible": eligible,
                "selection_key": (
                    [float(interval[0]), support, float(point)]
                    if interval is not None and point is not None
                    else None
                ),
                "optimize_strict_consensus": train_strict,
                "select_strict_consensus": dev_strict,
                "optimize_score": {
                    "path": str(score_paths[name]["optimize"]),
                    "sha256": sha256_file(score_paths[name]["optimize"]),
                },
                "select_score": {
                    "path": str(score_paths[name]["select"]),
                    "sha256": sha256_file(score_paths[name]["select"]),
                },
            }
        )

    eligible = [row for row in variants if row["eligible"]]
    chosen = None
    status = "failed_closed_no_eligible_adjudicator"
    if eligible:
        best_key = max(tuple(row["selection_key"]) for row in eligible)
        winners = [row for row in eligible if tuple(row["selection_key"]) == best_key]
        if len(winners) == 1:
            chosen = winners[0]
            status = "selected"
        else:
            status = "failed_closed_exact_selection_key_tie"

    return {
        "schema_version": "silver-match-v3-explicit-role-adjudicator-selection-v1",
        "task": plan["task"],
        "status": status,
        "selection_role": "prompt_dev",
        "selection_rule": thresholds.get("selection_rule"),
        "thresholds": {
            key: thresholds[key] for key in sorted(required_thresholds)
        },
        "chosen": chosen,
        "variants": variants,
        "inputs": {
            "command_plan": {"path": str(plan_path), "sha256": plan_sha256},
            "freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        },
        "test_or_blind_audit_consumed": False,
        "production_consumed": False,
        "outcomes_or_mi_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    freeze_path = (
        Path(args.freeze).resolve() if args.freeze else plan_path.with_name("FREEZE.json")
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = build_selection(plan_path, freeze_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
