#!/usr/bin/env python3
"""Select an exact/high verifier policy from a pre-inference frozen universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .score_verifier_calibration import wilson_interval


PLAN_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-plan-v1"
ROLE_FREEZE_SCHEMA = "silver-match-v3-explicit-role-task-local-gepa-freeze-v1"
POLICY_FREEZE_SCHEMA = "silver-match-v3-explicit-role-verifier-selection-freeze-v1"
ADJ_SELECTION_SCHEMA = "silver-match-v3-explicit-role-adjudicator-selection-v1"


def _arg_path(command: dict[str, Any], flag: str) -> Path:
    argv = list(command.get("argv") or [])
    option = f"--{flag}"
    if argv.count(option) != 1:
        raise ValueError(f"command must contain exactly one {option}")
    return Path(argv[argv.index(option) + 1]).resolve()


def _policy_counts(policy: dict[str, Any], path: Path) -> dict[str, Any]:
    n = int(policy.get("n") or 0)
    proposal_correct = int(policy.get("proposal_correct") or 0)
    retained = int(policy.get("retained") or 0)
    retained_true = int(policy.get("retained_true") or 0)
    point = policy.get("retained_precision")
    interval = policy.get("retained_precision_wilson_95")
    recomputed = wilson_interval(retained_true, retained)
    if (
        n < 0
        or proposal_correct < 0
        or proposal_correct > n
        or retained < 0
        or retained > n
        or retained_true < 0
        or retained_true > retained
        or retained_true > proposal_correct
        or (point is None) != (retained == 0)
        or (
            point is not None
            and abs(float(point) - retained_true / retained) > 1e-12
        )
        or interval != recomputed
    ):
        raise ValueError(f"verifier policy counts/interval are inconsistent: {path}")
    return {
        "n": n,
        "proposal_correct": proposal_correct,
        "retained": retained,
        "retained_true": retained_true,
        "retained_precision": point,
        "retained_precision_wilson_95": interval,
        "retained_recall_of_correct_proposals": policy.get(
            "retained_recall_of_correct_proposals"
        ),
        "wrong_proposal_rejection_rate": policy.get(
            "wrong_proposal_rejection_rate"
        ),
    }


def _load_score(
    cell: dict[str, Any], *, expected_role: str, expected_prompt_sha256: str
) -> tuple[Path, dict[str, Any]]:
    command = cell["command"]
    path = _arg_path(command, "output")
    score = json.loads(path.read_text(encoding="utf-8"))
    stage = str(cell["stage"])
    expected_schema = {
        "verifier_score_two_order": "silver-match-v3-two-order-verifier-score-v1",
        "verifier_score_three_order": "silver-match-v3-three-order-verifier-score-v1",
    }[stage]
    expected_split = "optimize" if expected_role == "optimize" else "dev"
    if (
        score.get("schema_version") != expected_schema
        or score.get("explicit_role") != expected_role
        or score.get("selection_split") != expected_split
    ):
        raise ValueError(f"verifier score role/schema differs from plan: {path}")
    input_keys = ["truth", "primary", "original", "hashed"]
    if stage == "verifier_score_three_order":
        input_keys.append("reverse")
    recorded = score.get("input_hashes") or {}
    if set(recorded) != set(input_keys):
        raise ValueError(f"verifier score input hash keys differ from plan: {path}")
    for key in input_keys:
        input_path = _arg_path(command, key)
        if not input_path.is_file() or sha256_file(input_path) != str(recorded[key]):
            raise ValueError(f"verifier score input is missing or hash-drifted: {input_path}")
        if key not in {"original", "hashed", "reverse"}:
            continue
        rows = list(read_jsonl(input_path))
        uids = [str(row.get("norm_uid") or "") for row in rows]
        if (
            len(uids) != len(set(uids))
            or "" in uids
            or any(row.get("prompt_sha256") != expected_prompt_sha256 for row in rows)
            or any(row.get("parse_error") for row in rows)
        ):
            raise ValueError(f"verifier inference prompt/UID contract differs: {input_path}")
    return path, score


def build_selection(
    plan_path: Path,
    role_freeze_path: Path,
    adjudicator_selection_path: Path,
    policy_freeze_path: Path,
) -> dict[str, Any]:
    paths = {
        "plan": plan_path.resolve(),
        "role_freeze": role_freeze_path.resolve(),
        "adjudicator_selection": adjudicator_selection_path.resolve(),
        "policy_freeze": policy_freeze_path.resolve(),
    }
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    role_freeze = json.loads(paths["role_freeze"].read_text(encoding="utf-8"))
    adjudicator_selection = json.loads(
        paths["adjudicator_selection"].read_text(encoding="utf-8")
    )
    policy_freeze = json.loads(paths["policy_freeze"].read_text(encoding="utf-8"))
    plan_sha256 = sha256_file(paths["plan"])
    role_freeze_sha256 = sha256_file(paths["role_freeze"])
    task = plan.get("task")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or role_freeze.get("schema_version") != ROLE_FREEZE_SCHEMA
        or role_freeze.get("task") != task
        or (role_freeze.get("command_plan") or {}).get("sha256") != plan_sha256
        or adjudicator_selection.get("schema_version") != ADJ_SELECTION_SCHEMA
        or adjudicator_selection.get("task") != task
        or (adjudicator_selection.get("inputs") or {})
        .get("command_plan", {})
        .get("sha256")
        != plan_sha256
        or policy_freeze.get("schema_version") != POLICY_FREEZE_SCHEMA
        or policy_freeze.get("status")
        != "FROZEN_BEFORE_ANY_VERIFIER_INFERENCE_OR_SCORE"
        or policy_freeze.get("task") != task
        or (policy_freeze.get("inputs") or {}).get("command_plan", {}).get("sha256")
        != plan_sha256
        or (policy_freeze.get("inputs") or {}).get("role_freeze", {}).get("sha256")
        != role_freeze_sha256
        or (policy_freeze.get("preinference_audit") or {}).get(
            "all_verifier_outputs_absent"
        )
        is not True
    ):
        raise ValueError("plan, role freeze, adjudicator selection, or policy freeze drifted")

    expected_candidates = [
        ("two_order_exact_high", "verifier_score_two_order", "policies.high_only"),
        ("all_three_order_exact_high", "verifier_score_three_order", "policy"),
    ]
    observed_candidates = [
        (row.get("name"), row.get("score_stage"), row.get("score_field"))
        for row in policy_freeze.get("candidate_policies") or []
        if row.get("production_eligible") is True
    ]
    if observed_candidates != expected_candidates:
        raise ValueError("policy freeze candidate universe differs from established exact/high set")
    thresholds = policy_freeze.get("thresholds") or {}
    if thresholds != {
        key: (plan.get("thresholds") or {}).get(key)
        for key in (
            "minimum_point_precision",
            "minimum_retained",
            "minimum_wilson_95_lower",
        )
    }:
        raise ValueError("policy freeze thresholds differ from command plan")

    common_inputs = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    if adjudicator_selection.get("status") != "selected":
        if adjudicator_selection.get("chosen") is not None:
            raise ValueError("failed-closed adjudicator selection unexpectedly has a winner")
        return {
            "schema_version": "silver-match-v3-explicit-role-verifier-selection-v1",
            "task": task,
            "status": "not_run_no_eligible_adjudicator",
            "selection_role": "prompt_dev",
            "selection_rule": policy_freeze.get("selection_rule"),
            "thresholds": thresholds,
            "chosen": None,
            "candidates": [],
            "diagnostics": [],
            "inputs": common_inputs,
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
            "outcomes_or_mi_used": False,
        }

    chosen_adj = str((adjudicator_selection.get("chosen") or {}).get("name") or "")
    declared_adj = {str(row["name"]) for row in plan.get("adjudicator_variants") or []}
    verifier_prompts = {
        str(row["name"]): str(row["combined_prompt_sha256"])
        for row in plan.get("verifier_variants") or []
    }
    if chosen_adj not in declared_adj or not verifier_prompts:
        raise ValueError("selected adjudicator or verifier variants are absent from plan")

    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cell in plan.get("commands") or []:
        stage = str(cell.get("stage") or "")
        if (
            stage not in {"verifier_score_two_order", "verifier_score_three_order"}
            or cell.get("conditional_on_selected_adjudicator_variant") != chosen_adj
        ):
            continue
        role = {"prompt_train": "optimize", "prompt_dev": "select"}.get(
            str(cell.get("role") or "")
        )
        verifier = str(cell.get("verifier_variant") or "")
        key = (verifier, role or "", stage)
        if verifier not in verifier_prompts or role is None or key in cells:
            raise ValueError("duplicate or malformed selected verifier score cell")
        cells[key] = cell

    candidates = []
    diagnostics = []
    for verifier, prompt_sha256 in verifier_prompts.items():
        loaded: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
        for role in ("optimize", "select"):
            for stage in ("verifier_score_two_order", "verifier_score_three_order"):
                key = (verifier, role, stage)
                if key not in cells:
                    raise ValueError(f"selected branch lacks frozen score cell: {key}")
                loaded[(role, stage)] = _load_score(
                    cells[key],
                    expected_role=role,
                    expected_prompt_sha256=prompt_sha256,
                )
        two_dev = loaded[("select", "verifier_score_two_order")][1]
        diagnostics.append(
            {
                "verifier_variant": verifier,
                "policy": "two_order_medium_or_high",
                "production_eligible": False,
                "select_metrics": _policy_counts(
                    two_dev["policies"]["medium_or_high"],
                    loaded[("select", "verifier_score_two_order")][0],
                ),
            }
        )
        for policy_name, stage, field in expected_candidates:
            train_path, train_score = loaded[("optimize", stage)]
            dev_path, dev_score = loaded[("select", stage)]
            if field == "policies.high_only":
                train_policy = train_score["policies"]["high_only"]
                dev_policy = dev_score["policies"]["high_only"]
            else:
                train_policy = train_score["policy"]
                dev_policy = dev_score["policy"]
            train_metrics = _policy_counts(train_policy, train_path)
            dev_metrics = _policy_counts(dev_policy, dev_path)
            point = dev_metrics["retained_precision"]
            interval = dev_metrics["retained_precision_wilson_95"]
            support = dev_metrics["retained"]
            eligible = (
                point is not None
                and interval is not None
                and float(point) >= float(thresholds["minimum_point_precision"])
                and float(interval[0]) >= float(thresholds["minimum_wilson_95_lower"])
                and support >= int(thresholds["minimum_retained"])
            )
            candidates.append(
                {
                    "adjudicator_variant": chosen_adj,
                    "verifier_variant": verifier,
                    "verifier_prompt_sha256": prompt_sha256,
                    "policy": policy_name,
                    "eligible": eligible,
                    "selection_key": (
                        [float(interval[0]), support, float(point)]
                        if interval is not None and point is not None
                        else None
                    ),
                    "optimize_metrics": train_metrics,
                    "select_metrics": dev_metrics,
                    "optimize_score": {
                        "path": str(train_path),
                        "sha256": sha256_file(train_path),
                    },
                    "select_score": {
                        "path": str(dev_path),
                        "sha256": sha256_file(dev_path),
                    },
                }
            )

    eligible = [row for row in candidates if row["eligible"]]
    chosen = None
    status = "failed_closed_no_eligible_verifier_policy"
    if eligible:
        best = max(tuple(row["selection_key"]) for row in eligible)
        winners = [row for row in eligible if tuple(row["selection_key"]) == best]
        if len(winners) == 1:
            chosen = winners[0]
            status = "selected"
        else:
            status = "failed_closed_exact_selection_key_tie"

    return {
        "schema_version": "silver-match-v3-explicit-role-verifier-selection-v1",
        "task": task,
        "status": status,
        "selection_role": "prompt_dev",
        "selection_rule": policy_freeze.get("selection_rule"),
        "thresholds": thresholds,
        "chosen": chosen,
        "candidates": candidates,
        "diagnostics": diagnostics,
        "inputs": common_inputs,
        "test_or_blind_audit_consumed": False,
        "production_consumed": False,
        "outcomes_or_mi_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--role-freeze")
    parser.add_argument("--adjudicator-selection", required=True)
    parser.add_argument("--policy-freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = build_selection(
        plan_path,
        Path(args.role_freeze).resolve()
        if args.role_freeze
        else plan_path.with_name("FREEZE.json"),
        Path(args.adjudicator_selection),
        Path(args.policy_freeze),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
