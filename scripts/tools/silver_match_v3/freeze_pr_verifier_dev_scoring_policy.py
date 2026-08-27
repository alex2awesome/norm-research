#!/usr/bin/env python3
"""Freeze exact-high PR verifier-dev scoring before any verifier output exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pair-freeze", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "plan": Path(args.plan).resolve(),
        "pair_freeze": Path(args.pair_freeze).resolve(),
        "prompt": Path(args.prompt).resolve(),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    pairs = json.loads(paths["pair_freeze"].read_text(encoding="utf-8"))
    if (
        plan.get("schema_version")
        != "silver-match-v3-pr-verifier-dev-inference-plan-v1"
        or plan.get("status")
        != "FROZEN_BEFORE_FIRST_FRESH_DEV_VERIFIER_INFERENCE"
        or plan.get("orders") != ["original", "hashed"]
        or pairs.get("status") != "FROZEN_BALANCED_BEFORE_VERIFIER_INFERENCE"
        or int(plan.get("selected_pair_count", -1))
        != int(pairs.get("selected_count", -2))
        or (plan.get("inputs") or {}).get("pair-freeze", {}).get("sha256")
        != sha256_file(paths["pair_freeze"])
        or (plan.get("inputs") or {}).get("prompt", {}).get("sha256")
        != sha256_file(paths["prompt"])
    ):
        raise ValueError("plan, pair universe, or prompt binding drift")
    for order, raw_path in (plan.get("outputs") or {}).items():
        path = Path(raw_path)
        if path.exists() or path.with_suffix(path.suffix + ".meta.json").exists():
            raise ValueError(f"cannot freeze scoring after {order} output exists")
    gates = plan.get("gates") or {}
    report = {
        "schema_version": "silver-match-v3-pr-verifier-dev-scoring-policy-v1",
        "status": "FROZEN_BEFORE_FIRST_FRESH_DEV_VERIFIER_OUTPUT",
        "task": plan["task"],
        "role": "verifier_dev_selection",
        "retention_rule": {
            "name": "two_order_exact_high",
            "required_orders": ["original", "hashed"],
            "both_decisions": "CONFIRM_MATCH",
            "both_metric_ids_equal_exact_proposal": True,
            "both_confidences": "high",
            "parse_error_allowed": False,
        },
        "gate_rule": {
            "minimum_retained_proposals": int(gates["minimum_retained_proposals"]),
            "minimum_exact_precision": float(gates["minimum_exact_precision"]),
            "minimum_wilson_lower_95": float(gates["minimum_wilson_lower_95"]),
            "all_three_must_pass": True,
        },
        "selection_rule": "sole predeclared prompt variant passes or fails as-is; no post-dev prompt or confidence-threshold choice",
        "failed_gate_action": gates["failed_gate_action"],
        "successful_gate_action": "eligible_for_new_independent_blind_audit_only",
        "contracts": {
            "medium_confidence_not_retained": True,
            "fresh_dev_cannot_mutate_prompt": True,
            "fresh_dev_not_final_blind_evidence": True,
            "no_outputs_read_by_freezer": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
