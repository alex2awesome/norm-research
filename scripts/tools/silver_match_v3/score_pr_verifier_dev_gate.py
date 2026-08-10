#!/usr/bin/env python3
"""Score the sole predeclared exact-high PR verifier-dev gate once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .score_verifier_calibration import safe_rate, wilson_interval


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    values = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in values or len(values) != len(rows):
        raise ValueError(f"invalid UID coverage: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-freeze", required=True)
    parser.add_argument("--output-freeze", required=True)
    parser.add_argument("--scoring-policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "pair_freeze": Path(args.pair_freeze).resolve(),
        "output_freeze": Path(args.output_freeze).resolve(),
        "scoring_policy": Path(args.scoring_policy).resolve(),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    pairs = json.loads(paths["pair_freeze"].read_text(encoding="utf-8"))
    frozen = json.loads(paths["output_freeze"].read_text(encoding="utf-8"))
    policy = json.loads(paths["scoring_policy"].read_text(encoding="utf-8"))
    if (
        pairs.get("status") != "FROZEN_BALANCED_BEFORE_VERIFIER_INFERENCE"
        or frozen.get("status") != "FROZEN_COMPLETE_BEFORE_FRESH_DEV_SCORING"
        or policy.get("status") != "FROZEN_BEFORE_FIRST_FRESH_DEV_VERIFIER_OUTPUT"
        or policy.get("retention_rule", {}).get("name") != "two_order_exact_high"
        or int(pairs.get("selected_count", -1)) != int(frozen.get("row_count", -2))
        or (policy.get("inputs") or {}).get("pair_freeze", {}).get("sha256")
        != sha256_file(paths["pair_freeze"])
    ):
        raise ValueError("pair/output/scoring freezes are incompatible")
    pair_outputs = pairs["outputs"]
    targets = _index(Path(pair_outputs["targets"]["path"]))
    primary = _index(Path(pair_outputs["primary"]["path"]))
    predictions = {
        order: _index(Path(frozen["outputs"][order]["predictions"]["path"]))
        for order in ("original", "hashed")
    }
    uids = set(targets)
    if set(primary) != uids or any(set(values) != uids for values in predictions.values()):
        raise ValueError("scoring inputs lack exact paired coverage")

    retained_uids: list[str] = []
    retained_true = 0
    order_decision_agreement = order_exact_agreement = 0
    for uid in sorted(uids):
        proposal = str(primary[uid]["metric_id"])
        left, right = predictions["original"][uid], predictions["hashed"][uid]
        order_decision_agreement += left.get("decision") == right.get("decision")
        order_exact_agreement += (
            left.get("decision"), left.get("metric_id")
        ) == (right.get("decision"), right.get("metric_id"))
        keep = (
            left.get("decision") == right.get("decision") == "CONFIRM_MATCH"
            and str(left.get("metric_id")) == str(right.get("metric_id")) == proposal
            and left.get("confidence") == right.get("confidence") == "high"
            and left.get("parse_error") is None
            and right.get("parse_error") is None
        )
        if keep:
            retained_uids.append(uid)
            retained_true += targets[uid]["target"] == "CONFIRM_MATCH"
    retained = len(retained_uids)
    false_retained = retained - retained_true
    positives = sum(row["target"] == "CONFIRM_MATCH" for row in targets.values())
    negatives = len(targets) - positives
    precision = safe_rate(retained_true, retained)
    precision_wilson = wilson_interval(retained_true, retained)
    wilson_lower = precision_wilson[0] if precision_wilson is not None else None
    gates = policy["gate_rule"]
    gate_results = {
        "minimum_retained_proposals": retained
        >= int(gates["minimum_retained_proposals"]),
        "minimum_exact_precision": precision is not None
        and precision >= float(gates["minimum_exact_precision"]),
        "minimum_wilson_lower_95": wilson_lower is not None
        and wilson_lower >= float(gates["minimum_wilson_lower_95"]),
    }
    passed = all(gate_results.values())
    report = {
        "schema_version": "silver-match-v3-pr-verifier-dev-gate-score-v1",
        "status": (
            "PASS_ELIGIBLE_FOR_NEW_INDEPENDENT_BLIND_AUDIT"
            if passed
            else "REJECTED_FRESH_DEV_GATE"
        ),
        "task": "press-releases",
        "role": "verifier_dev_selection",
        "policy": policy["retention_rule"],
        "n": len(uids),
        "target_counts": {"CONFIRM_MATCH": positives, "REJECT": negatives},
        "retained": retained,
        "retained_true": retained_true,
        "false_retained": false_retained,
        "retained_precision": precision,
        "retained_precision_wilson_95": precision_wilson,
        "retained_recall_of_correct_proposals": safe_rate(retained_true, positives),
        "wrong_proposal_rejection_rate": safe_rate(negatives - false_retained, negatives),
        "wrong_proposal_rejection_wilson_95": wilson_interval(
            negatives - false_retained, negatives
        ),
        "order_stability": {
            "decision_agreement": safe_rate(order_decision_agreement, len(uids)),
            "exact_decision_and_id_agreement": safe_rate(order_exact_agreement, len(uids)),
        },
        "gate_thresholds": gates,
        "gate_results": gate_results,
        "all_gates_pass": passed,
        "next_action": (
            policy["successful_gate_action"]
            if passed
            else policy["failed_gate_action"]
        ),
        "retained_uids": retained_uids,
        "contracts": {
            "scored_once": True,
            "no_post_dev_threshold_or_prompt_choice": True,
            "successful_gate_is_not_final_blind_evidence": True,
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
