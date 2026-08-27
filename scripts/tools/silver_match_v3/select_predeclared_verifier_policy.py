#!/usr/bin/env python3
"""Select between predeclared two- and three-order exact-high policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def _candidate(name: str, policy: dict, thresholds: dict, score: Path) -> dict:
    interval = policy.get("retained_precision_wilson_95") or [0.0, 0.0]
    eligible = (
        int(policy.get("retained") or 0) >= int(thresholds["minimum_retained"])
        and float(policy.get("retained_precision") or 0.0)
        >= float(thresholds["minimum_point_precision"])
        and float(interval[0]) >= float(thresholds["minimum_wilson_95_lower"])
    )
    return {
        "name": name,
        "eligible": eligible,
        "policy": policy,
        "score_path": str(score),
        "score_sha256": sha256_file(score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-freeze", required=True)
    parser.add_argument("--two-order-score", required=True)
    parser.add_argument("--three-order-score", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(getattr(args, name.replace("-", "_"))).resolve()
        for name in ("policy_freeze", "two_order_score", "three_order_score")
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze = json.loads(paths["policy_freeze"].read_text())
    two = json.loads(paths["two_order_score"].read_text())
    three = json.loads(paths["three_order_score"].read_text())
    if two.get("selection_split") != "dev" or three.get("selection_split") != "dev":
        raise ValueError("both scores must be on the frozen dev selection")
    expected = [row["name"] for row in freeze.get("candidate_policies") or []]
    if expected != ["two_order_exact_high", "all_three_order_exact_high"]:
        raise ValueError("policy freeze does not contain the expected candidates")
    thresholds = freeze["eligibility_gate"]
    candidates = [
        _candidate("two_order_exact_high", two["policies"]["high_only"], thresholds, paths["two_order_score"]),
        _candidate("all_three_order_exact_high", three["policy"], thresholds, paths["three_order_score"]),
    ]
    eligible = [row for row in candidates if row["eligible"]]
    chosen = None
    if eligible:
        chosen = max(
            eligible,
            key=lambda row: (
                row["policy"]["retained_precision_wilson_95"][0],
                row["policy"]["retained"],
                row["policy"]["retained_precision"],
            ),
        )
    payload = {
        "schema_version": "silver-match-v3-predeclared-verifier-policy-selection-v1",
        "task": freeze.get("task"),
        "selection_split": "dev",
        "status": "selected" if chosen else "failed_closed_no_eligible_policy",
        "policy_freeze": {"path": str(paths["policy_freeze"]), "sha256": sha256_file(paths["policy_freeze"])},
        "thresholds": thresholds,
        "candidates": candidates,
        "chosen": chosen,
        "permanent_blind_consumed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
