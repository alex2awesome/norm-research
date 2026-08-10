#!/usr/bin/env python3
"""Score exact two-pass full-bank consensus on a human dev anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file
from .score_verifier_calibration import safe_rate, wilson_interval


def _index(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate UIDs: {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    truth_path, first_path, second_path = map(
        lambda value: Path(value).resolve(), [args.truth, args.first, args.second]
    )
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    truth, first, second = _index(truth_path), _index(first_path), _index(second_path)
    common = set(truth) & set(first) & set(second)
    if common != set(first) or common != set(second):
        raise ValueError("truth and prediction UID sets differ")

    policies = {}
    for name in ("all_exact", "one_high", "both_high"):
        retained = []
        for uid in sorted(common):
            left, right = first[uid], second[uid]
            exact_repeat = (
                left.get("decision") == right.get("decision") == "MATCH"
                and left.get("metric_id") == right.get("metric_id")
            )
            if not exact_repeat:
                continue
            confidences = [left.get("confidence"), right.get("confidence")]
            if name == "one_high" and "high" not in confidences:
                continue
            if name == "both_high" and confidences != ["high", "high"]:
                continue
            retained.append(uid)
        exact = sum(
            truth[uid].get("decision") == "MATCH"
            and truth[uid].get("metric_id") == first[uid].get("metric_id")
            for uid in retained
        )
        binary = sum(truth[uid].get("decision") == "MATCH" for uid in retained)
        truth_matches = sum(row.get("decision") == "MATCH" for row in truth.values())
        policies[name] = {
            "retained": len(retained),
            "retained_exact": exact,
            "exact_precision": safe_rate(exact, len(retained)),
            "exact_precision_wilson_95": wilson_interval(exact, len(retained)),
            "binary_match_precision": safe_rate(binary, len(retained)),
            "exact_recall_of_truth_matches": safe_rate(exact, truth_matches),
        }
    decision_agreement = sum(
        first[uid].get("decision") == second[uid].get("decision") for uid in common
    )
    exact_agreement = sum(
        first[uid].get("decision") == second[uid].get("decision")
        and first[uid].get("metric_id") == second[uid].get("metric_id")
        for uid in common
    )
    report = {
        "schema_version": "silver-match-v3-two-pass-anchor-score-v1",
        "selection_split": "dev",
        "n": len(common),
        "truth_match_count": sum(
            row.get("decision") == "MATCH" for row in truth.values()
        ),
        "order_stability": {
            "decision_agreement": safe_rate(decision_agreement, len(common)),
            "exact_decision_and_id_agreement": safe_rate(exact_agreement, len(common)),
        },
        "policies": policies,
        "input_hashes": {
            "truth": sha256_file(truth_path),
            "first": sha256_file(first_path),
            "second": sha256_file(second_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
