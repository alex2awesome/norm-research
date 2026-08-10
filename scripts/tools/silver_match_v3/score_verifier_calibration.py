#!/usr/bin/env python3
"""Score Gemma proposal verification as a high-precision teacher filter."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .common import read_jsonl, sha256_file, write_jsonl


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def score(
    truth_rows: Sequence[dict[str, Any]],
    primary_rows: Sequence[dict[str, Any]],
    verification_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth = {str(row["norm_uid"]): row for row in truth_rows}
    primary = {str(row["norm_uid"]): row for row in primary_rows}
    verification = {str(row["norm_uid"]): row for row in verification_rows}
    if any(len(values) != len(rows) for values, rows in (
        (truth, truth_rows), (primary, primary_rows), (verification, verification_rows)
    )):
        raise ValueError("duplicate verifier calibration UID")
    if truth.keys() != primary.keys() or truth.keys() != verification.keys():
        raise ValueError("truth, proposal, and verification UID sets differ")
    counts: Counter[str] = Counter()
    errors = []
    for uid in sorted(truth):
        gold = str(truth[uid]["metric_id"])
        proposed = str(primary[uid]["metric_id"])
        prediction = verification[uid]
        decision = str(prediction.get("decision"))
        confidence = str(prediction.get("confidence"))
        proposal_correct = proposed == gold
        confirm = decision == "CONFIRM_MATCH" and prediction.get("metric_id") == proposed
        retained = confirm and confidence in {"high", "medium"}
        corrected = decision == "BETTER_CANDIDATE" and prediction.get("metric_id") == gold
        counts["n"] += 1
        counts["proposal_correct"] += int(proposal_correct)
        counts["proposal_wrong"] += int(not proposal_correct)
        counts[f"decision:{decision}"] += 1
        counts["confirm"] += int(confirm)
        counts["confirm_true"] += int(confirm and proposal_correct)
        counts["retained"] += int(retained)
        counts["retained_true"] += int(retained and proposal_correct)
        counts["wrong_proposal_rejected"] += int(not proposal_correct and not retained)
        counts["correct_proposal_retained"] += int(proposal_correct and retained)
        counts["conflict_corrected_to_gold"] += int(not proposal_correct and corrected)
        if retained != proposal_correct:
            errors.append(
                {
                    "norm_uid": uid,
                    "truth_metric_id": gold,
                    "proposal_metric_id": proposed,
                    "decision": decision,
                    "prediction_metric_id": prediction.get("metric_id"),
                    "confidence": confidence,
                    "reason": prediction.get("reason"),
                    "filter_error": "false_retain" if retained else "false_reject",
                }
            )
    report = {
        "counts": dict(sorted(counts.items())),
        "proposal_exact_accuracy": safe_rate(counts["proposal_correct"], counts["n"]),
        "confirm_precision": safe_rate(counts["confirm_true"], counts["confirm"]),
        "retained_precision": safe_rate(counts["retained_true"], counts["retained"]),
        "retained_precision_wilson_95": wilson_interval(
            counts["retained_true"], counts["retained"]
        ),
        "retained_recall_of_correct_proposals": safe_rate(
            counts["correct_proposal_retained"], counts["proposal_correct"]
        ),
        "wrong_proposal_rejection_rate": safe_rate(
            counts["wrong_proposal_rejected"], counts["proposal_wrong"]
        ),
        "wrong_proposal_rejection_wilson_95": wilson_interval(
            counts["wrong_proposal_rejected"], counts["proposal_wrong"]
        ),
        "conflict_exact_correction_rate": safe_rate(
            counts["conflict_corrected_to_gold"], counts["proposal_wrong"]
        ),
    }
    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors-output")
    args = parser.parse_args()
    paths = {key: Path(getattr(args, key)).resolve() for key in ("truth", "primary", "verification")}
    report, errors = score(
        list(read_jsonl(paths["truth"])),
        list(read_jsonl(paths["primary"])),
        list(read_jsonl(paths["verification"])),
    )
    report["input_hashes"] = {key: sha256_file(path) for key, path in paths.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.errors_output:
        write_jsonl(Path(args.errors_output), errors)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
