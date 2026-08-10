#!/usr/bin/env python3
"""Score fail-closed two-order verifier policies on human dev proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_jsonl, sha256_file
from .score_verifier_calibration import safe_rate, wilson_interval


def score_policy(
    truth: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Mapping[str, Any]],
    original: Mapping[str, Mapping[str, Any]],
    hashed: Mapping[str, Mapping[str, Any]],
    *,
    allowed_confidence: set[str],
) -> dict[str, Any]:
    proposal_correct = retained = retained_true = corrected = 0
    for row in truth:
        uid, gold = str(row["norm_uid"]), str(row["metric_id"])
        proposed = str(primary[uid]["metric_id"])
        is_correct = proposed == gold
        proposal_correct += is_correct
        left, right = original[uid], hashed[uid]
        keep = (
            left.get("decision") == right.get("decision") == "CONFIRM_MATCH"
            and left.get("metric_id") == right.get("metric_id") == proposed
            and left.get("confidence") in allowed_confidence
            and right.get("confidence") in allowed_confidence
        )
        retained += keep
        retained_true += keep and is_correct
        corrected += (
            not is_correct
            and left.get("decision") == right.get("decision") == "BETTER_CANDIDATE"
            and left.get("metric_id") == right.get("metric_id") == gold
        )
    wrong = len(truth) - proposal_correct
    return {
        "n": len(truth),
        "proposal_correct": proposal_correct,
        "proposal_exact_accuracy": safe_rate(proposal_correct, len(truth)),
        "retained": retained,
        "retained_true": retained_true,
        "retained_precision": safe_rate(retained_true, retained),
        "retained_precision_wilson_95": wilson_interval(retained_true, retained),
        "retained_recall_of_correct_proposals": safe_rate(retained_true, proposal_correct),
        "wrong_proposal_rejection_rate": safe_rate(wrong - (retained - retained_true), wrong),
        "wrong_proposal_rejection_wilson_95": wilson_interval(
            wrong - (retained - retained_true), wrong
        ),
        "two_order_exact_correction_count": corrected,
        "two_order_exact_correction_rate": safe_rate(corrected, wrong),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--explicit-role", choices=("optimize", "select"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        key: Path(getattr(args, key)).resolve()
        for key in ("truth", "primary", "original", "hashed")
    }
    truth = list(read_jsonl(paths["truth"]))
    values = {
        key: {str(row["norm_uid"]): row for row in read_jsonl(paths[key])}
        for key in ("primary", "original", "hashed")
    }
    uids = {str(row["norm_uid"]) for row in truth}
    if any(set(rows) != uids for rows in values.values()):
        raise ValueError("two-order verifier inputs lack exact paired coverage")
    if args.explicit_role and any(
        row.get("gepa_role") != args.explicit_role or row.get("split") != "train"
        for row in truth
    ):
        raise ValueError("verifier truth does not preserve its frozen explicit role")
    order_decision = sum(
        values["original"][uid].get("decision")
        == values["hashed"][uid].get("decision")
        for uid in uids
    )
    order_outcome = sum(
        (
            values["original"][uid].get("decision"),
            values["original"][uid].get("metric_id"),
        )
        == (
            values["hashed"][uid].get("decision"),
            values["hashed"][uid].get("metric_id"),
        )
        for uid in uids
    )
    report = {
        "schema_version": "silver-match-v3-two-order-verifier-score-v1",
        "selection_split": "dev" if args.explicit_role in (None, "select") else "optimize",
        "explicit_role": args.explicit_role,
        "order_stability": {
            "decision_agreement": safe_rate(order_decision, len(uids)),
            "exact_decision_and_id_agreement": safe_rate(order_outcome, len(uids)),
        },
        "policies": {
            "high_only": score_policy(
                truth,
                values["primary"],
                values["original"],
                values["hashed"],
                allowed_confidence={"high"},
            ),
            "medium_or_high": score_policy(
                truth,
                values["primary"],
                values["original"],
                values["hashed"],
                allowed_confidence={"high", "medium"},
            ),
        },
        "input_hashes": {key: sha256_file(path) for key, path in paths.items()},
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
