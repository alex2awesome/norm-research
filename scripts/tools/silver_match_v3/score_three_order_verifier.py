#!/usr/bin/env python3
"""Score a predeclared all-three-order, exact-high verifier policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file
from .score_verifier_calibration import safe_rate, wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--reverse", required=True)
    parser.add_argument("--selection-split", choices=("optimize", "dev"), default="optimize")
    parser.add_argument("--explicit-role", choices=("optimize", "select"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in ("truth", "primary", "original", "hashed", "reverse")
    }
    truth_rows = list(read_jsonl(paths["truth"]))
    values = {
        name: {str(row["norm_uid"]): row for row in read_jsonl(paths[name])}
        for name in ("primary", "original", "hashed", "reverse")
    }
    uids = {str(row["norm_uid"]) for row in truth_rows}
    if any(set(rows) != uids for rows in values.values()):
        raise ValueError("three-order inputs lack exact paired coverage")
    if args.explicit_role:
        expected_split = "optimize" if args.explicit_role == "optimize" else "dev"
        if args.selection_split != expected_split:
            raise ValueError("explicit verifier role does not match selection split")
        if any(
            row.get("gepa_role") != args.explicit_role or row.get("split") != "train"
            for row in truth_rows
        ):
            raise ValueError("verifier truth does not preserve its frozen explicit role")
    retained = retained_true = proposal_correct = 0
    for truth in truth_rows:
        uid = str(truth["norm_uid"])
        proposed = str(values["primary"][uid]["metric_id"])
        correct = truth.get("decision") == "MATCH" and str(truth.get("metric_id")) == proposed
        proposal_correct += correct
        outputs = [values[name][uid] for name in ("original", "hashed", "reverse")]
        keep = all(
            row.get("decision") == "CONFIRM_MATCH"
            and str(row.get("metric_id")) == proposed
            and row.get("confidence") == "high"
            and not row.get("parse_error")
            for row in outputs
        )
        retained += keep
        retained_true += keep and correct
    wrong = len(truth_rows) - proposal_correct
    false_retained = retained - retained_true
    report = {
        "schema_version": "silver-match-v3-three-order-verifier-score-v1",
        "selection_split": args.selection_split,
        "explicit_role": args.explicit_role,
        "policy": {
            "name": "all_three_exact_high",
            "n": len(truth_rows),
            "proposal_correct": proposal_correct,
            "proposal_exact_accuracy": safe_rate(proposal_correct, len(truth_rows)),
            "retained": retained,
            "retained_true": retained_true,
            "retained_precision": safe_rate(retained_true, retained),
            "retained_precision_wilson_95": wilson_interval(retained_true, retained),
            "retained_recall_of_correct_proposals": safe_rate(retained_true, proposal_correct),
            "wrong_proposal_rejection_rate": safe_rate(wrong - false_retained, wrong),
            "wrong_proposal_rejection_wilson_95": wilson_interval(wrong - false_retained, wrong),
        },
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
