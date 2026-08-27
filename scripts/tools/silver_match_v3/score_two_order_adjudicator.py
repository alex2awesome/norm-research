#!/usr/bin/env python3
"""Score original/hashed adjudicator outputs and their strict consensus."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_jsonl, sha256_file


def safe(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(
    truth: Sequence[Mapping[str, Any]],
    original: Mapping[str, Mapping[str, Any]],
    hashed: Mapping[str, Mapping[str, Any]],
    *,
    include_by_corpus: bool = True,
) -> dict[str, Any]:
    match_truth = [row for row in truth if row["decision"] == "MATCH"]
    abstain_truth = [row for row in truth if row["decision"] != "MATCH"]

    def mode_report(predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        predicted_matches = [row for row in truth if predictions[row["norm_uid"]]["decision"] == "MATCH"]
        exact_matches = [
            row
            for row in match_truth
            if predictions[row["norm_uid"]]["decision"] == "MATCH"
            and predictions[row["norm_uid"]].get("metric_id") == row.get("metric_id")
        ]
        typed = [
            row
            for row in abstain_truth
            if predictions[row["norm_uid"]]["decision"] == row["decision"]
        ]
        exact_all = len(exact_matches) + len(typed)
        return {
            "exact_label_correct": exact_all,
            "exact_label_accuracy": safe(exact_all, len(truth)),
            "predicted_match_count": len(predicted_matches),
            "binary_match_precision": safe(
                sum(row["decision"] == "MATCH" for row in predicted_matches),
                len(predicted_matches),
            ),
            "binary_match_recall": safe(
                sum(predictions[row["norm_uid"]]["decision"] == "MATCH" for row in match_truth),
                len(match_truth),
            ),
            "exact_id_correct": len(exact_matches),
            "exact_id_precision_among_predicted_matches": safe(
                len(exact_matches), len(predicted_matches)
            ),
            "exact_id_recall_of_truth_matches": safe(len(exact_matches), len(match_truth)),
            "typed_abstention_correct": len(typed),
            "typed_abstention_accuracy": safe(len(typed), len(abstain_truth)),
        }

    exact_order = decision_order = confirmed = correct_confirmed = 0
    strict_typed = 0
    for row in truth:
        uid = row["norm_uid"]
        left, right = original[uid], hashed[uid]
        decision_order += left["decision"] == right["decision"]
        exact_order += (left["decision"], left.get("metric_id")) == (
            right["decision"],
            right.get("metric_id"),
        )
        if left["decision"] == right["decision"] == "MATCH" and left.get(
            "metric_id"
        ) == right.get("metric_id"):
            confirmed += 1
            correct_confirmed += row["decision"] == "MATCH" and left.get(
                "metric_id"
            ) == row.get("metric_id")
        if (
            row["decision"] != "MATCH"
            and left["decision"] == right["decision"] == row["decision"]
        ):
            strict_typed += 1
    strict_correct = correct_confirmed + strict_typed
    report = {
        "n": len(truth),
        "truth_match_count": len(match_truth),
        "truth_typed_abstention_count": len(abstain_truth),
        "original": mode_report(original),
        "hashed": mode_report(hashed),
        "order": {
            "decision_agreement": safe(decision_order, len(truth)),
            "exact_decision_and_id_agreement": safe(exact_order, len(truth)),
            "disagreement_count": len(truth) - exact_order,
        },
        "strict_consensus": {
            "confirmed_match_count": confirmed,
            "correct_exact_id_count": correct_confirmed,
            "exact_id_precision": safe(correct_confirmed, confirmed),
            "exact_id_recall_of_truth_matches": safe(correct_confirmed, len(match_truth)),
            "strict_typed_abstention_correct": strict_typed,
            "strict_typed_abstention_accuracy": safe(strict_typed, len(abstain_truth)),
            "strict_exact_label_accuracy": safe(strict_correct, len(truth)),
            "unstable_or_rejected_count": len(truth) - confirmed - strict_typed,
        },
    }
    if include_by_corpus:
        by_corpus = {}
        for corpus in sorted({str(row.get("corpus") or "") for row in truth}):
            subset = [row for row in truth if str(row.get("corpus") or "") == corpus]
            by_corpus[corpus] = summarize(
                subset, original, hashed, include_by_corpus=False
            )["strict_consensus"]
        report["by_corpus"] = by_corpus
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--completed-marker", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        key: Path(value).resolve()
        for key, value in {
            "truth": args.truth,
            "original": args.original,
            "hashed": args.hashed,
            "completed_marker": args.completed_marker,
        }.items()
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    truth = list(read_jsonl(paths["truth"]))
    original = {str(row["norm_uid"]): row for row in read_jsonl(paths["original"])}
    hashed = {str(row["norm_uid"]): row for row in read_jsonl(paths["hashed"])}
    uids = {str(row["norm_uid"]) for row in truth}
    if len(uids) != len(truth) or set(original) != uids or set(hashed) != uids:
        raise ValueError("two-order score inputs do not have exact paired coverage")
    report = {
        "schema_version": "silver-match-v3-two-order-adjudicator-score-v1",
        "role": "sealed_frozen_test_reporting_only",
        "selection_performed": False,
        "metrics": summarize(truth, original, hashed),
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
