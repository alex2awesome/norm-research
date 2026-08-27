#!/usr/bin/env python3
"""Score the frozen three-order PR R4 output exactly once on select truth."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


ORDERS = ("original", "hashed", "reverse")


def safe(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize_three(
    truth: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    include_by_corpus: bool = True,
) -> dict[str, Any]:
    match_truth = [row for row in truth if row["decision"] == "MATCH"]
    abstain_truth = [row for row in truth if row["decision"] != "MATCH"]

    def mode_report(order: str) -> dict[str, Any]:
        values = predictions[order]
        predicted_matches = [row for row in truth if values[row["norm_uid"]]["decision"] == "MATCH"]
        exact_matches = [
            row
            for row in match_truth
            if values[row["norm_uid"]]["decision"] == "MATCH"
            and values[row["norm_uid"]].get("metric_id") == row.get("metric_id")
        ]
        typed = [
            row
            for row in abstain_truth
            if values[row["norm_uid"]]["decision"] == row["decision"]
        ]
        return {
            "exact_label_accuracy": safe(len(exact_matches) + len(typed), len(truth)),
            "predicted_match_count": len(predicted_matches),
            "binary_match_precision": safe(
                sum(row["decision"] == "MATCH" for row in predicted_matches),
                len(predicted_matches),
            ),
            "binary_match_recall": safe(
                sum(values[row["norm_uid"]]["decision"] == "MATCH" for row in match_truth),
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

    exact_order = decision_order = confirmed = correct_confirmed = strict_typed = 0
    for row in truth:
        uid = row["norm_uid"]
        values = [predictions[order][uid] for order in ORDERS]
        decisions = {value["decision"] for value in values}
        keys = {(value["decision"], value.get("metric_id")) for value in values}
        decision_order += len(decisions) == 1
        exact_order += len(keys) == 1
        if len(keys) == 1 and values[0]["decision"] == "MATCH":
            confirmed += 1
            correct_confirmed += (
                row["decision"] == "MATCH"
                and values[0].get("metric_id") == row.get("metric_id")
            )
        if (
            row["decision"] != "MATCH"
            and len(decisions) == 1
            and values[0]["decision"] == row["decision"]
        ):
            strict_typed += 1
    strict_correct = correct_confirmed + strict_typed
    report = {
        "n": len(truth),
        "truth_match_count": len(match_truth),
        "truth_typed_abstention_count": len(abstain_truth),
        "modes": {order: mode_report(order) for order in ORDERS},
        "order": {
            "all_three_decision_agreement": safe(decision_order, len(truth)),
            "all_three_exact_decision_and_id_agreement": safe(exact_order, len(truth)),
            "any_exact_disagreement_count": len(truth) - exact_order,
        },
        "strict_all_three_consensus": {
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
        report["by_corpus"] = {
            corpus: summarize_three(
                [row for row in truth if str(row.get("corpus") or "") == corpus],
                predictions,
                include_by_corpus=False,
            )["strict_all_three_consensus"]
            for corpus in sorted({str(row.get("corpus") or "") for row in truth})
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-freeze", required=True)
    parser.add_argument("--truth-release", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--global-policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    freeze_path = Path(args.output_freeze).resolve()
    release_path = Path(args.truth_release).resolve()
    truth_path = Path(args.truth).resolve()
    policy_path = Path(args.global_policy).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite one-shot R4 select score")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        freeze.get("status") != "FROZEN_COMPLETE_BEFORE_TRUTH_JOIN"
        or freeze.get("orders") != list(ORDERS)
        or release.get("status") != "FROZEN_COMPLETE_EXACT_TRUTH"
        or release.get("task") != "press-releases"
        or release.get("role") != "select"
        or release["artifacts"]["truth"]["sha256"] != sha256_file(truth_path)
        or policy.get("adjudicator_policy", {}).get("select_orders") != list(ORDERS)
    ):
        raise ValueError("R4 score inputs violate the frozen three-order select contract")
    truth = list(read_jsonl(truth_path))
    if any(
        row.get("task") != "press-releases"
        or row.get("gepa_role") != "select"
        or row.get("prompt_selection_eligible") is not True
        or row.get("prompt_gradient_eligible") is not False
        or row.get("evaluation_only") is not True
        for row in truth
    ):
        raise ValueError("truth is not wholly select-role evaluation evidence")
    predictions = {}
    for order in ORDERS:
        value = freeze["outputs"][order]["predictions"]
        path = Path(value["path"])
        if sha256_file(path) != value["sha256"]:
            raise ValueError(f"frozen prediction drift: {order}")
        predictions[order] = {str(row["norm_uid"]): row for row in read_jsonl(path)}
    truth_uids = {str(row["norm_uid"]) for row in truth}
    if any(set(values) != truth_uids for values in predictions.values()):
        raise ValueError("frozen predictions do not exactly cover select truth")
    report = {
        "schema_version": "silver-match-v3-pr-r4-select-three-order-score-v1",
        "status": "SCORED_ONCE_NO_FURTHER_PROMPT_ITERATION_ALLOWED",
        "task": "press-releases",
        "role": "select",
        "prompt_sha256": freeze["prompt_sha256"],
        "model": freeze["model"],
        "metrics": summarize_three(truth, predictions),
        "inputs": {
            "output_freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
            "truth_release": {"path": str(release_path), "sha256": sha256_file(release_path)},
            "truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "global_policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        },
        "selection_performed": False,
        "new_prompt_variant_after_score_permitted": False,
        "production_promotion": {
            "status": "REQUIRES_INDEPENDENT_VERIFIER_AND_BLIND_AUDIT",
            "reason": "three-order adjudicator select score alone is not a production-quality match claim",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "score_sha256": sha256_file(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
