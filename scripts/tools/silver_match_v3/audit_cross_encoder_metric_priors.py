#!/usr/bin/env python3
"""Freeze evidence that a pointwise CE learned metric-card exposure priors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _selected_metric(report: dict[str, Any], name: str) -> float:
    return float(report["selected_dev"][name])


def build_audit(
    report_path: Path, pairs_path: Path, bank_path: Path, task: str
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    bank_ids = [str(row["metric_id"]) for row in bank]
    if len(bank_ids) != len(set(bank_ids)):
        raise ValueError("bank metric IDs are not unique")
    bank_id_set = set(bank_ids)
    if report.get("task") != task:
        raise ValueError("training report task mismatch")
    if report.get("frozen_test_consumed") is not False:
        raise ValueError("postmortem requires a dev-only report with blind/test sealed")

    positive: Counter[str] = Counter()
    negative: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    pair_count = 0
    for row in read_jsonl(pairs_path):
        metric_id = str(row.get("metric_id") or "")
        if metric_id not in bank_id_set:
            raise ValueError(f"training pair metric outside bank: {metric_id!r}")
        label = float(row["label"])
        if label not in {0.0, 1.0}:
            raise ValueError(f"non-binary pair label: {label}")
        (positive if label == 1.0 else negative)[metric_id] += 1
        labels[str(label)] += 1
        kinds[str(row.get("kind") or "MISSING")] += 1
        pair_count += 1
    if not pair_count or not positive or not negative:
        raise ValueError("postmortem requires both positive and negative pairs")

    exposure = []
    for metric_id in bank_ids:
        pos, neg = int(positive[metric_id]), int(negative[metric_id])
        total = pos + neg
        exposure.append(
            {
                "metric_id": metric_id,
                "positive_pairs": pos,
                "negative_pairs": neg,
                "positive_pair_fraction": pos / total if total else None,
                "negative_to_positive_ratio": neg / pos if pos else None,
                "smoothed_log_positive_odds": math.log((pos + 1) / (neg + 1)),
            }
        )
    positive_only = [row for row in exposure if row["positive_pairs"] and not row["negative_pairs"]]
    unexposed = [row for row in exposure if not row["positive_pairs"] and not row["negative_pairs"]]
    severely_positive_skewed = [
        row
        for row in exposure
        if row["positive_pairs"]
        and row["positive_pair_fraction"] is not None
        and row["positive_pair_fraction"] >= 0.98
    ]

    base = report["base_dev"]
    selected = report["selected_dev"]
    base_score_median = float(base["top_score_quantiles"]["0.5"])
    selected_score_median = float(selected["top_score_quantiles"]["0.5"])
    base_margin_median = float(base["margin_quantiles"]["0.5"])
    selected_margin_median = float(selected["margin_quantiles"]["0.5"])
    score_saturated = selected_score_median >= 0.99
    margin_collapsed = selected_margin_median <= 0.001
    exposure_failure = bool(positive_only or severely_positive_skewed or unexposed)
    diagnosis_passes = score_saturated and margin_collapsed and exposure_failure

    return {
        "schema_version": "silver-match-v3-cross-encoder-metric-prior-postmortem-v1",
        "status": (
            "CONFIRMED_METRIC_CARD_PRIOR_COLLAPSE"
            if diagnosis_passes
            else "DIAGNOSIS_NOT_CONFIRMED"
        ),
        "task": task,
        "inputs": {
            "training_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
            "training_pairs": {
                "path": str(pairs_path),
                "sha256": sha256_file(pairs_path),
            },
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        },
        "role_audit": {
            "selection_data": "development_only",
            "frozen_test_consumed": False,
            "fresh_select_consumed": False,
            "permanent_blind_consumed": False,
        },
        "pair_audit": {
            "pair_count": pair_count,
            "label_counts": dict(sorted(labels.items())),
            "kind_counts": dict(sorted(kinds.items())),
            "bank_metric_count": len(bank_ids),
            "positive_metric_count": len(positive),
            "negative_metric_count": len(negative),
            "positive_only_metric_count": len(positive_only),
            "unexposed_metric_count": len(unexposed),
            "severely_positive_skewed_metric_count": len(severely_positive_skewed),
            "positive_only_metrics": positive_only,
            "unexposed_metrics": unexposed,
            "highest_smoothed_positive_odds": sorted(
                exposure,
                key=lambda row: (
                    -float(row["smoothed_log_positive_odds"]),
                    row["metric_id"],
                ),
            )[:30],
            "per_metric_exposure": exposure,
        },
        "model_audit": {
            "base_top_score_median": base_score_median,
            "selected_top_score_median": selected_score_median,
            "base_margin_median": base_margin_median,
            "selected_margin_median": selected_margin_median,
            "base_ungated_exact_recall_at_50": float(
                base["ungated_exact_recall_at_50"]
            ),
            "selected_ungated_exact_recall_at_50": _selected_metric(
                report, "ungated_exact_recall_at_50"
            ),
            "selected_predicted_match_count": int(selected["predicted_match_count"]),
            "score_saturated": score_saturated,
            "margin_collapsed": margin_collapsed,
        },
        "diagnostic_rule": {
            "score_saturated": "selected dev median top score >= 0.99",
            "margin_collapsed": "selected dev median top1-top2 margin <= 0.001",
            "exposure_failure": (
                "at least one bank card is positive-only, >=98% positive, or unexposed"
            ),
            "all_required": True,
            "passed": diagnosis_passes,
        },
        "required_remediation": {
            "append_only_new_objective": True,
            "global_metric_balanced_negatives": True,
            "no_positive_only_bank_metric": True,
            "minimum_negative_exposure_gate": True,
            "maximum_positive_fraction_gate": True,
            "evaluation_unit": "one norm ranked against the complete frozen bank",
            "report_macro_metric_recall": True,
            "report_prediction_concentration": True,
            "reuse_v1_weights": False,
            "fresh_select_and_permanent_blind_remain_sealed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--training-pairs", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = build_audit(
        Path(args.training_report).resolve(),
        Path(args.training_pairs).resolve(),
        Path(args.bank).resolve(),
        args.task,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output), **result},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
