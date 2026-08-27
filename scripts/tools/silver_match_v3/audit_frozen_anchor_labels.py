#!/usr/bin/env python3
"""Audit a previously frozen independent label artifact on hidden anchors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def _exact(predicted: dict[str, Any], gold: dict[str, Any]) -> bool:
    if predicted.get("decision") != gold.get("decision"):
        return False
    if gold.get("decision") == "MATCH":
        return predicted.get("metric_id") == gold.get("metric_id")
    return predicted.get("metric_id") is None


def audit(
    labels: list[dict[str, Any]], anchors: list[dict[str, Any]]
) -> dict[str, Any]:
    by_uid = {str(row["norm_uid"]): row for row in labels}
    if len(by_uid) != len(labels):
        raise ValueError("duplicate norm_uid in frozen labels")
    exact = decision = 0
    confusion: Counter[str] = Counter()
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    mismatches = []
    for gold in anchors:
        uid = str(gold["norm_uid"])
        if uid not in by_uid:
            raise ValueError(f"anchor absent from frozen labels: {uid}")
        predicted = by_uid[uid]
        decision_ok = predicted.get("decision") == gold.get("decision")
        exact_ok = _exact(predicted, gold)
        decision += int(decision_ok)
        exact += int(exact_ok)
        confusion[f"{gold.get('decision')}->{predicted.get('decision')}"] += 1
        stratum = str(gold.get("stratum") or "UNKNOWN")
        strata[stratum]["n"] += 1
        strata[stratum]["decision_correct"] += int(decision_ok)
        strata[stratum]["exact_correct"] += int(exact_ok)
        if not exact_ok:
            mismatches.append(
                {
                    "norm_uid": uid,
                    "stratum": stratum,
                    "gold_decision": gold.get("decision"),
                    "gold_metric_id": gold.get("metric_id"),
                    "predicted_decision": predicted.get("decision"),
                    "predicted_metric_id": predicted.get("metric_id"),
                    "predicted_confidence": predicted.get("confidence"),
                    "predicted_reason": predicted.get("reason"),
                }
            )
    total = len(anchors)
    return {
        "anchor_count": total,
        "decision_correct": decision,
        "decision_accuracy": decision / total if total else 0.0,
        "decision_accuracy_wilson95": wilson(decision, total),
        "exact_correct": exact,
        "exact_accuracy": exact / total if total else 0.0,
        "exact_accuracy_wilson95": wilson(exact, total),
        "confusion": dict(sorted(confusion.items())),
        "by_stratum": {
            key: dict(sorted(value.items())) for key, value in sorted(strata.items())
        },
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels_path = Path(args.labels).resolve()
    anchors_path = Path(args.anchors).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"immutable audit already exists: {output_path}")
    report = audit(list(read_jsonl(labels_path)), list(read_jsonl(anchors_path)))
    report.update(
        {
            "schema_version": "silver-match-v3-frozen-anchor-audit-v1",
            "frozen_labels": str(labels_path),
            "frozen_labels_sha256": sha256_file(labels_path),
            "hidden_anchors": str(anchors_path),
            "hidden_anchors_sha256": sha256_file(anchors_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "anchor_count", "decision_correct", "decision_accuracy",
        "exact_correct", "exact_accuracy", "frozen_labels_sha256",
    )}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
