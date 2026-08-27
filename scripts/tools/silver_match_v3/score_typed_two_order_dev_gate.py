#!/usr/bin/env python3
"""Score a conservative two-order typed exact-leaf gate on dev only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping


RANK = {"low": 0, "medium": 1, "high": 2}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index(path: Path, role: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows(path):
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in result or row.get("split") != "dev":
            raise ValueError(f"invalid {role} dev identity: {uid}")
        result[uid] = row
    if not result:
        raise ValueError(f"empty {role}")
    return result


def wilson(success: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = NormalDist().inv_cdf(0.975)
    p = success / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    r = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [max(0.0, c - r), min(1.0, c + r)]


def truth_ids(row: Mapping[str, Any]) -> set[str]:
    if row.get("decision") != "MATCH":
        return set()
    values = row.get("acceptable_metric_ids") or [row.get("metric_id")]
    return {str(value) for value in values if value not in (None, "")}


def statement(row: Mapping[str, Any]) -> str:
    content = str((row.get("messages") or [{"content": ""}])[0].get("content") or "")
    start = "HUMAN STATEMENT (verbatim):\n"
    end = "\nCONTEXT (capped at 600 characters):"
    if start in content and end in content:
        return content.split(start, 1)[1].split(end, 1)[0].strip()
    return content[:500]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--reordered", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--minimum-wilson-lower", type=float, default=0.85)
    parser.add_argument("--minimum-predictions", type=int, default=100)
    parser.add_argument("--examples-per-bucket", type=int, default=10)
    args = parser.parse_args()
    paths = {key: Path(value).resolve() for key, value in {
        "truth": args.truth, "original": args.original, "reordered": args.reordered
    }.items()}
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    truth, original, reordered = (
        index(paths["truth"], "truth"), index(paths["original"], "original"),
        index(paths["reordered"], "reordered")
    )
    if set(truth) != set(original) or set(truth) != set(reordered):
        raise ValueError("paired-order dev coverage differs from truth")
    gold_match = sum(bool(truth_ids(row)) for row in truth.values())
    reports = []
    accepted_by_threshold: dict[str, list[tuple[str, str]]] = {}
    for confidence in ("high", "medium", "low"):
        threshold = RANK[confidence]
        accepted = []
        for uid in sorted(truth):
            left, right = original[uid], reordered[uid]
            if (
                left.get("decision") == right.get("decision") == "MATCH"
                and left.get("metric_id") not in (None, "")
                and left.get("metric_id") == right.get("metric_id")
                and RANK.get(str(left.get("confidence") or ""), -1) >= threshold
                and RANK.get(str(right.get("confidence") or ""), -1) >= threshold
            ):
                accepted.append((uid, str(left["metric_id"])))
        correct = sum(metric in truth_ids(truth[uid]) for uid, metric in accepted)
        support = len(accepted)
        precision = correct / support if support else 0.0
        recall = correct / gold_match if gold_match else 0.0
        interval = wilson(correct, support)
        f05 = 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0
        passed = bool(
            support >= args.minimum_predictions and precision >= args.minimum_precision
            and interval and interval[0] >= args.minimum_wilson_lower
        )
        reports.append({
            "minimum_confidence": confidence, "accepted": support, "correct": correct,
            "precision": precision, "precision_wilson_95": interval,
            "recall_of_gold_matches": recall, "f_beta_0_5": f05, "gate_passed": passed,
        })
        accepted_by_threshold[confidence] = accepted
    feasible = [row for row in reports if row["gate_passed"]]
    if feasible:
        chosen = max(feasible, key=lambda row: (row["accepted"], row["precision_wilson_95"][0]))
        status = "PASS_VALIDATED_DEV_TWO_ORDER_GATE"
    else:
        chosen = max(reports, key=lambda row: (
            row["precision_wilson_95"][0] if row["precision_wilson_95"] else -1,
            row["precision"], row["accepted"],
        ))
        status = "FAIL_NO_DEV_TWO_ORDER_GATE_MEETS_CONTRACT"
    selected = accepted_by_threshold[chosen["minimum_confidence"]]
    selected_map = dict(selected)
    buckets: dict[str, list[dict[str, Any]]] = {"correct_accept": [], "false_accept": [], "unstable_gold_match": []}
    for uid in sorted(truth):
        base = {
            "norm_uid": uid, "human_statement": statement(truth[uid]),
            "gold_decision": truth[uid].get("decision"), "gold_metric_id": truth[uid].get("metric_id"),
            "original": {key: original[uid].get(key) for key in ("decision", "metric_id", "confidence", "reason")},
            "reordered": {key: reordered[uid].get(key) for key in ("decision", "metric_id", "confidence", "reason")},
        }
        if uid in selected_map:
            bucket = "correct_accept" if selected_map[uid] in truth_ids(truth[uid]) else "false_accept"
        elif truth_ids(truth[uid]):
            bucket = "unstable_gold_match"
        else:
            continue
        if len(buckets[bucket]) < args.examples_per_bucket:
            buckets[bucket].append(base)
    order_exact = sum(
        (original[uid].get("decision"), original[uid].get("metric_id"))
        == (reordered[uid].get("decision"), reordered[uid].get("metric_id"))
        for uid in truth
    )
    report = {
        "schema_version": "silver-match-v3-humor-typed-two-order-dev-gate-v1",
        "status": status, "role": "dev_selection_only", "test_or_blind_rows_read": 0,
        "n": len(truth), "gold_match_count": gold_match,
        "order_exact_agreement": order_exact / len(truth),
        "original_decisions": dict(sorted(Counter(row.get("decision") for row in original.values()).items())),
        "reordered_decisions": dict(sorted(Counter(row.get("decision") for row in reordered.values()).items())),
        "threshold_reports": reports, "selected_gate": chosen,
        "examples": buckets,
        "policy": {"minimum_precision": args.minimum_precision, "minimum_wilson_lower": args.minimum_wilson_lower, "minimum_predictions": args.minimum_predictions},
        "inputs": {key: {"path": str(path), "sha256": sha(path)} for key, path in paths.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "selected_gate": chosen, "order_exact_agreement": report["order_exact_agreement"]}, sort_keys=True))


if __name__ == "__main__":
    main()
