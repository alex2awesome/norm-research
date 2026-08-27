#!/usr/bin/env python3
"""Audit a frozen binary CE on development pairs without touching heldout data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist, mean, median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if str(row.get("split", "")).lower() != "dev":
                    raise ValueError(f"non-dev row at {path}:{number}")
                rows.append(row)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("empty quantile input")
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def count_buckets(counts: list[int]) -> dict[str, int]:
    result = {"0": 0, "1": 0, "2": 0, "3-5": 0, "6+": 0}
    for value in counts:
        key = "0" if value == 0 else "1" if value == 1 else "2" if value == 2 else "3-5" if value <= 5 else "6+"
        result[key] += 1
    return result


def concentration(counter: Counter[str], names: dict[str, str], limit: int = 20) -> dict[str, Any]:
    total = sum(counter.values())
    shares = [count / total for count in counter.values()] if total else []
    return {
        "total_retained_pair_predictions": total,
        "unique_metrics": len(counter),
        "largest_metric_share": max(shares, default=0.0),
        "herfindahl_index": sum(share * share for share in shares),
        "top": [
            {"metric_id": metric, "name": names.get(metric), "count": count, "share": count / total}
            for metric, count in counter.most_common(limit)
        ] if total else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-threshold", type=float, default=0.9960545301437378)
    parser.add_argument("--additional-threshold", type=float, action="append", default=[])
    args = parser.parse_args()

    score_path, bank_path, output = map(lambda value: Path(value).resolve(), (args.scores, args.bank, args.output))
    if output.exists():
        raise FileExistsError(output)
    rows = read_jsonl(score_path)
    bank_doc = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = bank_doc["metrics"]
    names = {str(row["metric_id"]): str(row["name"]) for row in metrics}
    ancestry = {str(row["metric_id"]): [str(x) for x in row.get("source_r2_cluster_ids", [])] for row in metrics}

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str]] = set()
    probabilities = []
    for row in rows:
        uid, metric = str(row.get("norm_uid", "")), str(row.get("metric_id", ""))
        if not uid or not metric or (uid, metric) in identities:
            raise ValueError(f"missing/duplicate pair: {(uid, metric)}")
        identities.add((uid, metric))
        if metric not in names:
            raise ValueError(f"score metric absent from bank: {metric}")
        probs = row.get("probabilities") or {}
        if set(probs) != {"EXACT", "REJECT"}:
            raise ValueError(f"invalid probability labels: {(uid, metric)}")
        exact, reject = float(probs["EXACT"]), float(probs["REJECT"])
        if not (0 <= exact <= 1 and 0 <= reject <= 1 and abs(exact + reject - 1) <= 1e-4):
            raise ValueError(f"invalid probabilities: {(uid, metric)}")
        if str(row.get("gold_relation")) not in {"EXACT", "FAMILY", "REJECT"}:
            raise ValueError(f"invalid gold relation: {(uid, metric)}")
        row = dict(row)
        row["exact_probability"] = exact
        groups[uid].append(row)
        probabilities.append(exact)

    gold: dict[str, set[str]] = {
        uid: {str(row["metric_id"]) for row in candidates if row["gold_relation"] == "EXACT"}
        for uid, candidates in groups.items()
    }
    gold_positive = sum(bool(ids) for ids in gold.values())
    pair_gold_positive = sum(len(ids) for ids in gold.values())
    top1 = {uid: max(candidates, key=lambda row: (row["exact_probability"], str(row["metric_id"]))) for uid, candidates in groups.items()}
    top1_correct = sum(str(row["metric_id"]) in gold[uid] for uid, row in top1.items())
    top1_metrics = Counter(str(row["metric_id"]) for row in top1.values())

    fixed = [0.0, 0.5, 0.9, 0.95, 0.98, 0.9871788024902344, 0.99, 0.995,
             args.frozen_threshold, 0.997, 0.999, 0.9995, 0.9999, 1.0]
    thresholds = sorted(set(fixed + args.additional_threshold))
    surface = []
    for threshold in thresholds:
        retained = {
            uid: [row for row in candidates if row["exact_probability"] >= threshold]
            for uid, candidates in groups.items()
        }
        counts = [len(retained[uid]) for uid in groups]
        accepted = sum(value > 0 for value in counts)
        retained_pairs = sum(counts)
        true_pairs = sum(
            str(row["metric_id"]) in gold[uid] for uid, candidates in retained.items() for row in candidates
        )
        retained_family_pairs = sum(
            row["gold_relation"] == "FAMILY" for candidates in retained.values() for row in candidates
        )
        any_correct = sum(any(str(row["metric_id"]) in gold[uid] for row in candidates) for uid, candidates in retained.items())
        exact_sets = sum({str(row["metric_id"]) for row in retained[uid]} == gold[uid] for uid in groups)
        pair_precision = true_pairs / retained_pairs if retained_pairs else 0.0
        pair_recall = true_pairs / pair_gold_positive if pair_gold_positive else 0.0
        norm_precision = any_correct / accepted if accepted else 0.0
        norm_recall = any_correct / gold_positive if gold_positive else 0.0
        metric_counts = Counter(str(row["metric_id"]) for candidates in retained.values() for row in candidates)
        false_abstentions = sum(bool(gold[uid]) and not retained[uid] for uid in groups)
        surface.append({
            "threshold": threshold,
            "retained_count_distribution": {
                "buckets": count_buckets(counts),
                "mean": mean(counts), "median": median(counts), "max": max(counts),
            },
            "predicted_positive_pairs": retained_pairs,
            "retained_gold_family_pairs_counted_as_binary_false_positive": retained_family_pairs,
            "predicted_positive_pair_base_rate": retained_pairs / len(rows),
            "accepted_norms": accepted,
            "coverage": accepted / len(groups),
            "multi_metric_norms": sum(value > 1 for value in counts),
            "fraction_accepted_with_any_gold_retained": norm_precision,
            "pair": {"true_positive": true_pairs, "false_positive": retained_pairs - true_pairs,
                     "precision": pair_precision, "precision_wilson_95": wilson(true_pairs, retained_pairs),
                     "recall": pair_recall, "f1": f1(pair_precision, pair_recall)},
            "norm_any_correct": {"correct": any_correct, "precision": norm_precision,
                                 "precision_wilson_95": wilson(any_correct, accepted),
                                 "recall": norm_recall, "f1": f1(norm_precision, norm_recall)},
            "false_abstention": {"count": false_abstentions,
                                  "rate_among_gold_positive_norms": false_abstentions / gold_positive if gold_positive else 0.0},
            "exact_retained_set": {"count": exact_sets, "rate_all_norms": exact_sets / len(groups)},
            "metric_concentration": concentration(metric_counts, names),
            "degenerate_checks": {
                "all_pairs_positive": retained_pairs == len(rows),
                "all_pairs_zero": retained_pairs == 0,
                "all_accepted_norms_same_single_metric": bool(accepted) and len(metric_counts) == 1,
            },
        })

    frozen = next(row for row in surface if row["threshold"] == args.frozen_threshold)
    # R2 IDs are leaf ancestry, not an explicit family taxonomy.  Report their
    # frequency only as an exploded ancestry diagnostic, with no family claim.
    frozen_metric_counts = Counter(
        str(row["metric_id"]) for uid, candidates in groups.items()
        for row in candidates if row["exact_probability"] >= args.frozen_threshold
    )
    ancestry_counts: Counter[str] = Counter()
    for metric, count in frozen_metric_counts.items():
        for r2_id in ancestry[metric]:
            ancestry_counts[r2_id] += count

    report = {
        "schema_version": "silver-match-v3-binary-ce-dev-surface-audit-v1",
        "role": "dev_analysis_only",
        "test_or_blind_rows_read": 0,
        "inputs": {"scores": {"path": str(score_path), "sha256": sha256_file(score_path)},
                   "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)}},
        "coverage": {"pair_rows": len(rows), "norms": len(groups), "bank_metrics": len(metrics),
                     "scored_unique_metrics": len({metric for _, metric in identities}),
                     "gold_relation_pair_counts": dict(Counter(str(row["gold_relation"]) for row in rows)),
                     "gold_positive_pairs": pair_gold_positive, "gold_positive_norms": gold_positive,
                     "pairs_per_norm": {"mean": mean(len(x) for x in groups.values()),
                                        "median": median(len(x) for x in groups.values()),
                                        "min": min(len(x) for x in groups.values()), "max": max(len(x) for x in groups.values())}},
        "score_distribution": {str(q): quantile(probabilities, q) for q in (0, .001, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999, 1)},
        "top1_exact": {"correct": top1_correct, "all_norm_precision": top1_correct / len(groups),
                       "recall_of_gold_positive_norms": top1_correct / gold_positive if gold_positive else 0.0,
                       "unique_top1_metrics": len(top1_metrics), "largest_top1_metric_share": max(top1_metrics.values()) / len(groups)},
        "frozen_threshold": args.frozen_threshold,
        "frozen_summary": frozen,
        "family_concentration_note": "No explicit metric-family field exists in the frozen bank; source_r2_cluster_ids are exploded ancestry, not asserted families.",
        "frozen_exploded_r2_ancestry": {"unique_r2_ids": len(ancestry_counts), "top": ancestry_counts.most_common(20)},
        "threshold_surface": surface,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"coverage": report["coverage"], "top1_exact": report["top1_exact"],
                      "frozen_summary": frozen, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
