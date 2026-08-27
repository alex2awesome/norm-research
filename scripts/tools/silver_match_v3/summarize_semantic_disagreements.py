#!/usr/bin/env python3
"""Summarize pairwise decision and exact-leaf conflicts in a semantic audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import read_jsonl, sha256_file


PAIRS = (
    ("semantic_pass1", "strict_three_pass"),
    ("semantic_pass1", "resolver_pass2"),
    ("strict_three_pass", "resolver_pass2"),
)


def key(row: dict) -> tuple[str, str | None]:
    decision = str(row.get("decision") or "")
    return decision, str(row["metric_id"]) if decision == "MATCH" else None


def category(left: tuple[str, str | None], right: tuple[str, str | None]) -> str:
    if left == right:
        return "exact_agreement"
    if left[0] == right[0] == "MATCH":
        return "exact_leaf_conflict"
    if "MATCH" in (left[0], right[0]):
        other = right[0] if left[0] == "MATCH" else left[0]
        return f"match_vs_{other.lower()}"
    return "typed_decision_conflict"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disagreements", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spot-uid", action="append", default=[])
    args = parser.parse_args()
    source, output = Path(args.disagreements).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = list(read_jsonl(source))
    pair_categories: dict[str, Counter[str]] = {}
    pair_leaf_confusions: dict[str, Counter[str]] = {}
    pair_decision_confusions: dict[str, Counter[str]] = {}
    for left_name, right_name in PAIRS:
        name = f"{left_name}_vs_{right_name}"
        categories: Counter[str] = Counter()
        leaf: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        for row in rows:
            predictions = row["source_predictions"]
            left, right = predictions.get(left_name), predictions.get(right_name)
            if left is None or right is None:
                continue
            left_key, right_key = key(left), key(right)
            value = category(left_key, right_key)
            categories[value] += 1
            if value != "exact_agreement":
                decisions[f"{left_key[0]}->{right_key[0]}"] += 1
            if value == "exact_leaf_conflict":
                leaf[f"{left_key[1]}->{right_key[1]}"] += 1
        pair_categories[name] = categories
        pair_leaf_confusions[name] = leaf
        pair_decision_confusions[name] = decisions
    by_availability = Counter(
        "+".join(
            name
            for name in ("semantic_pass1", "strict_three_pass", "resolver_pass2")
            if row["source_predictions"].get(name) is not None
        )
        for row in rows
    )
    spots = {
        uid: next((row for row in rows if str(row["norm_uid"]).startswith(uid)), None)
        for uid in args.spot_uid
    }
    report = {
        "schema_version": "silver-match-v3-semantic-disagreement-taxonomy-v1",
        "disagreement_rows": len(rows),
        "source_availability": dict(sorted(by_availability.items())),
        "pairwise_category_counts": {
            name: dict(sorted(values.items()))
            for name, values in pair_categories.items()
        },
        "pairwise_decision_confusions": {
            name: [
                {"confusion": value, "count": count}
                for value, count in values.most_common(20)
            ]
            for name, values in pair_decision_confusions.items()
        },
        "pairwise_top_exact_leaf_confusions": {
            name: [
                {"confusion": value, "count": count}
                for value, count in values.most_common(30)
            ]
            for name, values in pair_leaf_confusions.items()
        },
        "spot_rows": spots,
        "input": {"path": str(source), "sha256": sha256_file(source)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
