#!/usr/bin/env python3
"""Create a deterministic EXACT-vs-rest CE train subset.

The release retains every known exact positive pair and exactly equal-sized
frozen hard/easy negative budgets.  FAMILY rows are adjacent-family or
retrieval-confusion negatives; easy rows must come from the builder's explicit
global-balanced-negative lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


SCHEMA = "silver-match-v3-binary-exact-train-subset-v1"


def _key(row: dict[str, Any], seed: int, role: str) -> str:
    value = "\x1f".join(
        (str(seed), role, str(row.get("norm_uid")), str(row.get("metric_id")))
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def materialize(
    source: Path, *, negative_count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []
    easy: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for row in read_jsonl(source):
        relation = str(row.get("relation") or "").upper()
        source_counts[relation] += 1
        if relation == "EXACT":
            positives.append(row)
        elif relation == "FAMILY":
            hard.append(row)
        elif relation == "REJECT" and "global_balanced_negative" in set(
            row.get("candidate_lanes") or []
        ):
            easy.append(row)
    if len(hard) < negative_count or len(easy) < negative_count or not positives:
        raise ValueError("source lacks the frozen positive/hard/easy budgets")
    hard = sorted(hard, key=lambda row: _key(row, seed, "hard"))[:negative_count]
    easy = sorted(easy, key=lambda row: _key(row, seed, "easy"))[:negative_count]
    selected = positives + hard + easy
    selected.sort(key=lambda row: _key(row, seed, "output"))
    keys = [(str(row.get("norm_uid")), str(row.get("metric_id"))) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("selected binary train pairs are not unique")
    report = {
        "schema_version": SCHEMA,
        "status": "FROZEN_ALL_POSITIVES_PLUS_BALANCED_HARD_EASY_NEGATIVES",
        "seed": seed,
        "binary_semantics": {"positive": ["EXACT"], "negative": ["FAMILY", "REJECT"]},
        "selection": {
            "all_exact_positives": len(positives),
            "hard_family_or_retrieval_confusions": len(hard),
            "easy_global_balanced": len(easy),
            "total": len(selected),
        },
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "relation_counts": dict(sorted(source_counts.items())),
        },
        "test_or_blind_rows_read": 0,
        "dev_rows_read": 0,
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--negative-count", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=2026071500)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite binary subset outputs")
    rows, report = materialize(
        Path(args.source).resolve(),
        negative_count=args.negative_count,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    report["output"] = {
        "path": str(output),
        "sha256": sha256_file(output),
        "count": len(rows),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
