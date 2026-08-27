#!/usr/bin/env python3
"""Render self-contained candidate-card shards for independent labelers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, write_jsonl


def render(
    items: list[dict],
    candidate_by_uid: dict[str, dict],
    metric_by_id: dict[str, dict],
    candidate_k: int,
    include_examples: bool = True,
) -> list[dict]:
    output = []
    for item in items:
        uid = item["norm_uid"]
        if uid not in candidate_by_uid:
            raise KeyError(f"missing candidates for {uid}")
        cards = []
        for candidate in candidate_by_uid[uid].get("candidates", [])[:candidate_k]:
            metric = metric_by_id.get(candidate["metric_id"])
            if metric is None:
                raise KeyError(f"unknown metric {candidate['metric_id']} for {uid}")
            card = {
                "rank": candidate["rank"],
                "metric_id": metric["metric_id"],
                "name": metric["name"],
                "description": metric["description"],
            }
            if include_examples:
                card["examples"] = metric.get("examples") or []
            cards.append(card)
        output.append(
            {
                "norm_uid": uid,
                "corpus": item["corpus"],
                "task": item["task"],
                "norm": item["norm"],
                "context": item.get("context"),
                "aspect": item.get("aspect"),
                "polarity": item.get("polarity"),
                "split": item.get("split"),
                "candidates": cards,
                "label": {
                    "decision": None,
                    "metric_id": None,
                    "confidence": None,
                    "reason": None,
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--shard-size", type=int, default=60)
    parser.add_argument(
        "--omit-examples",
        action="store_true",
        help="Omit bank examples from candidate cards for compact labeling shards.",
    )
    args = parser.parse_args()

    items = sorted(read_jsonl(Path(args.items)), key=lambda row: row["norm_uid"])
    candidates = {row["norm_uid"]: row for row in read_jsonl(Path(args.candidates))}
    bank_payload = json.loads(Path(args.bank).read_text(encoding="utf-8"))
    metrics = bank_payload["metrics"]
    metric_by_id = {metric["metric_id"]: metric for metric in metrics}
    rows = render(
        items,
        candidates,
        metric_by_id,
        args.candidate_k,
        include_examples=not args.omit_examples,
    )
    output_root = Path(args.output_root)
    task = rows[0]["task"] if rows else bank_payload["task"]
    paths = []
    for start in range(0, len(rows), args.shard_size):
        path = output_root / f"{task}.part-{start // args.shard_size:03d}.jsonl"
        write_jsonl(path, rows[start : start + args.shard_size])
        paths.append(str(path))
    summary = {
        "task": task,
        "count": len(rows),
        "candidate_k": args.candidate_k,
        "include_examples": not args.omit_examples,
        "shard_size": args.shard_size,
        "parts": paths,
        "full_bank": str(Path(args.bank)),
        "full_bank_count": len(metrics),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{task}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
