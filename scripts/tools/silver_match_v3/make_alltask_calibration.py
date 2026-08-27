#!/usr/bin/env python3
"""Freeze a source-disjoint, corpus-covered calibration sample for all tasks."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_jsonl
from .config import DEFAULT_OUTPUT_ROOT
from .make_calibration import split_for, split_group_for


def rank_value(group: str, uid: str) -> int:
    return int(hashlib.sha256(f"{group}\0{uid}".encode("utf-8")).hexdigest(), 16)


def push_bottom_k(heap: list, row: dict[str, Any], k: int) -> None:
    """Retain the k lowest deterministic ranks in a max-heap."""
    value = rank_value(row["split_group"], row["norm_uid"])
    item = (-value, row["norm_uid"], row)
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def select_rows(
    by_corpus: dict[str, list[dict[str, Any]]],
    corpus_to_task: dict[str, str],
    *,
    per_task: int,
    min_per_corpus: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_uids: set[str] = set()
    tasks: dict[str, list[str]] = defaultdict(list)
    for corpus, task in corpus_to_task.items():
        tasks[task].append(corpus)
    for task, corpora in sorted(tasks.items()):
        task_rows = []
        for corpus in sorted(corpora):
            ordered = sorted(
                by_corpus.get(corpus, []),
                key=lambda row: (rank_value(row["split_group"], row["norm_uid"]), row["norm_uid"]),
            )
            for row in ordered[:min_per_corpus]:
                if row["norm_uid"] not in selected_uids:
                    selected.append(row)
                    selected_uids.add(row["norm_uid"])
            task_rows.extend(ordered[min_per_corpus:])
        remaining = max(0, per_task - sum(row["task"] == task for row in selected))
        for row in sorted(
            task_rows,
            key=lambda row: (rank_value(row["split_group"], row["norm_uid"]), row["norm_uid"]),
        )[:remaining]:
            if row["norm_uid"] not in selected_uids:
                selected.append(row)
                selected_uids.add(row["norm_uid"])
    return sorted(selected, key=lambda row: (row["task"], row["corpus"], row["norm_uid"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "alltask_calibration"))
    parser.add_argument("--per-task", type=int, default=240)
    parser.add_argument("--min-per-corpus", type=int, default=20)
    parser.add_argument(
        "--validity-stratum",
        choices=("all", "old_valid", "new_faithful_only"),
        default="all",
        help="optionally isolate the old narrow extraction-validity strata",
    )
    args = parser.parse_args()
    if args.per_task <= 0 or args.min_per_corpus < 0:
        parser.error("sample sizes must be non-negative, with --per-task > 0")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heaps: dict[str, list] = defaultdict(list)
    seen_groups: set[str] = set()
    # Keep enough candidates per corpus to satisfy its floor and still allow a
    # task-wide fill without loading the 1.35M-row universe into memory.
    heap_size = args.per_task + args.min_per_corpus
    for corpus, meta in sorted(manifest["corpora"].items()):
        for row in read_jsonl(Path(meta["path"])):
            extraction_valid = int(row.get("extraction_valid", 1))
            if args.validity_stratum == "old_valid" and extraction_valid != 1:
                continue
            if args.validity_stratum == "new_faithful_only" and extraction_valid != 0:
                continue
            group = split_group_for(row)
            if group in seen_groups:
                continue
            seen_groups.add(group)
            item = {
                **row,
                "split_group": group,
                "split": split_for(group),
            }
            push_bottom_k(heaps[corpus], item, heap_size)
    by_corpus = {
        corpus: [entry[2] for entry in heap]
        for corpus, heap in heaps.items()
    }
    selected = select_rows(
        by_corpus,
        {corpus: meta["task"] for corpus, meta in manifest["corpora"].items()},
        per_task=args.per_task,
        min_per_corpus=args.min_per_corpus,
    )

    output_root = Path(args.output_root)
    write_jsonl(output_root / "items.jsonl", selected)
    by_task: dict[str, list[dict]] = defaultdict(list)
    by_corpus_selected: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_task[row["task"]].append(row)
        by_corpus_selected[row["corpus"]].append(row)
    for task, rows in sorted(by_task.items()):
        write_jsonl(output_root / "tasks" / f"{task}.jsonl", rows)
    for corpus, rows in sorted(by_corpus_selected.items()):
        uid_path = output_root / "uids" / f"{corpus}.txt"
        uid_path.parent.mkdir(parents=True, exist_ok=True)
        uid_path.write_text("".join(f"{row['norm_uid']}\n" for row in rows), encoding="utf-8")

    summary = {
        "manifest": str(manifest_path),
        "manifest_total_norms": manifest["total_norms"],
        "per_task_target": args.per_task,
        "min_per_corpus_target": args.min_per_corpus,
        "validity_stratum": args.validity_stratum,
        "total": len(selected),
        "unique_split_groups": len({row["split_group"] for row in selected}),
        "by_task": {task: len(rows) for task, rows in sorted(by_task.items())},
        "by_corpus": {
            corpus: len(rows) for corpus, rows in sorted(by_corpus_selected.items())
        },
        "by_split": {
            split: sum(row["split"] == split for row in selected)
            for split in ("train", "dev", "test")
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
