#!/usr/bin/env python3
"""Validate task-specific retriever teachers without loading the 8B model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file, write_jsonl
from .config import DEFAULT_OUTPUT_ROOT
from .train_nemotron_lora import load_universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--teachers", action="append", required=True)
    parser.add_argument("--teacher-manifest")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-seed", type=int, default=73129)
    parser.add_argument("--train-percent", type=int, default=80)
    parser.add_argument("--dev-percent", type=int, default=10)
    parser.add_argument("--allow-unhashed-teachers", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    teachers = tuple(Path(path).resolve() for path in args.teachers)
    teacher_manifest = Path(args.teacher_manifest).resolve() if args.teacher_manifest else None
    universe = load_universe(
        manifest,
        teachers,
        args.task,
        split_seed=args.split_seed,
        train_percent=args.train_percent,
        dev_percent=args.dev_percent,
        require_bank_hash=not args.allow_unhashed_teachers,
        teacher_manifest_path=teacher_manifest,
    )
    output_root = Path(args.output_root)
    rows = [
        {
            "norm_uid": row.norm_uid,
            "corpus": row.corpus,
            "task": row.task,
            "source_group": row.source_group,
            "split": row.split,
            "metric_id": row.metric_id,
            "acceptable_metric_ids": list(row.acceptable_metric_ids),
            "teacher_sources": list(row.teacher_sources),
            "supervision_strength": row.supervision_strength,
        }
        for row in universe.labels
    ]
    label_path = output_root / f"{args.task}.audited_labels.jsonl"
    write_jsonl(label_path, rows)
    report = {
        "task": args.task,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "teachers": {str(path): sha256_file(path) for path in teachers},
        "teacher_manifest": str(teacher_manifest) if teacher_manifest else None,
        "teacher_audit": universe.teacher_audit,
        "split_audit": universe.split_audit,
        "selected": len(rows),
        "strong": sum(row["supervision_strength"] == "strong" for row in rows),
        "weak_forced_top3": sum(
            row["supervision_strength"] == "weak_forced_top3" for row in rows
        ),
        "labels_sha256": sha256_file(label_path),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{args.task}.teacher_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
