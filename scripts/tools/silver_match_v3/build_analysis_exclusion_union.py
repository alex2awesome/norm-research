#!/usr/bin/env python3
"""Build an immutable UID union for leakage-safe task analysis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task")
    args = parser.parse_args()
    inputs = [Path(value).resolve() for value in args.input]
    if len(inputs) != len(set(inputs)) or any(not path.exists() for path in inputs):
        raise ValueError("exclusion inputs must be distinct existing files")
    output = Path(args.output).resolve()
    meta = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta.exists():
        raise FileExistsError(output)

    sources: dict[str, set[str]] = defaultdict(set)
    source_counts: dict[str, int] = {}
    for path in inputs:
        count = 0
        for row in read_jsonl(path):
            if args.task and row.get("task") not in {None, args.task}:
                continue
            uid = str(row.get("norm_uid") or "")
            if not uid:
                raise ValueError(f"missing norm_uid in exclusion input: {path}")
            sources[uid].add(str(path))
            count += 1
        source_counts[str(path)] = count
    if not sources:
        raise ValueError("empty analysis exclusion union")

    write_jsonl(
        output,
        (
            {
                "schema_version": "silver-match-v3-analysis-exclusion-union-v1",
                "norm_uid": uid,
                "task": args.task,
                "exclusion_sources": sorted(paths),
            }
            for uid, paths in sorted(sources.items())
        ),
    )
    report = {
        "schema_version": "silver-match-v3-analysis-exclusion-union-report-v1",
        "task": args.task,
        "count": len(sources),
        "inputs": {
            str(path): {
                "sha256": sha256_file(path),
                "rows_read": source_counts[str(path)],
            }
            for path in inputs
        },
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    meta.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "meta_sha256": sha256_file(meta)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
