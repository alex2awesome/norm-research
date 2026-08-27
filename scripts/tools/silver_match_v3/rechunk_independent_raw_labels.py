#!/usr/bin/env python3
"""Merge valid structured-label subchunks into one frozen target chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-chunk", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.target_chunk).resolve()
    inputs = [Path(value).resolve() for value in args.input]
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite rechunked labels")
    if len(inputs) < 2 or len(inputs) != len(set(inputs)):
        raise ValueError("provide at least two distinct raw-label subchunks")

    expected = [str(row["norm_uid"]) for row in read_jsonl(target)]
    if len(expected) != len(set(expected)):
        raise ValueError("target chunk contains duplicate UIDs")
    by_uid = {}
    source_chunks = {}
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task") != args.task or not isinstance(payload.get("labels"), list):
            raise ValueError(f"raw-label task/schema mismatch: {path}")
        values = payload["labels"]
        source_chunks[str(path)] = {
            "chunk_id": payload.get("chunk_id"),
            "count": len(values),
            "sha256": sha256_file(path),
        }
        for row in values:
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in by_uid:
                raise ValueError(f"raw-label subchunks contain missing/duplicate UID: {uid!r}")
            by_uid[uid] = row
    if set(by_uid) != set(expected):
        missing = sorted(set(expected) - set(by_uid))
        extra = sorted(set(by_uid) - set(expected))
        raise ValueError(
            f"subchunks do not exactly cover frozen target; missing={missing[:3]} extra={extra[:3]}"
        )

    payload = {
        "task": args.task,
        "chunk_id": target.stem,
        "labels": [by_uid[uid] for uid in expected],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "silver-match-v3-rechunked-independent-raw-labels-v1",
        "task": args.task,
        "target_chunk": {
            "path": str(target),
            "chunk_id": target.stem,
            "count": len(expected),
            "sha256": sha256_file(target),
        },
        "source_subchunks": source_chunks,
        "exact_uid_coverage": True,
        "label_rows_unmodified": True,
        "output": {
            "path": str(output),
            "count": len(expected),
            "sha256": sha256_file(output),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
