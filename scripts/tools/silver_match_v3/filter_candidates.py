#!/usr/bin/env python3
"""Freeze a candidate subset from labeled UIDs/splits for GEPA rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", choices=("all", "train", "dev", "test"), default="all")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate_path, label_path, output_path = map(
        Path, (args.candidates, args.labels, args.output)
    )
    if output_path.exists():
        raise FileExistsError(output_path)
    allowed = {
        row["norm_uid"]
        for row in read_jsonl(label_path)
        if args.split == "all" or row.get("split") == args.split
    }
    rows = [row for row in read_jsonl(candidate_path) if row["norm_uid"] in allowed]
    if len(rows) != len(allowed):
        found = {row["norm_uid"] for row in rows}
        missing = sorted(allowed - found)
        raise ValueError(f"missing {len(missing)} labeled candidates; first={missing[:3]}")
    write_jsonl(output_path, rows)
    meta = {
        "candidates_sha256": sha256_file(candidate_path),
        "labels_sha256": sha256_file(label_path),
        "split": args.split,
        "count": len(rows),
        "output_sha256": sha256_file(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
