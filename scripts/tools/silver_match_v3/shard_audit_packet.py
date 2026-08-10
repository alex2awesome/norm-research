#!/usr/bin/env python3
"""Deterministically shard a blinded manual-audit packet by norm UID."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def shard_id(uid: str, count: int) -> int:
    return int(hashlib.sha256(uid.encode()).hexdigest()[:16], 16) % count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    source, root = Path(args.input).resolve(), Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {root}")
    root.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in range(args.num_shards)]
    for row in read_jsonl(source):
        shards[shard_id(str(row["norm_uid"]), args.num_shards)].append(row)
    outputs = {}
    for index, rows in enumerate(shards):
        path = root / f"shard-{index:02d}-of-{args.num_shards:02d}.jsonl"
        write_jsonl(path, rows)
        outputs[path.name] = {"count": len(rows), "sha256": sha256_file(path)}
    report = {
        "input": str(source),
        "input_sha256": sha256_file(source),
        "num_shards": args.num_shards,
        "outputs": outputs,
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
