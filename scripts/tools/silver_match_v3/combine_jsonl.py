#!/usr/bin/env python3
"""Atomically combine JSONL inputs while enforcing a unique key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default="norm_uid")
    args = parser.parse_args()
    seen = set()
    inputs = [Path(path).resolve() for path in args.inputs]
    output = Path(args.output).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite combined artifact: {output}")
    if len(set(inputs)) != len(inputs) or any(not path.exists() for path in inputs):
        raise ValueError("combine inputs must be distinct existing files")
    input_hashes = {str(path): sha256_file(path) for path in inputs}
    input_counts = {str(path): 0 for path in inputs}

    def rows():
        for source in inputs:
            for row in read_jsonl(source):
                key = row.get(args.key)
                if key in (None, ""):
                    raise ValueError(f"missing {args.key} in {source}")
                if key in seen:
                    raise ValueError(f"duplicate {args.key}={key}")
                seen.add(key)
                input_counts[str(source)] += 1
                yield row

    count = write_jsonl(output, rows())
    summary = {
        "schema_version": "silver-match-v3-combined-jsonl-v1",
        "key": args.key,
        "inputs": {
            path: {"count": input_counts[path], "sha256": digest}
            for path, digest in input_hashes.items()
        },
        "output": str(output),
        "count": count,
        "sha256": sha256_file(output),
    }
    meta_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
