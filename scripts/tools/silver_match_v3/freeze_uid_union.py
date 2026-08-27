#!/usr/bin/env python3
"""Freeze the sorted unique norm_uid union of one or more JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import normalize_space, read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = [Path(value).resolve() for value in args.input]
    output = Path(args.output).resolve()
    meta = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta.exists():
        raise FileExistsError(output)
    uids: set[str] = set()
    rows = 0
    for path in inputs:
        for row in read_jsonl(path):
            rows += 1
            uid = normalize_space(row.get("norm_uid"))
            if not uid:
                raise ValueError(f"missing norm_uid in {path}")
            uids.add(uid)
    if not uids:
        raise ValueError("empty UID union")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{uid}\n" for uid in sorted(uids)), encoding="utf-8")
    payload = {
        "schema_version": "silver-match-v3-frozen-uid-union-v1",
        "inputs": {str(path): sha256_file(path) for path in inputs},
        "input_rows": rows,
        "unique_uids": len(uids),
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    meta.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload, "meta_sha256": sha256_file(meta)}, sort_keys=True))


if __name__ == "__main__":
    main()
