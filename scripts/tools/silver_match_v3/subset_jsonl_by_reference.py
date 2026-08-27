#!/usr/bin/env python3
"""Project a unique-key JSONL artifact onto an immutable reference UID set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(rows: list[dict[str, Any]], key: str, path: Path) -> dict[str, dict[str, Any]]:
    output = {str(row[key]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate {key} values: {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default="norm_uid")
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    reference_path = Path(args.reference).resolve()
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite JSONL subset: {output_path}")
    inputs = list(read_jsonl(input_path))
    reference = list(read_jsonl(reference_path))
    by_key = _index(inputs, args.key, input_path)
    reference_by_key = _index(reference, args.key, reference_path)
    missing = sorted(set(reference_by_key) - set(by_key))
    if missing:
        raise ValueError(f"input does not cover reference keys: {missing[:3]}")
    rows = [by_key[str(row[args.key])] for row in reference]
    write_jsonl(output_path, rows)
    report = {
        "schema_version": "silver-match-v3-jsonl-reference-subset-v1",
        "key": args.key,
        "input_count": len(inputs),
        "reference_count": len(reference),
        "output_count": len(rows),
        "inputs": {
            "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
            "reference": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
