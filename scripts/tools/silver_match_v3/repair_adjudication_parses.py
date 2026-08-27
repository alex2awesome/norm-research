#!/usr/bin/env python3
"""Create an immutable repaired copy of adjudications with parser-only failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adjudicate_gemma import parse_response
from .common import read_jsonl, sha256_file, write_jsonl


def repair_row(row: dict) -> tuple[dict, bool]:
    if row.get("decision") != "INVALID_OUTPUT" or not row.get("raw_response"):
        return row, False
    parsed, error = parse_response(
        str(row["raw_response"]), set(row.get("candidate_ids") or [])
    )
    if parsed is None:
        return {**row, "repair_parse_error": error}, False
    return {
        **row,
        "decision": parsed["decision"],
        "metric_id": parsed["metric_id"],
        "confidence": parsed["confidence"],
        "reason": parsed["reason"],
        "parse_error": None,
        "repaired_from_parse_error": row.get("parse_error"),
    }, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    output, repaired = [], 0
    for row in read_jsonl(input_path):
        value, changed = repair_row(row)
        output.append(value)
        repaired += changed
    write_jsonl(output_path, output)
    meta = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "count": len(output),
        "repaired": repaired,
        "remaining_invalid": sum(row.get("decision") == "INVALID_OUTPUT" for row in output),
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
