#!/usr/bin/env python3
"""Exclude reference UIDs or source groups from a JSONL file with provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--exclude", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-key", default="source_group")
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)).resolve() for name in ("input", "exclude")}
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    excluded_rows = list(read_jsonl(paths["exclude"]))
    excluded_uids = {str(row["norm_uid"]) for row in excluded_rows}
    excluded_groups = {
        str(row[args.group_key])
        for row in excluded_rows
        if row.get(args.group_key)
    }
    input_rows = list(read_jsonl(paths["input"]))
    kept = [
        row
        for row in input_rows
        if str(row["norm_uid"]) not in excluded_uids
        and (
            not row.get(args.group_key)
            or str(row[args.group_key]) not in excluded_groups
        )
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, kept)
    report = {
        "schema_version": "silver-match-v3-jsonl-exclusion-v1",
        "input_count": len(input_rows),
        "output_count": len(kept),
        "excluded_count": len(input_rows) - len(kept),
        "excluded_uid_count": len(excluded_uids),
        "excluded_group_count": len(excluded_groups),
        "group_key": args.group_key,
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "output_sha256": sha256_file(output),
    }
    report_path = output.with_suffix(output.suffix + ".meta.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
