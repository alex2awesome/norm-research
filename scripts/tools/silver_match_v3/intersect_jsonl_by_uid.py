#!/usr/bin/env python3
"""Freeze the ordered UID intersection of a JSONL source and reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, reference, output = map(
        lambda value: Path(value).resolve(), (args.source, args.reference, args.output)
    )
    meta = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta.exists():
        raise FileExistsError(output)
    reference_rows = list(read_jsonl(reference))
    reference_uids = [str(row.get("norm_uid") or "") for row in reference_rows]
    if not reference_rows or "" in reference_uids or len(reference_uids) != len(set(reference_uids)):
        raise ValueError("reference has empty, missing, or duplicate norm_uid values")
    source_rows = list(read_jsonl(source))
    source_uids = [str(row.get("norm_uid") or "") for row in source_rows]
    if not source_rows or "" in source_uids or len(source_uids) != len(set(source_uids)):
        raise ValueError("source has empty, missing, or duplicate norm_uid values")
    wanted = set(reference_uids)
    rows = [row for row in source_rows if str(row["norm_uid"]) in wanted]
    if not rows:
        raise ValueError("source/reference UID intersection is empty")
    write_jsonl(output, rows)
    report = {
        "schema_version": "silver-match-v3-jsonl-uid-intersection-v1",
        "source_count": len(source_rows),
        "reference_count": len(reference_rows),
        "intersection_count": len(rows),
        "source_rows_outside_reference": len(source_rows) - len(rows),
        "reference_rows_absent_from_source": len(wanted - set(source_uids)),
        "inputs": {
            "source": {"path": str(source), "sha256": sha256_file(source)},
            "reference": {"path": str(reference), "sha256": sha256_file(reference)},
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    meta.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(meta)}, sort_keys=True))


if __name__ == "__main__":
    main()
