#!/usr/bin/env python3
"""Extract one task's immutable MI certificate rows from a combined table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--canonical-task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = json.loads(source.read_text(encoding="utf-8"))
    table = payload.get("table") if isinstance(payload, dict) else payload
    rows = [
        row for row in table
        if isinstance(row, dict) and row.get("task") == args.source_task
        and row.get("opt_omega_bits") is not None
    ]
    if not rows:
        raise ValueError("no task MI rows found")
    result = {
        "schema_version": "silver-match-v3-extracted-mi-certificate-v1",
        "task": args.canonical_task,
        "source_task": args.source_task,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "table": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "task": args.canonical_task,
        "rows": len(rows),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
