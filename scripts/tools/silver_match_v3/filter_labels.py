#!/usr/bin/env python3
"""Write an immutable task/split subset of a labeled JSONL artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def filter_rows(
    rows: list[dict], *, task: str | None, split: str | None, where: dict[str, str]
) -> list[dict]:
    return [
        row
        for row in rows
        if (task is None or row.get("task") == task)
        and (split is None or row.get("split") == split)
        and all(str(row.get(key)) == value for key, value in where.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task")
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Require an exact string-valued field match; may be repeated.",
    )
    args = parser.parse_args()
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".meta.json").exists():
        raise FileExistsError(output)
    where = {}
    for expression in args.where:
        if "=" not in expression:
            raise ValueError(f"--where must be KEY=VALUE: {expression!r}")
        key, value = expression.split("=", 1)
        if not key or key in where:
            raise ValueError(f"invalid/duplicate --where key: {key!r}")
        where[key] = value
    rows = filter_rows(
        list(read_jsonl(source)), task=args.task, split=args.split, where=where
    )
    if not rows:
        raise ValueError("label filter selected no rows")
    write_jsonl(output, rows)
    meta = {
        "input": str(source),
        "input_sha256": sha256_file(source),
        "task": args.task,
        "split": args.split,
        "where": where,
        "count": len(rows),
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
