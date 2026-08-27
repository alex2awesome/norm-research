#!/usr/bin/env python3
"""Materialize an immutable JSONL tail with provenance metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--start-line", type=int, required=True, help="one-based inclusive line")
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with source.open("r", encoding="utf-8", errors="replace") as src, tmp.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, 1):
            if line_no >= args.start_line:
                dst.write(line)
                count += 1
    tmp.replace(output)
    meta = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "start_line": args.start_line,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rows": count,
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

