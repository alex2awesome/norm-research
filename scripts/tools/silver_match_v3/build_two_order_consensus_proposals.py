#!/usr/bin/env python3
"""Keep only exact MATCH proposals stable under two candidate orders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {key: Path(getattr(args, key)).resolve() for key in ("original", "hashed")}
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(output)
    rows = {
        key: {str(row["norm_uid"]): row for row in read_jsonl(path)}
        for key, path in paths.items()
    }
    if set(rows["original"]) != set(rows["hashed"]):
        raise ValueError("two order proposal inputs have different coverage")
    selected = []
    for uid in sorted(rows["original"]):
        original, hashed = rows["original"][uid], rows["hashed"][uid]
        if original.get("task") != args.task or hashed.get("task") != args.task:
            continue
        if (
            original.get("decision") == hashed.get("decision") == "MATCH"
            and original.get("metric_id") == hashed.get("metric_id")
        ):
            selected.append(
                {
                    **original,
                    "consensus_order_modes": ["original", "hashed"],
                    "consensus_metric_id": original["metric_id"],
                    "hashed_confidence": hashed.get("confidence"),
                    "hashed_reason": hashed.get("reason"),
                    "hashed_output_sha256": sha256_file(paths["hashed"]),
                }
            )
    write_jsonl(output, selected)
    report = {
        "schema_version": "silver-match-v3-two-order-consensus-proposals-v1",
        "task": args.task,
        "input_count": len(rows["original"]),
        "consensus_match_count": len(selected),
        "inputs": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()},
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
