#!/usr/bin/env python3
"""Export an immutable UID file for full-bank retrieval of provisional nonmatches."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import read_jsonl, sha256_file


def export_uids(
    *, input_paths: list[Path], output_path: Path, include_decisions: set[str] | None
) -> dict:
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(output_path)
    seen = set()
    selected = []
    counts = Counter()
    for path in input_paths:
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in seen:
                raise ValueError(f"missing/duplicate norm_uid across final inputs: {uid!r}")
            seen.add(uid)
            decision = str(row.get("decision") or "")
            counts[decision or "MISSING"] += 1
            eligible = (
                decision in include_decisions
                if include_decisions is not None
                else decision != "MATCH"
            )
            if eligible:
                selected.append(uid)
    if not selected:
        raise ValueError("no provisional nonmatch UIDs selected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{uid}\n" for uid in selected), encoding="utf-8")
    report = {
        "schema_version": "silver-match-v3-rescue-uid-export-v1",
        "inputs": {str(path): sha256_file(path) for path in input_paths},
        "input_count": len(seen),
        "decision_counts": dict(sorted(counts.items())),
        "include_decisions": sorted(include_decisions) if include_decisions else None,
        "selection_rule": (
            "listed decisions" if include_decisions is not None else "decision != MATCH"
        ),
        "selected_count": len(selected),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-decision", action="append", default=[])
    args = parser.parse_args()
    report = export_uids(
        input_paths=[Path(path).resolve() for path in args.input],
        output_path=Path(args.output).resolve(),
        include_decisions=set(args.include_decision) or None,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
