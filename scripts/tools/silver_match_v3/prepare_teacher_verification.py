#!/usr/bin/env python3
"""Prepare high-confidence proposal rows and compact contrastive slates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .common import read_jsonl, sha256_file, write_jsonl


def select_proposals(
    rows: Sequence[dict[str, Any]], task: str, confidences: set[str]
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("task") == task
        and row.get("decision") == "MATCH"
        and row.get("confidence") in confidences
        and row.get("label_source") == "sonnet_full"
    ]
    uids = [str(row["norm_uid"]) for row in selected]
    if len(uids) != len(set(uids)):
        raise ValueError("duplicate selected proposal UID")
    return sorted(selected, key=lambda row: str(row["norm_uid"]))


def compact_candidates(
    proposals: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates = {str(row["norm_uid"]): row for row in candidate_rows}
    if len(candidates) != len(candidate_rows):
        raise ValueError("duplicate candidate UID")
    output = []
    for proposal in proposals:
        uid, primary = str(proposal["norm_uid"]), str(proposal["metric_id"])
        if uid not in candidates:
            raise KeyError(f"proposal lacks retrieved candidates: {uid}")
        source = candidates[uid]
        values = list(source.get("candidates") or [])
        by_id = {str(row["metric_id"]): row for row in values}
        primary_row = by_id.get(primary, {"metric_id": primary, "injected_primary": True})
        compact, seen = [primary_row], {primary}
        for row in values:
            metric_id = str(row["metric_id"])
            if metric_id in seen:
                continue
            compact.append(row)
            seen.add(metric_id)
            if len(compact) == limit:
                break
        if len(compact) != limit:
            raise ValueError(f"candidate slate shorter than {limit}: {uid}")
        output.append(
            {
                **source,
                "primary_was_injected": bool(primary_row.get("injected_primary")),
                "candidates": compact,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--task", required=True)
    select.add_argument("--teacher-set", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--confidence", action="append", default=["high"])
    compact = subparsers.add_parser("compact")
    compact.add_argument("--proposals", required=True)
    compact.add_argument("--candidates", required=True)
    compact.add_argument("--output", required=True)
    compact.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "select":
        source = Path(args.teacher_set).resolve()
        proposals = select_proposals(
            list(read_jsonl(source)), args.task, set(args.confidence)
        )
        write_jsonl(output, proposals)
        report = {
            "command": "select",
            "task": args.task,
            "confidence": sorted(set(args.confidence)),
            "count": len(proposals),
            "metrics": len({row["metric_id"] for row in proposals}),
            "input_hash": sha256_file(source),
            "output_hash": sha256_file(output),
        }
    else:
        proposal_path = Path(args.proposals).resolve()
        candidate_path = Path(args.candidates).resolve()
        proposals = list(read_jsonl(proposal_path))
        compacted = compact_candidates(
            proposals, list(read_jsonl(candidate_path)), limit=args.limit
        )
        write_jsonl(output, compacted)
        report = {
            "command": "compact",
            "count": len(compacted),
            "limit": args.limit,
            "injected_primary": sum(row["primary_was_injected"] for row in compacted),
            "input_hashes": {
                "proposals": sha256_file(proposal_path),
                "candidates": sha256_file(candidate_path),
            },
            "output_hash": sha256_file(output),
        }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
