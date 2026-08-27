#!/usr/bin/env python3
"""Keep only exact, independently repeated full-bank MATCH labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate UIDs: {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--policy", choices=["all_exact", "one_high", "both_high"], default="one_high"
    )
    parser.add_argument(
        "--pending-blind-audit",
        action="store_true",
        help="Lock retained rows out of gradients until a disjoint audit promotes them.",
    )
    args = parser.parse_args()

    first_path, second_path = Path(args.first).resolve(), Path(args.second).resolve()
    output_path, report_path = Path(args.output).resolve(), Path(args.report).resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite independent consensus outputs")
    first, second = _index(first_path), _index(second_path)
    if not set(second).issubset(first):
        raise ValueError("second-pass UIDs are not a subset of first-pass labels")

    decisions = Counter()
    consensus = []
    for uid, right in second.items():
        left = first[uid]
        if left.get("task") != right.get("task"):
            raise ValueError(f"task mismatch: {uid}")
        if left.get("current_bank_source_sha256") != right.get(
            "current_bank_source_sha256"
        ):
            raise ValueError(f"bank mismatch: {uid}")
        exact = (
            left.get("decision") == right.get("decision") == "MATCH"
            and left.get("metric_id") == right.get("metric_id")
        )
        if not exact:
            decisions["disagreement_or_abstention"] += 1
            continue
        confidences = [str(left.get("confidence")), str(right.get("confidence"))]
        eligible = {
            "all_exact": True,
            "one_high": "high" in confidences,
            "both_high": confidences == ["high", "high"],
        }[args.policy]
        decisions["exact_consensus"] += 1
        decisions[f"exact_confidence:{'/'.join(confidences)}"] += 1
        if not eligible:
            decisions["exact_below_policy"] += 1
            continue
        consensus.append(
            {
                **left,
                "label_source": "independent_codex_two_pass_full_bank_consensus",
                "training_eligible": not args.pending_blind_audit,
                "training_blocked_pending_blind_audit": args.pending_blind_audit,
                "consensus_policy": args.policy,
                "first_confidence": left.get("confidence"),
                "second_confidence": right.get("confidence"),
                "first_reason": left.get("reason"),
                "second_reason": right.get("reason"),
                "first_label_sha256": sha256_file(first_path),
                "second_label_sha256": sha256_file(second_path),
            }
        )
        decisions["retained"] += 1

    write_jsonl(output_path, consensus)
    report = {
        "schema_version": "silver-match-v3-independent-relabel-consensus-v1",
        "task": next(iter(first.values())).get("task") if first else None,
        "policy": args.policy,
        "pending_blind_audit": args.pending_blind_audit,
        "first_count": len(first),
        "second_count": len(second),
        "counts": dict(sorted(decisions.items())),
        "retained_count": len(consensus),
        "inputs": {
            "first": {"path": str(first_path), "sha256": sha256_file(first_path)},
            "second": {"path": str(second_path), "sha256": sha256_file(second_path)},
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
