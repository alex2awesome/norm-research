#!/usr/bin/env python3
"""Promote a predeclared consensus pool only if a final hidden-ID audit clears."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl
from .score_verifier_calibration import safe_rate, wilson_interval


def _index(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if len(rows) != len(output):
        raise ValueError(f"duplicate UIDs: {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--min-point-precision", type=float, default=0.90)
    parser.add_argument("--min-wilson-lower", type=float, default=0.80)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    proposal_path, audit_path = Path(args.proposals).resolve(), Path(args.audit).resolve()
    output, report_path = Path(args.output).resolve(), Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite final blind-audit promotion")
    proposals, audit = _index(proposal_path), _index(audit_path)
    if set(proposals) != set(audit):
        raise ValueError("final blind audit must cover the exact proposal pool")
    exact = [
        uid
        for uid, proposal in proposals.items()
        if audit[uid].get("decision") == "MATCH"
        and audit[uid].get("metric_id") == proposal.get("metric_id")
    ]
    success, support = len(exact), len(proposals)
    point = safe_rate(success, support)
    interval = wilson_interval(success, support)
    lower = interval[0] if interval else 0.0
    cleared = (
        support >= args.min_support
        and (point or 0.0) >= args.min_point_precision
        and lower >= args.min_wilson_lower
    )
    promoted = []
    if cleared:
        for uid in sorted(exact):
            promoted.append(
                {
                    **proposals[uid],
                    "label_source": "independent_consensus_final_blind_audit",
                    "training_eligible": True,
                    "final_audit_confidence": audit[uid].get("confidence"),
                    "final_audit_reason": audit[uid].get("reason"),
                }
            )
    write_jsonl(output, promoted)
    report = {
        "schema_version": "silver-match-v3-final-blind-audit-promotion-v1",
        "task": next(iter(proposals.values())).get("task") if proposals else None,
        "proposal_count": support,
        "audit_exact": success,
        "audit_exact_rate": point,
        "audit_exact_wilson_95": interval,
        "thresholds": {
            "minimum_point_precision": args.min_point_precision,
            "minimum_wilson_lower": args.min_wilson_lower,
            "minimum_support": args.min_support,
        },
        "promotion_cleared": cleared,
        "known_disagreements_dropped": support - success,
        "promoted_count": len(promoted),
        "inputs": {
            "proposals": {"path": str(proposal_path), "sha256": sha256_file(proposal_path)},
            "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
