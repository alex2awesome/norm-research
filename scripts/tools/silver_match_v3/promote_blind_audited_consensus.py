#!/usr/bin/env python3
"""Gate strict proposal/verifier consensus with a hidden-ID final audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl
from .score_verifier_calibration import safe_rate, wilson_interval


def _index(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate UIDs: {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--verifier-original", required=True)
    parser.add_argument("--verifier-hashed", required=True)
    parser.add_argument("--min-point-precision", type=float, default=0.90)
    parser.add_argument("--min-wilson-lower", type=float, default=0.80)
    parser.add_argument("--min-audited-support", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    paths = {
        "proposals": Path(args.proposals).resolve(),
        "audit": Path(args.audit).resolve(),
        "verifier_original": Path(args.verifier_original).resolve(),
        "verifier_hashed": Path(args.verifier_hashed).resolve(),
    }
    output, report_path = Path(args.output).resolve(), Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite blind-audit promotion outputs")
    proposals, audit = _index(paths["proposals"]), _index(paths["audit"])
    original, hashed = _index(paths["verifier_original"]), _index(paths["verifier_hashed"])
    if not set(proposals).issubset(audit):
        raise ValueError("blind audit does not cover every proposal")
    if set(original) != set(proposals) or set(hashed) != set(proposals):
        raise ValueError("verifier orders do not cover the proposal set exactly")

    verifier_retained = []
    for uid, proposal in proposals.items():
        metric_id = proposal.get("metric_id")
        pair = (original[uid], hashed[uid])
        confirm = all(
            row.get("decision") == "CONFIRM_MATCH"
            and row.get("metric_id") == metric_id
            and row.get("confidence") == "high"
            and not row.get("parse_error")
            for row in pair
        )
        if confirm:
            verifier_retained.append(uid)

    audited_exact = [
        uid
        for uid in verifier_retained
        if audit[uid].get("decision") == "MATCH"
        and audit[uid].get("metric_id") == proposals[uid].get("metric_id")
    ]
    support, success = len(verifier_retained), len(audited_exact)
    point = safe_rate(success, support)
    interval = wilson_interval(success, support)
    lower = interval[0] if interval is not None else 0.0
    cleared = (
        support >= args.min_audited_support
        and (point or 0.0) >= args.min_point_precision
        and lower >= args.min_wilson_lower
        and len(audited_exact) >= args.min_audited_support
    )
    promoted = []
    if cleared:
        for uid in sorted(audited_exact):
            proposal = proposals[uid]
            promoted.append(
                {
                    **proposal,
                    "label_source": "blind_audited_three_stage_exact_consensus",
                    "training_eligible": True,
                    "blind_audit_metric_id": audit[uid].get("metric_id"),
                    "blind_audit_confidence": audit[uid].get("confidence"),
                    "blind_audit_reason": audit[uid].get("reason"),
                    "verifier_orders": ["original", "hashed"],
                }
            )
    write_jsonl(output, promoted)
    report = {
        "schema_version": "silver-match-v3-blind-audited-consensus-promotion-v1",
        "task": next(iter(proposals.values())).get("task") if proposals else None,
        "proposal_count": len(proposals),
        "verifier_retained": support,
        "blind_audit_exact": success,
        "blind_audit_exact_rate": point,
        "blind_audit_exact_wilson_95": interval,
        "thresholds": {
            "minimum_point_precision": args.min_point_precision,
            "minimum_wilson_lower": args.min_wilson_lower,
            "minimum_audited_support": args.min_audited_support,
        },
        "promotion_cleared": cleared,
        "promoted_count": len(promoted),
        "known_audit_disagreements_dropped": support - success,
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
