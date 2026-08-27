#!/usr/bin/env python3
"""Promote non-audit three-pass teachers only if a disjoint blind audit clears."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
    parser.add_argument("--training-candidates", required=True)
    parser.add_argument("--audit-proposals", required=True)
    parser.add_argument("--audit-labels", required=True)
    parser.add_argument("--min-point-precision", type=float, default=0.90)
    parser.add_argument("--min-wilson-lower", type=float, default=0.80)
    parser.add_argument("--min-support", type=int, default=60)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    paths = {
        "training_candidates": Path(args.training_candidates).resolve(),
        "audit_proposals": Path(args.audit_proposals).resolve(),
        "audit_labels": Path(args.audit_labels).resolve(),
    }
    output, report_path = Path(args.output).resolve(), Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite three-pass promotion outputs")
    train = _index(paths["training_candidates"])
    proposals = _index(paths["audit_proposals"])
    audit = _index(paths["audit_labels"])
    if set(proposals) != set(audit):
        raise ValueError(
            "blind audit must cover the exact predeclared audit proposal set"
        )
    if set(train) & set(proposals):
        raise ValueError("training candidates overlap blind audit by UID")
    train_groups = {str(row["split_group"]) for row in train.values()}
    audit_groups = {str(row["split_group"]) for row in proposals.values()}
    if train_groups & audit_groups:
        raise ValueError("training candidates overlap blind audit by source group")
    tasks = {
        str(row.get("task"))
        for row in [*train.values(), *proposals.values(), *audit.values()]
    }
    bank_hashes = {
        str(row.get("current_bank_source_sha256"))
        for row in [*train.values(), *proposals.values(), *audit.values()]
    }
    if len(tasks) != 1 or len(bank_hashes) != 1:
        raise ValueError("task or bank identity mismatch across promotion inputs")
    if any(row.get("training_eligible") for row in train.values()):
        raise ValueError(
            "training candidates were already marked eligible before audit"
        )

    exact = [
        uid
        for uid, proposal in proposals.items()
        if audit[uid].get("decision") == "MATCH"
        and audit[uid].get("metric_id") == proposal.get("metric_id")
    ]
    high_confidence_exact = [
        uid for uid in exact if audit[uid].get("confidence") == "high"
    ]
    support, success = len(proposals), len(exact)
    point = safe_rate(success, support)
    interval = wilson_interval(success, support)
    lower = interval[0] if interval else 0.0
    weights = [
        float(proposals[uid].get("audit_design_weight", 1.0)) for uid in proposals
    ]
    total_weight = sum(weights)
    success_weight = sum(
        float(proposals[uid].get("audit_design_weight", 1.0)) for uid in exact
    )
    effective_n = (
        total_weight * total_weight / sum(weight * weight for weight in weights)
        if weights
        else 0.0
    )
    weighted_point = success_weight / total_weight if total_weight else None
    weighted_interval = (
        wilson_interval(weighted_point * effective_n, effective_n)
        if weighted_point is not None
        else None
    )
    weighted_lower = weighted_interval[0] if weighted_interval else 0.0
    cleared = (
        support >= args.min_support
        and (point or 0.0) >= args.min_point_precision
        and lower >= args.min_wilson_lower
        and (weighted_point or 0.0) >= args.min_point_precision
        and weighted_lower >= args.min_wilson_lower
    )
    promoted = []
    if cleared:
        for uid in sorted(train):
            promoted.append(
                {
                    **train[uid],
                    "label_source": "independent_three_pass_consensus_blind_audit_promoted",
                    "training_eligible": True,
                    "training_blocked_pending_blind_audit": False,
                    "blind_audit_point_precision": point,
                    "blind_audit_wilson_95": interval,
                    "blind_audit_support": support,
                    "blind_audit_design_weighted_precision": weighted_point,
                    "blind_audit_design_weighted_wilson_95": weighted_interval,
                    "blind_audit_labels_sha256": sha256_file(paths["audit_labels"]),
                }
            )
    write_jsonl(output, promoted)
    report = {
        "schema_version": "silver-match-v3-three-pass-consensus-promotion-v1",
        "task": next(iter(tasks)),
        "training_candidate_count": len(train),
        "audit_support": support,
        "audit_exact": success,
        "audit_success_policy": "same exact MATCH metric_id; confidence reported separately",
        "audit_high_confidence_exact": len(high_confidence_exact),
        "audit_confidence_counts": dict(
            sorted(Counter(str(row.get("confidence")) for row in audit.values()).items())
        ),
        "audit_exact_rate": point,
        "audit_exact_wilson_95": interval,
        "audit_design_weighted_exact_rate": weighted_point,
        "audit_design_weighted_approximate_wilson_95": weighted_interval,
        "audit_design_effective_sample_size": effective_n,
        "thresholds": {
            "minimum_point_precision": args.min_point_precision,
            "minimum_wilson_lower": args.min_wilson_lower,
            "minimum_support": args.min_support,
        },
        "promotion_cleared": cleared,
        "promoted_count": len(promoted),
        "audit_items_promoted": 0,
        "audit_permanently_excluded_from_gradients": True,
        "weighted_interval_caveat": (
            "design-weighted interval uses Kish effective sample size and is approximate; "
            "within-stratum selection deliberately favors metric-leaf diversity"
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
