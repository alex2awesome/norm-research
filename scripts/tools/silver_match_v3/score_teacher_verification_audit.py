#!/usr/bin/env python3
"""Score independently completed teacher-verification audit packets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .common import read_jsonl, sha256_file


MANUAL_DECISIONS = {
    "EXACT_MATCH",
    "WRONG_METRIC",
    "NO_EXPLICIT_CRITERION",
    "GENERIC_VERDICT",
    "CONTEXT_NEEDED",
    "NOISE",
    "UNCERTAIN",
}


def wilson(successes: float, total: float, z: float = 1.96) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def ratio(successes: int, total: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "n": total,
        "estimate": successes / total if total else None,
        "wilson_95": wilson(successes, total),
    }


def weighted_ratio(rows: Sequence[dict[str, Any]], predicate) -> dict[str, Any]:
    weights = [float(row.get("audit_design_weight", 1.0)) for row in rows]
    total_weight = sum(weights)
    success_weight = sum(
        weight for row, weight in zip(rows, weights) if predicate(row)
    )
    effective_n = (
        total_weight * total_weight / sum(weight * weight for weight in weights)
        if weights
        else 0.0
    )
    estimate = success_weight / total_weight if total_weight else None
    return {
        "weighted_successes": success_weight,
        "weighted_n": total_weight,
        "estimate": estimate,
        "effective_sample_size": effective_n,
        "approximate_wilson_95": (
            wilson(estimate * effective_n, effective_n) if estimate is not None else None
        ),
    }


def join_key(
    rows: Sequence[dict[str, Any]], key_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_uid = {str(row["norm_uid"]): row for row in rows}
    key_by_uid = {str(row["norm_uid"]): row for row in key_rows}
    if len(by_uid) != len(rows) or len(key_by_uid) != len(key_rows):
        raise ValueError("duplicate UID in audit or audit key")
    if by_uid.keys() != key_by_uid.keys():
        raise ValueError("audit and audit key UID sets differ")
    return [{**key_by_uid[uid], **by_uid[uid]} for uid in sorted(by_uid)]


def score(
    rows: Sequence[dict[str, Any]], key_rows: Sequence[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if key_rows is not None:
        rows = join_key(rows, key_rows)
    completed = []
    for row in rows:
        decision = row.get("manual_decision")
        if not decision:
            continue
        if decision not in MANUAL_DECISIONS:
            raise ValueError(f"unknown manual_decision for {row.get('norm_uid')}: {decision}")
        if decision != "UNCERTAIN" and not row.get("manual_reason"):
            raise ValueError(f"completed audit lacks manual_reason: {row.get('norm_uid')}")
        completed.append(row)
    determinate = [row for row in completed if row["manual_decision"] != "UNCERTAIN"]
    retained = [row for row in determinate if row["gemma_outcome"] == "retained"]
    rejected = [row for row in determinate if row["gemma_outcome"] != "retained"]
    exact = lambda row: row["manual_decision"] == "EXACT_MATCH"
    injected_exact = [
        row
        for row in determinate
        if exact(row) and row["proposal_retrieval_status"] == "injected_for_verification"
    ]
    all_exact = [row for row in determinate if exact(row)]
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in determinate:
        by_stratum["|".join(row["audit_stratum"])].append(row)
    return {
        "packet_count": len(rows),
        "completed": len(completed),
        "determinate": len(determinate),
        "uncertain": len(completed) - len(determinate),
        "manual_decisions": dict(sorted(Counter(row["manual_decision"] for row in completed).items())),
        "retained_exact_precision": ratio(sum(exact(row) for row in retained), len(retained)),
        "retained_exact_precision_design_weighted": weighted_ratio(retained, exact),
        "rejected_wrong_proposal_rate": ratio(sum(not exact(row) for row in rejected), len(rejected)),
        "rejected_wrong_proposal_rate_design_weighted": weighted_ratio(
            rejected, lambda row: not exact(row)
        ),
        "exact_proposal_injected_rate": ratio(len(injected_exact), len(all_exact)),
        "exact_proposal_injected_rate_design_weighted": weighted_ratio(
            all_exact,
            lambda row: row["proposal_retrieval_status"]
            == "injected_for_verification",
        ),
        "by_stratum": {
            key: {
                "exact_proposal_rate": ratio(sum(exact(row) for row in values), len(values)),
                "manual_decisions": dict(
                    sorted(Counter(row["manual_decision"] for row in values).items())
                ),
            }
            for key, values in sorted(by_stratum.items())
        },
        "weighted_interval_caveat": (
            "design-weighted intervals use effective sample size and are approximate; "
            "within-stratum sampling deliberately favors source/metric diversity"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--key", help="unblinded key emitted by the packet builder")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.audit).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    key_path = Path(args.key).resolve() if args.key else None
    report = score(
        list(read_jsonl(source)),
        list(read_jsonl(key_path)) if key_path else None,
    )
    report["audit"] = str(source)
    report["audit_sha256"] = sha256_file(source)
    if key_path:
        report["key"] = str(key_path)
        report["key_sha256"] = sha256_file(key_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
