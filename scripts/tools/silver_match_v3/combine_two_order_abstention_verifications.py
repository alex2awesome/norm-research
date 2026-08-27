#!/usr/bin/env python3
"""Combine typed-abstention verifier orders under a fail-closed consensus rule."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .verify_abstention_gemma import TYPED_DECISIONS


def _unique(path: Path, kind: str) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in output or len(output) != len(rows):
        raise ValueError(f"missing/duplicate norm_uid in {kind}")
    return output


def combine(
    *, audits_path: Path, original_path: Path, hashed_path: Path, output_path: Path
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    audits = _unique(audits_path, "audits")
    original = _unique(original_path, "original")
    hashed = _unique(hashed_path, "hashed")
    if set(audits) != set(original) or set(audits) != set(hashed):
        raise ValueError("typed-abstention two-order coverage mismatch")
    counts: Counter[str] = Counter()
    output = []
    for uid in sorted(audits):
        audit, left, right = audits[uid], original[uid], hashed[uid]
        if not audit.get("rescue_exhaustive"):
            raise ValueError(f"non-exhaustive abstention audit: {uid}")
        if {left.get("order_mode"), right.get("order_mode")} != {
            "original",
            "hashed",
        }:
            raise ValueError(f"abstention verifier orders differ: {uid}")
        for label, row in (("original", left), ("hashed", right)):
            if (
                row.get("task") != audit.get("task")
                or row.get("corpus") != audit.get("corpus")
                or row.get("bank_source_sha256") != audit.get("bank_source_sha256")
                or row.get("provisional_decision") != audit.get("provisional_decision")
            ):
                raise ValueError(f"{label} abstention verifier provenance mismatch: {uid}")
        if left.get("prompt_sha256") != right.get("prompt_sha256"):
            raise ValueError(f"abstention verifier prompt mismatch: {uid}")
        if left.get("model") != right.get("model"):
            raise ValueError(f"abstention verifier model mismatch: {uid}")
        possible = bool(
            left.get("possible_exact_bank_match")
            or right.get("possible_exact_bank_match")
        )
        consensus = (
            left.get("confirmed_decision")
            if left.get("confirmed_decision") == right.get("confirmed_decision")
            and left.get("confirmed_decision") in TYPED_DECISIONS
            and left.get("confidence") == right.get("confidence") == "high"
            and not left.get("parse_error")
            and not right.get("parse_error")
            else None
        )
        if possible:
            decision = "POSSIBLE_EXACT_BANK_MATCH"
            confirmed = None
            confidence = "low"
            reason = "at_least_one_order_flags_possible_exact_bank_match"
            counts["possible_exact_bank_match"] += 1
        elif consensus:
            decision = str(consensus)
            confirmed = str(consensus)
            confidence = "high"
            reason = "both_orders_high_confidence_same_typed_abstention"
            counts[f"confirmed:{consensus}"] += 1
        else:
            decision = "UNRESOLVED_ABSTENTION"
            confirmed = None
            confidence = "low"
            reason = "two_order_typed_abstention_policy_not_satisfied"
            counts["unresolved"] += 1
        output.append(
            {
                "schema_version": "silver-match-v3-two-order-abstention-verification-v1",
                "norm_uid": uid,
                "corpus": audit.get("corpus"),
                "task": audit.get("task"),
                "row": audit.get("row"),
                "provisional_decision": audit.get("provisional_decision"),
                "decision": decision,
                "confirmed_decision": confirmed,
                "possible_exact_bank_match": possible,
                "metric_id": None,
                "confidence": confidence,
                "reason": reason,
                "bank_source_sha256": audit.get("bank_source_sha256"),
                "prompt_sha256": left.get("prompt_sha256"),
                "model": left.get("model"),
                "verification_orders": ["original", "hashed"],
                "strict_two_order_abstention": bool(consensus),
                "rescue_bank_count": audit.get("rescue_bank_count"),
                "rescue_coverage_repeats": audit.get("rescue_coverage_repeats", 1),
                "rescue_reincludes_primary": audit.get(
                    "rescue_reincludes_primary", False
                ),
                "original": left,
                "hashed": right,
            }
        )
    write_jsonl(output_path, output)
    report = {
        "schema_version": "silver-match-v3-two-order-abstention-verification-report-v1",
        "complete": len(output) == len(audits),
        "count": len(output),
        "counts": dict(sorted(counts.items())),
        "inputs": {
            "audits": {"path": str(audits_path), "sha256": sha256_file(audits_path)},
            "original": {
                "path": str(original_path),
                "sha256": sha256_file(original_path),
            },
            "hashed": {"path": str(hashed_path), "sha256": sha256_file(hashed_path)},
        },
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--hashed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = combine(
        audits_path=Path(args.audits).resolve(),
        original_path=Path(args.original).resolve(),
        hashed_path=Path(args.hashed).resolve(),
        output_path=Path(args.output).resolve(),
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
