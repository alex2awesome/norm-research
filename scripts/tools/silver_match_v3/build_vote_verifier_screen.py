#!/usr/bin/env python3
"""Build a truth-blind screen from a minimum number of exact verifier votes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build_union_verifier_screen import _index, _parse_variant
from .common import sha256_file, write_jsonl


def build(
    *,
    task: str,
    primary_path: Path,
    variants: list[tuple[str, Path, Path]],
    minimum_confirmations: int,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if len(variants) < 2 or len({name for name, _, _ in variants}) != len(variants):
        raise ValueError("at least two uniquely named verifier variants are required")
    total_votes = 2 * len(variants)
    if not 1 <= minimum_confirmations <= total_votes:
        raise ValueError("minimum confirmations must be within available vote count")
    primary = _index(primary_path)
    if any(
        row.get("task") != task
        or row.get("decision") != "MATCH"
        or not row.get("metric_id")
        for row in primary.values()
    ):
        raise ValueError("primary must contain only task-matched MATCH proposals")
    expected = set(primary)
    votes: list[tuple[str, str, dict[str, dict[str, Any]]]] = []
    input_refs: dict[str, Any] = {
        "primary": {"path": str(primary_path), "sha256": sha256_file(primary_path)}
    }
    for name, original_path, hashed_path in variants:
        input_refs[name] = {}
        for order, path in (("original", original_path), ("hashed", hashed_path)):
            indexed = _index(path)
            if set(indexed) != expected:
                raise ValueError(f"variant {name}/{order} lacks exact primary coverage")
            votes.append((name, order, indexed))
            input_refs[name][order] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    selected: list[dict[str, Any]] = []
    confirmation_histogram = {str(value): 0 for value in range(total_votes + 1)}
    for uid in sorted(expected):
        proposal = primary[uid]
        proposed = str(proposal["metric_id"])
        passed: list[str] = []
        for name, order, indexed in votes:
            row = indexed[uid]
            if (
                row.get("task") != task
                or row.get("order_mode") != order
                or str(row.get("primary_metric_id") or "") != proposed
                or row.get("candidate_bank_source_sha256")
                != proposal.get("candidate_bank_source_sha256")
            ):
                raise ValueError(f"variant provenance mismatch: {name}/{uid}/{order}")
            if (
                row.get("decision") == "CONFIRM_MATCH"
                and str(row.get("metric_id") or "") == proposed
                and str(row.get("confidence") or "").lower() == "high"
                and not row.get("parse_error")
            ):
                passed.append(f"{name}:{order}")
        confirmation_histogram[str(len(passed))] += 1
        if len(passed) >= minimum_confirmations:
            selected.append(
                {
                    **proposal,
                    "proposal_screen": "minimum_exact_high_verifier_votes",
                    "proposal_screen_minimum_confirmations": minimum_confirmations,
                    "proposal_screen_confirmations": passed,
                    "proposal_screen_accepts_match": False,
                    "requires_independent_strong_verifier": True,
                }
            )
    output_root.mkdir(parents=True, exist_ok=False)
    output = output_root / "screened_primary.jsonl"
    write_jsonl(output, selected)
    report = {
        "schema_version": "silver-match-v3-vote-verifier-screen-v1",
        "status": "FROZEN_TRUTH_BLIND_SCREEN_REQUIRES_STRONG_VERIFIER",
        "task": task,
        "input_count": len(primary),
        "variant_count": len(variants),
        "available_vote_count": total_votes,
        "minimum_confirmations": minimum_confirmations,
        "selected_count": len(selected),
        "confirmation_histogram": confirmation_histogram,
        "selection_rule": (
            "retain when at least the frozen minimum number of independent variant/order "
            "views high-confidence-confirm the exact primary metric ID"
        ),
        "development_contract": (
            "minimum confirmation count is an optimize-only design choice; a new "
            "source-disjoint select panel is mandatory"
        ),
        "contracts": {
            "truth_or_targets_read": False,
            "screen_is_not_a_match_acceptance": True,
            "independent_strong_verifier_required": True,
            "downstream_gate_must_be_frozen_before_strong_inference": True,
        },
        "inputs": input_refs,
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    freeze = output_root / "SCREEN_FREEZE.json"
    freeze.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "screen_freeze_sha256": sha256_file(freeze)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--minimum-confirmations", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                task=args.task,
                primary_path=Path(args.primary).resolve(),
                variants=[_parse_variant(value) for value in args.variant],
                minimum_confirmations=args.minimum_confirmations,
                output_root=Path(args.output_root).resolve(),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
