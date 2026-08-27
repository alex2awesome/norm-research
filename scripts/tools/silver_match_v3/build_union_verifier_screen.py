#!/usr/bin/env python3
"""Build a truth-blind union of strict two-order verifier confirmations.

This is a proposal screen, not a verifier selection result.  It may reduce the
number of examples sent to a stronger checker, but it never accepts a MATCH on
its own.  In particular, the implementation does not read gold truth or target
labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"invalid norm_uid coverage: {path}")
    return indexed


def _parse_variant(value: str) -> tuple[str, Path, Path]:
    fields = value.split("=", 1)
    if len(fields) != 2 or not fields[0]:
        raise ValueError("--variant must be NAME=ORIGINAL,HASHED")
    paths = fields[1].split(",", 1)
    if len(paths) != 2:
        raise ValueError("--variant must be NAME=ORIGINAL,HASHED")
    return fields[0], Path(paths[0]).resolve(), Path(paths[1]).resolve()


def build(
    *,
    task: str,
    primary_path: Path,
    variants: list[tuple[str, Path, Path]],
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if len(variants) < 2 or len({name for name, _, _ in variants}) != len(variants):
        raise ValueError("at least two uniquely named verifier variants are required")

    primary = _index(primary_path)
    if any(
        row.get("task") != task
        or row.get("decision") != "MATCH"
        or not row.get("metric_id")
        for row in primary.values()
    ):
        raise ValueError("primary must contain only task-matched MATCH proposals")

    pair_rows: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    input_refs: dict[str, Any] = {
        "primary": {"path": str(primary_path), "sha256": sha256_file(primary_path)}
    }
    expected = set(primary)
    for name, original_path, hashed_path in variants:
        original, hashed = _index(original_path), _index(hashed_path)
        if set(original) != expected or set(hashed) != expected:
            raise ValueError(f"variant {name} lacks exact primary UID coverage")
        pair_rows[name] = original, hashed
        input_refs[name] = {
            "original": {
                "path": str(original_path),
                "sha256": sha256_file(original_path),
            },
            "hashed": {
                "path": str(hashed_path),
                "sha256": sha256_file(hashed_path),
            },
        }

    selected: list[dict[str, Any]] = []
    selected_by_variant = {name: 0 for name, _, _ in variants}
    for uid in sorted(expected):
        proposal = primary[uid]
        proposed = str(proposal["metric_id"])
        passed: list[str] = []
        for name, _, _ in variants:
            original, hashed = pair_rows[name][0][uid], pair_rows[name][1][uid]
            for order, row in (("original", original), ("hashed", hashed)):
                if (
                    row.get("task") != task
                    or row.get("order_mode") != order
                    or str(row.get("primary_metric_id") or "") != proposed
                    or row.get("candidate_bank_source_sha256")
                    != proposal.get("candidate_bank_source_sha256")
                ):
                    raise ValueError(f"variant provenance mismatch: {name}/{uid}/{order}")
            accepted = all(
                row.get("decision") == "CONFIRM_MATCH"
                and str(row.get("metric_id") or "") == proposed
                and str(row.get("confidence") or "").lower() == "high"
                and not row.get("parse_error")
                for row in (original, hashed)
            )
            if accepted:
                passed.append(name)
                selected_by_variant[name] += 1
        if passed:
            selected.append(
                {
                    **proposal,
                    "proposal_screen": "union_of_strict_two_order_high_confirmations",
                    "proposal_screen_variants": passed,
                    "proposal_screen_accepts_match": False,
                    "requires_independent_strong_verifier": True,
                }
            )

    output_root.mkdir(parents=True, exist_ok=False)
    output = output_root / "screened_primary.jsonl"
    write_jsonl(output, selected)
    report = {
        "schema_version": "silver-match-v3-union-verifier-screen-v1",
        "status": "FROZEN_TRUTH_BLIND_SCREEN_REQUIRES_STRONG_VERIFIER",
        "task": task,
        "input_count": len(primary),
        "variant_count": len(variants),
        "selected_count": len(selected),
        "selected_by_variant": selected_by_variant,
        "selection_rule": (
            "union across variants; within each variant both original and hashed "
            "must high-confidence-confirm the same primary metric ID"
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
    report_path = output_root / "SCREEN_FREEZE.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**report, "screen_freeze_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = build(
        task=args.task,
        primary_path=Path(args.primary).resolve(),
        variants=[_parse_variant(value) for value in args.variant],
        output_root=Path(args.output_root).resolve(),
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
