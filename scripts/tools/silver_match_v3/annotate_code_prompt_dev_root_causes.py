#!/usr/bin/env python3
"""Attach a frozen manual root-cause audit to Code prompt-dev errors.

This is diagnostic only.  It does not change truth, candidates, prompts, model
weights, thresholds, or any production label.  The input is the mechanically
complete prompt-dev error audit emitted by
``audit_explicit_role_adjudicator_errors``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import sha256_file


SCHEMA = "silver-match-v3-code-prompt-dev-root-cause-audit-v1"

# Prefixes are unique within the immutable 60-row Code select panel.  The
# script resolves and writes the complete norm_uid after validating uniqueness.
ROOT_CAUSES: dict[str, tuple[str, str]] = {
    "0c6cbf2c": ("bank_leaf_ambiguity", "Trailing-newline evidence genuinely overlaps the whitespace-specific and broader layout leaves."),
    "0f8600c8": ("bank_gap_no_leaf", "The frozen truth finds no resource-lifecycle leaf; the model overextended generic robustness to scanner closure."),
    "190fa59a": ("bank_leaf_ambiguity", "Mixed responsibilities plausibly invokes architectural layering, cohesion, or general complexity."),
    "2102b2ff": ("bank_leaf_ambiguity", "The exact layout leaf sits in a crowded formatting/whitespace family and was conservatively abstained."),
    "21317c42": ("adjudicator_error", "Package/folder organization is explicit and its exact candidate was present, but the model under-retained it."),
    "2d26b8d4": ("bank_leaf_ambiguity", "The two API leaves substantially duplicate stewardship, ergonomics, and least-astonishment semantics."),
    "2f7f3caa": ("family_only_boundary", "Frozen truth deliberately withholds a leaf among expression readability and function-level decomposition."),
    "31713de0": ("family_only_boundary", "Frozen truth deliberately withholds a leaf among overlapping naming-convention leaves."),
    "35a00ffc": ("bank_leaf_ambiguity", "A flawed weighting algorithm plausibly invokes both overall design appropriateness and functional correctness."),
    "3c8f31fb": ("bank_gap_no_leaf", "Frozen truth finds no exact default-mapping leaf; the model overextended broad API ergonomics."),
    "44273612": ("adjudicator_error", "The exact simplicity candidate was rank 1; selecting generic refactoring was avoidable leaf drift."),
    "4b2754a1": ("adjudicator_error", "The exact composition/inheritance candidate was present and the model only under-retained it."),
    "4b70fe42": ("bank_leaf_ambiguity", "A bad minimum value can be framed as a correctness defect or preventive input validation."),
    "4c1b11fc": ("bank_leaf_ambiguity", "Needless ugly code overlaps general simplicity and the narrower overengineering leaf."),
    "4d2b5304": ("adjudicator_error", "The prompt addendum misread stale auth state requiring update as unused-state deletion; maintainability was present at rank 3."),
    "50ad7c2f": ("adjudicator_error", "The logging criterion is explicit and its exact candidate was present, but the model under-retained it."),
    "52b082ac": ("candidate_miss", "The frozen exact conventions leaf was absent from top 50; downstream adjudication could not select it."),
    "569dcbb2": ("bank_leaf_ambiguity", "Avoiding a needless class overlaps broad KISS/YAGNI and the narrower speculative-generality leaf."),
    "646cc4b8": ("adjudicator_error", "The request for an additional test case explicitly invokes test design; its candidate was present."),
    "68d71c6f": ("bank_leaf_ambiguity", "Unchecked failures overlap two near-duplicate robust/error-prevention leaves."),
    "6aebec4f": ("adjudicator_error", "Unspecified map iteration order is a correctness risk; API stewardship was an avoidable drift."),
    "6cd81e50": ("family_only_boundary", "Frozen truth deliberately withholds a leaf for a request to add an explanatory comment."),
    "722beba8": ("bank_leaf_ambiguity", "Avoiding namespace std is both a C++-safety rule and an ecosystem convention."),
    "7724c955": ("bank_leaf_ambiguity", "Failure of close overlaps robust handling, prevention, and resilience leaves with weak bank boundaries."),
    "7802880e": ("adjudicator_error", "Sharing an implementation directly invokes duplication/reuse; information hiding was avoidable drift."),
    "7d0c9110": ("candidate_miss", "The exact existing-codebase-consistency leaf was absent from top 50."),
    "96bf2bba": ("family_only_boundary", "Frozen truth deliberately withholds a leaf within a crowded naming family."),
    "97d94b15": ("family_only_boundary", "Frozen truth deliberately withholds a leaf for comment necessity/quality."),
    "9837d1cf": ("family_only_boundary", "Frozen truth deliberately withholds a leaf for the proposed API-method shape."),
    "acfffa33": ("adjudicator_error", "Lower execution time explicitly invokes performance; the exact candidate was present."),
    "b682f46c": ("family_only_boundary", "Frozen truth deliberately withholds a leaf between SRP/cohesion and small-function decomposition."),
    "bc70ba59": ("bank_leaf_ambiguity", "Replaceable behavior simultaneously invokes evolvability and composition/dependency injection."),
    "bdd11ca0": ("bank_leaf_ambiguity", "Passing context for a logger overlaps architectural layering, design appropriateness, and API minimality."),
    "c12b6f37": ("family_only_boundary", "Frozen truth deliberately withholds an API-usability leaf despite a plausible exact API candidate."),
    "c6cfa422": ("family_only_boundary", "Frozen truth deliberately withholds a leaf for documenting why responsibilities remain combined."),
    "ca0b32fd": ("adjudicator_error", "Const qualification directly invokes the C++-safety leaf; expression clarity was avoidable drift."),
    "cbcb2a70": ("candidate_miss", "The frozen exact cohesive-interface leaf was absent from top 50."),
    "cbcdc4bf": ("adjudicator_error", "The explicit bad setting is a correctness claim; the present exact candidate was under-retained."),
    "d3fa223d": ("bank_leaf_ambiguity", "Replacing a manual accumulator with reduce invokes both established-component reuse and idiomatic control flow."),
    "d5e91db9": ("adjudicator_error", "A switch/default recommendation is control-flow structure; expression-level clarity was avoidable drift."),
    "d6fd1307": ("bank_leaf_ambiguity", "Future search-time improvement overlaps algorithmic efficiency and optimization discipline."),
    "e3da51d1": ("adjudicator_error", "Keeping transaction handling out of feeHandler explicitly invokes responsibility separation."),
    "e479141a": ("bank_leaf_ambiguity", "Rejecting an abstraction whose cost exceeds DRY benefit straddles pragmatic DRY and overengineering."),
    "e619ee0a": ("family_only_boundary", "Frozen truth deliberately withholds a leaf for missing why-comments."),
    "e9a125f1": ("adjudicator_error", "Using std:: is an explicit C++ rule and the exact candidate was present."),
    "eaede9e4": ("bank_leaf_ambiguity", "Passing a master password explicitly overlaps configuration design and API parameter ergonomics."),
    "ec00a3fc": ("generic_verdict_boundary", "The frozen truth contains only a generic better/worse verdict; the model invented a control-flow criterion."),
    "ef6a5d65": ("adjudicator_error", "Reducing regex handling to one conditional explicitly invokes simple control flow."),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source = Path(args.errors).resolve()
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    if output.exists() or report.exists():
        raise FileExistsError("root-cause audit outputs are append-only")
    rows = _read_jsonl(source)
    if len(rows) != 48 or len({row["norm_uid"] for row in rows}) != 48:
        raise ValueError("expected the immutable 48-UID Code prompt-dev error audit")
    by_prefix: dict[str, dict[str, Any]] = {}
    for row in rows:
        prefix = str(row["norm_uid"])[:8]
        if prefix in by_prefix:
            raise ValueError(f"non-unique norm_uid prefix: {prefix}")
        by_prefix[prefix] = row
    if set(by_prefix) != set(ROOT_CAUSES):
        raise ValueError("manual root-cause map does not exactly cover the input audit")

    annotated: list[dict[str, Any]] = []
    for prefix in sorted(by_prefix):
        row = by_prefix[prefix]
        category, reason = ROOT_CAUSES[prefix]
        consensus_errors = {
            variant: view["consensus"]
            for variant, view in sorted(row["variant_views"].items())
            if view["consensus"].get("present")
            and view["consensus"].get("error_category")
        }
        annotated.append(
            {
                "schema_version": SCHEMA,
                "task": "code-review",
                "norm_uid": row["norm_uid"],
                "corpus": row["corpus"],
                "audit_scope": "prompt_dev_evaluation_only_no_gradient",
                "primary_root_cause": category,
                "root_cause_reason": reason,
                "truth_decision": row["truth"]["decision"],
                "truth_metric_id": row["truth"].get("metric_id"),
                "truth_candidate_present": row["truth_candidate_present"],
                "truth_candidate_rank": row["truth_candidate_rank"],
                "consensus_retained_error": bool(consensus_errors),
                "consensus_errors": consensus_errors,
                "source_error_audit_sha256": sha256_file(source),
                "test_or_blind_audit_used": False,
                "production_used": False,
                "outcomes_or_mi_used": False,
                "gradient_eligible": False,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotated),
        encoding="utf-8",
    )
    all_counts = Counter(row["primary_root_cause"] for row in annotated)
    retained = [row for row in annotated if row["consensus_retained_error"]]
    payload = {
        "schema_version": f"{SCHEMA}-report",
        "task": "code-review",
        "scope": {
            "role": "prompt_dev",
            "evaluation_only": True,
            "gradient_eligible": False,
            "test_or_blind_audit_used": False,
            "production_used": False,
            "outcomes_or_mi_used": False,
        },
        "input": {"path": str(source), "sha256": sha256_file(source), "count": 48},
        "output": {"path": str(output), "sha256": sha256_file(output), "count": 48},
        "all_error_uid_root_causes": dict(sorted(all_counts.items())),
        "consensus_retained_error_uid_count": len(retained),
        "consensus_retained_root_causes": dict(
            sorted(Counter(row["primary_root_cause"] for row in retained).items())
        ),
        "interpretation": (
            "Counts diagnose the frozen prompt-dev seam only; they do not revise truth "
            "or constitute an unbiased estimate of production error rates."
        ),
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload, "report_sha256": sha256_file(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
