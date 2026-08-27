#!/usr/bin/env python3
"""Project inspectable pair certificates from the frozen Math-a12 operation.

The worker replays the exact traversal used by ``analyze_document`` while
retaining the already-computed per-pair outputs of ``verify_expression_pair``.
It receives sanitized ctext under opaque aliases and no prompt reference.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import sympy as sp

from methods.metric_seam.hybrids.ops_symbolic_steps_v1 import (
    MAX_PAIR_CANDIDATES,
    SCHEMA as ANALYSIS_SCHEMA,
    MathOps,
    _answer_only,
    _clean_expression,
    _equation_rows,
    _parse_rational_expression,
    analyze_document,
    verify_expression_pair,
)


REQUEST_SCHEMA = "metric-seam.math-a12-pair-certificate-request.v1"
RESULT_SCHEMA = "metric-seam.math-a12-pair-certificate-worker-result.v1"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _analysis_from_pairs(
    pairs: list[Mapping[str, Any]], *, equality_rows: int, budget_exhausted: bool
) -> dict[str, Any]:
    statuses = Counter(pair["status"] for pair in pairs)
    parsed_count = len(pairs) - statuses["parse_noncoverage"]
    return {
        "schema": ANALYSIS_SCHEMA,
        "relation_id": "explicit_rational_equality_preservation",
        "equality_rows_seen": equality_rows,
        "pair_candidate_count": len(pairs),
        "parsed_rational_pair_count": parsed_count,
        "verified_rational_identity_count": statuses["verified_rational_identity"],
        "exact_nonidentity_witness_count": statuses["exact_nonidentity_witness"],
        "universal_identity_counterexample_count": statuses[
            "universal_identity_counterexample"
        ],
        "symbolically_unresolved_count": statuses["symbolically_unresolved"],
        "parse_noncoverage_count": statuses["parse_noncoverage"],
        "positive_code_witness_count": sum(
            pair.get("positive_code_witness") is True for pair in pairs
        ),
        "criterion_defect_witness_count": 0,
        "abstained": parsed_count == 0,
        "pair_budget_exhausted": budget_exhausted,
        "whole_criterion_fidelity": "UNAVAILABLE",
        "whole_criterion_scalar": None,
    }


def _canonical_expression(source: str) -> str | None:
    try:
        return sp.sstr(_parse_rational_expression(_clean_expression(source)))
    except Exception:
        return None


def project_document(text: str) -> dict[str, Any]:
    """Return pair-level projection and prove it collapses to frozen v1 output."""

    certificates: list[dict[str, Any]] = []
    equality_rows = 0
    budget_exhausted = False
    span_index = 0
    for _kind, span in MathOps.extract_math_spans(_answer_only(text)):
        span_index += 1
        for parts in _equation_rows(span):
            equality_rows += 1
            for adjacency_index, (lhs, rhs) in enumerate(zip(parts, parts[1:]), 1):
                if len(certificates) >= MAX_PAIR_CANDIDATES:
                    budget_exhausted = True
                    break
                verified = verify_expression_pair(lhs, rhs)
                certificate = {
                    "pair_index": len(certificates) + 1,
                    "span_index": span_index,
                    "equality_row_index": equality_rows,
                    "adjacency_index": adjacency_index,
                    "pair_sha256": verified["pair_sha256"],
                    "status": verified["status"],
                    "expression_pair": {
                        "lhs_latex_sanitized": _clean_expression(lhs),
                        "rhs_latex_sanitized": _clean_expression(rhs),
                        "lhs_sympy_canonical": _canonical_expression(lhs),
                        "rhs_sympy_canonical": _canonical_expression(rhs),
                    },
                    "domain_nonzero_obligations": verified.get(
                        "domain_nonzero_obligations"
                    ),
                    "counterexample_assignment": verified.get(
                        "counterexample_assignment"
                    ),
                    "positive_code_witness": verified.get(
                        "positive_code_witness", False
                    ),
                    "criterion_defect_witness": verified.get(
                        "criterion_defect_witness", False
                    ),
                    "declared_universal_scope": verified.get(
                        "declared_universal_scope", False
                    ),
                    "claim_scope_required": verified.get("claim_scope_required"),
                    "reason": verified.get("reason"),
                }
                certificates.append(certificate)
            if budget_exhausted:
                break
        if budget_exhausted:
            break

    projected_analysis = _analysis_from_pairs(
        certificates,
        equality_rows=equality_rows,
        budget_exhausted=budget_exhausted,
    )
    frozen_analysis = analyze_document(text)
    if projected_analysis != frozen_analysis:
        raise ValueError("pair projection does not collapse to frozen analyze_document")
    return {
        "analysis": projected_analysis,
        "certificates": certificates,
        "dynamic_max_contributing_depth": (
            3 if projected_analysis["parsed_rational_pair_count"] > 0 else 1
        ),
    }


def execute(request: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unexpected pair-certificate request schema")
    if request.get("reference_values_present") is not False:
        raise ValueError("reference values must be absent")
    if request.get("source_identifiers_present") is not False:
        raise ValueError("source identifiers must be absent")
    rows = request.get("eval_items")
    if not isinstance(rows, list) or not rows:
        raise ValueError("pair-certificate request has no rows")
    expected = [f"heldout_{index:04d}" for index in range(1, len(rows) + 1)]
    outputs = []
    for expected_key, row in zip(expected, rows):
        if not isinstance(row, dict) or set(row) != {"ctext", "item_key"}:
            raise ValueError("projection row exceeds the ctext/item_key allowlist")
        if row.get("item_key") != expected_key or not isinstance(row.get("ctext"), str):
            raise ValueError("projection row has invalid alias or ctext")
        outputs.append(
            {"item_key": expected_key, "projection": project_document(row["ctext"])}
        )
    return {
        "schema": RESULT_SCHEMA,
        "relation_id": "explicit_rational_equality_preservation",
        "n_items": len(outputs),
        "reference_values_present": False,
        "source_identifiers_present": False,
        "outputs": outputs,
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} REQUEST OUTPUT")
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output_path.write_bytes(_canonical_bytes(execute(request)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
