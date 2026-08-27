#!/usr/bin/env python3
"""Opaque held-out worker for the frozen Math-a12 symbolic relation.

The worker receives sanitized ``ctext`` under opaque aliases.  It receives no
source identifiers, prompt/reference values, outcome labels, residuals, or
parent-criterion aggregation.  The operation emits count-only relation-local
evidence and deliberately has no document-quality scalar.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

from methods.metric_seam.hybrids.ops_symbolic_steps_v1 import analyze_document


REQUEST_SCHEMA = "metric-seam.math-a12-symbolic-heldout-request.v1"
RESULT_SCHEMA = "metric-seam.math-a12-symbolic-heldout-worker-result.v1"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def execute(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the frozen count-only operation over opaque held-out rows."""

    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unexpected symbolic held-out request schema")
    if request.get("reference_values_present") is not False:
        raise ValueError("reference values must be absent from candidate execution")
    if request.get("source_identifiers_present") is not False:
        raise ValueError("source identifiers must be absent from candidate execution")
    if request.get("parent_scalar_requested") is not False:
        raise ValueError("a parent scalar is forbidden for this relation-local run")
    rows = request.get("eval_items")
    if not isinstance(rows, list) or not rows:
        raise ValueError("held-out request has no rows")

    outputs: list[dict[str, Any]] = []
    expected_keys = [f"heldout_{index:04d}" for index in range(1, len(rows) + 1)]
    for expected_key, row in zip(expected_keys, rows):
        if not isinstance(row, dict) or set(row) != {"ctext", "item_key"}:
            raise ValueError("candidate row exceeds the ctext/item_key allowlist")
        if row.get("item_key") != expected_key:
            raise ValueError("held-out aliases are not canonical")
        ctext = row.get("ctext")
        if not isinstance(ctext, str):
            raise ValueError("held-out ctext must be a string")
        analysis = analyze_document(ctext)
        if analysis.get("whole_criterion_scalar") is not None:
            raise ValueError("symbolic operation unexpectedly emitted a parent scalar")
        if analysis.get("whole_criterion_fidelity") != "UNAVAILABLE":
            raise ValueError("symbolic operation changed its whole-criterion boundary")
        if analysis.get("criterion_defect_witness_count") != 0:
            raise ValueError("ungated nonidentity was promoted to a criterion defect")
        outputs.append({"item_key": expected_key, "analysis": analysis})

    return {
        "schema": RESULT_SCHEMA,
        "relation_id": "explicit_rational_equality_preservation",
        "n_items": len(outputs),
        "reference_values_present": False,
        "source_identifiers_present": False,
        "parent_scalar_emitted": False,
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
