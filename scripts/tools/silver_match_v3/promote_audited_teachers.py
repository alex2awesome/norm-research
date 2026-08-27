#!/usr/bin/env python3
"""Promote order-stable teachers only after an independent precision audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .common import read_jsonl, sha256_file, write_jsonl


def gate(
    teachers: Sequence[dict[str, Any]],
    audit: dict[str, Any],
    *,
    min_retained_audit_n: int,
    min_point_precision: float,
    min_ci_lower: float,
) -> dict[str, Any]:
    raw = audit.get("retained_exact_precision") or {}
    weighted = audit.get("retained_exact_precision_design_weighted") or {}
    n = int(raw.get("n") or 0)
    point = weighted.get("estimate")
    interval = weighted.get("approximate_wilson_95")
    ci_lower = interval[0] if interval else None
    checks = {
        "nonempty_teacher_set": bool(teachers),
        "retained_audit_n": n >= min_retained_audit_n,
        "design_weighted_point_precision": point is not None
        and point >= min_point_precision,
        "approximate_ci_lower": ci_lower is not None and ci_lower >= min_ci_lower,
        "teacher_rows_previously_locked": all(
            row.get("gradient_eligible") is False for row in teachers
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "retained_audit_n": n,
            "design_weighted_point_precision": point,
            "approximate_wilson_95": interval,
        },
        "thresholds": {
            "min_retained_audit_n": min_retained_audit_n,
            "min_point_precision": min_point_precision,
            "min_ci_lower": min_ci_lower,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", required=True)
    parser.add_argument("--audit-score", required=True)
    parser.add_argument("--selection-record", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-retained-audit-n", type=int, default=30)
    parser.add_argument("--min-point-precision", type=float, default=0.90)
    parser.add_argument("--min-ci-lower", type=float, default=0.80)
    args = parser.parse_args()
    paths = {
        "retained": Path(args.retained).resolve(),
        "audit_score": Path(args.audit_score).resolve(),
        "selection_record": Path(args.selection_record).resolve(),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    teachers = list(read_jsonl(paths["retained"]))
    audit = json.loads(paths["audit_score"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection_record"].read_text(encoding="utf-8"))
    result = gate(
        teachers,
        audit,
        min_retained_audit_n=args.min_retained_audit_n,
        min_point_precision=args.min_point_precision,
        min_ci_lower=args.min_ci_lower,
    )
    if not result["passed"]:
        raise ValueError(f"independent teacher audit gate failed: {result}")
    task = str(selection.get("task") or "")
    if not task or any(str(row.get("task")) != task for row in teachers):
        raise ValueError("retained teacher task does not match the selection record")
    prompt_hash = str(selection.get("chosen", {}).get("prompt_sha256") or "")
    if not prompt_hash or any(
        row.get("verification_prompt_sha256") != prompt_hash for row in teachers
    ):
        raise ValueError("retained teachers do not match the selected verifier prompt")
    audit_hash = sha256_file(paths["audit_score"])
    promoted = []
    for source in teachers:
        row = dict(source)
        row.update(
            {
                "gradient_eligible": True,
                "promotion_source": "independent_blind_stratified_audit",
                "independent_audit_score_sha256": audit_hash,
                "audited_design_weighted_precision": result["observed"][
                    "design_weighted_point_precision"
                ],
                "audited_precision_interval": result["observed"][
                    "approximate_wilson_95"
                ],
            }
        )
        promoted.append(row)
    write_jsonl(output, promoted)
    report = {
        "count": len(promoted),
        "gate": result,
        "input_hashes": {key: sha256_file(path) for key, path in paths.items()},
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
