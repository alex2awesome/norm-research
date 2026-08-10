"""Merge the disjoint guarded R1/R2 and R3 math cross-audit overlays.

Each source overlay retains its own byte/source/program guards and validator.
This module validates both, requires complete disjoint coverage of all retrieved
R1/R2/R3 candidates, and emits one overlay for the canonical fidelity merger.
It performs no candidate execution and reads no items or outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam import (
    adjudicate_math_construct_fidelity_r1_r2_cross_audit as r12,
)
from methods.metric_seam import adjudicate_math_construct_fidelity_r3_cross_audit as r3


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "metric-seam.math-static-construct-fidelity-cross-adjudication-merged.v1"
SOURCE_SCHEMA = "metric-seam.math-static-construct-fidelity-cross-adjudication.v1"
LEVELS = ("R1", "R2", "R3")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(ROOT.resolve())
    return {"path": str(relative), "sha256": _sha256(path)}


def _counts(rows: Sequence[Mapping]) -> dict[str, Any]:
    retrieved = [row for row in rows if row.get("candidate") is not None]
    eligible = [row for row in retrieved if row["eligible_for_relation_local_execution"]]
    return {
        "retrieved_candidates": len(retrieved),
        "retrieved_verdicts": dict(sorted(Counter(row["verdict"] for row in retrieved).items())),
        "retrieved_depths": dict(
            sorted(Counter(str(row["audited_depth"]) for row in retrieved).items())
        ),
        "eligible_depths": dict(
            sorted(Counter(str(row["audited_depth"]) for row in eligible).items())
        ),
        "eligible_for_relation_local_execution": len(eligible),
    }


def _apply(rows: Sequence[Mapping], changes: Sequence[Mapping]) -> list[dict]:
    output = {str(row["cell_id"]): copy.deepcopy(dict(row)) for row in rows}
    for change in changes:
        cell_id = str(change["cell_id"])
        if cell_id not in output:
            raise ValueError(f"cross-audit change is outside its source rows: {cell_id}")
        row = output[cell_id]
        for field, value in change["before"].items():
            if row.get(field) != value:
                raise ValueError(f"stale guarded before value for {cell_id}:{field}")
        row.update(copy.deepcopy(change["after"]))
    return list(output.values())


def merge_cross_audits(
    source_audits: Sequence[Mapping],
    overlays: Sequence[Mapping],
    *,
    source_audit_paths: Sequence[Path] | None = None,
    overlay_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    if len(source_audits) != 2 or len(overlays) != 2:
        raise ValueError("expected disjoint R1/R2 and R3 source audits/overlays")
    by_levels = {tuple(overlay.get("levels", [])): index for index, overlay in enumerate(overlays)}
    if set(by_levels) != {("R1", "R2"), ("R3",)}:
        raise ValueError("cross-audit overlays must cover disjoint R1/R2 and R3 levels")
    r12_index, r3_index = by_levels[("R1", "R2")], by_levels[("R3",)]
    r12.validate(dict(overlays[r12_index]), dict(source_audits[r12_index]))
    r3.validate(dict(overlays[r3_index]), dict(source_audits[r3_index]))

    ordered_indices = (r12_index, r3_index)
    ordered_sources = [source_audits[index] for index in ordered_indices]
    ordered_overlays = [overlays[index] for index in ordered_indices]
    rows = [copy.deepcopy(row) for source in ordered_sources for row in source["rows"]]
    if len(rows) != 90 or len({row["cell_id"] for row in rows}) != 90:
        raise ValueError("source audits do not close over 90 unique math cells")
    retrieved = [row for row in rows if row.get("candidate") is not None]
    if len(retrieved) != 47:
        raise ValueError("source audits must contain the frozen 47 retrieved candidates")
    for overlay in ordered_overlays:
        if (
            overlay.get("schema") != SOURCE_SCHEMA
            or overlay.get("status") != "complete_guarded_static_cross_audit"
            or overlay.get("task") != "math-stackexchange"
            or overlay.get("design_scope")
            != "outcome_blind_static_code_only_cross_adjudication"
            or overlay.get("forbidden_inputs_used") is not False
            or overlay.get("candidate_execution_performed") is not False
            or overlay.get("candidate_import_performed") is not False
            or overlay.get("model_or_api_calls_performed") is not False
            or overlay.get("review_coverage", {}).get("all_retrieved_candidates_reviewed")
            is not True
        ):
            raise ValueError("source overlay is not a complete sealed static cross-audit")

    changes = [
        copy.deepcopy(change)
        for overlay in ordered_overlays
        for change in overlay["changes"]
    ]
    if len(changes) != 21 or len({change["cell_id"] for change in changes}) != 21:
        raise ValueError("expected 21 unique guarded changes across the two overlays")
    patched = _apply(rows, changes)
    before = _counts(rows)
    after = _counts(patched)
    if before != {
        "retrieved_candidates": 47,
        "retrieved_verdicts": {"mismatch": 13, "partial": 34},
        "retrieved_depths": {"1": 20, "2": 27},
        "eligible_depths": {"1": 10, "2": 24},
        "eligible_for_relation_local_execution": 34,
    }:
        raise ValueError("pre-cross-audit math counts drifted")
    if after != {
        "retrieved_candidates": 47,
        "retrieved_verdicts": {"mismatch": 14, "partial": 33},
        "retrieved_depths": {"1": 20, "2": 27},
        "eligible_depths": {"1": 10, "2": 23},
        "eligible_for_relation_local_execution": 33,
    }:
        raise ValueError(f"post-cross-audit math counts drifted: {after}")

    source_records = []
    if source_audit_paths is not None or overlay_paths is not None:
        if source_audit_paths is None or overlay_paths is None:
            raise ValueError("source and overlay path records must be supplied together")
        if len(source_audit_paths) != 2 or len(overlay_paths) != 2:
            raise ValueError("path records must align to the two input pairs")
        source_records = [
            {
                "levels": list(overlays[index]["levels"]),
                "source_audit": _source_record(source_audit_paths[index]),
                "source_overlay": _source_record(overlay_paths[index]),
            }
            for index in ordered_indices
        ]
    artifact = {
        "schema": SCHEMA,
        "status": "complete_guarded_static_cross_audit",
        "task": "math-stackexchange",
        "levels": list(LEVELS),
        "design_scope": "outcome_blind_static_code_only_cross_adjudication",
        "source_records": source_records,
        "forbidden_inputs": list(ordered_overlays[0]["forbidden_inputs"]),
        "forbidden_inputs_used": False,
        "candidate_execution_performed": False,
        "candidate_import_performed": False,
        "model_or_api_calls_performed": False,
        "accelerators_used": False,
        "review_coverage": {
            "source_rows": 90,
            "retrieved_candidates_reviewed": 47,
            "changed_rows": len(changes),
            "unchanged_retrieved_rows": 47 - len(changes),
            "all_retrieved_candidates_reviewed": True,
        },
        "before_counts": before,
        "after_counts_if_overlay_applied": after,
        "changes": sorted(changes, key=lambda change: change["cell_id"]),
        "interpretation": (
            "Complete static cross-audit of all 47 retrieved math candidates. The overlay "
            "corrects presence/function, matched depth, clamps, dead paths, and channel leakage. "
            "It performs no execution and neither mismatch nor non-discovery establishes tacitness."
        ),
    }
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, action="append", required=True)
    parser.add_argument("--overlay", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in args.source_audit]
    overlays = [json.loads(path.read_text(encoding="utf-8")) for path in args.overlay]
    artifact = merge_cross_audits(
        sources,
        overlays,
        source_audit_paths=args.source_audit,
        overlay_paths=args.overlay,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(artifact["after_counts_if_overlay_applied"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
