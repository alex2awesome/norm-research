"""Freeze legal relation-program selection from compiler train only.

The gate selects a relation program when it measured at least 30 compiler-train
items and produced a nonconstant finite output.  It then selects every audited
cell mapping that uses that relation.  It does not load heldout items, prompt
outputs, references, outcomes, or external supervision.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.adjudicate_legal_hierarchy_construct_fidelity import (
    SCHEMA as FIDELITY_SCHEMA,
)
from methods.metric_seam.hierarchy_legal_runner import (
    SCHEMA as EXECUTION_SCHEMA,
    TASK,
    validate_fidelity,
)


SCHEMA = "metric-seam.hierarchy-legal-train-gate.v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_EXECUTION = DEFAULT_BASE / "legal_compiler_train_execution_v1.json"
DEFAULT_FIDELITY = DEFAULT_BASE / "legal_construct_fidelity_v1.json"
DEFAULT_OUTPUT = DEFAULT_BASE / "legal_train_gate_v1.json"
MIN_MEASURED = 30
MIN_OFF_MODE = 5


class LegalTrainGateError(ValueError):
    """Raised if compiler-train inputs do not prove a sealed selection."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_execution(execution: Mapping[str, Any], *, fidelity_source: Path) -> None:
    if (
        execution.get("schema") != EXECUTION_SCHEMA
        or execution.get("phase") != "compiler_train"
        or execution.get("task") != TASK
    ):
        raise LegalTrainGateError("unexpected legal compiler-train execution")
    design = execution.get("design", {})
    if design.get("input_fields") != ["item_key", "ctext"] or design.get("exact_presented_ctext_used") is not True:
        raise LegalTrainGateError("compiler train did not use exact text-only inputs")
    for field in (
        "source_text_beyond_presented_ctext_loaded",
        "historical_600_character_construct_definitions_modified",
        "historical_h0_programs_modified_or_executed",
        "outcome_or_reference_values_loaded",
        "prompt_outputs_loaded",
        "external_supervision_used",
        "network_or_api_used",
        "accelerator_used",
        "whole_criterion_score_emitted",
        "prompt_articulability_measured",
        "reconstruction_measured",
        "isomorphism_measured",
    ):
        if design.get(field) is not False:
            raise LegalTrainGateError(f"compiler train violates {field}")
    sources = execution.get("sources", {})
    fidelity_binding = sources.get("fidelity", {})
    if fidelity_binding.get("path") != _relative(fidelity_source) or fidelity_binding.get("sha256") != _sha256(fidelity_source):
        raise LegalTrainGateError("compiler train is not bound to the supplied fidelity audit")
    if sources.get("selection_gate") is not None:
        raise LegalTrainGateError("compiler train unexpectedly received a selection gate")
    summary = execution.get("summary", {})
    if summary.get("n_items") != 150 or summary.get("status_counts") != {"measured_exact_presented_ctext": 150}:
        raise LegalTrainGateError("compiler train did not complete all 150 items")
    if summary.get("failure_types") != {}:
        raise LegalTrainGateError("compiler train has failures")
    rows = execution.get("rows")
    if not isinstance(rows, list) or len(rows) != 150:
        raise LegalTrainGateError("compiler-train rows drifted")
    for row in rows:
        if set(row) != {"item_key", "ctext_sha256", "ctext_chars", "status", "error_type", "result"}:
            raise LegalTrainGateError("compiler row exposes unexpected fields")
        if not row["item_key"].startswith("train_") or row["status"] != "measured_exact_presented_ctext":
            raise LegalTrainGateError("compiler row is not a successful opaque train row")


def build_train_gate(
    execution: Mapping[str, Any],
    fidelity: Mapping[str, Any],
    *,
    execution_source: Path,
    fidelity_source: Path,
) -> dict[str, Any]:
    if fidelity.get("schema") != FIDELITY_SCHEMA or fidelity.get("task") != TASK:
        raise LegalTrainGateError("unexpected construct-fidelity input")
    audited_relation_ids, audited_mappings = validate_fidelity(fidelity)
    _validate_execution(execution, fidelity_source=fidelity_source)
    if execution.get("relation_ids") != audited_relation_ids or execution.get("relation_mappings") != audited_mappings:
        raise LegalTrainGateError("compiler train relation plan differs from the audit")
    measurements = execution["summary"].get("relation_measurement", {})
    if set(measurements) != set(audited_relation_ids):
        raise LegalTrainGateError("compiler train relation summary is incomplete")
    profiles = []
    selected_relation_ids = []
    for relation_id in audited_relation_ids:
        profile = measurements[relation_id]
        n_measured = profile.get("n_measured")
        nonconstant = profile.get("nonconstant")
        minimum, maximum = profile.get("minimum"), profile.get("maximum")
        if (
            isinstance(n_measured, bool)
            or not isinstance(n_measured, int)
            or not 0 <= n_measured <= 150
            or not isinstance(nonconstant, bool)
            or (n_measured and (not _finite(minimum) or not _finite(maximum)))
        ):
            raise LegalTrainGateError(f"invalid relation profile: {relation_id}")
        n_off_mode = profile.get("n_off_mode")
        if isinstance(n_off_mode, bool) or not isinstance(n_off_mode, int) or not 0 <= n_off_mode <= n_measured:
            raise LegalTrainGateError(f"invalid relation off-mode support: {relation_id}")
        selected = n_measured >= MIN_MEASURED and nonconstant and n_off_mode >= MIN_OFF_MODE
        if selected:
            selected_relation_ids.append(relation_id)
        profiles.append(
            {
                "relation_id": relation_id,
                "n_measured": n_measured,
                "n_abstained": profile["n_abstained"],
                "n_unique_values": profile["n_unique_values"],
                "largest_tie_count": profile["largest_tie_count"],
                "n_off_mode": n_off_mode,
                "minimum": minimum,
                "maximum": maximum,
                "nonconstant": nonconstant,
                "selected": selected,
                "selection_reason": (
                    "measured_at_least_30_and_nonconstant"
                    if selected
                    else "insufficient_measured_coverage"
                    if n_measured < MIN_MEASURED
                    else "insufficient_off_mode_support"
                    if n_off_mode < MIN_OFF_MODE
                    else "constant_on_compiler_train"
                ),
            }
        )
    selected_set = set(selected_relation_ids)
    selected_mappings = [row for row in audited_mappings if row["relation_id"] in selected_set]
    selected_cells = {row["cell_id"] for row in selected_mappings}
    by_level = Counter(row["level"] for row in selected_mappings)
    by_depth = Counter(row["effective_code_depth"] for row in selected_mappings)
    selected_cell_by_level = Counter(
        next(row["level"] for row in selected_mappings if row["cell_id"] == cell_id)
        for cell_id in selected_cells
    )
    return {
        "schema": SCHEMA,
        "status": "frozen-before-heldout-pre-reference-execution",
        "task": TASK,
        "source_execution": {
            "path": _relative(execution_source),
            "sha256": _sha256(execution_source),
            "schema": EXECUTION_SCHEMA,
        },
        "source_fidelity": {
            "path": _relative(fidelity_source),
            "sha256": _sha256(fidelity_source),
            "schema": FIDELITY_SCHEMA,
        },
        "selection_rule": {
            "minimum_compiler_train_measurements": MIN_MEASURED,
            "requires_nonconstant_finite_output": True,
            "minimum_compiler_train_off_mode_support": MIN_OFF_MODE,
            "uses_only_compiler_train_execution": True,
            "whole_construct_selection": False,
        },
        "separation": {
            "heldout_items_loaded": False,
            "prompt_outputs_loaded": False,
            "reference_or_outcome_values_loaded": False,
            "external_supervision_used": False,
            "articulability_measured": False,
            "reconstruction_measured": False,
            "isomorphism_measured": False,
        },
        "selected_relation_ids": selected_relation_ids,
        "selected_mappings": selected_mappings,
        "summary": {
            "n_audited_relation_programs": len(audited_relation_ids),
            "n_selected_relation_programs": len(selected_relation_ids),
            "n_audited_relation_mappings": len(audited_mappings),
            "n_selected_relation_mappings": len(selected_mappings),
            "n_selected_cells": len(selected_cells),
            "selected_mapping_by_level": {level: by_level.get(level, 0) for level in ("R1", "R2", "R3")},
            "selected_cell_by_level": {level: selected_cell_by_level.get(level, 0) for level in ("R1", "R2", "R3")},
            "selected_mapping_by_depth": {str(depth): count for depth, count in sorted(by_depth.items())},
        },
        "relation_profiles": profiles,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--fidelity", type=Path, default=DEFAULT_FIDELITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    execution = json.loads(args.execution.read_text(encoding="utf-8"))
    fidelity = json.loads(args.fidelity.read_text(encoding="utf-8"))
    payload = build_train_gate(
        execution,
        fidelity,
        execution_source=args.execution,
        fidelity_source=args.fidelity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
