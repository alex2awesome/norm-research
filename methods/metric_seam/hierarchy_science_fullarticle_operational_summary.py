"""Summarize the additive science static -> train -> heldout code funnel.

The executable sample is the new abstract+extracted-body representation, not the
canonical abstract-only hierarchy sample.  The output therefore reports two distinct
denominators: relation-mapping prevalence over the frozen 90-cell hierarchy panel and
item-level measurability over the additive 150/150 execution splits.  It never treats
the latter as a codability rate or as directly comparable canonical execution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.hierarchy_science_claim_prevalence import (
    _validate_sampling_frame,
)
from methods.metric_seam.hierarchy_science_fullarticle_runner import (
    EXECUTION_SCHEMA,
    GATE_SCHEMA,
    MAX_TRAIN_FAILED,
    MIN_DISTINCT_MEASURED_STATUSES,
    MIN_TRAIN_COVERAGE,
    MIN_TRAIN_MEASURED,
    _summarize_execution,
    _validate_execution_rows,
    _validate_science_contract,
    build_train_gate,
)


SCHEMA = "metric-seam.science-fullarticle-operational-prevalence.v1"
TASK = "peer-review"
LEVELS = ("R1", "R2", "R3")
EXPANSION_KEY = "eligible_inventory_stratum_expansion"
STAGES = (
    "static_relation_local_witness",
    "train_operational_fullarticle_section_verifier",
    "heldout_measurable_fullarticle_section_verifier",
)


class ScienceOperationalSummaryError(ValueError):
    """Raised when the additive science execution funnel does not close."""


def _rate(rows: Sequence[Mapping], stage: str, *, weighted: bool) -> dict[str, Any]:
    weights = [float(row["design_weight"]) if weighted else 1.0 for row in rows]
    denominator = sum(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ScienceOperationalSummaryError("science stage denominator is invalid")
    numerator = sum(
        weight * bool(row[stage]) for weight, row in zip(weights, rows)
    )
    return {
        "n_sampled_nodes": len(rows),
        "expanded_population_nodes": round(denominator, 6),
        "expanded_positive_nodes": round(numerator, 6),
        "rate": round(numerator / denominator, 6),
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _scope(rows: Sequence[Mapping]) -> dict:
    return {
        "n_sampled_nodes": len(rows),
        "balanced_panel": {
            stage: _rate(rows, stage, weighted=False) for stage in STAGES
        },
        EXPANSION_KEY: {
            stage: _rate(rows, stage, weighted=True) for stage in STAGES
        },
        "stage_retention": {
            "train_given_static": {
                "numerator": sum(row[STAGES[1]] for row in rows),
                "denominator": sum(row[STAGES[0]] for row in rows),
                "fraction": _fraction(
                    sum(row[STAGES[1]] for row in rows),
                    sum(row[STAGES[0]] for row in rows),
                ),
            },
            "heldout_given_train_operational": {
                "numerator": sum(row[STAGES[2]] for row in rows),
                "denominator": sum(row[STAGES[1]] for row in rows),
                "fraction": _fraction(
                    sum(row[STAGES[2]] for row in rows),
                    sum(row[STAGES[1]] for row in rows),
                ),
            },
        },
    }


def _validate_execution(
    execution: Mapping,
    seed: Mapping,
    fidelity: Mapping,
    *,
    phase: str,
) -> tuple[list[dict], dict]:
    if (
        execution.get("schema") != EXECUTION_SCHEMA
        or execution.get("status") != "execution_complete_pre_prompt_pre_reference"
        or execution.get("phase") != phase
    ):
        raise ScienceOperationalSummaryError(
            f"science {phase} execution is not complete"
        )
    try:
        eligible = _validate_science_contract(seed, fidelity)
        rows = _validate_execution_rows(execution.get("rows"))
    except ValueError as error:
        raise ScienceOperationalSummaryError(str(error)) from error
    if execution.get("relation_mappings") != eligible:
        raise ScienceOperationalSummaryError(
            f"science {phase} relation mappings drifted"
        )
    policy = execution.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise ScienceOperationalSummaryError(
            f"science {phase} execution policy is missing"
        )
    for field in (
        "reference_values_loaded",
        "outcome_values_loaded",
        "prompt_or_reconstruction_outputs_loaded",
        "external_supervision_used",
        "models_or_apis_called",
        "accelerators_used",
    ):
        if policy.get(field) is not False:
            raise ScienceOperationalSummaryError(
                f"science {phase} execution crossed forbidden boundary: {field}"
            )
    representation = execution.get("representation")
    if (
        not isinstance(representation, Mapping)
        or representation.get("same_ctext_bytes_for_future_prompt_and_code") is not True
        or representation.get("canonical_hierarchy_items") is not False
        or representation.get("complete_pdf_claimed") is not False
    ):
        raise ScienceOperationalSummaryError(
            f"science {phase} representation provenance drifted"
        )
    expected_summary = _summarize_execution(rows, n_relations=len(eligible))
    if execution.get("summary") != expected_summary:
        raise ScienceOperationalSummaryError(
            f"science {phase} execution summary drifted"
        )
    return eligible, expected_summary


def _measurability_decision(summary: Mapping) -> dict:
    states = summary["three_state_totals_unique_items"]
    criteria = {
        "minimum_measured_items": states["measured"] >= MIN_TRAIN_MEASURED,
        "minimum_measured_coverage": summary["measured_coverage"]
        >= MIN_TRAIN_COVERAGE,
        "minimum_distinct_measured_statuses": summary[
            "n_distinct_measured_verifier_statuses"
        ]
        >= MIN_DISTINCT_MEASURED_STATUSES,
        "maximum_failed_items": states["failed"] <= MAX_TRAIN_FAILED,
    }
    return {"passes": all(criteria.values()), "criteria_passed": criteria}


def build_science_operational_summary(
    panel: Mapping,
    seed: Mapping,
    fidelity: Mapping,
    train_execution: Mapping,
    train_gate: Mapping,
    heldout_execution: Mapping,
    *,
    sources: Mapping | None = None,
) -> dict:
    """Build the noncanonical representation's relation-mapping execution funnel."""

    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise ScienceOperationalSummaryError("unexpected hierarchy panel schema")
    if fidelity.get("source_panel_content_sha256") != panel.get(
        "panel_content_sha256"
    ):
        raise ScienceOperationalSummaryError(
            "science construct-fidelity audit is bound to another panel"
        )
    cells = {
        str(cell["id"]): cell
        for cell in panel.get("cells", [])
        if cell.get("task") == TASK
    }
    fidelity_rows = {
        str(row["cell_id"]): row for row in fidelity.get("rows", [])
    }
    if len(cells) != 90 or len(fidelity_rows) != 90 or set(cells) != set(fidelity_rows):
        raise ScienceOperationalSummaryError(
            "science panel and construct-fidelity rows do not close at 90 cells"
        )
    try:
        sampling_frame = _validate_sampling_frame(panel, cells)
    except ValueError as error:
        raise ScienceOperationalSummaryError(str(error)) from error
    train_relations, train_summary = _validate_execution(
        train_execution, seed, fidelity, phase="compiler_train"
    )
    heldout_relations, heldout_summary = _validate_execution(
        heldout_execution, seed, fidelity, phase="heldout_pre_reference"
    )
    if train_relations != heldout_relations:
        raise ScienceOperationalSummaryError("science train/heldout mappings drifted")
    try:
        rebuilt_gate = build_train_gate(train_execution)
    except ValueError as error:
        raise ScienceOperationalSummaryError(str(error)) from error
    if train_gate != rebuilt_gate or train_gate.get("schema") != GATE_SCHEMA:
        raise ScienceOperationalSummaryError(
            "science train gate is not the deterministic train-only rebuild"
        )
    if train_gate.get("selected_relation_mappings") != train_relations:
        raise ScienceOperationalSummaryError(
            "science train gate did not select the frozen relation mappings"
        )
    heldout_gate = heldout_execution.get("train_gate")
    if not isinstance(heldout_gate, Mapping) or heldout_gate != {
        "schema": GATE_SCHEMA,
        "selected": True,
        "n_selected_relation_mappings": 6,
        "selection_used_heldout": False,
    }:
        raise ScienceOperationalSummaryError(
            "science heldout execution is not bound to the train-only gate"
        )

    static_cells = {
        row["cell_id"]
        for row in fidelity["rows"]
        if row.get("eligible_for_later_relation_local_execution") is True
    }
    train_cells = {
        row["cell_id"] for row in train_gate["selected_relation_mappings"]
    }
    heldout_decision = _measurability_decision(heldout_summary)
    heldout_cells = train_cells if heldout_decision["passes"] else set()
    if not heldout_cells <= train_cells <= static_cells:
        raise ScienceOperationalSummaryError(
            "science operational relation mappings are not nested"
        )

    joined = []
    for cell_id, cell in cells.items():
        audit = fidelity_rows[cell_id]
        if (
            audit.get("level") != cell.get("level")
            or audit.get("metric_name") != cell.get("construct")
        ):
            raise ScienceOperationalSummaryError(
                f"{cell_id}: science panel/audit metadata drifted"
            )
        joined.append(
            {
                "cell_id": cell_id,
                "level": cell["level"],
                "source_kind": cell["source_kind"],
                "breadth_stratum": cell["breadth_stratum"],
                "design_weight": cell["design_weight"],
                STAGES[0]: cell_id in static_cells,
                STAGES[1]: cell_id in train_cells,
                STAGES[2]: cell_id in heldout_cells,
            }
        )

    return {
        "schema": SCHEMA,
        "status": "additive_representation_static_train_heldout_funnel_complete",
        "task": TASK,
        "sources": dict(sources or {}),
        "representation": {
            "canonical_hierarchy_items": False,
            "sample": (
                "new current-stage outcome-blind 300-paper split from a historical "
                "outcome-stratified 2,400-paper evidence corpus"
            ),
            "ctext": (
                "abstract plus upstream-capped extracted methods/results/evaluation body"
            ),
            "same_bytes_for_future_prompt_and_current_code": True,
            "direct_comparison_to_canonical_abstract_only_execution": False,
            "complete_pdf_claimed": False,
            "upstream_corpus_historically_outcome_stratified": True,
            "outcome_values_used_by_current_split_gate_or_execution": False,
        },
        "scientific_object": {
            "code_relation": (
                "document-internal numeric/comparative abstract-claim consistency with "
                "distinct retrieved body sentences"
            ),
            "relation_unit": (
                "one of six independently construct-fidelity-approved hierarchy-cell mappings"
            ),
            "effective_code_depth": 3,
            "external_scientific_truth": False,
            "whole_peer_review_construct": False,
        },
        "channel_contract": {
            "program_execution_outputs_read": True,
            "item_text_loaded_by_summary": False,
            "reference_values_loaded": False,
            "outcome_values_loaded": False,
            "prompt_or_reconstruction_outputs_loaded": False,
            "external_supervision_used": False,
            "models_or_apis_called": False,
            "accelerators_used": False,
        },
        "validation": {
            "stage_relation_mapping_counts": {
                STAGES[0]: len(static_cells),
                STAGES[1]: len(train_cells),
                STAGES[2]: len(heldout_cells),
            },
            "compiler_train": train_summary,
            "train_only_gate": train_gate["summary"],
            "heldout_pre_reference": heldout_summary,
            "heldout_measurability": heldout_decision,
        },
        "sampling_frame": sampling_frame,
        "estimands": {
            "balanced_panel": (
                "descriptive relation-mapping rate over the frozen 90-cell peer-review panel"
            ),
            EXPANSION_KEY: (
                "conditional stratum-expansion point estimate over 675 eligible native "
                "peer-review action-node records"
            ),
            "item_measurability": (
                "three-state coverage over each additive 150-item split; not a codability rate"
            ),
            "sampling_uncertainty": "not estimated; descriptive point estimates only",
            "source_frame": (
                "item rates describe the historically outcome-stratified 2,400-paper "
                "evidence frame and are not population prevalence estimates"
            ),
        },
        "pooled_eligible_action_nodes": _scope(joined),
        "by_level": {
            level: _scope([row for row in joined if row["level"] == level])
            for level in LEVELS
        },
        "item_execution": {
            "compiler_train": train_summary,
            "heldout_pre_reference": heldout_summary,
        },
        "uncertainty_intervals_emitted": False,
        "claim_limits": [
            "The 6/90 relation-mapping rate is partial subrelation coverage, not whole-metric codability.",
            "The 118/150 train and 108/150 heldout measured rates are applicability and execution coverage, not codability.",
            "This additive representation/sample is not directly comparable to canonical abstract-only hierarchy execution.",
            "A support certificate is document-internal consistency, not external scientific truth.",
            "Insufficient and abstained outputs do not establish lack of support or tacitness.",
            "No prompt articulability, reconstruction, isomorphism, or code-beats-LLM claim is measured.",
            "No reference judgement, outcome value, external supervision, model/API call, or accelerator was used by this split, gate, or execution.",
            "The upstream evidence corpus was historically outcome-stratified even though this split, gate, and execution never loaded outcome values.",
        ],
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "panel": args.panel,
        "seed_map": args.seed,
        "construct_fidelity": args.fidelity,
        "compiler_train_execution": args.train,
        "train_gate": args.gate,
        "heldout_pre_reference_execution": args.heldout,
    }
    result = build_science_operational_summary(
        _load(args.panel),
        _load(args.seed),
        _load(args.fidelity),
        _load(args.train),
        _load(args.gate),
        _load(args.heldout),
        sources={name: str(path) for name, path in paths.items()},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
