"""Balanced and weighted static sensitivity for math symbolic-step coverage.

This artifact leaves the canonical 33/90 result untouched.  It reports how a
previously manual SymPy/Lark rational-equality capability would change static
relation-local coverage after an independent five-dimension source audit.
Nothing in this module imports or executes the verifier or reads items,
certificates, prompt/reference outputs, outcomes, scores, or correlations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.adjudicate_math_symbolic_capability_expansion import (
    SCHEMA as ADJUDICATION_SCHEMA,
)
from methods.metric_seam.hierarchy_math_symbolic_capability_mapper import TASK


SCHEMA = "metric-seam.math-symbolic-capability-expansion-prevalence.v1"
LEVELS = ("R1", "R2", "R3")
EXPANSION_KEY = "eligible_inventory_stratum_expansion"
OUTCOMES = (
    "canonical_relation_local_unchanged",
    "formal_symbolic_relation_local",
    "newly_covered_by_formal_symbolic_relation",
    "existing_cell_adds_formal_symbolic_relation",
    "additive_sensitivity_union_relation_local",
    "whole_construct_exact",
)


class SymbolicExpansionPrevalenceError(ValueError):
    """Raised when the frozen sensitivity frame cannot be joined exactly."""


def _rate(rows: Sequence[Mapping[str, Any]], outcome: str, *, weighted: bool) -> dict:
    if not rows:
        return {
            "n_sampled_nodes": 0,
            "expanded_population_nodes": 0.0,
            "expanded_positive_nodes": 0.0,
            "rate": None,
        }
    weights = [float(row["design_weight"]) if weighted else 1.0 for row in rows]
    denominator = sum(weights)
    numerator = sum(
        weight * bool(row[outcome]) for weight, row in zip(weights, rows, strict=True)
    )
    if not math.isfinite(denominator) or denominator <= 0:
        raise SymbolicExpansionPrevalenceError("invalid sensitivity denominator")
    return {
        "n_sampled_nodes": len(rows),
        "expanded_population_nodes": round(denominator, 6),
        "expanded_positive_nodes": round(numerator, 6),
        "rate": round(numerator / denominator, 6),
    }


def _scope(rows: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "n_sampled_nodes": len(rows),
        "balanced_panel": {
            outcome: _rate(rows, outcome, weighted=False) for outcome in OUTCOMES
        },
        EXPANSION_KEY: {
            outcome: _rate(rows, outcome, weighted=True) for outcome in OUTCOMES
        },
    }


def _validate_sampling_frame(
    panel: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]]
) -> dict:
    inventory_rows = [
        row for row in panel.get("inventory", []) if row.get("task") == TASK
    ]
    if {row.get("level") for row in inventory_rows} != set(LEVELS):
        raise SymbolicExpansionPrevalenceError("math inventory lacks R1/R2/R3")
    inventory = {str(row["level"]): row for row in inventory_rows}
    strata: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells.values():
        strata[
            (
                str(cell["level"]),
                str(cell["source_kind"]),
                str(cell["breadth_stratum"]),
            )
        ].append(cell)
    for key, rows in strata.items():
        populations = {int(row["stratum_population_n"]) for row in rows}
        selected = {int(row["stratum_selected_n"]) for row in rows}
        probabilities = {float(row["inclusion_probability"]) for row in rows}
        weights = {float(row["design_weight"]) for row in rows}
        if len(populations) != 1 or len(selected) != 1:
            raise SymbolicExpansionPrevalenceError(f"inconsistent stratum counts: {key}")
        population_n = next(iter(populations))
        selected_n = next(iter(selected))
        if selected_n != len(rows) or not 0 < selected_n <= population_n:
            raise SymbolicExpansionPrevalenceError(f"invalid selected count: {key}")
        if (
            len(probabilities) != 1
            or len(weights) != 1
            or not math.isclose(
                next(iter(probabilities)), selected_n / population_n, abs_tol=1e-12
            )
            or not math.isclose(
                next(iter(weights)), population_n / selected_n, abs_tol=1e-12
            )
        ):
            raise SymbolicExpansionPrevalenceError(f"design weight drift: {key}")
    stratum_population: Counter[str] = Counter()
    for (level, _kind, _breadth), rows in strata.items():
        stratum_population[level] += int(rows[0]["stratum_population_n"])
    eligible_by_level = {
        level: int(inventory[level]["n_eligible_nodes"]) for level in LEVELS
    }
    complete_by_level = {
        level: int(inventory[level]["n_complete_nodes"]) for level in LEVELS
    }
    if dict(stratum_population) != eligible_by_level:
        raise SymbolicExpansionPrevalenceError("strata do not sum to eligible inventory")
    weighted_total = sum(float(cell["design_weight"]) for cell in cells.values())
    eligible_total = sum(eligible_by_level.values())
    if not math.isclose(weighted_total, eligible_total, abs_tol=1e-9):
        raise SymbolicExpansionPrevalenceError(
            "design weights do not expand to eligible inventory"
        )
    complete_total = sum(complete_by_level.values())
    return {
        "n_complete_action_node_records": complete_total,
        "n_eligible_action_node_records": eligible_total,
        "n_excluded_by_frozen_eligibility_rule": complete_total - eligible_total,
        "complete_by_level": complete_by_level,
        "eligible_by_level": eligible_by_level,
        "n_sampling_strata": len(strata),
        "selected_per_stratum": sorted({len(rows) for rows in strata.values()}),
        "eligibility_rule": (
            "nonempty name, at least 8 description words, and at least 1 child"
        ),
    }


def build_symbolic_expansion_prevalence(
    panel: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    *,
    sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise SymbolicExpansionPrevalenceError("unexpected hierarchy panel schema")
    panel_sha = panel.get("panel_content_sha256")
    if not isinstance(panel_sha, str):
        raise SymbolicExpansionPrevalenceError("panel content identity is missing")
    if (
        adjudication.get("schema") != ADJUDICATION_SCHEMA
        or adjudication.get("task") != TASK
    ):
        raise SymbolicExpansionPrevalenceError("unexpected symbolic adjudication")
    if (
        adjudication.get("status")
        != "static_five_dimension_adjudication_complete_pre_execution"
    ):
        raise SymbolicExpansionPrevalenceError("symbolic adjudication is incomplete")
    if adjudication.get("panel_content_sha256") != panel_sha:
        raise SymbolicExpansionPrevalenceError("adjudication is bound to another panel")
    if adjudication.get("canonical_artifact_modified") is not False:
        raise SymbolicExpansionPrevalenceError("canonical result was not preserved")
    for field in (
        "programs_or_items_executed",
        "certificate_counts_loaded",
        "prompt_outputs_loaded",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "correlations_or_reconstruction_loaded",
        "models_apis_or_gpus_used",
    ):
        if adjudication.get(field) is not False:
            raise SymbolicExpansionPrevalenceError(
                f"adjudication crossed forbidden boundary: {field}"
            )
    depth = adjudication.get("matched_relation_depth")
    if depth != {
        "depth": 3,
        "meaning": "positive formal relation via computer algebra",
        "isolation_or_test_execution_adds_depth": False,
    }:
        raise SymbolicExpansionPrevalenceError("formal relation depth receipt drifted")

    cells = {
        str(cell["id"]): cell
        for cell in panel.get("cells", [])
        if cell.get("task") == TASK
    }
    audits = {
        str(row.get("cell_id")): row for row in adjudication.get("rows", [])
    }
    if len(cells) != 90 or len(audits) != 90 or set(cells) != set(audits):
        raise SymbolicExpansionPrevalenceError(
            "panel/adjudication identities do not match exactly"
        )
    frame = _validate_sampling_frame(panel, cells)
    joined: list[dict[str, Any]] = []
    for cell_id, cell in cells.items():
        audit = audits[cell_id]
        if (
            audit.get("level") != cell.get("level")
            or audit.get("metric_name") != cell.get("construct")
            or audit.get("metric_description") != cell.get("description")
        ):
            raise SymbolicExpansionPrevalenceError(f"{cell_id}: audit source drift")
        retrieved = audit.get("retrieved_candidate")
        canonical = audit.get("canonical_relation_local_before")
        symbolic = audit.get("symbolic_relation_local_static_fidelity")
        exact = audit.get("whole_construct_exact")
        if not all(isinstance(value, bool) for value in (retrieved, canonical, symbolic, exact)):
            raise SymbolicExpansionPrevalenceError(f"{cell_id}: invalid outcome flag")
        dimensions = audit.get("dimension_audit")
        if retrieved:
            if not isinstance(dimensions, dict) or set(dimensions) != {
                "object",
                "relation",
                "polarity",
                "applicability",
                "aggregation",
            }:
                raise SymbolicExpansionPrevalenceError(
                    f"{cell_id}: incomplete five-dimension audit"
                )
        elif dimensions is not None:
            raise SymbolicExpansionPrevalenceError(
                f"{cell_id}: noncandidate has an adjudication"
            )
        if symbolic:
            if (
                not retrieved
                or audit.get("verdict") != "partial_relation_local"
                or audit.get("matched_relation_depth") != 3
                or exact
            ):
                raise SymbolicExpansionPrevalenceError(
                    f"{cell_id}: symbolic witness contract drift"
                )
        elif audit.get("matched_relation_depth") is not None:
            raise SymbolicExpansionPrevalenceError(
                f"{cell_id}: unmatched row has a relation depth"
            )
        joined.append(
            {
                "cell_id": cell_id,
                "level": cell["level"],
                "source_kind": cell["source_kind"],
                "breadth_stratum": cell["breadth_stratum"],
                "design_weight": cell["design_weight"],
                "canonical_relation_local_unchanged": canonical,
                "formal_symbolic_relation_local": symbolic,
                "newly_covered_by_formal_symbolic_relation": symbolic and not canonical,
                "existing_cell_adds_formal_symbolic_relation": symbolic and canonical,
                "additive_sensitivity_union_relation_local": symbolic or canonical,
                "whole_construct_exact": exact,
            }
        )
    counts = {outcome: sum(bool(row[outcome]) for row in joined) for outcome in OUTCOMES}
    expected_counts = {
        "canonical_relation_local_unchanged": 33,
        "formal_symbolic_relation_local": 7,
        "newly_covered_by_formal_symbolic_relation": 5,
        "existing_cell_adds_formal_symbolic_relation": 2,
        "additive_sensitivity_union_relation_local": 38,
        "whole_construct_exact": 0,
    }
    if counts != expected_counts:
        raise SymbolicExpansionPrevalenceError("symbolic sensitivity counts drifted")

    pooled = _scope(joined)
    by_level = {
        level: _scope([row for row in joined if row["level"] == level])
        for level in LEVELS
    }
    return {
        "schema": SCHEMA,
        "status": "static_additive_sensitivity_complete_pre_execution",
        "task": TASK,
        "sources": dict(sources or {}),
        "panel_content_sha256": panel_sha,
        "sampling_frame": frame,
        "capability_provenance": adjudication["capability_selection_provenance"],
        "relation_depth_receipt": {
            **depth,
            "formal_symbolic_matched_cells": 7,
            "newly_covered_at_depth3": 5,
            "already_covered_adding_depth3_relation": 2,
            "capability_runtime_receipt": adjudication["capability_runtime_receipt"],
        },
        "outcome_definitions": {
            "canonical_relation_local_unchanged": (
                "the frozen cross-audited 33-cell static result; read-only and unchanged"
            ),
            "formal_symbolic_relation_local": (
                "independently adjudicated rational equality preservation/nonidentity for "
                "a presented step at frozen formal-relation depth 3"
            ),
            "newly_covered_by_formal_symbolic_relation": (
                "formal-symbolic match on a cell absent from the canonical 33"
            ),
            "existing_cell_adds_formal_symbolic_relation": (
                "formal-symbolic relation added to a canonically covered cell without "
                "replacing its prior relation or depth"
            ),
            "additive_sensitivity_union_relation_local": (
                "canonical static relation-local cells union accepted symbolic mappings"
            ),
            "whole_construct_exact": "exact whole-construct fidelity; always false here",
        },
        "estimands": {
            "balanced_panel": (
                "unweighted descriptive sensitivity in the balanced 30-cell-per-level panel"
            ),
            EXPANSION_KEY: (
                "conditional stratum-expansion point sensitivity over 1,185 eligible native "
                "action-node records, assuming deterministic SHA rank is exchangeable within "
                "source-kind x breadth x level strata"
            ),
            "sampling_uncertainty": (
                "not estimated; no randomized-design replicate weights or alternate "
                "capability-adjudication samples exist"
            ),
        },
        "pooled_eligible_action_nodes": pooled,
        "by_level": by_level,
        "uncertainty_intervals_emitted": False,
        "program_or_item_execution_emitted": False,
        "prompt_reference_outcome_or_reconstruction_stages_emitted": False,
        "canonical_artifact_modified": False,
        "claim_limits": [
            "The canonical static result remains 33/90; 38/90 is an additive capability sensitivity only.",
            "The SymPy/Lark verifier is a manually designed historical pipeline seed, not an automatic discovery.",
            "All seven symbolic matches are only presented-step rational equality relations at frozen depth 3.",
            "Isolation wrappers and test presence are runtime metadata and do not increase relation depth.",
            "No whole-proof correctness, every-step aggregation, execution performance, prompt articulability, reconstruction, isomorphism, or codability is estimated.",
            "The weighted expansion is conditional and has no design-based uncertainty interval.",
            "Failure to retrieve or match is bounded non-discovery, never evidence of tacitness.",
        ],
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    payload = build_symbolic_expansion_prevalence(
        _load(args.panel),
        _load(args.adjudication),
        sources={
            "panel": str(args.panel),
            "symbolic_construct_fidelity": str(args.adjudication),
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["pooled_eligible_action_nodes"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
