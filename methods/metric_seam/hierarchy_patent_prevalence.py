"""Descriptive static patent witness rates over the hierarchy panel.

This module joins the frozen 90-cell patent panel to the static fidelity audit.
It never executes a program or reads items, outcomes, judge scores, program
outputs, correlations, or reconstruction results.  Its conditional expansion
is a point estimate under within-stratum exchangeability of the deterministic,
outcome-blind SHA sample; it is not a design-unbiased estimate and has no
confidence interval.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.adjudicate_patent_construct_fidelity import SCHEMA as FIDELITY_SCHEMA


SCHEMA = "metric-seam.patent-static-witness-prevalence.v1"
TASK = "patents"
LEVELS = ("R1", "R2", "R3")
EXPANSION_KEY = "eligible_inventory_stratum_expansion"
OUTCOMES = (
    "retrieved_candidate",
    "relation_local_static_fidelity",
    "depth3_evidence_relation",
    "pure_code_witness",
    "whole_construct_exact",
)


class PatentPrevalenceError(ValueError):
    """Raised when the panel/audit descriptive join fails closed."""


def _rate(rows: Sequence[Mapping], outcome: str, *, weighted: bool) -> dict:
    if not rows:
        return {
            "n_sampled_nodes": 0,
            "expanded_population_nodes": 0.0,
            "expanded_positive_nodes": 0.0,
            "rate": None,
        }
    weights = [float(row["design_weight"]) if weighted else 1.0 for row in rows]
    denominator = sum(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        raise PatentPrevalenceError("descriptive denominator must be finite and positive")
    numerator = sum(weight * bool(row[outcome]) for weight, row in zip(weights, rows))
    return {
        "n_sampled_nodes": len(rows),
        "expanded_population_nodes": round(denominator, 6),
        "expanded_positive_nodes": round(numerator, 6),
        "rate": round(numerator / denominator, 6),
    }


def _scope(rows: Sequence[Mapping]) -> dict:
    return {
        "n_sampled_nodes": len(rows),
        "balanced_panel": {outcome: _rate(rows, outcome, weighted=False) for outcome in OUTCOMES},
        EXPANSION_KEY: {outcome: _rate(rows, outcome, weighted=True) for outcome in OUTCOMES},
    }


def _validate_sampling_frame(panel: Mapping, cells: Mapping[str, Mapping]) -> dict:
    inventory_rows = [row for row in panel.get("inventory", []) if row.get("task") == TASK]
    if {row.get("level") for row in inventory_rows} != set(LEVELS):
        raise PatentPrevalenceError("patent inventory must contain exactly R1/R2/R3")
    inventory = {str(row["level"]): row for row in inventory_rows}
    strata: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
    for cell in cells.values():
        strata[(
            str(cell["level"]),
            str(cell["source_kind"]),
            str(cell["breadth_stratum"]),
        )].append(cell)
    for key, rows in strata.items():
        try:
            populations = {int(row["stratum_population_n"]) for row in rows}
            selected = {int(row["stratum_selected_n"]) for row in rows}
            probabilities = {float(row["inclusion_probability"]) for row in rows}
            weights = {float(row["design_weight"]) for row in rows}
        except (KeyError, TypeError, ValueError) as error:
            raise PatentPrevalenceError(f"invalid patent stratum metadata for {key}") from error
        if len(populations) != 1 or len(selected) != 1:
            raise PatentPrevalenceError(f"inconsistent patent stratum counts for {key}")
        population_n, selected_n = next(iter(populations)), next(iter(selected))
        if selected_n != len(rows) or not 0 < selected_n <= population_n:
            raise PatentPrevalenceError(f"invalid patent selected count for {key}")
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
            raise PatentPrevalenceError(
                f"patent inclusion fraction/design weight drifted for {key}"
            )
    stratum_population = Counter()
    for (level, _kind, _breadth), rows in strata.items():
        stratum_population[level] += int(rows[0]["stratum_population_n"])
    eligible_by_level = {
        level: int(inventory[level]["n_eligible_nodes"]) for level in LEVELS
    }
    complete_by_level = {
        level: int(inventory[level]["n_complete_nodes"]) for level in LEVELS
    }
    if dict(stratum_population) != eligible_by_level:
        raise PatentPrevalenceError("patent strata do not sum to the eligible inventory")
    weighted_total = sum(float(cell["design_weight"]) for cell in cells.values())
    eligible_total = sum(eligible_by_level.values())
    if not math.isclose(weighted_total, eligible_total, abs_tol=1e-9):
        raise PatentPrevalenceError("patent weights do not expand to the eligible inventory")
    complete_total = sum(complete_by_level.values())
    return {
        "n_complete_action_node_records": complete_total,
        "n_eligible_action_node_records": eligible_total,
        "n_excluded_by_frozen_eligibility_rule": complete_total - eligible_total,
        "complete_by_level": complete_by_level,
        "eligible_by_level": eligible_by_level,
        "n_sampling_strata": len(strata),
        "selected_per_stratum": sorted({len(rows) for rows in strata.values()}),
        "eligibility_rule": "nonempty name, at least 8 description words, and at least 1 child",
    }


def build_patent_prevalence(
    panel: Mapping, fidelity: Mapping, *, sources: Mapping | None = None
) -> dict:
    # Validate the frozen task-local sampling frame below.  Do not invoke the
    # evolving global panel validator here: newer generators require metadata
    # that was intentionally added after this frozen v3 sample and would make
    # an unrelated task's migration state alter the patent estimate.
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise PatentPrevalenceError("unexpected hierarchy panel schema")
    if not isinstance(panel.get("panel_content_sha256"), str):
        raise PatentPrevalenceError("hierarchy panel has no content identity")
    if fidelity.get("schema") != FIDELITY_SCHEMA or fidelity.get("task") != TASK:
        raise PatentPrevalenceError("unexpected patent construct-fidelity artifact")
    if fidelity.get("status") != "static-relation-local-adjudication-complete":
        raise PatentPrevalenceError("patent fidelity audit is not complete")
    if fidelity.get("source_panel_content_sha256") != panel.get("panel_content_sha256"):
        raise PatentPrevalenceError("patent fidelity audit is bound to another panel")
    for field in (
        "execution_performed",
        "items_loaded",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "program_outputs_loaded",
        "external_supervision_loaded_for_this_audit",
    ):
        if fidelity.get(field) is not False:
            raise PatentPrevalenceError(f"patent static audit crossed forbidden boundary: {field}")
    audit_rows = fidelity.get("rows")
    if not isinstance(audit_rows, list) or len(audit_rows) != 90:
        raise PatentPrevalenceError("patent fidelity audit must contain exactly 90 rows")
    audits = {str(row.get("cell_id")): row for row in audit_rows}
    if len(audits) != 90:
        raise PatentPrevalenceError("patent fidelity rows contain duplicate IDs")
    cells = {
        str(cell["id"]): cell for cell in panel["cells"] if cell.get("task") == TASK
    }
    if len(cells) != 90 or set(cells) != set(audits):
        raise PatentPrevalenceError("patent panel/fidelity identities do not match exactly")
    frame = _validate_sampling_frame(panel, cells)

    joined = []
    for cell_id, cell in cells.items():
        audit = audits[cell_id]
        if audit.get("level") != cell.get("level") or audit.get("metric_name") != cell.get(
            "construct"
        ):
            raise PatentPrevalenceError(f"{cell_id}: panel/audit source metadata drifted")
        verdict = audit.get("verdict")
        if verdict not in {"no_candidate", "partial_relation_local"}:
            raise PatentPrevalenceError(f"{cell_id}: unexpected patent fidelity verdict")
        retrieved = audit.get("candidate_aspect_id") is not None
        relation_local = verdict == "partial_relation_local"
        if retrieved != relation_local:
            raise PatentPrevalenceError(
                f"{cell_id}: v1 patent bank requires every retrieved seed to be adjudicated partial"
            )
        depths = audit.get("eligible_relation_local_depths")
        if not isinstance(depths, list) or any(
            isinstance(depth, bool) or not isinstance(depth, int) or depth not in range(5)
            for depth in depths
        ):
            raise PatentPrevalenceError(f"{cell_id}: invalid matched-relation depths")
        if relation_local != bool(depths):
            raise PatentPrevalenceError(f"{cell_id}: verdict/depth mismatch")
        joined.append({
            "cell_id": cell_id,
            "level": cell["level"],
            "source_kind": cell["source_kind"],
            "breadth_stratum": cell["breadth_stratum"],
            "design_weight": cell["design_weight"],
            "retrieved_candidate": retrieved,
            "relation_local_static_fidelity": relation_local,
            "depth3_evidence_relation": 3 in depths,
            "pure_code_witness": bool(audit.get("pure_code_witness")),
            "whole_construct_exact": bool(audit.get("exact_whole_construct_fidelity")),
        })

    by_level = {
        level: _scope([row for row in joined if row["level"] == level]) for level in LEVELS
    }
    source_kind_specific = {
        level: {
            kind: _scope([
                row
                for row in joined
                if row["level"] == level and row["source_kind"] == kind
            ])
            for kind in sorted({
                row["source_kind"] for row in joined if row["level"] == level
            })
        }
        for level in LEVELS
    }
    merged = [
        row for row in joined if row["source_kind"] in {"merged_tree", "merged_group"}
    ]
    return {
        "schema": SCHEMA,
        "status": "static_descriptive_rates_complete",
        "task": TASK,
        "sources": dict(sources or {}),
        "panel_content_sha256": panel["panel_content_sha256"],
        "sampling_frame": frame,
        "outcome_definitions": {
            "retrieved_candidate": "source-only seed retrieval; not construct fidelity",
            "relation_local_static_fidelity": (
                "static partial match to at least one criterion sub-relation; not codability"
            ),
            "depth3_evidence_relation": (
                "matched relation uses the precomputed prior-art evidence operation; not pure code"
            ),
            "pure_code_witness": "whole candidate is an executable code-only witness",
            "whole_construct_exact": "exact static whole-construct relation fidelity",
        },
        "estimands": {
            "balanced_panel": "unweighted descriptive rate in the balanced 30-node-per-level panel",
            EXPANSION_KEY: (
                "conditional stratum-expansion point estimate over 1,368 eligible native "
                "action-node records, treating deterministic outcome-blind SHA rank as pseudo-"
                "random/exchangeable within source-kind x breadth x level strata"
            ),
            "sampling_uncertainty": (
                "not estimated; no randomized-design replicate weights or audited alternate "
                "samples exist"
            ),
        },
        "pooled_eligible_action_nodes": _scope(joined),
        "by_level": by_level,
        "point_sensitivities": {
            "source_kind_specific": source_kind_specific,
            "merged_only": _scope(merged),
        },
        "channel_provenance": {
            "historical_programs": "manual hybrids",
            "prior_art_candidates": "examiner/oracle conditioned",
            "disclosure_relations": "precomputed reading-model verdicts",
            "autonomous_retrieval": False,
            "pure_code": False,
        },
        "uncertainty_intervals_emitted": False,
        "execution_or_outcome_stages_emitted": False,
        "claim_limits": [
            "No patent candidate program was executed in this hierarchy pass.",
            "No prompt references, outcomes, correlations, reconstruction, or isomorphism were loaded.",
            "Rates describe static witnesses in a four-program historical bank, not patent-metric codability.",
            "Depth 3 denotes an external evidence pipeline, not pure code.",
            "The evidence candidates were examiner/oracle conditioned and are not autonomous retrieval.",
            "No-candidate rows are bounded non-discovery, never evidence of tacitness.",
            "The expansion covers eligible action-node records, not unique constructs or raw rubrics.",
            "R1/R2/R3 point differences do not establish an abstraction or hierarchy-round trend.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    fidelity = json.loads(args.fidelity.read_text(encoding="utf-8"))
    result = build_patent_prevalence(
        panel,
        fidelity,
        sources={"panel": str(args.panel), "construct_fidelity": str(args.fidelity)},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
