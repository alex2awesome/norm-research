"""Descriptive static math witness rates; never execute candidate programs.

The conditional expansion is a point estimate over eligible native action-node
records under hash-as-random within-stratum exchangeability. No confidence
interval, operational stage, held-out stage, or reconstruction claim is emitted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.hierarchy_math_fidelity_merge import SCHEMA as FIDELITY_SCHEMA


SCHEMA = "metric-seam.math-static-witness-prevalence.v1"
TASK = "math-stackexchange"
LEVELS = ("R1", "R2", "R3")
OUTCOMES = (
    "retrieved_candidate",
    "relation_local_static_fidelity",
    "whole_construct_exact",
)
EXPANSION_KEY = "eligible_inventory_stratum_expansion"


class MathPrevalenceError(ValueError):
    """Raised when the descriptive frame cannot be joined exactly."""


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(rows: Sequence[Mapping], outcome: str, *, weighted: bool) -> dict:
    if not rows:
        return {"n_sampled_nodes": 0, "expanded_population_nodes": 0.0, "rate": None}
    weights = [float(row["design_weight"]) if weighted else 1.0 for row in rows]
    denominator = sum(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        raise MathPrevalenceError("descriptive denominator must be finite and positive")
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
        raise MathPrevalenceError("math inventory must contain exactly R1/R2/R3")
    inventory = {str(row["level"]): row for row in inventory_rows}
    strata: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
    for cell in cells.values():
        strata[(
            str(cell["level"]), str(cell["source_kind"]), str(cell["breadth_stratum"])
        )].append(cell)
    for key, rows in strata.items():
        try:
            populations = {int(row["stratum_population_n"]) for row in rows}
            selected = {int(row["stratum_selected_n"]) for row in rows}
            probabilities = {float(row["inclusion_probability"]) for row in rows}
            weights = {float(row["design_weight"]) for row in rows}
        except (KeyError, TypeError, ValueError) as error:
            raise MathPrevalenceError(f"invalid math stratum metadata for {key}") from error
        if len(populations) != 1 or len(selected) != 1:
            raise MathPrevalenceError(f"inconsistent math stratum counts for {key}")
        population_n, selected_n = next(iter(populations)), next(iter(selected))
        if selected_n != len(rows) or not 0 < selected_n <= population_n:
            raise MathPrevalenceError(f"invalid math selected count for {key}")
        if (
            len(probabilities) != 1 or len(weights) != 1
            or not math.isclose(
                next(iter(probabilities)), selected_n / population_n, abs_tol=1e-12
            )
            or not math.isclose(next(iter(weights)), population_n / selected_n, abs_tol=1e-12)
        ):
            raise MathPrevalenceError(f"math inclusion fraction/design weight drifted for {key}")
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
        raise MathPrevalenceError("math strata do not sum to the eligible inventory")
    weighted_total = sum(float(cell["design_weight"]) for cell in cells.values())
    eligible_total = sum(eligible_by_level.values())
    if not math.isclose(weighted_total, eligible_total, abs_tol=1e-9):
        raise MathPrevalenceError("math weights do not expand to the eligible inventory")
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


def build_math_prevalence(
    panel: Mapping, fidelity: Mapping, *, sources: Mapping | None = None,
) -> dict:
    # Validate the frozen task-local frame below rather than coupling this
    # artifact to later global-panel metadata migrations in unrelated tasks.
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise MathPrevalenceError("unexpected hierarchy panel schema")
    if not isinstance(panel.get("panel_content_sha256"), str):
        raise MathPrevalenceError("hierarchy panel has no content identity")
    if fidelity.get("schema") != FIDELITY_SCHEMA or fidelity.get("task") != TASK:
        raise MathPrevalenceError("unexpected merged math construct-fidelity artifact")
    if fidelity.get("panel_content_sha256") != panel.get("panel_content_sha256"):
        raise MathPrevalenceError("math fidelity audit is bound to another panel")
    if fidelity.get("status") not in {
        "provisional_static_merge_pending_cross_audit",
        "static_construct_fidelity_complete_pre_execution",
    }:
        raise MathPrevalenceError("math fidelity artifact has an invalid review state")
    for field in (
        "execution_performed", "items_loaded", "reference_values_loaded",
        "outcome_labels_loaded", "program_outputs_loaded", "external_supervision",
    ):
        if fidelity.get(field) is not False:
            raise MathPrevalenceError(f"math static artifact crossed forbidden boundary: {field}")
    rows = fidelity.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise MathPrevalenceError("math fidelity artifact must contain exactly 90 rows")
    audits = {str(row.get("cell_id")): row for row in rows}
    if len(audits) != 90:
        raise MathPrevalenceError("math fidelity rows have duplicate IDs")
    cells = {
        str(cell["id"]): cell for cell in panel["cells"] if cell.get("task") == TASK
    }
    if len(cells) != 90 or set(cells) != set(audits):
        raise MathPrevalenceError("math panel/fidelity identities do not match exactly")
    frame = _validate_sampling_frame(panel, cells)

    joined = []
    for cell_id, cell in cells.items():
        audit = audits[cell_id]
        if audit.get("level") != cell.get("level"):
            raise MathPrevalenceError(f"{cell_id}: level drift")
        verdict = audit.get("verdict")
        eligible = audit.get("eligible_for_relation_local_execution")
        if not isinstance(eligible, bool) or eligible != (verdict in {"partial", "exact"}):
            raise MathPrevalenceError(f"{cell_id}: fidelity eligibility drift")
        joined.append({
            "cell_id": cell_id,
            "level": cell["level"],
            "source_kind": cell["source_kind"],
            "breadth_stratum": cell["breadth_stratum"],
            "design_weight": cell["design_weight"],
            "retrieved_candidate": audit.get("candidate") is not None,
            "relation_local_static_fidelity": eligible,
            "whole_construct_exact": verdict == "exact",
            "audited_depth": audit.get("audited_depth"),
        })

    by_level = {
        level: _scope([row for row in joined if row["level"] == level]) for level in LEVELS
    }
    source_kind_specific = {
        level: {
            kind: _scope([
                row for row in joined if row["level"] == level and row["source_kind"] == kind
            ])
            for kind in sorted({row["source_kind"] for row in joined if row["level"] == level})
        }
        for level in LEVELS
    }
    merged = [
        row for row in joined if row["source_kind"] in {"merged_tree", "merged_group"}
    ]
    depth_witnesses = {}
    for depth in range(5):
        depth_rows = [
            {**row, "depth_witness": (
                row["relation_local_static_fidelity"] and row["audited_depth"] == depth
            )}
            for row in joined
        ]
        depth_witnesses[str(depth)] = {
            "balanced_panel": _rate(depth_rows, "depth_witness", weighted=False),
            EXPANSION_KEY: _rate(depth_rows, "depth_witness", weighted=True),
        }
    provisional = fidelity["cross_audit"]["status"] != "complete"
    return {
        "schema": SCHEMA,
        "status": (
            "provisional_static_rates_pending_cross_audit"
            if provisional else "static_descriptive_rates_cross_audited"
        ),
        "task": TASK,
        "sources": dict(sources or {}),
        "panel_content_sha256": panel["panel_content_sha256"],
        "cross_audit": fidelity["cross_audit"],
        "sampling_frame": frame,
        "outcome_definitions": {
            "retrieved_candidate": "frozen static candidate retrieval; not construct fidelity",
            "relation_local_static_fidelity": (
                "audited static match to at least one construct subrelation; not codability"
            ),
            "whole_construct_exact": "audited exact static whole-construct relation fidelity",
        },
        "estimands": {
            "balanced_panel": "unweighted descriptive rate in the balanced 30-node-per-level panel",
            EXPANSION_KEY: (
                "conditional stratum-expansion point estimate over 1,185 eligible native "
                "action-node records, treating deterministic outcome-blind SHA rank as "
                "pseudo-random/exchangeable within source-kind x breadth x level strata"
            ),
            "sampling_uncertainty": (
                "not estimated; no randomized-design replicate weights or audited alternate "
                "samples exist"
            ),
        },
        "pooled_eligible_action_nodes": _scope(joined),
        "by_level": by_level,
        "eligible_static_witness_by_audited_depth": depth_witnesses,
        "point_sensitivities": {
            "source_kind_specific": source_kind_specific,
            "merged_only": _scope(merged),
        },
        "uncertainty_intervals_emitted": False,
        "execution_or_outcome_stages_emitted": False,
        "claim_limits": [
            "This artifact is provisional until the independent cross-audit is complete."
            if provisional else "The independent static cross-audit is complete.",
            "No math candidate program was executed.",
            "No prompt references, outcomes, correlations, reconstruction, or isomorphism were loaded.",
            "Rates describe existing historical-library static witnesses, not metric codability.",
            "Mismatch and no-candidate rows are bounded non-discovery, never evidence of tacitness.",
            "The conditional expansion covers eligible action-node records, not unique constructs or raw rubrics.",
            "R1/R2/R3 point differences do not establish an abstraction or hierarchy-round trend.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    payload = build_math_prevalence(
        _load(args.panel), _load(args.fidelity),
        sources={"panel": str(args.panel), "construct_fidelity": str(args.fidelity)},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        outcome: payload["pooled_eligible_action_nodes"][EXPANSION_KEY][outcome]["rate"]
        for outcome in OUTCOMES
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
