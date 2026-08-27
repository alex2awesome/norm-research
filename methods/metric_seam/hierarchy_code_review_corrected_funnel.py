"""Build an additive code-review funnel after the independent static audit.

The historical fidelity, execution, train-gate, heldout-readiness, prompt, and
prevalence artifacts remain untouched.  This readout applies the guarded
static overlay as a filter: a mapping cannot remain train-operational or
heldout-ready after its construct-fidelity verdict becomes mismatch.

Only panel, construct-fidelity, train-selection, heldout-readiness, and prior
point-estimate metadata are read.  No item text, code vector, prompt value,
reference judgment, outcome, correlation, or reconstruction result is loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.adjudicate_code_review_construct_fidelity_cross_audit import (
    SCHEMA as CROSS_AUDIT_SCHEMA,
    validate_cross_audit,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
DEFAULT_PANEL = BASE / "panel_v3.json"
DEFAULT_FIDELITY = BASE / "code_review_construct_fidelity_v2.json"
DEFAULT_CROSS_AUDIT = (
    BASE / "code_review_construct_fidelity_independent_cross_audit_v1.json"
)
DEFAULT_TRAIN_GATE = BASE / "code_review_train_gate_v1.json"
DEFAULT_HELDOUT = BASE / "code_review_heldout_readiness_v1.json"
DEFAULT_PREVALENCE = BASE / "code_review_witness_prevalence_v3.json"
DEFAULT_OUTPUT = BASE / "code_review_corrected_funnel_v1.json"

SCHEMA = "metric-seam.code-review-corrected-funnel.v1"
FIDELITY_SCHEMA = "metric-seam.code-review-construct-fidelity-merged.v1"
TRAIN_SCHEMA = "metric-seam.hierarchy-code-train-gate.v1"
HELDOUT_SCHEMA = "metric-seam.hierarchy-code-heldout-readiness.v1"
PREVALENCE_SCHEMA = "metric-seam.hierarchy-witness-prevalence.v2"
LEVELS = ("R1", "R2", "R3")
STAGES = (
    "retrieved_candidate",
    "relation_local_static_fidelity",
    "train_operational_relation_witness",
    "heldout_confirmatory_reconstruction_evaluable",
)


class CorrectedFunnelError(ValueError):
    """Raised when source metadata cannot support a guarded corrected join."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectedFunnelError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorrectedFunnelError(f"{path}: expected a JSON object")
    return value


def _fraction(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        raise CorrectedFunnelError("rate inputs must be finite with a positive denominator")
    return round(numerator / denominator, 6)


def _validate_panel(panel: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells_raw = panel.get("cells")
    if not isinstance(cells_raw, list):
        raise CorrectedFunnelError("panel has no cells list")
    rows = [row for row in cells_raw if isinstance(row, Mapping) and row.get("task") == "code-review"]
    if len(rows) != 90:
        raise CorrectedFunnelError("panel must contain exactly 90 code-review cells")
    cells: dict[str, Mapping[str, Any]] = {}
    strata: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cell_id = row.get("id")
        level = row.get("level")
        if not isinstance(cell_id, str) or cell_id in cells:
            raise CorrectedFunnelError("panel contains a missing or duplicate code-review ID")
        if level not in LEVELS:
            raise CorrectedFunnelError(f"{cell_id}: invalid level {level!r}")
        required = (
            "source_kind",
            "breadth_stratum",
            "stratum_population_n",
            "stratum_selected_n",
            "design_weight",
        )
        if any(field not in row for field in required):
            raise CorrectedFunnelError(f"{cell_id}: missing sampling metadata")
        try:
            weight = float(row["design_weight"])
            population = int(row["stratum_population_n"])
            selected = int(row["stratum_selected_n"])
        except (TypeError, ValueError) as exc:
            raise CorrectedFunnelError(f"{cell_id}: invalid sampling metadata") from exc
        if not math.isfinite(weight) or weight <= 0 or population <= 0 or selected <= 0:
            raise CorrectedFunnelError(f"{cell_id}: non-positive sampling metadata")
        cells[cell_id] = row
        strata[(str(level), str(row["source_kind"]), str(row["breadth_stratum"]))].append(row)
    if Counter(str(row["level"]) for row in rows) != Counter({level: 30 for level in LEVELS}):
        raise CorrectedFunnelError("panel is not balanced at 30 cells per level")
    for key, members in strata.items():
        populations = {int(row["stratum_population_n"]) for row in members}
        selected_counts = {int(row["stratum_selected_n"]) for row in members}
        weights = {float(row["design_weight"]) for row in members}
        if len(populations) != 1 or len(selected_counts) != 1 or len(weights) != 1:
            raise CorrectedFunnelError(f"inconsistent stratum metadata for {key}")
        population = next(iter(populations))
        selected = next(iter(selected_counts))
        weight = next(iter(weights))
        if selected != len(members) or not math.isclose(weight, population / selected):
            raise CorrectedFunnelError(f"invalid design expansion for stratum {key}")
    total_weight = sum(float(row["design_weight"]) for row in rows)
    if not math.isclose(total_weight, 1128.0, abs_tol=1e-9):
        raise CorrectedFunnelError(f"code-review design weights sum to {total_weight}, not 1128")
    return cells


def _validate_fidelity(
    fidelity: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]], panel: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    if fidelity.get("schema") != FIDELITY_SCHEMA:
        raise CorrectedFunnelError("unexpected construct-fidelity schema")
    if fidelity.get("status") != "static_construct_fidelity_complete_pre_execution":
        raise CorrectedFunnelError("construct fidelity is not the complete pre-execution table")
    if fidelity.get("panel_content_sha256") != panel.get("panel_content_sha256"):
        raise CorrectedFunnelError("panel and construct-fidelity bindings differ")
    for flag in (
        "execution_performed",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "external_supervision",
    ):
        if fidelity.get(flag) is not False:
            raise CorrectedFunnelError(f"construct fidelity violates sealed flag {flag}")
    rows = fidelity.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise CorrectedFunnelError("construct fidelity must contain exactly 90 rows")
    audits = {str(row.get("cell_id")): row for row in rows if isinstance(row, Mapping)}
    if len(audits) != 90 or set(audits) != set(cells):
        raise CorrectedFunnelError("panel/fidelity cell identities differ")
    return audits


def _candidate_identity(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    candidate = row.get("candidate")
    if not isinstance(candidate, Mapping):
        return None
    try:
        return (
            str(candidate["aspect_id"]),
            str(candidate["source_path"]),
            str(candidate["source_sha256"]),
        )
    except KeyError as exc:
        raise CorrectedFunnelError(f"candidate identity is incomplete for {row.get('cell_id')}") from exc


def _program_scope(
    programs: Any,
    audits: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> tuple[set[str], dict[tuple[str, str, str], set[str]]]:
    if not isinstance(programs, list):
        raise CorrectedFunnelError(f"{label} has no program list")
    cell_ids: set[str] = set()
    by_program: dict[tuple[str, str, str], set[str]] = {}
    for program in programs:
        if not isinstance(program, Mapping):
            raise CorrectedFunnelError(f"{label} contains a malformed program")
        try:
            identity = (
                str(program["aspect_id"]),
                str(program["source_path"]),
                str(program["source_sha256"]),
            )
            owned = {str(cell_id) for cell_id in program["cell_ids"]}
        except (KeyError, TypeError) as exc:
            raise CorrectedFunnelError(f"{label} contains an incomplete program") from exc
        if identity in by_program or not owned:
            raise CorrectedFunnelError(f"{label} repeats or empties a program scope")
        for cell_id in owned:
            if cell_id in cell_ids or cell_id not in audits:
                raise CorrectedFunnelError(f"{label} contains duplicate/unknown cell {cell_id}")
            if _candidate_identity(audits[cell_id]) != identity:
                raise CorrectedFunnelError(f"{label} candidate identity drift for {cell_id}")
        cell_ids.update(owned)
        by_program[identity] = owned
    return cell_ids, by_program


def _validate_selection_sources(
    train_gate: Mapping[str, Any],
    heldout: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
) -> tuple[set[str], set[str], dict[tuple[str, str, str], set[str]], dict[tuple[str, str, str], set[str]]]:
    if train_gate.get("schema") != TRAIN_SCHEMA or train_gate.get("status") != (
        "frozen_before_heldout_program_execution"
    ):
        raise CorrectedFunnelError("unexpected or unfrozen train gate")
    for flag in ("reference_values_used", "outcome_labels_used", "heldout_items_or_outputs_used"):
        if train_gate.get(flag) is not False:
            raise CorrectedFunnelError(f"train gate violates sealed flag {flag}")
    if heldout.get("schema") != HELDOUT_SCHEMA or heldout.get("status") != (
        "frozen_before_prompt_reference_scoring"
    ):
        raise CorrectedFunnelError("unexpected or unfrozen heldout readiness")
    for flag in ("reference_values_used", "outcome_labels_used", "prompt_outputs_used"):
        if heldout.get(flag) is not False:
            raise CorrectedFunnelError(f"heldout readiness violates sealed flag {flag}")

    train, train_programs = _program_scope(
        train_gate.get("selected_programs"), audits, label="train gate"
    )
    confirmatory, heldout_programs = _program_scope(
        heldout.get("confirmatory_programs"), audits, label="heldout readiness"
    )
    historical_static = {
        cell_id
        for cell_id, row in audits.items()
        if row.get("eligible_for_relation_local_execution") is True
    }
    if not confirmatory <= train <= historical_static:
        raise CorrectedFunnelError("historical funnel is not nested")
    if not set(heldout_programs) <= set(train_programs):
        raise CorrectedFunnelError("heldout contains a program not selected on train")
    for identity, owned in heldout_programs.items():
        if not owned <= train_programs[identity]:
            raise CorrectedFunnelError("heldout program cell scope exceeds train scope")

    train_summary = train_gate.get("summary", {})
    heldout_summary = heldout.get("summary", {})
    expected_train = {
        "n_selected_programs": len(train_programs),
        "n_static_relation_mappings": len(historical_static),
        "n_selected_relation_mappings": len(train),
    }
    expected_heldout = {
        "n_train_selected_programs": len(train_programs),
        "n_confirmatory_programs": len(heldout_programs),
        "n_confirmatory_relation_mappings": len(confirmatory),
    }
    for field, expected in expected_train.items():
        if train_summary.get(field) != expected:
            raise CorrectedFunnelError(f"train summary drift at {field}")
    for field, expected in expected_heldout.items():
        if heldout_summary.get(field) != expected:
            raise CorrectedFunnelError(f"heldout summary drift at {field}")
    return train, confirmatory, train_programs, heldout_programs


def _apply_cross_audit(
    cross_audit: Mapping[str, Any],
    fidelity: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if cross_audit.get("schema") != CROSS_AUDIT_SCHEMA:
        raise CorrectedFunnelError("unexpected independent cross-audit schema")
    try:
        validate_cross_audit(cross_audit, fidelity)
    except ValueError as exc:
        raise CorrectedFunnelError(f"independent cross-audit failed validation: {exc}") from exc
    states = {
        cell_id: {
            "verdict": row.get("verdict"),
            "scope": row.get("scope"),
            "eligible_for_relation_local_execution": row.get(
                "eligible_for_relation_local_execution"
            ),
            "audited_depth": row.get("audited_depth"),
        }
        for cell_id, row in audits.items()
    }
    reviews = cross_audit.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 68:
        raise CorrectedFunnelError("independent cross-audit does not cover 68 retrieved rows")
    for review in reviews:
        cell_id = str(review.get("cell_id"))
        if cell_id not in states or review.get("before") != states[cell_id]:
            raise CorrectedFunnelError(f"cross-audit before-state drift for {cell_id}")
        after = review.get("after")
        if not isinstance(after, Mapping):
            raise CorrectedFunnelError(f"cross-audit after-state missing for {cell_id}")
        states[cell_id] = copy.deepcopy(dict(after))
    return states


def _stage_point(
    selected: set[str],
    population: Sequence[str],
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    denominator = len(population)
    selected_local = set(population) & selected
    weighted_denominator = sum(float(cells[cell_id]["design_weight"]) for cell_id in population)
    weighted_numerator = sum(float(cells[cell_id]["design_weight"]) for cell_id in selected_local)
    return {
        "balanced_panel": {
            "n_positive": len(selected_local),
            "denominator": denominator,
            "rate": _fraction(len(selected_local), denominator),
        },
        "conditional_eligible_inventory_expansion": {
            "estimated_positive_nodes": round(weighted_numerator, 6),
            "estimated_population_nodes": round(weighted_denominator, 6),
            "rate": _fraction(weighted_numerator, weighted_denominator),
        },
    }


def _readout(
    stage_sets: Mapping[str, set[str]],
    cells: Mapping[str, Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    all_ids = list(cells)
    stages = {
        stage: _stage_point(selected, all_ids, cells)
        for stage, selected in stage_sets.items()
    }
    by_level: dict[str, Any] = {}
    for level in LEVELS:
        level_ids = [cell_id for cell_id in all_ids if cells[cell_id]["level"] == level]
        by_level[level] = {
            stage: _stage_point(selected, level_ids, cells)
            for stage, selected in stage_sets.items()
        }

    by_depth: dict[str, Any] = {}
    for stage in STAGES[1:]:
        selected = stage_sets[stage]
        counts = Counter(str(states[cell_id]["audited_depth"]) for cell_id in selected)
        weighted = Counter()
        for cell_id in selected:
            weighted[str(states[cell_id]["audited_depth"])] += float(
                cells[cell_id]["design_weight"]
            )
        by_depth[stage] = {
            depth: {
                "n_positive": counts[depth],
                "estimated_positive_nodes": round(weighted[depth], 6),
                "rate_contribution_to_all_eligible_inventory": _fraction(
                    weighted[depth], 1128.0
                ),
            }
            for depth in ("1", "2")
        }

    static_n = len(stage_sets["relation_local_static_fidelity"])
    stages["relation_local_static_fidelity"]["fraction_of_corrected_static"] = 1.0
    for stage in STAGES[2:]:
        stages[stage]["fraction_of_corrected_static"] = _fraction(
            len(stage_sets[stage]), static_n
        )
    return {"stages": stages, "by_level": by_level, "by_depth": by_depth}


def _validate_prior_point_estimates(
    prevalence: Mapping[str, Any], historical: Mapping[str, Any]
) -> None:
    if prevalence.get("schema") != PREVALENCE_SCHEMA or prevalence.get("task") != "code-review":
        raise CorrectedFunnelError("unexpected historical prevalence artifact")
    scope = prevalence.get("pooled_eligible_action_nodes", {})
    balanced = scope.get("balanced_panel", {})
    expanded = scope.get("eligible_inventory_stratum_expansion", {})
    for stage in STAGES:
        expected = historical["stages"][stage]
        if balanced.get(stage, {}).get("rate") != expected["balanced_panel"]["rate"]:
            raise CorrectedFunnelError(f"historical balanced point drift for {stage}")
        if expanded.get(stage, {}).get("rate") != expected[
            "conditional_eligible_inventory_expansion"
        ]["rate"]:
            raise CorrectedFunnelError(f"historical expansion point drift for {stage}")


def _removed_records(
    removed: set[str],
    cells: Mapping[str, Mapping[str, Any]],
    audits: Mapping[str, Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for cell_id in sorted(removed, key=lambda value: (cells[value]["level"], value)):
        candidate = audits[cell_id]["candidate"]
        result.append(
            {
                "cell_id": cell_id,
                "level": cells[cell_id]["level"],
                "metric_name": audits[cell_id]["metric_name"],
                "candidate_aspect_id": candidate["aspect_id"],
                "candidate_source_path": candidate["source_path"],
                "historical_audited_depth": audits[cell_id]["audited_depth"],
                "corrected_matched_relation_depth": (
                    states[cell_id]["audited_depth"]
                    if states[cell_id]["eligible_for_relation_local_execution"]
                    else None
                ),
                "design_weight": cells[cell_id]["design_weight"],
                "removal_reason": "independent static construct-fidelity verdict is mismatch",
            }
        )
    return result


def build_corrected_funnel(
    panel: Mapping[str, Any],
    fidelity: Mapping[str, Any],
    cross_audit: Mapping[str, Any],
    train_gate: Mapping[str, Any],
    heldout: Mapping[str, Any],
    prevalence: Mapping[str, Any],
    *,
    sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the corrected metadata-only funnel and expansion point estimates."""

    cells = _validate_panel(panel)
    audits = _validate_fidelity(fidelity, cells, panel)
    train_before, heldout_before, train_programs, heldout_programs = (
        _validate_selection_sources(train_gate, heldout, audits)
    )
    states = _apply_cross_audit(cross_audit, fidelity, audits)

    retrieved = {cell_id for cell_id, row in audits.items() if row.get("candidate") is not None}
    static_before = {
        cell_id
        for cell_id, row in audits.items()
        if row.get("eligible_for_relation_local_execution") is True
    }
    static_after = {
        cell_id
        for cell_id, state in states.items()
        if state.get("eligible_for_relation_local_execution") is True
    }
    train_after = train_before & static_after
    heldout_after = heldout_before & train_after
    if not heldout_after <= train_after <= static_after <= retrieved:
        raise CorrectedFunnelError("corrected funnel is not nested")

    historical_sets = {
        "retrieved_candidate": retrieved,
        "relation_local_static_fidelity": static_before,
        "train_operational_relation_witness": train_before,
        "heldout_confirmatory_reconstruction_evaluable": heldout_before,
    }
    corrected_sets = {
        "retrieved_candidate": retrieved,
        "relation_local_static_fidelity": static_after,
        "train_operational_relation_witness": train_after,
        "heldout_confirmatory_reconstruction_evaluable": heldout_after,
    }
    historical = _readout(historical_sets, cells, {
        cell_id: {
            "audited_depth": audits[cell_id]["audited_depth"],
            "eligible_for_relation_local_execution": audits[cell_id][
                "eligible_for_relation_local_execution"
            ],
        }
        for cell_id in audits
    })
    corrected = _readout(corrected_sets, cells, states)
    _validate_prior_point_estimates(prevalence, historical)

    removed_static = static_before - static_after
    removed_train = train_before - train_after
    removed_heldout = heldout_before - heldout_after
    expected_removed_train = {
        "TB::code-review::general::R1::merged_tree::171::33b7ed9b7e4e601644ef",
        "TB::code-review::general::R2::merged_group::131::43ed2014b9a1669be3ca",
        "TB::code-review::general::R3::grandparent::3::681c2abce3bef33e3781",
    }
    if len(removed_static) != 6 or removed_train != expected_removed_train:
        raise CorrectedFunnelError("corrected removal scope differs from the guarded audit")
    if removed_heldout != expected_removed_train:
        raise CorrectedFunnelError("heldout removal scope differs from the guarded audit")

    remaining_train_programs = {
        identity for identity, owned in train_programs.items() if owned & train_after
    }
    remaining_heldout_programs = {
        identity for identity, owned in heldout_programs.items() if owned & heldout_after
    }
    depth_change = next(
        change
        for change in cross_audit["changes"]
        if change["changed_fields"] == ["audited_depth"]
    )
    depth_cell = depth_change["cell_id"]
    if depth_cell not in train_after or depth_cell in heldout_after:
        raise CorrectedFunnelError(
            "Documentation IA depth correction should remain train-operational but not confirmatory"
        )

    comparisons = {}
    for stage in STAGES:
        before_balanced = historical["stages"][stage]["balanced_panel"]
        after_balanced = corrected["stages"][stage]["balanced_panel"]
        before_expanded = historical["stages"][stage][
            "conditional_eligible_inventory_expansion"
        ]
        after_expanded = corrected["stages"][stage][
            "conditional_eligible_inventory_expansion"
        ]
        comparisons[stage] = {
            "balanced_n_before": before_balanced["n_positive"],
            "balanced_n_after": after_balanced["n_positive"],
            "balanced_rate_before": before_balanced["rate"],
            "balanced_rate_after": after_balanced["rate"],
            "balanced_rate_change": round(
                after_balanced["rate"] - before_balanced["rate"], 6
            ),
            "expanded_rate_before": before_expanded["rate"],
            "expanded_rate_after": after_expanded["rate"],
            "expanded_rate_change": round(
                after_expanded["rate"] - before_expanded["rate"], 6
            ),
        }

    return {
        "schema": SCHEMA,
        "status": "corrected_static_gate_propagated_without_reexecution",
        "task": "code-review",
        "design_scope": "unsupervised_metadata_only_funnel_correction",
        "sources": dict(sources or {}),
        "sealed_inputs": {
            "item_text_loaded": False,
            "program_execution_performed": False,
            "program_score_vectors_loaded": False,
            "prompt_or_model_calls_performed": False,
            "llm_judgments_loaded": False,
            "references_loaded": False,
            "outcomes_loaded": False,
            "correlations_loaded": False,
            "reconstruction_results_loaded": False,
            "external_supervision_used": False,
            "gpu_used": False,
        },
        "propagation_rule": (
            "Corrected train-operational = historical train-selected intersect corrected "
            "static fidelity. Corrected heldout-ready = historical confirmatory intersect "
            "corrected train-operational. Historical selection and execution are not rerun."
        ),
        "historical_readout": historical,
        "corrected_readout": corrected,
        "before_after": comparisons,
        "program_counts": {
            "static_unique_eligible_before": len({
                _candidate_identity(audits[cell_id]) for cell_id in static_before
            }),
            "static_unique_eligible_after": len({
                _candidate_identity(audits[cell_id]) for cell_id in static_after
            }),
            "train_selected_before": len(train_programs),
            "train_selected_after_static_filter": len(remaining_train_programs),
            "heldout_confirmatory_before": len(heldout_programs),
            "heldout_confirmatory_after_static_filter": len(remaining_heldout_programs),
        },
        "removed_mappings": {
            "static": _removed_records(removed_static, cells, audits, states),
            "train_operational": _removed_records(removed_train, cells, audits, states),
            "heldout_confirmatory": _removed_records(
                removed_heldout, cells, audits, states
            ),
        },
        "depth_corrections": [
            {
                "cell_id": depth_cell,
                "level": cells[depth_cell]["level"],
                "metric_name": audits[depth_cell]["metric_name"],
                "candidate_aspect_id": audits[depth_cell]["candidate"]["aspect_id"],
                "before_depth": depth_change["before"]["audited_depth"],
                "after_matched_relation_depth": depth_change["after"]["audited_depth"],
                "train_operational_after": depth_cell in train_after,
                "heldout_confirmatory_after": depth_cell in heldout_after,
                "reason": depth_change["reason"],
            }
        ],
        "interpretation": (
            "These are corrected relation-local static/availability readouts, not "
            "codability, reconstruction, isomorphism, whole-construct verifiability, "
            "prompt articulability, or tacitness estimates. Conditional inventory "
            "expansion inherits deterministic within-stratum exchangeability assumptions "
            "and has no design-based uncertainty interval."
        ),
    }


def validate_corrected_funnel(
    artifact: Mapping[str, Any],
    panel: Mapping[str, Any],
    fidelity: Mapping[str, Any],
    cross_audit: Mapping[str, Any],
    train_gate: Mapping[str, Any],
    heldout: Mapping[str, Any],
    prevalence: Mapping[str, Any],
) -> None:
    expected = build_corrected_funnel(
        panel,
        fidelity,
        cross_audit,
        train_gate,
        heldout,
        prevalence,
        sources=artifact.get("sources") if isinstance(artifact.get("sources"), Mapping) else {},
    )
    if artifact != expected:
        raise CorrectedFunnelError("corrected funnel differs from guarded rebuild")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--fidelity", type=Path, default=DEFAULT_FIDELITY)
    parser.add_argument("--cross-audit", type=Path, default=DEFAULT_CROSS_AUDIT)
    parser.add_argument("--train-gate", type=Path, default=DEFAULT_TRAIN_GATE)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--prevalence", type=Path, default=DEFAULT_PREVALENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite additive artifact: {args.out}")
    source_paths = {
        "panel": _relative(args.panel),
        "construct_fidelity": _relative(args.fidelity),
        "independent_cross_audit": _relative(args.cross_audit),
        "train_gate": _relative(args.train_gate),
        "heldout_readiness": _relative(args.heldout),
        "historical_prevalence": _relative(args.prevalence),
    }
    artifact = build_corrected_funnel(
        _load(args.panel),
        _load(args.fidelity),
        _load(args.cross_audit),
        _load(args.train_gate),
        _load(args.heldout),
        _load(args.prevalence),
        sources=source_paths,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["before_after"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
