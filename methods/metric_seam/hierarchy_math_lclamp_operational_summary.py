"""Validate and summarize the sealed math constant-L execution funnel.

This artifact concerns conditional slices ``g_c(x) = f(x, c)`` of retrospective
historical hybrids.  Train-operational means that a program had a train-only,
nonconstant, sufficiently covered sentinel slice selected by the frozen gate.
Heldout-measurable means that exact frozen slice remained nonconstant and
failure-free on the pre-reference heldout items.  Neither stage evaluates the
original hybrid, a pure-code rewrite, a whole construct, prompt articulability,
reconstruction, isomorphism, codability, or tacitness.

The builder reads program execution outputs but never loads item text,
references, outcomes, LLM values, model outputs, or accelerator state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.hierarchy_math_lclamp_gate import (
    _EXECUTION_TOP_FIELDS,
    _PROGRAM_FIELDS,
    _validate_profile_result,
    build_train_profile_gate,
)
from methods.metric_seam.hierarchy_math_lclamp_runner import (
    EXECUTION_SCHEMA,
    SENTINELS,
    _capability_sources_from_audit,
    _content_fingerprint,
    apply_profile_selection,
    build_execution_plan,
    build_sentinel_profiles,
    validate_merged_audit,
    validate_profiles,
)
from methods.metric_seam.hierarchy_math_prevalence import _validate_sampling_frame


SCHEMA = "metric-seam.math-lclamp-operational-prevalence.v1"
TASK = "math-stackexchange"
LEVELS = ("R1", "R2", "R3")
DEPTHS = tuple(str(depth) for depth in range(5))
EXPANSION_KEY = "eligible_inventory_stratum_expansion"
STAGES = (
    "static_relation_local_witness",
    "train_operational_constant_l_slice",
    "heldout_measurable_constant_l_slice",
)
CANONICAL_GATE_THRESHOLDS = {
    "min_measured": 10,
    "min_coverage": 0.05,
    "min_unique_scores": 2,
    "max_failed": 0,
    "profile_tie_break": "lowest fixed profile_index",
}


class MathLClampSummaryError(ValueError):
    """Raised when the sealed execution funnel does not close exactly."""


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _rate(rows: Sequence[Mapping], outcome: str, *, weighted: bool) -> dict[str, Any]:
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
        raise MathLClampSummaryError("stage denominator must be finite and positive")
    numerator = sum(weight * bool(row[outcome]) for weight, row in zip(weights, rows))
    return {
        "n_sampled_nodes": len(rows),
        "expanded_population_nodes": round(denominator, 6),
        "expanded_positive_nodes": round(numerator, 6),
        "rate": round(numerator / denominator, 6),
    }


def _scope(rows: Sequence[Mapping]) -> dict[str, Any]:
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


def _program_summary(program: Mapping, *, n_items: int) -> dict[str, Any]:
    profiles = program["profiles"]
    profile_statuses = Counter(profile["measurement_status"] for profile in profiles)
    states = Counter()
    for profile in profiles:
        states.update(profile["summary"]["state_counts"])
    return {
        "n_items": n_items,
        "n_profiles": len(profiles),
        "profile_measurement_status_counts": dict(sorted(profile_statuses.items())),
        "three_state_totals": {
            state: states[state] for state in ("measured", "abstained", "failed")
        },
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks, right_ranks = _average_ranks(left), _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left_ranks)
    right_ss = sum((y - right_mean) ** 2 for y in right_ranks)
    if left_ss == 0 or right_ss == 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _sentinel_sensitivity(train_execution: Mapping) -> dict[str, Any]:
    """Describe train ordering stability across fixed, nonconstant L profiles."""

    program_rows = []
    pooled_rhos: list[float] = []
    pooled_pairs = 0
    pooled_abstained = 0
    pooled_identical = 0
    for program in train_execution["programs"]:
        profiles = [
            profile
            for profile in program["profiles"]
            if profile["measurement_status"] == "nondegenerate_measurement"
        ]
        rhos: list[float] = []
        n_pairs = 0
        n_abstained = 0
        n_identical = 0
        for left, right in itertools.combinations(profiles, 2):
            n_pairs += 1
            left_scores = {
                row["item_key"]: float(row["score"])
                for row in left["rows"]
                if row["measurement_state"] == "measured"
            }
            right_scores = {
                row["item_key"]: float(row["score"])
                for row in right["rows"]
                if row["measurement_state"] == "measured"
            }
            common = sorted(set(left_scores) & set(right_scores))
            rho = _spearman(
                [left_scores[item] for item in common],
                [right_scores[item] for item in common],
            )
            if rho is None:
                n_abstained += 1
                continue
            rhos.append(rho)
            if (
                set(left_scores) == set(right_scores)
                and all(left_scores[item] == right_scores[item] for item in left_scores)
            ):
                n_identical += 1
        pooled_rhos.extend(rhos)
        pooled_pairs += n_pairs
        pooled_abstained += n_abstained
        pooled_identical += n_identical
        program_rows.append(
            {
                "aspect_id": program["aspect_id"],
                "n_nondegenerate_profiles": len(profiles),
                "n_profile_pairs": n_pairs,
                "n_spearman_pairs": len(rhos),
                "n_abstained_pairs": n_abstained,
                "n_identical_vector_pairs": n_identical,
                "identical_vector_pair_rate": _fraction(n_identical, len(rhos)),
                "spearman_median": (
                    round(_median(rhos), 6) if _median(rhos) is not None else None
                ),
                "spearman_min": round(min(rhos), 6) if rhos else None,
                "spearman_max": round(max(rhos), 6) if rhos else None,
            }
        )
    return {
        "status": "descriptive_train_only_no_target",
        "profile_scope": "nondegenerate fixed-sentinel compiler-train profiles only",
        "constant_profiles": "abstained before pairing",
        "pair_rule": (
            "Spearman agreement on common measured item keys; abstain when either rank "
            "vector is constant on the common support"
        ),
        "pooling_caveat": (
            "Pooled summaries are pair-weighted and descriptive; profile pairs within a "
            "program are dependent, and programs with more L fields contribute more pairs."
        ),
        "used_for_train_gate_selection": False,
        "used_for_heldout_decisions": False,
        "reference_values_used": False,
        "outcome_labels_used": False,
        "score_direction_or_target_used": False,
        "pooled_pair_weighted": {
            "n_programs": len(program_rows),
            "n_nondegenerate_profiles": sum(
                row["n_nondegenerate_profiles"] for row in program_rows
            ),
            "n_profile_pairs": pooled_pairs,
            "n_spearman_pairs": len(pooled_rhos),
            "n_abstained_pairs": pooled_abstained,
            "n_identical_vector_pairs": pooled_identical,
            "identical_vector_pair_rate": _fraction(
                pooled_identical, len(pooled_rhos)
            ),
            "spearman_median": (
                round(_median(pooled_rhos), 6)
                if _median(pooled_rhos) is not None
                else None
            ),
            "spearman_min": round(min(pooled_rhos), 6) if pooled_rhos else None,
            "spearman_max": round(max(pooled_rhos), 6) if pooled_rhos else None,
        },
        "by_program": sorted(program_rows, key=lambda row: row["aspect_id"]),
        "interpretation": (
            "This measures whether code-side item ordering changes when the L channel is "
            "clamped to different constants. It uses no target and is not reconstruction, "
            "isomorphism, prompt articulability, or evidence that a sentinel is semantically valid."
        ),
    }


def _validate_execution_common(
    execution: Mapping,
    audit: Mapping,
    plans_and_profiles: Sequence[tuple[Mapping, Sequence[Mapping]]],
    *,
    phase: str,
) -> dict[str, Any]:
    """Validate a completed execution without importing or rerunning a program."""

    if not isinstance(execution, Mapping) or set(execution) != _EXECUTION_TOP_FIELDS:
        raise MathLClampSummaryError(f"{phase} execution top-level shape drifted")
    if (
        execution.get("schema") != EXECUTION_SCHEMA
        or execution.get("status") != "conditional_slice_execution_complete"
        or execution.get("phase") != phase
    ):
        raise MathLClampSummaryError(f"{phase} execution is not complete")
    if execution.get("construct_fidelity_fingerprint") != _content_fingerprint(audit):
        raise MathLClampSummaryError(f"{phase} execution is bound to another audit")
    capability_runtime = _capability_sources_from_audit(audit)
    if execution.get("capability_runtime") != capability_runtime:
        raise MathLClampSummaryError(f"{phase} capability runtime drifted")
    false_fields = (
        "original_hybrid_execution",
        "pure_code_rewrite_claimed",
        "whole_construct_fidelity_claimed",
        "reference_fields_passed_to_worker",
        "outcome_fields_passed_to_worker",
        "actual_llm_extractions_passed_to_worker",
        "models_or_apis_called_by_runner",
        "credentials_inherited_by_worker",
        "accelerators_visible_to_worker",
        "worker_filesystem_isolated",
        "worker_network_isolated",
        "ops_corpus_or_retrieval_state_loaded",
    )
    if any(execution.get(field) is not False for field in false_fields):
        raise MathLClampSummaryError(f"{phase} execution crossed a forbidden boundary")
    if (
        execution.get("worker_process_isolated") is not True
        or execution.get("constant_profiles_frozen_within_each_run") is not True
        or execution.get("sentinel_grid") != list(SENTINELS)
    ):
        raise MathLClampSummaryError(f"{phase} execution policy drifted")
    expected_item_file = (
        "compiler_train.json" if phase == "compiler_train" else "sealed_heldout.json"
    )
    if Path(str(execution.get("items_path", ""))).name != expected_item_file:
        raise MathLClampSummaryError(f"{phase} execution used an unexpected item split")
    n_items = execution.get("n_items")
    if isinstance(n_items, bool) or not isinstance(n_items, int) or n_items <= 1:
        raise MathLClampSummaryError(f"{phase} execution has invalid item count")
    if phase == "compiler_train":
        if execution.get("profile_selection_source") is not None:
            raise MathLClampSummaryError("compiler train unexpectedly used a profile gate")
    elif not isinstance(execution.get("profile_selection_source"), str) or not execution[
        "profile_selection_source"
    ]:
        raise MathLClampSummaryError("heldout execution has no frozen profile-gate source")

    expected = {}
    for plan, profiles in plans_and_profiles:
        identity = (
            plan["aspect_id"],
            plan["source_path"],
            plan["program_sha256"],
            plan["selected_revision"],
            tuple(plan["llm_field_names"]),
        )
        if identity in expected:
            raise MathLClampSummaryError(f"{phase} plan contains duplicate programs")
        expected[identity] = (plan, list(profiles))
    programs = execution.get("programs")
    if not isinstance(programs, list) or len(programs) != len(expected):
        raise MathLClampSummaryError(f"{phase} execution/program count drifted")

    seen = set()
    common_item_keys: list[str] | None = None
    for program in programs:
        if not isinstance(program, Mapping) or set(program) != _PROGRAM_FIELDS:
            raise MathLClampSummaryError(f"{phase} program record shape drifted")
        identity = (
            program.get("aspect_id"),
            program.get("source_path"),
            program.get("program_sha256"),
            program.get("selected_revision"),
            tuple(program.get("llm_field_names", [])),
        )
        pair = expected.get(identity)
        if pair is None or identity in seen:
            raise MathLClampSummaryError(f"{phase} program identity drifted")
        seen.add(identity)
        plan, expected_profiles = pair
        if program.get("worker_status") != "completed":
            raise MathLClampSummaryError(f"{phase} contains an incomplete worker")
        if program.get("relations") != plan["relations"]:
            raise MathLClampSummaryError(f"{phase} relation mappings drifted")
        profiles = program.get("profiles")
        if not isinstance(profiles, list) or len(profiles) != len(expected_profiles):
            raise MathLClampSummaryError(f"{phase} profile count drifted")
        observed_identities = []
        for profile in profiles:
            if not isinstance(profile, Mapping):
                raise MathLClampSummaryError(f"{phase} profile record is invalid")
            _validate_profile_result(profile, n_items=n_items)
            observed_identities.append(
                {
                    "profile_id": profile["profile_id"],
                    "profile_index": profile["profile_index"],
                    "assignments": profile["assignments"],
                }
            )
            item_keys = [row["item_key"] for row in profile["rows"]]
            if common_item_keys is None:
                common_item_keys = item_keys
            elif item_keys != common_item_keys:
                raise MathLClampSummaryError(f"{phase} profile item identities drifted")
        require_complete = phase == "compiler_train"
        validate_profiles(
            observed_identities,
            plan["llm_field_names"],
            require_complete_grid=require_complete,
        )
        if observed_identities != expected_profiles:
            raise MathLClampSummaryError(f"{phase} sentinel profile selection drifted")
        expected_measurement_status = (
            "at_least_one_nondegenerate_profile"
            if any(
                profile["measurement_status"] == "nondegenerate_measurement"
                for profile in profiles
            )
            else "no_nondegenerate_profile"
        )
        if (
            program.get("measurement_status") != expected_measurement_status
            or program.get("summary") != _program_summary(program, n_items=n_items)
        ):
            raise MathLClampSummaryError(f"{phase} program summary drifted")
    if seen != set(expected):
        raise MathLClampSummaryError(f"{phase} execution omitted a program")

    worker_statuses = Counter(program["worker_status"] for program in programs)
    profile_statuses = Counter(
        profile["measurement_status"]
        for program in programs
        for profile in program["profiles"]
    )
    states = Counter()
    for program in programs:
        for profile in program["profiles"]:
            states.update(profile["summary"]["state_counts"])
    expected_summary = {
        "n_unique_programs": len(programs),
        "n_profile_runs": sum(len(program["profiles"]) for program in programs),
        "n_relation_mappings": sum(len(plan["relations"]) for plan, _ in expected.values()),
        "worker_status_counts": dict(sorted(worker_statuses.items())),
        "profile_measurement_status_counts": dict(sorted(profile_statuses.items())),
        "three_state_totals": {
            state: states[state] for state in ("measured", "abstained", "failed")
        },
    }
    if execution.get("summary") != expected_summary:
        raise MathLClampSummaryError(f"{phase} aggregate summary drifted")
    return expected_summary


def _depth_scope(rows: Sequence[Mapping], depth: str) -> dict[str, Any]:
    projected = [
        {
            **row,
            **{
                stage: bool(row[stage] and str(row["audited_depth"]) == depth)
                for stage in STAGES
            },
        }
        for row in rows
    ]
    result = _scope(projected)
    result["interpretation"] = (
        "Rates use the full enclosing panel/inventory denominator; this depth contributes "
        "only positive relation mappings whose deepest audited decision-contributing "
        "operation equals this value."
    )
    return result


def build_math_lclamp_operational_summary(
    panel: Mapping,
    audit: Mapping,
    train_execution: Mapping,
    train_gate: Mapping,
    heldout_execution: Mapping,
    *,
    sources: Mapping | None = None,
) -> dict[str, Any]:
    """Build a descriptive static -> train -> heldout constant-L funnel."""

    try:
        validate_merged_audit(audit)
        plans = build_execution_plan(audit)
    except ValueError as exc:
        raise MathLClampSummaryError(str(exc)) from exc
    if panel.get("schema") != "tacit_breadth_metric_panel/v1":
        raise MathLClampSummaryError("unexpected hierarchy panel schema")
    if audit.get("panel_content_sha256") != panel.get("panel_content_sha256"):
        raise MathLClampSummaryError("construct audit is bound to another panel")
    cells = {
        str(cell["id"]): cell
        for cell in panel.get("cells", [])
        if cell.get("task") == TASK
    }
    audit_rows = {str(row["cell_id"]): row for row in audit["rows"]}
    if len(cells) != 90 or len(audit_rows) != 90 or set(cells) != set(audit_rows):
        raise MathLClampSummaryError("panel and construct audit do not close at 90 cells")
    try:
        sampling_frame = _validate_sampling_frame(panel, cells)
    except ValueError as exc:
        raise MathLClampSummaryError(str(exc)) from exc

    train_plans = [
        (plan, build_sentinel_profiles(plan["llm_field_names"])) for plan in plans
    ]
    train_summary = _validate_execution_common(
        train_execution, audit, train_plans, phase="compiler_train"
    )
    thresholds = train_gate.get("thresholds")
    if thresholds != CANONICAL_GATE_THRESHOLDS:
        raise MathLClampSummaryError("train gate thresholds differ from the frozen policy")
    try:
        rebuilt_gate = build_train_profile_gate(
            train_execution,
            audit,
            min_measured=thresholds.get("min_measured"),
            min_coverage=thresholds.get("min_coverage"),
            min_unique_scores=thresholds.get("min_unique_scores"),
            max_failed=thresholds.get("max_failed"),
            execution_source=train_gate.get("training_execution_source"),
            audit_source=train_gate.get("construct_fidelity_source"),
        )
    except (TypeError, ValueError) as exc:
        raise MathLClampSummaryError(str(exc)) from exc
    if rebuilt_gate != train_gate:
        raise MathLClampSummaryError("train gate is not the deterministic train-only rebuild")
    capability_runtime = _capability_sources_from_audit(audit)
    try:
        heldout_plans = apply_profile_selection(
            plans,
            train_gate,
            audit_fingerprint=_content_fingerprint(audit),
            capability_runtime=capability_runtime,
        )
    except ValueError as exc:
        raise MathLClampSummaryError(str(exc)) from exc
    heldout_summary = _validate_execution_common(
        heldout_execution,
        audit,
        heldout_plans,
        phase="heldout_pre_reference",
    )

    train_operational_cells = {
        cell_id
        for selected in train_gate["selected_program_profiles"]
        for cell_id in selected["cell_ids"]
    }
    heldout_programs = {program["aspect_id"]: program for program in heldout_execution["programs"]}
    heldout_measurable_cells: set[str] = set()
    for plan, _profiles in heldout_plans:
        program = heldout_programs[plan["aspect_id"]]
        profile = program["profiles"][0]
        summary = profile["summary"]
        measurable = bool(
            profile["measurement_status"] == "nondegenerate_measurement"
            and summary["state_counts"]["failed"] == 0
            and summary["n_measured"] >= 2
            and summary["n_unique_scores"] >= 2
        )
        if measurable:
            heldout_measurable_cells.update(
                relation["cell_id"] for relation in plan["relations"]
            )
    static_cells = {
        row["cell_id"]
        for row in audit["rows"]
        if row["eligible_for_relation_local_execution"]
    }
    if not heldout_measurable_cells <= train_operational_cells <= static_cells:
        raise MathLClampSummaryError("operational stage relation mappings are not nested")

    joined = []
    for cell_id, cell in cells.items():
        row = audit_rows[cell_id]
        for audit_field, cell_field in (
            ("task", "task"),
            ("level", "level"),
            ("metric_name", "construct"),
            ("metric_description", "description"),
        ):
            if row.get(audit_field) != cell.get(cell_field):
                raise MathLClampSummaryError(f"{cell_id}: panel/audit metadata drifted")
        joined.append(
            {
                "cell_id": cell_id,
                "level": cell["level"],
                "source_kind": cell["source_kind"],
                "breadth_stratum": cell["breadth_stratum"],
                "design_weight": cell["design_weight"],
                "audited_depth": row["audited_depth"],
                STAGES[0]: cell_id in static_cells,
                STAGES[1]: cell_id in train_operational_cells,
                STAGES[2]: cell_id in heldout_measurable_cells,
            }
        )

    by_level = {
        level: _scope([row for row in joined if row["level"] == level])
        for level in LEVELS
    }
    by_depth = {depth: _depth_scope(joined, depth) for depth in DEPTHS}
    by_level_and_depth = {
        level: {
            depth: _depth_scope(
                [row for row in joined if row["level"] == level], depth
            )
            for depth in DEPTHS
        }
        for level in LEVELS
    }
    return {
        "schema": SCHEMA,
        "status": "complete_static_train_and_pre_reference_heldout_funnel",
        "task": TASK,
        "sources": dict(sources or {}),
        "panel_content_sha256": panel["panel_content_sha256"],
        "scientific_object": {
            "executable_object": "constant-L conditional slices g_c(x)=f(x,c)",
            "relation_unit": "one audited relation-local mapping from a hierarchy cell to a historical hybrid",
            "train_operational": (
                "at least one profile selected solely by train coverage, nonconstancy, and failure thresholds"
            ),
            "heldout_measurable": (
                "the exact train-selected profile is failure-free and has at least two distinct heldout scores"
            ),
        },
        "channel_contract": {
            "program_execution_outputs_read": True,
            "item_text_loaded": False,
            "reference_values_loaded": False,
            "outcome_labels_loaded": False,
            "prompt_or_llm_values_loaded": False,
            "models_or_apis_called": False,
            "accelerators_used": False,
            "score_direction_or_target_used_for_selection": False,
        },
        "validation": {
            "construct_fidelity": {
                "n_cells": len(audit_rows),
                "n_static_relation_mappings": len(static_cells),
                "n_unique_programs": len(plans),
                "cross_audit": audit["cross_audit"],
            },
            "compiler_train": train_summary,
            "train_only_profile_gate": train_gate["summary"],
            "heldout_pre_reference": heldout_summary,
            "stage_relation_mapping_counts": {
                STAGES[0]: len(static_cells),
                STAGES[1]: len(train_operational_cells),
                STAGES[2]: len(heldout_measurable_cells),
            },
        },
        "sampling_frame": sampling_frame,
        "stage_definitions": {
            STAGES[0]: "static subrelation fidelity after independent cross-audit",
            STAGES[1]: "static mapping whose program passed the frozen train-only profile gate",
            STAGES[2]: "train-operational mapping whose frozen profile remained nonconstant and failure-free heldout",
        },
        "estimands": {
            "balanced_panel": "unweighted descriptive rate over the balanced 90-cell panel",
            EXPANSION_KEY: (
                "conditional stratum-expansion point estimate over 1,185 eligible native action-node records"
            ),
            "sampling_uncertainty": "not estimated; these are descriptive point estimates",
        },
        "pooled_eligible_action_nodes": _scope(joined),
        "by_level": by_level,
        "by_audited_depth": by_depth,
        "by_level_and_audited_depth": by_level_and_depth,
        "unsupervised_sentinel_sensitivity": _sentinel_sensitivity(train_execution),
        "uncertainty_intervals_emitted": False,
        "claim_limits": [
            "Operational and heldout-measurable refer only to constant-L conditional slices.",
            "The sentinel values are controls, not reconstructed prompt judgments.",
            "This is not execution of the original hybrid and not a pure-code rewrite.",
            "No whole-construct verifiability, prompt articulability, reconstruction, isomorphism, or codability is estimated.",
            "No references, outcomes, score targets, model calls, or accelerators were used.",
            "Rates describe relation mappings available in a retrospective manual historical library.",
            "The conditional expansion covers eligible action-node records, not unique constructs or raw rubrics.",
            "R1/R2/R3 differences are descriptive and do not establish a hierarchy-round trend.",
        ],
    }


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--train-execution", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--heldout-execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    paths = {
        "panel": args.panel,
        "construct_fidelity": args.audit,
        "compiler_train_execution": args.train_execution,
        "train_profile_gate": args.train_gate,
        "heldout_pre_reference_execution": args.heldout_execution,
    }
    payload = build_math_lclamp_operational_summary(
        _load(args.panel),
        _load(args.audit),
        _load(args.train_execution),
        _load(args.train_gate),
        _load(args.heldout_execution),
        sources={name: _source_record(path) for name, path in paths.items()},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
