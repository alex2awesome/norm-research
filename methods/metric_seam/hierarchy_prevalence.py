"""Estimate code-review hierarchy witness prevalence without calling it codability.

The balanced 30-per-level panel and the eligible native action-node inventory answer
different questions. This module reports both, keeps progressively stronger code
outcomes separate, and treats the deterministic stratum expansion as conditional on
hash-as-random within-stratum exchangeability. Its one-way block perturbations are
diagnostics, not confidence intervals or a complete dependence analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from methods.metric_seam.hierarchy_code_runner import TRAIN_GATE_SCHEMA
from methods.metric_seam.hierarchy_fidelity_merge import SCHEMA as FIDELITY_SCHEMA
from methods.metric_seam.hierarchy_heldout_readiness import SCHEMA as HELDOUT_SCHEMA
from methods.metric_seam.hierarchy_panel_compat import validate_hierarchy_panel


SCHEMA = "metric-seam.hierarchy-witness-prevalence.v2"
OUTCOMES = (
    "retrieved_candidate",
    "relation_local_static_fidelity",
    "train_operational_relation_witness",
    "heldout_confirmatory_reconstruction_evaluable",
)
EXPANSION_KEY = "eligible_inventory_stratum_expansion"


class PrevalenceError(ValueError):
    """Raised when source artifacts cannot be joined without ambiguity."""


def _rate(rows: Sequence[Mapping], outcome: str, *, weighted: bool) -> dict:
    if not rows:
        return {"n_sampled_nodes": 0, "estimated_population_nodes": 0.0, "rate": None}
    weights = [float(row["design_weight"]) if weighted else 1.0 for row in rows]
    denominator = sum(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        raise PrevalenceError("prevalence denominator must be finite and positive")
    numerator = sum(weight * bool(row[outcome]) for weight, row in zip(weights, rows))
    return {
        "n_sampled_nodes": len(rows),
        "estimated_population_nodes": round(denominator, 6),
        "estimated_positive_nodes": round(numerator, 6),
        "rate": round(numerator / denominator, 6),
    }


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _one_way_observed_block_perturbation(
    rows: Sequence[Mapping], outcome: str, block_field: str, *,
    n_resamples: int, seed_material: str,
) -> dict:
    """Perturb observed one-way blocks; deliberately make no CI claim."""
    blocks: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        blocks[str(row[block_field])].append(row)
    keys = sorted(blocks)
    if not keys:
        raise PrevalenceError("cannot perturb an empty block set")
    rng = random.Random(int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16))
    rates = []
    for _ in range(n_resamples):
        sampled_keys = [rng.choice(keys) for _ in keys]
        sampled = [row for key in sampled_keys for row in blocks[key]]
        rates.append(_rate(sampled, outcome, weighted=True)["rate"])
    return {
        "block_field": block_field,
        "n_observed_blocks": len(keys),
        "n_resamples": n_resamples,
        "p025": round(_percentile(rates, 0.025), 6),
        "median": round(_percentile(rates, 0.5), 6),
        "p975": round(_percentile(rates, 0.975), 6),
        "method": "uniform pairs perturbation of observed one-way blocks",
        "interpretation": (
            f"{block_field.removesuffix('_component_id').replace('_', ' ')} one-way "
            "observed-block perturbation range; it does not preserve the stratified sampling "
            "design, include unobserved component members, combine dependence partitions, or "
            "form an iid/design-based confidence interval"
        ),
    }


def _scope(rows: Sequence[Mapping], *, n_resamples: int, label: str) -> dict:
    return {
        "n_sampled_nodes": len(rows),
        "n_dependency_components_sampled": len({row["dependency_component_id"] for row in rows}),
        "n_level_local_provenance_components_sampled": len({
            row["provenance_component_id"] for row in rows
        }),
        "balanced_panel": {
            outcome: _rate(rows, outcome, weighted=False) for outcome in OUTCOMES
        },
        EXPANSION_KEY: {
            outcome: {
                **_rate(rows, outcome, weighted=True),
                "dependency_one_way_observed_block_perturbation": (
                    _one_way_observed_block_perturbation(
                        rows, outcome, "dependency_component_id", n_resamples=n_resamples,
                        seed_material=f"{label}|{outcome}|dependency",
                    )
                ),
                "level_local_provenance_one_way_observed_block_perturbation": (
                    _one_way_observed_block_perturbation(
                        rows, outcome, "provenance_component_id", n_resamples=n_resamples,
                        seed_material=f"{label}|{outcome}|provenance",
                    )
                ),
            }
            for outcome in OUTCOMES
        },
    }


def _leaf_support_ids(leaves: Sequence[Mapping]) -> set[str]:
    """Recover the frozen raw-support identity without exposing leaf text downstream."""
    result: set[str] = set()
    for leaf in leaves:
        if not isinstance(leaf, Mapping):
            continue
        for field in ("key", "r2_cluster_id", "cluster_id"):
            if leaf.get(field) is not None:
                result.add(f"{field}:{leaf[field]}")
                break
        else:
            payload = {
                "name": str(leaf.get("name") or leaf.get("medoid_name") or "").strip(),
                "description": str(
                    leaf.get("description") or leaf.get("medoid_description") or ""
                ).strip(),
            }
            if payload["name"]:
                digest = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()[:24]
                result.add(f"content:{digest}")
    return result


def _component_partition(
    rows: Sequence[Mapping], *, label: str,
    feature_fn: Callable[[Mapping], set[str]],
) -> tuple[dict[str, str], dict]:
    """Return transitive connected components induced by shared named features."""
    ids = [str(row["cell_id"]) for row in rows]
    parent = {cell_id: cell_id for cell_id in ids}

    def root(cell_id: str) -> str:
        while parent[cell_id] != cell_id:
            parent[cell_id] = parent[parent[cell_id]]
            cell_id = parent[cell_id]
        return cell_id

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_owner: dict[str, str] = {}
    for row in rows:
        cell_id = str(row["cell_id"])
        for feature in sorted(feature_fn(row)):
            if feature in first_owner:
                union(cell_id, first_owner[feature])
            else:
                first_owner[feature] = cell_id

    members: dict[str, list[str]] = defaultdict(list)
    for cell_id in ids:
        members[root(cell_id)].append(cell_id)
    component_ids: dict[str, str] = {}
    levels_by_id = {str(row["cell_id"]): str(row["level"]) for row in rows}
    sizes = []
    cross_level_sizes = []
    for component_members in members.values():
        component_members.sort()
        digest = hashlib.sha256(
            json.dumps(component_members, ensure_ascii=False).encode()
        ).hexdigest()[:20]
        component_id = f"code-review::{label}::{digest}"
        for cell_id in component_members:
            component_ids[cell_id] = component_id
        sizes.append(len(component_members))
        if len({levels_by_id[cell_id] for cell_id in component_members}) > 1:
            cross_level_sizes.append(len(component_members))
    size_counts = Counter(sizes)
    return component_ids, {
        "n_components": len(sizes),
        "largest_component": max(sizes, default=0),
        "n_singleton_components": size_counts.get(1, 0),
        "component_size_counts": {
            str(size): count for size, count in sorted(size_counts.items())
        },
        "n_cross_level_components": len(cross_level_sizes),
        "largest_cross_level_component": max(cross_level_sizes, default=0),
    }


def _dependence_diagnostics(rows: Sequence[Mapping]) -> dict:
    raw_assignments, raw_summary = _component_partition(
        rows,
        label="cross-level-raw-support",
        feature_fn=lambda row: {f"raw:{value}" for value in row["raw_support_ids"]},
    )
    program_assignments, program_summary = _component_partition(
        rows,
        label="shared-candidate-program",
        feature_fn=lambda row: (
            {f"program:{row['candidate_source_path']}"}
            if row["candidate_source_path"] is not None else set()
        ),
    )
    joint_assignments, joint_summary = _component_partition(
        rows,
        label="dependency-raw-program-union",
        feature_fn=lambda row: (
            {f"dependency:{row['dependency_component_id']}"}
            | {f"raw:{value}" for value in row["raw_support_ids"]}
            | (
                {f"program:{row['candidate_source_path']}"}
                if row["candidate_source_path"] is not None else set()
            )
        ),
    )
    program_reuse = {}
    for outcome in OUTCOMES:
        positive = [row for row in rows if row[outcome]]
        programs = {
            row["candidate_source_path"] for row in positive
            if row["candidate_source_path"] is not None
        }
        program_reuse[outcome] = {
            "n_positive_mappings": len(positive),
            "n_unique_candidate_programs": len(programs),
        }
    return {
        "status": "component_structure_only_no_interval",
        "cross_level_raw_support": {
            **raw_summary,
            "interpretation": (
                "connects sampled cells sharing raw-support identity across R1/R2/R3; unlike "
                "the frozen provenance ID, it is not reset at each hierarchy level"
            ),
        },
        "shared_candidate_program": {
            **program_summary,
            "interpretation": (
                "connects mappings produced by the same historical code program; this is an "
                "instrument dependence diagnostic, not construct identity"
            ),
        },
        "joint_dependency_raw_program_union": {
            **joint_summary,
            "interpretation": (
                "transitive union of level-local immediate dependency, cross-level raw support, "
                "and shared program edges; deliberately reported as component structure only"
            ),
        },
        "program_reuse_by_outcome": program_reuse,
        "cell_assignments": [
            {
                "cell_id": row["cell_id"],
                "cross_level_raw_support_component_id": raw_assignments[row["cell_id"]],
                "shared_candidate_program_component_id": program_assignments[row["cell_id"]],
                "joint_dependency_raw_program_union_component_id": joint_assignments[
                    row["cell_id"]
                ],
            }
            for row in rows
        ],
        "not_an_interval": True,
    }


def _program_ownership(
    programs: Sequence[Mapping], audits: Mapping[str, Mapping], *, label: str,
) -> tuple[set[str], dict[tuple[str, str, str], set[str]]]:
    ownership: dict[str, tuple[str, str, str]] = {}
    by_program: dict[tuple[str, str, str], set[str]] = {}
    for program in programs:
        try:
            identity = (
                str(program["aspect_id"]),
                str(program["source_path"]),
                str(program["source_sha256"]),
            )
            program_cells = [str(cell_id) for cell_id in program["cell_ids"]]
        except (KeyError, TypeError) as error:
            raise PrevalenceError(f"{label} has an incomplete program record") from error
        if not program_cells or len(program_cells) != len(set(program_cells)):
            raise PrevalenceError(f"{label} program has missing or duplicate cell IDs")
        if identity in by_program:
            raise PrevalenceError(f"{label} repeats one candidate program")
        by_program[identity] = set(program_cells)
        for cell_id in program_cells:
            if cell_id in ownership:
                raise PrevalenceError(f"{label} assigns {cell_id} to multiple programs")
            if cell_id not in audits:
                raise PrevalenceError(f"{label} contains unknown cell {cell_id}")
            candidate = audits[cell_id].get("candidate")
            expected = (
                str(candidate.get("aspect_id")),
                str(candidate.get("source_path")),
                str(candidate.get("source_sha256")),
            ) if isinstance(candidate, Mapping) else None
            if expected != identity:
                raise PrevalenceError(
                    f"{label} program identity does not match audit candidate for {cell_id}"
                )
            ownership[cell_id] = identity
    return set(ownership), by_program


def _validate_summary_counts(
    cells: Mapping[str, Mapping], audits: Mapping[str, Mapping],
    train_gate: Mapping, heldout_readiness: Mapping,
    operational: set[str], confirmatory: set[str],
    train_programs: Mapping[tuple[str, str, str], set[str]],
    heldout_programs: Mapping[tuple[str, str, str], set[str]],
) -> None:
    eligible = {
        cell_id for cell_id, row in audits.items()
        if row.get("eligible_for_relation_local_execution") is True
    }
    train_summary = train_gate.get("summary", {})
    heldout_summary = heldout_readiness.get("summary", {})
    expected_train = {
        "n_candidate_programs": len({
            (
                str(audits[cell_id]["candidate"]["aspect_id"]),
                str(audits[cell_id]["candidate"]["source_path"]),
                str(audits[cell_id]["candidate"]["source_sha256"]),
            )
            for cell_id in eligible
        }),
        "n_selected_programs": len(train_programs),
        "n_static_relation_mappings": len(eligible),
        "n_selected_relation_mappings": len(operational),
    }
    expected_heldout = {
        "n_train_selected_programs": len(train_programs),
        "n_confirmatory_programs": len(heldout_programs),
        "n_confirmatory_relation_mappings": len(confirmatory),
    }
    for field, expected in expected_train.items():
        if train_summary.get(field) != expected:
            raise PrevalenceError(f"train-gate summary drifted at {field}")
    for field, expected in expected_heldout.items():
        if heldout_summary.get(field) != expected:
            raise PrevalenceError(f"heldout-readiness summary drifted at {field}")
    for label, selected, summary, field in (
        ("train", operational, train_summary, "selected_relation_mappings_by_level"),
        ("heldout", confirmatory, heldout_summary, "confirmatory_relation_mappings_by_level"),
    ):
        observed = Counter(str(cells[cell_id]["level"]) for cell_id in selected)
        if summary.get(field) != {level: observed[level] for level in ("R1", "R2", "R3")}:
            raise PrevalenceError(f"{label} level summary drifted")
    for label, selected, summary, field in (
        ("train", operational, train_summary, "selected_relation_mappings_by_depth"),
        ("heldout", confirmatory, heldout_summary, "confirmatory_relation_mappings_by_depth"),
    ):
        observed = Counter(str(audits[cell_id]["audited_depth"]) for cell_id in selected)
        if summary.get(field) != {depth: observed[depth] for depth in ("1", "2")}:
            raise PrevalenceError(f"{label} depth summary drifted")


def _validate_artifact_bindings(
    train_gate: Mapping, heldout_readiness: Mapping,
    train_programs: Mapping[tuple[str, str, str], set[str]],
    heldout_programs: Mapping[tuple[str, str, str], set[str]],
    sources: Mapping[str, str] | None,
) -> None:
    if train_gate.get("status") != "frozen_before_heldout_program_execution":
        raise PrevalenceError("train gate is not frozen before heldout execution")
    if heldout_readiness.get("status") != "frozen_before_prompt_reference_scoring":
        raise PrevalenceError("heldout readiness is not frozen before prompt scoring")
    if train_gate.get("reference_values_used") is not False:
        raise PrevalenceError("train gate reports use of reference values")
    if heldout_readiness.get("prompt_outputs_used") is not False:
        raise PrevalenceError("heldout readiness reports use of prompt outputs")
    if Path(str(train_gate.get("training_execution_source", ""))).name != (
        "code_review_train_execution_v2.json"
    ):
        raise PrevalenceError("train gate is not bound to the canonical parser replay")
    if Path(str(train_gate.get("construct_fidelity_source", ""))).name != (
        "code_review_construct_fidelity_v1.json"
    ):
        raise PrevalenceError("train gate is not bound to its frozen construct-fidelity audit")
    if Path(str(heldout_readiness.get("heldout_execution_source", ""))).name != (
        "code_review_heldout_execution_v1.json"
    ):
        raise PrevalenceError("heldout readiness is not bound to canonical execution")
    declared_gate = Path(str(heldout_readiness.get("compiler_train_gate_source", ""))).name
    expected_gate = Path(str((sources or {}).get(
        "train_gate", "code_review_train_gate_v1.json"
    ))).name
    if declared_gate != expected_gate:
        raise PrevalenceError("heldout readiness is bound to another train gate")
    if not set(heldout_programs) <= set(train_programs):
        raise PrevalenceError("heldout readiness contains a program not selected on train")
    for identity, cell_ids in heldout_programs.items():
        if not cell_ids <= train_programs[identity]:
            raise PrevalenceError("heldout program cell scope exceeds its train-selected scope")


def _validate_sampling_design(panel: Mapping, cells: Mapping[str, Mapping]) -> dict:
    inventory_rows = [
        row for row in panel.get("inventory", []) if row.get("task") == "code-review"
    ]
    if {row.get("level") for row in inventory_rows} != {"R1", "R2", "R3"}:
        raise PrevalenceError("code-review inventory must contain exactly R1/R2/R3")
    inventory_by_level = {str(row["level"]): row for row in inventory_rows}
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
            raise PrevalenceError(f"invalid stratum metadata for {key}") from error
        if len(populations) != 1 or len(selected) != 1:
            raise PrevalenceError(f"inconsistent population/sample counts in stratum {key}")
        population_n, selected_n = next(iter(populations)), next(iter(selected))
        if selected_n != len(rows) or not 0 < selected_n <= population_n:
            raise PrevalenceError(f"invalid selected count in stratum {key}")
        expected_probability = selected_n / population_n
        expected_weight = population_n / selected_n
        if (
            len(probabilities) != 1 or len(weights) != 1
            or not math.isclose(next(iter(probabilities)), expected_probability, abs_tol=1e-12)
            or not math.isclose(next(iter(weights)), expected_weight, abs_tol=1e-12)
        ):
            raise PrevalenceError(f"inclusion fraction/design weight drifted in stratum {key}")
    by_level_population = Counter()
    for (level, _kind, _breadth), rows in strata.items():
        by_level_population[level] += int(rows[0]["stratum_population_n"])
    eligible_by_level = {
        level: int(row["n_eligible_nodes"]) for level, row in inventory_by_level.items()
    }
    complete_by_level = {
        level: int(row["n_complete_nodes"]) for level, row in inventory_by_level.items()
    }
    if dict(by_level_population) != eligible_by_level:
        raise PrevalenceError("stratum populations do not sum to the eligible inventory")
    weighted_total = sum(float(cell["design_weight"]) for cell in cells.values())
    eligible_total = sum(eligible_by_level.values())
    if not math.isclose(weighted_total, eligible_total, abs_tol=1e-9):
        raise PrevalenceError("design weights do not expand to the eligible inventory")
    return {
        "n_complete_action_node_records": sum(complete_by_level.values()),
        "n_eligible_action_node_records": eligible_total,
        "n_excluded_by_frozen_eligibility_rule": (
            sum(complete_by_level.values()) - eligible_total
        ),
        "complete_by_level": complete_by_level,
        "eligible_by_level": eligible_by_level,
        "n_sampling_strata": len(strata),
        "selected_per_stratum": sorted({len(rows) for rows in strata.values()}),
        "eligibility_rule": "nonempty name, at least 8 description words, and at least 1 child",
    }


def build_prevalence(
    panel: Mapping, audit: Mapping, train_gate: Mapping,
    heldout_readiness: Mapping, *, n_resamples: int = 2000,
    sources: Mapping[str, str] | None = None,
) -> dict:
    errors = validate_hierarchy_panel(panel)
    if errors:
        raise PrevalenceError(f"invalid hierarchy panel: {errors}")
    if audit.get("schema") != FIDELITY_SCHEMA or len(audit.get("rows", [])) != 90:
        raise PrevalenceError("expected canonical 90-row construct-fidelity audit")
    if audit.get("panel_content_sha256") != panel.get("panel_content_sha256"):
        raise PrevalenceError("construct-fidelity audit is bound to another panel")
    if train_gate.get("schema") != TRAIN_GATE_SCHEMA:
        raise PrevalenceError("expected frozen training gate")
    if heldout_readiness.get("schema") != HELDOUT_SCHEMA:
        raise PrevalenceError("expected heldout readiness")
    if n_resamples < 100:
        raise PrevalenceError("at least 100 block perturbations are required")

    cells = {
        str(cell["id"]): cell for cell in panel["cells"] if cell["task"] == "code-review"
    }
    audits = {str(row["cell_id"]): row for row in audit["rows"]}
    if len(cells) != 90 or set(audits) != set(cells):
        raise PrevalenceError("panel/audit code-review cell identities do not match exactly")
    for cell_id, row in audits.items():
        if not isinstance(row.get("eligible_for_relation_local_execution"), bool):
            raise PrevalenceError(f"audit eligibility is not boolean for {cell_id}")

    operational, train_programs = _program_ownership(
        train_gate.get("selected_programs", []), audits, label="train gate"
    )
    confirmatory, heldout_programs = _program_ownership(
        heldout_readiness.get("confirmatory_programs", []), audits,
        label="heldout readiness",
    )
    eligible = {
        cell_id for cell_id, row in audits.items()
        if row["eligible_for_relation_local_execution"]
    }
    retrieved = {
        cell_id for cell_id, row in audits.items() if row.get("candidate") is not None
    }
    if not confirmatory <= operational <= eligible <= retrieved:
        raise PrevalenceError(
            "outcome funnel is not nested: confirmatory <= operational <= static <= retrieved"
        )
    _validate_artifact_bindings(
        train_gate, heldout_readiness, train_programs, heldout_programs, sources
    )
    _validate_summary_counts(
        cells, audits, train_gate, heldout_readiness, operational, confirmatory,
        train_programs, heldout_programs,
    )
    sampling_frame = _validate_sampling_design(panel, cells)

    joined = []
    for cell_id, cell in cells.items():
        audit_row = audits[cell_id]
        raw_support_ids = _leaf_support_ids(cell.get("children", []))
        if not raw_support_ids:
            raise PrevalenceError(f"{cell_id} has no recoverable raw-support identity")
        candidate = audit_row.get("candidate")
        joined.append({
            "cell_id": cell_id,
            "level": cell["level"],
            "source_kind": cell["source_kind"],
            "breadth_stratum": cell["breadth_stratum"],
            "design_weight": cell["design_weight"],
            "dependency_component_id": cell["dependency_component_id"],
            "provenance_component_id": cell["provenance_component_id"],
            "raw_support_ids": raw_support_ids,
            "candidate_source_path": (
                str(candidate["source_path"]) if isinstance(candidate, Mapping) else None
            ),
            "retrieved_candidate": cell_id in retrieved,
            "relation_local_static_fidelity": cell_id in eligible,
            "train_operational_relation_witness": cell_id in operational,
            "heldout_confirmatory_reconstruction_evaluable": cell_id in confirmatory,
        })

    by_level = {
        level: _scope(
            [row for row in joined if row["level"] == level],
            n_resamples=n_resamples, label=f"code-review|{level}",
        )
        for level in ("R1", "R2", "R3")
    }
    source_kind = {}
    for level in ("R1", "R2", "R3"):
        level_rows = [row for row in joined if row["level"] == level]
        source_kind[level] = {
            kind: _scope(
                [row for row in level_rows if row["source_kind"] == kind],
                n_resamples=n_resamples, label=f"code-review|{level}|{kind}",
            )
            for kind in sorted({row["source_kind"] for row in level_rows})
        }
    merged_rows = [
        row for row in joined if row["source_kind"] in {"merged_tree", "merged_group"}
    ]
    return {
        "schema": SCHEMA,
        "status": "pre_reconstruction_code_witness_prevalence",
        "task": "code-review",
        "panel_content_sha256": panel["panel_content_sha256"],
        "sources": dict(sources or {}),
        "supersedes": {
            "artifact": (
                "outputs/metric_seam_pilot/hierarchy_r123/"
                "code_review_witness_prevalence_v2.json"
            ),
            "scope": "estimand labels, validation, and dependence diagnostics",
            "point_estimates_changed": False,
            "disposition": "v2 point arithmetic retained; v2 labels and sensitivity scope superseded",
        },
        "sampling_frame": sampling_frame,
        "outcome_definitions": {
            "retrieved_candidate": "static candidate retrieval; not construct fidelity",
            "relation_local_static_fidelity": (
                "audited match to at least one construct subrelation; never whole-metric codability"
            ),
            "train_operational_relation_witness": (
                "relation-local candidate passed the frozen train-only measurement gate"
            ),
            "heldout_confirmatory_reconstruction_evaluable": (
                "preselected program has at least 30 heldout code scores; no prompt result yet"
            ),
        },
        "estimands": {
            "balanced_panel": "unweighted rate in the balanced 30-node-per-level sample",
            EXPANSION_KEY: (
                "stratum-expansion point estimate over 1,128 eligible native action-node "
                "records, conditional on treating deterministic outcome-blind SHA rank as "
                "pseudo-random/exchangeable within source-kind x breadth x level strata"
            ),
            "pooled_scope": (
                "eligible-record-population-weighted pooling of 856 R1, 217 R2, and 55 R3 "
                "nodes; not an equal-level average and not the full 1,132-record inventory"
            ),
            "sampling_uncertainty": (
                "not estimated: the frozen salt is deterministic, and no design-respecting "
                "replicate weights or audited alternate samples exist"
            ),
        },
        "whole_construct_exact": {"n": 0, "denominator": 90, "rate": 0.0},
        "pooled_eligible_action_nodes": _scope(
            joined, n_resamples=n_resamples, label="code-review|pooled"
        ),
        "by_level": by_level,
        "sensitivities": {
            "source_kind_specific": source_kind,
            "merged_only": _scope(
                merged_rows, n_resamples=n_resamples, label="code-review|merged-only"
            ),
            "dependence_component_diagnostics": _dependence_diagnostics(joined),
            "tightest_first_terminal_frontier": {
                "status": "not_yet_measured",
                "reason": (
                    "the frontier is a distinct construct population and has not received an "
                    "outcome-blind 30-node audit; it cannot be inferred from the native-node sample"
                ),
            },
        },
        "outstanding_sensitivities": [
            "design-respecting sampling uncertainty under an explicit randomized selection design",
            "joint or multiway dependence sensitivity across hierarchy, raw support, and program",
            "outcome-blind tightest-first terminal-frontier audit",
        ],
        "claim_limits": [
            "No prompt references have been scored, so prompt articulability is unestimated.",
            "No code-prompt reconstruction statistic or isomorphism adjudication exists yet.",
            "The 21/90 heldout rate is evaluability among sampled constructs, not codability.",
            "All executable matches are partial subrelations; whole-construct code fidelity is 0/90.",
            "Expansion estimates cover eligible action-node records, not unique constructs or raw rubrics.",
            "One-way observed-block perturbations are not confidence intervals or complete dependence sensitivity.",
            "R1/R2/R3 point differences do not establish a hierarchy-round or abstraction trend.",
        ],
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--heldout-readiness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    paths = {
        "panel": str(args.panel),
        "construct_fidelity": str(args.audit),
        "train_gate": str(args.train_gate),
        "heldout_readiness": str(args.heldout_readiness),
    }
    payload = build_prevalence(
        _load(args.panel), _load(args.audit), _load(args.train_gate),
        _load(args.heldout_readiness), n_resamples=args.n_resamples, sources=paths,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    headline = payload["pooled_eligible_action_nodes"][EXPANSION_KEY]
    print(json.dumps({outcome: headline[outcome]["rate"] for outcome in OUTCOMES}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
