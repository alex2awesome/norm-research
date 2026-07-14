#!/usr/bin/env python
"""Run direct 3B-articulation -> 8B-name policy-isomorphism analysis on isolated shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import (
    sha256_bytes,
    sha256_file,
)
from methods.codability.experiments.common_target_ladder import (
    policy_cell_identity,
    require_same_policy_cell_identity,
    uses_breadth_cell_identity,
    validate_policy_cell_panel,
)
from methods.codability.experiments.policy_data import (
    PUBLIC_DEVELOPMENT_PARTITIONS,
    _align_orbit,
    _average_repetitions,
    _orbits,
    _resolve_declared_path,
    authorize_policy_partition,
    load_partition_source_groups,
    load_public_index,
    require_partition,
    selection_required_for_phase,
    validate_additional_artifacts,
    validate_frozen_environment,
    validate_frozen_implementation,
    validate_lockbox_release,
    validate_executor_prompt_arms,
    validate_policy_articulation_selection_provenance,
)
from methods.codability.experiments.score_fresh_name_arms import (
    load_lockbox_selection,
    selection_scope_for_manifest,
)
from methods.codability.experiments.policy_isomorphism import (
    PolicyBootstrapContext,
    articulation_distance,
    certify_pairwise_policy_fidelity,
    certify_policy_isomorphism,
    certify_scale_step_substitution,
    compare_articulation_to_matched_control,
    oracle_mean_shift_diagnostic,
    summarize_isomorphism_fiber,
)


DEFAULT_FUNCTIONAL_FLOOR_PROFILE = (0.60, 0.65, 0.70, 0.75, 0.80)
CONTROL_PROVENANCES = frozenset({"wrong_construct_control", "inert_length_control"})
REQUIRED_SPECIFICITY_CONTROL_PROVENANCES = (
    "inert_length_control",
    "wrong_construct_control",
)
SOURCE_HIERARCHY_PROVENANCES = frozenset({
    "source_hierarchy_definition",
    "source_hierarchy_immediate_children",
    "source_hierarchy_children",
    "source_hierarchy_leaf_signals",
    "source_definition_plus_children",
    "source_definition_children_and_leaf_signals",
    "source_address_prefix",
    "source_address_prefix_full",
})
BREADTH_DESIGN_FIELDS = (
    "breadth_stratum",
    "stratum_population_n",
    "stratum_selected_n",
    "stratum_coverage_fraction",
    "nominal_poststratification_weight",
    "dependency_component_id",
    "dependency_component_size",
    "dependency_degree",
    "source_assignment_multiplicity_max",
    "provenance_component_id",
    "provenance_component_size",
    "provenance_overlap_degree",
    "provenance_assignment_multiplicity_max",
    "task_raw_provenance_component_id",
    "task_raw_provenance_component_size",
    "task_raw_provenance_overlap_degree",
    "task_raw_provenance_assignment_multiplicity_max",
    "source_kind",
    "source_index",
    "source_path",
    "source_sha256",
    "leaf_support_count",
    "leaf_support_sha256",
)
_JOINT_FUNCTIONAL_RESULT_KEY = (
    "joint_fixed_target_and_endpoint_functional_isomorphic_scale_substitution"
)
_JOINT_EQUIVALENT_RESULT_KEY = (
    "joint_fixed_target_and_endpoint_functional_equivalent_scale_substitution"
)


def _identity_payload(cell: dict, *, context: str) -> dict:
    """Copy cell identity and optional sampling-design metadata into an emitted report."""
    payload = {
        key: value
        for key, value in policy_cell_identity(cell, context=context).items()
        if key != "identity_mode"
    }
    payload.update({key: cell[key] for key in BREADTH_DESIGN_FIELDS if key in cell})
    return payload


def _validate_scored_breadth_identity(
    meta: list[dict], cell: dict, *, label: str
) -> dict:
    """Bind hierarchy score rows to their bank node; legacy H49 remains hash/cell-id bound."""
    if not uses_breadth_cell_identity(cell):
        return {
            "valid": True,
            "identity_mode": "legacy_domain_gi",
            "cell_id": cell["id"],
            "legacy_compatibility": True,
        }
    expected = policy_cell_identity(cell, context=f"{label} bank cell")
    rows = [row for row in meta if row.get("cell_id") == cell["id"]]
    if not rows:
        raise ValueError(f"{label} scores omit breadth cell {cell['id']!r}")
    for row in rows:
        observed = policy_cell_identity(row, context=f"{label} score metadata")
        if observed != expected:
            changed = sorted(
                key for key in set(expected) | set(observed)
                if expected.get(key) != observed.get(key)
            )
            raise ValueError(
                f"{label} score/bank breadth identity mismatch for {cell['id']!r}; "
                f"changed={changed}"
            )
    return {
        "valid": True,
        "identity_mode": "hierarchy_node",
        "cell_id": cell["id"],
        "node_id": cell["node_id"],
        "task": cell["task"],
        "level": cell["level"],
        "bucket": cell["bucket"],
        "n_score_rows": len(rows),
    }


def _validate_frozen_scored_arm_panel(
        *, observed_arm_ids: set[str], cell: dict, model_job: dict,
        label: str, selected_arm_ids: set[str] | None = None) -> dict:
    """Require a frozen job to score its complete declared bank-arm policy."""
    bank_ids = {arm.get("id") for arm in cell.get("arms", [])}
    if None in bank_ids or len(bank_ids) != len(cell.get("arms", [])):
        raise ValueError(f"{label} arm bank contains missing or duplicate ids")
    if selected_arm_ids is not None:
        expected = set(selected_arm_ids)
        policy = "frozen_selection"
    else:
        policy = model_job.get("arm_policy")
        if policy == "all":
            expected = bank_ids
        elif policy == "name_only":
            expected = {"name"}
        else:
            raise ValueError(f"{label} has unsupported frozen arm policy {policy!r}")
    if observed_arm_ids != expected:
        raise ValueError(
            f"{label} scored-arm panel differs from frozen {policy} policy: "
            f"missing={sorted(expected - observed_arm_ids)} "
            f"extra={sorted(observed_arm_ids - expected)}"
        )
    return {
        "valid": True,
        "arm_policy": policy,
        "n_expected_arms": len(expected),
        "expected_arm_ids": sorted(expected),
    }


def _is_control_row(row: dict) -> bool:
    return bool(
        row.get("control_for")
        or row.get("provenance") in CONTROL_PROVENANCES
    )


_BREADTH_SELECTION_POLICY_SCHEMA = "tacit_breadth_selection_policy/v2"
_BREADTH_SELECTION_ROLES = (
    "best_functional_rank",
    "best_vector_identity",
    "best_component_distinct_route_within_rank_tolerance",
    "best_address_dose",
)


def _portable_selection_path(path: str | Path) -> str:
    """Record repo-local artifacts relatively so a frozen selection survives server sync."""
    resolved = Path(path).resolve()
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _selection_shape_error(message: str) -> ValueError:
    return ValueError(f"breadth search report shape cannot support frozen selection: {message}")


def _finite_selection_number(value, *, field: str, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not np.isfinite(value)):
        raise _selection_shape_error(f"{field} is missing or non-finite")
    return float(value)


def _selection_policy(manifest: dict) -> dict:
    policy = manifest.get("selection_policy")
    if not isinstance(policy, dict) or policy.get("schema") != (
            _BREADTH_SELECTION_POLICY_SCHEMA):
        raise ValueError("search manifest omits the supported frozen breadth selection policy")
    if policy.get("roles_in_order") != list(_BREADTH_SELECTION_ROLES):
        raise ValueError("search manifest changes the supported selection role order")
    minimum = policy.get("minimum_candidates_per_cell")
    maximum = policy.get("maximum_candidates_per_cell")
    tolerance = policy.get("rank_diversity_tolerance")
    if (not isinstance(minimum, int) or isinstance(minimum, bool)
            or not isinstance(maximum, int) or isinstance(maximum, bool)
            or minimum != 1 or not minimum <= maximum <= len(_BREADTH_SELECTION_ROLES)):
        raise ValueError("breadth selection policy must retain one to four candidates per cell")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not np.isfinite(tolerance) or tolerance < 0.0):
        raise ValueError("breadth selection policy has an invalid diversity tolerance")
    for key in (
            "primary_order", "vector_order", "diversity_rule", "dose_rule",
            "null_cell_rule", "control_rule"):
        if not isinstance(policy.get(key), str) or not policy[key].strip():
            raise ValueError(f"breadth selection policy omits {key}")
    return policy


def _artifact_rows_for_selection(
        manifest: dict, *, manifest_path: Path, metric_panel_path: Path,
        additional_artifact_paths: tuple[Path, ...]) -> tuple[dict, list[dict]]:
    validation = validate_additional_artifacts(
        manifest, manifest_path=manifest_path)
    if validation.get("valid") is not True:
        raise ValueError("search manifest additional-artifact validation failed")
    rows = manifest.get("additional_artifacts", [])
    roles = [row.get("role") for row in rows]
    if (None in roles or len(roles) != len(set(roles))
            or roles.count("metric_panel") != 1):
        raise ValueError(
            "search manifest additional artifacts need unique roles and one metric_panel"
        )
    declared = []
    for row in rows:
        path = _resolve_declared_path(row["path"], manifest_path=manifest_path).resolve()
        declared.append((row, path))
    panel_row, declared_panel_path = next(
        (row, path) for row, path in declared if row["role"] == "metric_panel")
    if metric_panel_path.resolve() != declared_panel_path:
        raise ValueError("explicit metric panel differs from the search-manifest binding")
    if sha256_file(metric_panel_path) != panel_row["sha256"]:
        raise ValueError("metric panel differs from the search-manifest hash")
    expected_extra = {
        path: row for row, path in declared if row["role"] != "metric_panel"
    }
    provided_extra = [path.resolve() for path in additional_artifact_paths]
    if len(provided_extra) != len(set(provided_extra)):
        raise ValueError("explicit additional-artifact paths contain duplicates")
    if set(provided_extra) != set(expected_extra):
        raise ValueError(
            "explicit additional-artifact paths differ from the search-manifest bindings"
        )
    recorded_panel = {
        "role": "metric_panel", "path": panel_row["path"],
        "sha256": panel_row["sha256"],
    }
    recorded_extra = [{
        "role": expected_extra[path]["role"],
        "path": expected_extra[path]["path"],
        "sha256": expected_extra[path]["sha256"],
    } for path in sorted(provided_extra, key=str)]
    return recorded_panel, recorded_extra


def _validate_selection_partition(
        manifest: dict, *, manifest_path: Path, packet: dict,
        selected_partition: str) -> None:
    protocol_path = _resolve_declared_path(
        manifest.get("protocol_manifest_path", ""), manifest_path=manifest_path)
    if (not protocol_path.is_file()
            or sha256_file(protocol_path) != manifest.get("protocol_manifest_sha256")):
        raise ValueError("search-bound protocol is missing or changed")
    protocol = json.loads(protocol_path.read_text())
    specs = {
        row.get("id"): row for row in protocol.get("partitions", [])
        if isinstance(row, dict)
    }
    if selected_partition not in specs:
        raise ValueError(
            f"selected partition {selected_partition!r} is absent from the search-bound protocol"
        )
    domains = set(manifest.get("domains", []))
    if set(specs[selected_partition].get("domains", [])) != domains:
        raise ValueError("selected protocol partition does not cover every frozen domain")
    packet_domains = {
        row.get("domain"): row for row in packet.get("domains", [])
        if isinstance(row, dict)
    }
    if set(packet_domains) != domains:
        raise ValueError("search-bound packet domain panel differs from the execution manifest")
    missing = sorted(
        domain for domain, row in packet_domains.items()
        if selected_partition not in {
            partition.get("id") for partition in row.get("partitions", [])
            if isinstance(partition, dict)
        }
    )
    if missing:
        raise ValueError(
            f"selected partition is absent from packet domains: {missing}"
        )


def _validate_search_report_binding(
        report: dict, *, report_path: Path, manifest: dict, manifest_path: Path,
        arm_bank_sha256: str, packet_manifest_sha256: str) -> tuple[str, str]:
    if report.get("schema") != _POLICY_REPORT_SCHEMA:
        raise ValueError("breadth selection requires the current policy-isomorphism report schema")
    phases = manifest.get("phases", {})
    if len(phases) != 1:
        raise ValueError("breadth search manifest must declare exactly one search phase")
    search_phase, partitions = next(iter(phases.items()))
    if not isinstance(partitions, list) or len(partitions) != 1:
        raise ValueError("breadth search phase must contain exactly one partition")
    search_partition = partitions[0]
    authorization = report.get("partition_authorization", {})
    source_group = report.get("source_group_inference", {})
    if (report.get("partition") != search_partition
            or report.get("arm_bank_sha256") != arm_bank_sha256
            or authorization.get("phase") != search_phase
            or authorization.get("execution_manifest_sha256") != sha256_file(manifest_path)
            or authorization.get("selection_artifact_sha256") is not None
            or source_group.get("packet_manifest_sha256") != packet_manifest_sha256):
        raise ValueError("search report is not bound to the exact frozen search execution")
    if report.get("frozen_invocation_validation", {}).get("valid") is not True:
        raise ValueError("search report lacks a valid frozen invocation certificate")
    if report.get("additional_artifact_validation", {}).get("valid") is not True:
        raise ValueError("search report lacks valid additional-artifact authentication")
    expected_additional_files = [
        {"path": row.get("path"), "sha256": row.get("sha256")}
        for row in manifest.get("additional_artifacts", [])
    ]
    if report.get("additional_artifact_validation", {}).get(
            "files") != expected_additional_files:
        raise ValueError("search report did not authenticate the frozen metric panel")
    expected_files = manifest.get("implementation", {}).get("analysis", {}).get("files")
    if report.get("analysis_implementation", {}).get("files") != expected_files:
        raise ValueError("search report analysis implementation differs from its manifest")
    if (manifest.get("analysis", {}).get("runner", {}).get("include_controls") is not True
            or report.get("config", {}).get("include_controls") is not True):
        raise ValueError("breadth search selection requires matched-control analysis")
    if report.get("cell_panel_identity_validation", {}).get("valid") is not True:
        raise ValueError("search report lacks valid cell-panel identity authentication")
    for cell in report.get("cells", []):
        provenance = cell.get("score_provenance_validation")
        if not isinstance(provenance, dict) or set(provenance) != {"small", "target"}:
            raise ValueError("search report lacks complete production score provenance")
        if any(
            validation.get("valid") is not True
            or validation.get("fake_backend") is not False
            for validation in provenance.values()
        ):
            raise ValueError("search report contains unauthenticated or fake score provenance")
        if (cell.get("executor_prompt_bank_validation", {}).get("valid") is not True
                or cell.get("target_prompt_bank_validation", {}).get("valid") is not True):
            raise ValueError("search report lacks complete executor/target prompt authentication")
        arm_panel = cell.get("scored_arm_panel_validation", {})
        if (arm_panel.get("small", {}).get("valid") is not True
                or arm_panel.get("target", {}).get("valid") is not True):
            raise ValueError("search report lacks complete executor/target arm-panel closure")
    if not report_path.is_file():
        raise ValueError("search report does not exist")
    return search_phase, search_partition


def _selection_candidate_features(
        *, cell_id: str, arm: dict, report_row: dict, grade: dict) -> dict:
    certificate = report_row.get("certificate")
    if not isinstance(certificate, dict):
        raise _selection_shape_error(f"{cell_id}/{arm['id']} omits its certificate")
    functional = certificate.get("functional")
    point = certificate.get("point")
    differences = certificate.get("differences")
    margins = certificate.get("margins")
    observed_grade = grade.get("grades", {}).get("observed")
    if not all(isinstance(value, dict) for value in (
            functional, point, differences, margins, observed_grade)):
        raise _selection_shape_error(
            f"{cell_id}/{arm['id']} omits functional, vector, or control fields"
        )
    candidate = point.get("candidate_robust", {})
    adverse_rho = _finite_selection_number(
        functional.get("adverse_rho_point"),
        field=f"{cell_id}/{arm['id']} adverse rho", optional=True)
    quotient_rho = _finite_selection_number(
        functional.get("quotient_rho_point"),
        field=f"{cell_id}/{arm['id']} quotient rho", optional=True)
    rank_floor = (
        min(adverse_rho, quotient_rho)
        if adverse_rho is not None and quotient_rho is not None else None
    )
    mae = _finite_selection_number(
        candidate.get("mae_tvd"), field=f"{cell_id}/{arm['id']} adverse MAE")
    flip = _finite_selection_number(
        candidate.get("binary_flip_rate"), field=f"{cell_id}/{arm['id']} flip rate")
    bias = _finite_selection_number(
        candidate.get("absolute_bias"), field=f"{cell_id}/{arm['id']} absolute bias")
    vector_specs = (
        ("mae_tvd", "mae_excess_over_target_self", "upper"),
        ("spearman", "rho_minus_target_self", "lower"),
        ("binary_flip_rate", "flip_excess_over_target_self", "upper"),
        ("absolute_bias", "bias_excess_over_target_self", "upper"),
    )
    normalized_excess = {}
    vector_complete = True
    for coordinate, difference_key, direction in vector_specs:
        margin = _finite_selection_number(
            margins.get({
                "mae_tvd": "mae", "spearman": "rho",
                "binary_flip_rate": "flip", "absolute_bias": "bias",
            }[coordinate]), field=f"{cell_id}/{arm['id']} {coordinate} margin")
        if margin <= 0.0:
            raise _selection_shape_error(
                f"{cell_id}/{arm['id']} {coordinate} margin must be positive"
            )
        difference = differences.get(difference_key)
        if not isinstance(difference, dict) or "point" not in difference:
            raise _selection_shape_error(
                f"{cell_id}/{arm['id']} omits {difference_key}"
            )
        value = _finite_selection_number(
            difference.get("point"), field=f"{cell_id}/{arm['id']} {difference_key}",
            optional=True)
        if value is None:
            vector_complete = False
            normalized_excess[coordinate] = None
        elif direction == "upper":
            normalized_excess[coordinate] = max(0.0, value - margin) / margin
        else:
            normalized_excess[coordinate] = max(0.0, -margin - value) / margin
    semantic_words = arm.get("semantic_content_word_count")
    if (not isinstance(semantic_words, int) or isinstance(semantic_words, bool)
            or semantic_words < 0):
        raise _selection_shape_error(
            f"{cell_id}/{arm['id']} has an invalid semantic word count"
        )
    components = arm.get("components", [])
    if (not isinstance(components, list)
            or any(not isinstance(value, str) or not value for value in components)
            or len(components) != len(set(components))):
        raise _selection_shape_error(
            f"{cell_id}/{arm['id']} has invalid component identities"
        )
    control_superiority = observed_grade.get(
        "better_than_every_required_control_on_rank_and_mae")
    observed_functional = functional.get("observed_functional_policy_substitution")
    if not isinstance(control_superiority, bool) or not isinstance(observed_functional, bool):
        raise _selection_shape_error(
            f"{cell_id}/{arm['id']} omits point control/functional gates"
        )
    return {
        "arm_id": arm["id"],
        "channel": arm.get("channel"),
        "components": components,
        "n_address_units": arm.get("n_address_units"),
        "content_specific_point_superiority": control_superiority,
        "observed_functional_policy_substitution": observed_functional,
        "adverse_rho_point": adverse_rho,
        "quotient_rho_point": quotient_rho,
        "rank_floor": rank_floor,
        "adverse_mae_tvd": mae,
        "binary_flip_rate": flip,
        "absolute_bias": bias,
        "semantic_content_word_count": semantic_words,
        "normalized_vector_excess": normalized_excess,
        "maximum_normalized_vector_excess": (
            max(normalized_excess.values()) if vector_complete else None
        ),
    }


def _primary_selection_key(features: dict) -> tuple:
    rank_floor = features["rank_floor"]
    return (
        -rank_floor if rank_floor is not None else float("inf"),
        0 if features["observed_functional_policy_substitution"] else 1,
        0 if features["content_specific_point_superiority"] else 1,
        features["adverse_mae_tvd"],
        features["binary_flip_rate"],
        features["absolute_bias"],
        features["semantic_content_word_count"],
        features["arm_id"],
    )


def _vector_selection_key(features: dict) -> tuple:
    excess = features["maximum_normalized_vector_excess"]
    return (excess if excess is not None else float("inf"), *_primary_selection_key(features))


def _select_cell_articulations(
        *, cell: dict, report_cell: dict, policy: dict) -> dict:
    cell_id = cell["id"]
    bank_arms = {arm.get("id"): arm for arm in cell.get("arms", [])}
    report_rows = {row.get("arm_id"): row for row in report_cell.get("rows", [])}
    grade_rows = {
        row.get("arm_id"): row
        for row in report_cell.get("content_specific_scale_step", {}).get(
            "arm_grades", [])
    }
    if (None in bank_arms or len(bank_arms) != len(cell.get("arms", []))
            or None in report_rows or len(report_rows) != len(report_cell.get("rows", []))
            or None in grade_rows):
        raise _selection_shape_error(f"{cell_id} has missing or duplicate arm ids")
    candidates = [
        arm for arm in cell["arms"]
        if arm.get("id") != "name" and not _is_control_row(arm)
    ]
    if not candidates:
        raise _selection_shape_error(f"{cell_id} contains no explicit-content candidates")
    features_by_id = {}
    for arm in candidates:
        arm_id = arm["id"]
        row = report_rows.get(arm_id)
        grade = grade_rows.get(arm_id)
        if row is None or grade is None:
            raise _selection_shape_error(
                f"{cell_id}/{arm_id} is absent from search rows or control grades"
            )
        for key in ("channel", "provenance", "components", "semantic_content_word_count"):
            if row.get(key) != arm.get(key):
                raise _selection_shape_error(
                    f"{cell_id}/{arm_id} changes bank field {key} in the search report"
                )
        features_by_id[arm_id] = _selection_candidate_features(
            cell_id=cell_id, arm=arm, report_row=row, grade=grade)
    primary_order = sorted(features_by_id, key=lambda arm_id: _primary_selection_key(
        features_by_id[arm_id]))
    vector_order = sorted(features_by_id, key=lambda arm_id: _vector_selection_key(
        features_by_id[arm_id]))
    primary_ranks = {arm_id: index + 1 for index, arm_id in enumerate(primary_order)}
    vector_ranks = {arm_id: index + 1 for index, arm_id in enumerate(vector_order)}
    primary_id = primary_order[0]
    primary = features_by_id[primary_id]
    tolerance = float(policy["rank_diversity_tolerance"])
    primary_components = set(primary["components"])
    diversity_eligible = []
    for arm_id in primary_order:
        if arm_id == primary_id:
            continue
        candidate = features_by_id[arm_id]
        components = set(candidate["components"])
        if (candidate["channel"] == primary["channel"]
                or not primary_components or not components
                or primary_components <= components or components <= primary_components
                or primary["rank_floor"] is None or candidate["rank_floor"] is None
                or candidate["rank_floor"] < primary["rank_floor"] - tolerance):
            continue
        diversity_eligible.append(arm_id)
    diversity_eligible.sort(key=lambda arm_id: (
        0 if features_by_id[arm_id]["content_specific_point_superiority"] else 1,
        _primary_selection_key(features_by_id[arm_id]),
    ))
    address_order = [
        arm_id for arm_id in primary_order
        if features_by_id[arm_id]["channel"] == "address_dose"
    ]
    role_choices = {
        "best_functional_rank": primary_id,
        "best_vector_identity": vector_order[0],
        "best_component_distinct_route_within_rank_tolerance": (
            diversity_eligible[0] if diversity_eligible else None),
        "best_address_dose": address_order[0] if address_order else None,
    }
    role_orders = {
        "best_functional_rank": primary_order,
        "best_vector_identity": vector_order,
        "best_component_distinct_route_within_rank_tolerance": diversity_eligible,
        "best_address_dose": address_order,
    }
    reason_templates = {
        "best_functional_rank": "highest frozen adverse/quotient rank-floor articulation",
        "best_vector_identity": "smallest maximum normalized target-self-band excess",
        "best_component_distinct_route_within_rank_tolerance": (
            "content-specific-first different-channel component-incomparable route within rank "
            "tolerance"),
        "best_address_dose": "best primary-order address-dose articulation",
    }
    role_assignments = []
    selected_ids = []
    for role in policy["roles_in_order"]:
        arm_id = role_choices[role]
        assignment = {
            "role": role,
            "status": "assigned" if arm_id is not None else "not_available",
            "arm_id": arm_id,
            "role_rank": (role_orders[role].index(arm_id) + 1
                          if arm_id is not None else None),
            "selection_reason": (
                reason_templates[role] if arm_id is not None
                else f"no arm satisfies the frozen {role} eligibility rule"),
        }
        role_assignments.append(assignment)
        if arm_id is not None and arm_id not in selected_ids:
            selected_ids.append(arm_id)
    maximum = int(policy["maximum_candidates_per_cell"])
    if len(selected_ids) > maximum:
        selected_ids = selected_ids[:maximum]
    if len(selected_ids) < int(policy["minimum_candidates_per_cell"]):
        raise _selection_shape_error(f"{cell_id} failed the frozen null-cell retention rule")
    required_provenances = (
        "wrong_construct_control", "inert_length_control",
    )
    controls_by_candidate = {}
    control_ids = []
    for arm_id in selected_ids:
        controls = [
            arm for arm in cell["arms"] if arm.get("control_for") == arm_id
        ]
        by_provenance = {arm.get("provenance"): arm for arm in controls}
        if (len(controls) != 2 or set(by_provenance) != set(required_provenances)
                or len(by_provenance) != len(controls)):
            raise _selection_shape_error(
                f"{cell_id}/{arm_id} lacks exactly one inert and wrong control"
            )
        ids = [by_provenance[provenance]["id"] for provenance in required_provenances]
        controls_by_candidate[arm_id] = ids
        control_ids.extend(ids)
    candidate_selections = []
    for arm_id in selected_ids:
        assignments = [row for row in role_assignments if row["arm_id"] == arm_id]
        candidate_selections.append({
            "arm_id": arm_id,
            "roles": [row["role"] for row in assignments],
            "role_ranks": {row["role"]: row["role_rank"] for row in assignments},
            "selection_reasons": [row["selection_reason"] for row in assignments],
            "primary_rank": primary_ranks[arm_id],
            "vector_rank": vector_ranks[arm_id],
            "matched_control_ids": controls_by_candidate[arm_id],
            "selection_features": features_by_id[arm_id],
        })
    return {
        "cell_id": cell_id,
        "allowed_arm_ids": ["name", *selected_ids, *control_ids],
        "candidate_arm_ids": selected_ids,
        "control_ids": control_ids,
        "required_control_provenances": list(required_provenances),
        "candidate_selections": candidate_selections,
        "role_assignments": role_assignments,
        "selection_reason": (
            "deterministic application of the frozen search-manifest role policy; null cells "
            "retain their best explicit arm"
        ),
    }


def build_policy_articulation_selection(
        *, search_execution_manifest_path: str | Path,
        search_report_path: str | Path, arm_bank_path: str | Path,
        packet_manifest_path: str | Path, metric_panel_path: str | Path,
        additional_artifact_paths: tuple[str | Path, ...],
        selected_phase: str, selected_partition: str) -> dict:
    """Compile one deterministic, fully hash-bound search-to-validation selection."""
    paths = {
        "manifest": Path(search_execution_manifest_path),
        "report": Path(search_report_path),
        "bank": Path(arm_bank_path),
        "packet": Path(packet_manifest_path),
        "panel": Path(metric_panel_path),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = sorted(key for key, path in paths.items() if not path.is_file())
        raise ValueError(f"selection inputs are missing: {missing}")
    if not isinstance(selected_phase, str) or not selected_phase:
        raise ValueError("selected_phase must be a nonempty string")
    if not isinstance(selected_partition, str) or not selected_partition:
        raise ValueError("selected_partition must be a nonempty string")
    manifest = json.loads(paths["manifest"].read_text())
    if (manifest.get("schema") != "fresh_name_execution_manifest/v2"
            or not str(manifest.get("status", "")).startswith("frozen-before-")):
        raise ValueError("selection requires a frozen v2 search execution manifest")
    policy = _selection_policy(manifest)
    panel_binding, additional_bindings = _artifact_rows_for_selection(
        manifest,
        manifest_path=paths["manifest"],
        metric_panel_path=paths["panel"],
        additional_artifact_paths=tuple(Path(path) for path in additional_artifact_paths),
    )
    bank_sha = sha256_file(paths["bank"])
    packet_sha = sha256_file(paths["packet"])
    if (bank_sha != manifest.get("arm_bank_sha256")
            or packet_sha != manifest.get("packet_manifest_sha256")):
        raise ValueError("explicit arm bank or packet differs from the search manifest")
    packet = json.loads(paths["packet"].read_text())
    _validate_selection_partition(
        manifest,
        manifest_path=paths["manifest"],
        packet=packet,
        selected_partition=selected_partition,
    )
    report = json.loads(paths["report"].read_text())
    search_phase, search_partition = _validate_search_report_binding(
        report,
        report_path=paths["report"],
        manifest=manifest,
        manifest_path=paths["manifest"],
        arm_bank_sha256=bank_sha,
        packet_manifest_sha256=packet_sha,
    )
    if selected_phase == search_phase or selected_partition == search_partition:
        raise ValueError("selection target must differ from the inspected search phase/partition")
    bank = json.loads(paths["bank"].read_text())
    bank_rows = bank.get("cells", [])
    report_rows = report.get("cells", [])
    bank_ids = [cell.get("id") for cell in bank_rows]
    report_ids = [cell.get("cell_id") for cell in report_rows]
    expected_cell_ids = manifest.get("analysis", {}).get("runner", {}).get("cell_ids")
    if (not isinstance(expected_cell_ids, list) or not expected_cell_ids
            or None in bank_ids or None in report_ids
            or len(bank_ids) != len(set(bank_ids))
            or len(report_ids) != len(set(report_ids))
            or expected_cell_ids != bank_ids or expected_cell_ids != report_ids
            or len(expected_cell_ids) != len(set(expected_cell_ids))):
        raise ValueError("search manifest, report, and arm bank use different cell panels")
    bank_cells = {cell["id"]: cell for cell in bank_rows}
    report_cells = {cell["cell_id"]: cell for cell in report_rows}
    cells = [
        _select_cell_articulations(
            cell=bank_cells[cell_id], report_cell=report_cells[cell_id], policy=policy)
        for cell_id in expected_cell_ids
    ]
    payload = {
        "schema": "policy_articulation_selection/v1",
        "status": "frozen-after-search-before-validation-scoring",
        "search_phase": search_phase,
        "search_partition": search_partition,
        "selected_phase": selected_phase,
        "selected_partition": selected_partition,
        "search_execution_manifest_path": _portable_selection_path(paths["manifest"]),
        "search_execution_manifest_sha256": sha256_file(paths["manifest"]),
        "search_report_path": _portable_selection_path(paths["report"]),
        "search_report_sha256": sha256_file(paths["report"]),
        "arm_bank_path": _portable_selection_path(paths["bank"]),
        "arm_bank_sha256": bank_sha,
        "packet_manifest_path": _portable_selection_path(paths["packet"]),
        "packet_manifest_sha256": packet_sha,
        "metric_panel_path": panel_binding["path"],
        "metric_panel_sha256": panel_binding["sha256"],
        "additional_artifacts": additional_bindings,
        "selection_policy": policy,
        "selection_policy_sha256": sha256_bytes(json.dumps(
            policy, sort_keys=True, separators=(",", ":")).encode()),
        "n_cells": len(cells),
        "candidate_count_range": [
            min(len(cell["candidate_arm_ids"]) for cell in cells),
            max(len(cell["candidate_arm_ids"]) for cell in cells),
        ],
        "cells": cells,
    }
    payload["selection_content_sha256"] = sha256_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode())
    allowed = load_lockbox_selection(
        payload,
        arm_bank_sha256=bank_sha,
        packet_manifest_sha256=packet_sha,
        expected_phase=selected_phase,
        expected_partition=selected_partition,
        arm_bank=bank,
    )
    if set(allowed) != set(expected_cell_ids):
        raise ValueError("shared selector did not validate the complete frozen cell panel")
    return payload


def write_policy_articulation_selection(
        *, out_path: str | Path, **selection_inputs) -> dict:
    """Validate the complete selection in memory, then atomically expose the final artifact."""
    payload = build_policy_articulation_selection(**selection_inputs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    return {
        "path": str(out),
        "sha256": sha256_file(out),
        "n_cells": payload["n_cells"],
        "candidate_count_range": payload["candidate_count_range"],
    }


def _content_specific_scale_memberships(
        rows: list[dict], matched_control_certificates: list[dict]) -> dict:
    """Intersect joint scale grades with both required matched-control contrasts.

    A source is eligible only when at least one inert-length and one wrong-construct
    control are present.  If a source has multiple controls of either type, it must beat
    every one on both rank and MAE.  The simultaneous tier deliberately uses only the
    certificates scored over the *union* of scale candidates and matched-control pairs;
    separately Bonferroni-adjusted families cannot be combined into a familywise claim.
    """
    required = set(REQUIRED_SPECIFICITY_CONTROL_PROVENANCES)
    controls_by_source: dict[str, list[dict]] = {}
    for contrast in matched_control_certificates:
        if contrast.get("control_provenance") not in required:
            continue
        controls_by_source.setdefault(contrast["source_arm_id"], []).append(contrast)

    grades = {
        "observed": (
            "scale_step_certificate",
            "observed",
            "certificate",
            "point",
        ),
        "certified": (
            "scale_step_certificate",
            "certified",
            "certificate",
            "CI",
        ),
        "simultaneous_certified": (
            "scale_step_specificity_simultaneous_certificate",
            "certified",
            "specificity_simultaneous_certificate",
            "CI",
        ),
    }
    membership = {
        f"{grade}_joint_fixed_target_endpoint_isomorphic_members": []
        for grade in grades
    }
    membership.update({
        f"{grade}_joint_fixed_target_endpoint_equivalent_members": []
        for grade in grades
    })
    arm_grades = []
    for row in rows:
        if _is_control_row(row):
            continue
        arm_id = row["arm_id"]
        controls = controls_by_source.get(arm_id, [])
        present = sorted({control["control_provenance"] for control in controls})
        missing = sorted(required - set(present))
        coverage_complete = not missing
        arm_result = {
            "arm_id": arm_id,
            "required_control_provenances": list(
                REQUIRED_SPECIFICITY_CONTROL_PROVENANCES),
            "present_control_provenances": present,
            "missing_control_provenances": missing,
            "control_coverage_complete": coverage_complete,
            "n_required_matched_controls": len(controls),
            "failure_reasons": (
                [f"missing matched control provenance: {value}" for value in missing]
                if missing else []
            ),
            "grades": {},
        }
        for grade, (
                scale_certificate_key, evidence_grade,
                control_certificate_key, gate_suffix) in grades.items():
            scale_certificate = row.get(scale_certificate_key)
            scale_evidence = (
                scale_certificate.get("evidence", {}).get(evidence_grade, {})
                if scale_certificate else {}
            )
            control_details = []
            for control in controls:
                certificate = control.get(control_certificate_key)
                gates = certificate.get("gates", {}) if certificate else {}
                rank_gate = bool(gates.get(f"source_rank_better_{gate_suffix}", False))
                mae_gate = bool(gates.get(f"source_mae_better_{gate_suffix}", False))
                control_details.append({
                    "control_arm_id": control["control_arm_id"],
                    "control_provenance": control["control_provenance"],
                    "certificate_available": certificate is not None,
                    "source_rank_better": rank_gate,
                    "source_mae_better": mae_gate,
                    "source_better_on_rank_and_mae": bool(rank_gate and mae_gate),
                })
            control_superiority = bool(
                coverage_complete
                and controls
                and all(control["source_better_on_rank_and_mae"]
                        for control in control_details)
            )
            functional_base = bool(scale_evidence.get(_JOINT_FUNCTIONAL_RESULT_KEY, False))
            equivalent_base = bool(scale_evidence.get(_JOINT_EQUIVALENT_RESULT_KEY, False))
            functional_member = bool(functional_base and control_superiority)
            equivalent_member = bool(equivalent_base and control_superiority)
            arm_result["grades"][grade] = {
                "joint_fixed_target_endpoint_isomorphic_base": functional_base,
                "joint_fixed_target_endpoint_equivalent_base": equivalent_base,
                "better_than_every_required_control_on_rank_and_mae": (
                    control_superiority),
                "joint_fixed_target_endpoint_isomorphic_content_specific": (
                    functional_member),
                "joint_fixed_target_endpoint_equivalent_content_specific": (
                    equivalent_member),
                "controls": control_details,
            }
            if functional_member:
                membership[
                    f"{grade}_joint_fixed_target_endpoint_isomorphic_members"
                ].append(arm_id)
            if equivalent_member:
                membership[
                    f"{grade}_joint_fixed_target_endpoint_equivalent_members"
                ].append(arm_id)
        arm_grades.append(arm_result)

    return {
        "estimand": (
            "joint fixed-target/direct-endpoint scale substitution that is superior on "
            "both rank and MAE to every matched inert-length and wrong-construct control"
        ),
        "required_control_provenances": list(
            REQUIRED_SPECIFICITY_CONTROL_PROVENANCES),
        "coverage_rule": (
            "at least one matched control of each required provenance; when multiple are "
            "present, the source must beat every control on both rank and MAE"
        ),
        "simultaneous_rule": (
            "union-family Bonferroni over all eligible non-control scale candidates and all "
            "matched source-control pairs within the cell"
        ),
        "arm_grades": arm_grades,
        **membership,
    }


def _content_specific_joint_fiber(
        *,
        candidate_arm_ids: list[str],
        content_specific_membership: dict,
        arm_specs: dict[str, dict],
        arm_orbits: dict[str, dict[str, np.ndarray]],
        bootstrap_clusters: list[str] | None,
        n_boot: int,
        seed: int,
        confidence: float,
        mutual_rho_floor: float,
        mutual_rho_sensitivity_floor: float,
        min_rank_valid_fraction: float,
        mutual_mae_margin: float,
        mutual_flip_margin: float,
        mutual_bias_margin: float,
        distinctness_floor: float,
        use_bootstrap_cache: bool = True) -> dict:
    """Intersect content-specific H_J membership with a certified mutual-policy gate."""
    component_sets = {
        arm_id: set(arm_specs[arm_id].get("components") or [])
        for arm_id in candidate_arm_ids
    }
    component_minimal = {
        arm_id
        for arm_id, components in component_sets.items()
        if components and not any(
            other_components < components
            for other_id, other_components in component_sets.items()
            if other_id != arm_id and other_components
        )
    }
    atomic_route_arms = {
        arm_id for arm_id, components in component_sets.items() if not components
    }
    pairs = [
        (left, right)
        for left_index, left in enumerate(sorted(candidate_arm_ids))
        for right in sorted(candidate_arm_ids)[left_index + 1:]
    ]
    pair_confidence = (
        1.0 - (1.0 - confidence) / len(pairs) if pairs else confidence
    )
    membership_sets = {
        "observed": set(content_specific_membership[
            "observed_joint_fixed_target_endpoint_isomorphic_members"]),
        "certified": set(content_specific_membership[
            "certified_joint_fixed_target_endpoint_isomorphic_members"]),
        "simultaneous_certified": set(content_specific_membership[
            "simultaneous_certified_joint_fixed_target_endpoint_isomorphic_members"]),
    }
    equivalent_sets = {
        "observed": set(content_specific_membership[
            "observed_joint_fixed_target_endpoint_equivalent_members"]),
        "certified": set(content_specific_membership[
            "certified_joint_fixed_target_endpoint_equivalent_members"]),
        "simultaneous_certified": set(content_specific_membership[
            "simultaneous_certified_joint_fixed_target_endpoint_equivalent_members"]),
    }
    result_pairs = []
    for pair_index, (left, right) in enumerate(pairs):
        distance = articulation_distance(arm_specs[left], arm_specs[right])
        both_have_components = bool(component_sets[left] and component_sets[right])
        both_are_atomic_routes = left in atomic_route_arms and right in atomic_route_arms
        components_incomparable = (
            not (
                component_sets[left] <= component_sets[right]
                or component_sets[right] <= component_sets[left]
            )
            if both_have_components else None
        )
        channels_distinct = (
            arm_specs[left].get("channel") != arm_specs[right].get("channel")
        )
        if both_have_components:
            structural_basis = "declared_component_topology"
            route_structure_gate = bool(
                left in component_minimal
                and right in component_minimal
                and components_incomparable
            )
        elif both_are_atomic_routes:
            structural_basis = "frozen_atomic_routes_with_distinct_channels"
            route_structure_gate = channels_distinct
        else:
            structural_basis = "incompatible_component_metadata"
            route_structure_gate = False
        structural_gate = bool(
            route_structure_gate and distance >= distinctness_floor
        )
        pair_seed = seed + pair_index
        pair_context = (
            PolicyBootstrapContext(
                n_items=len(next(iter(arm_orbits[left].values()))),
                n_boot=n_boot,
                seed=pair_seed,
                bootstrap_clusters=bootstrap_clusters,
            )
            if use_bootstrap_cache else None
        )
        certificate = certify_pairwise_policy_fidelity(
            arm_orbits[left],
            arm_orbits[right],
            bootstrap_clusters=bootstrap_clusters,
            rho_floor=mutual_rho_floor,
            rho_sensitivity_floor=mutual_rho_sensitivity_floor,
            min_rank_valid_fraction=min_rank_valid_fraction,
            mae_margin=mutual_mae_margin,
            flip_margin=mutual_flip_margin,
            bias_margin=mutual_bias_margin,
            n_boot=n_boot,
            seed=pair_seed,
            confidence=pair_confidence,
            bootstrap_context=pair_context,
        )
        point_gate = certificate["gates"]["point_at_least_primary_floor"]
        interval_gate = certificate["gates"]["lower_CI_at_least_primary_floor"]
        point_sensitivity = certificate["gates"][
            "point_at_least_sensitivity_floor"]
        interval_sensitivity = certificate["gates"][
            "lower_CI_at_least_sensitivity_floor"]
        grades = {}
        for grade in ("observed", "certified", "simultaneous_certified"):
            both_hj = {left, right} <= membership_sets[grade]
            both_equivalent = {left, right} <= equivalent_sets[grade]
            mutual_gate = point_gate if grade == "observed" else interval_gate
            sensitivity_gate = (
                point_sensitivity if grade == "observed" else interval_sensitivity
            )
            vector_gate = certificate["gates"][
                "point_vector_equivalent"
                if grade == "observed" else "certified_vector_equivalent"
            ]
            grades[grade] = {
                "both_members_pass_content_specific_H_J": both_hj,
                "both_members_also_pass_H_J_eq": both_equivalent,
                "mutual_rank_gate": mutual_gate,
                "mutual_rank_sensitivity_gate": sensitivity_gate,
                "mutual_quotient_vector_equivalence_gate": vector_gate,
                "H_fiber": bool(structural_gate and both_hj and mutual_gate),
                "H_fiber_sensitivity": bool(
                    structural_gate and both_hj and sensitivity_gate),
                "H_fiber_eq": bool(
                    structural_gate and both_equivalent and mutual_gate),
                "H_fiber_vec": bool(
                    structural_gate and both_hj and vector_gate),
                "H_fiber_vec_eq": bool(
                    structural_gate and both_equivalent and vector_gate),
            }
        result_pairs.append({
            "left": left,
            "right": right,
            "structural_basis": structural_basis,
            "component_minimal": (
                left in component_minimal and right in component_minimal
                if both_have_components else None
            ),
            "components_incomparable": components_incomparable,
            "both_frozen_atomic_routes": both_are_atomic_routes,
            "channels_distinct": channels_distinct,
            "articulation_surface_distance": distance,
            "distinctness_floor": distinctness_floor,
            "structural_gate": structural_gate,
            "mutual_policy_certificate": certificate,
            "grades": grades,
        })

    def passing(grade: str, key: str) -> list[dict]:
        return [
            {"left": row["left"], "right": row["right"]}
            for row in result_pairs if row["grades"][grade][key]
        ]

    return {
        "schema": "content_specific_joint_articulation_fiber/v1",
        "estimand": (
            "equal-but-different explicit articulations that independently pass content-specific "
            "joint fixed-target/direct-endpoint scale substitution and a mutual rank interval"
        ),
        "candidate_arm_ids": sorted(candidate_arm_ids),
        "component_minimal_arm_ids": sorted(component_minimal),
        "atomic_route_arm_ids": sorted(atomic_route_arms),
        "mutual_rho_floor": mutual_rho_floor,
        "mutual_rho_sensitivity_floor": mutual_rho_sensitivity_floor,
        "min_rank_valid_fraction": min_rank_valid_fraction,
        "mutual_quotient_vector_equivalence_margins": {
            "mae_tvd": mutual_mae_margin,
            "binary_flip_rate": mutual_flip_margin,
            "absolute_bias": mutual_bias_margin,
        },
        "pairwise_multiplicity": {
            "method": (
                "Bonferroni over the frozen candidate-pair family using central two-sided "
                "intervals. Ordinal H_fiber consumes the rank lower edge. H_fiber^vec is an "
                "intersection-union test over the rank lower edge and the MAE/flip/bias upper "
                "edges, so no within-pair alpha split across co-primary coordinates is needed"
            ),
            "family_size": len(pairs),
            "per_comparison_central_interval_confidence": pair_confidence,
            "one_sided_pair_family_alpha_bound": (1.0 - confidence) / 2.0,
            "composite_with_content_specific_H_J_FWER_bound": 1.0 - confidence,
            "composite_error_control": (
                "union of two <=alpha/2 families: content-specific H_J/control and mutual "
                "candidate pairs. Pair-vector coordinates are co-primary intersection-union "
                "gates; pair multiplicity, not coordinate count, sets their Bonferroni divisor"
            ),
        },
        "pair_certificates": result_pairs,
        "observed_H_fiber_pairs": passing("observed", "H_fiber"),
        "certified_H_fiber_pairs": passing("certified", "H_fiber"),
        "simultaneous_certified_H_fiber_pairs": passing(
            "simultaneous_certified", "H_fiber"),
        "simultaneous_certified_H_fiber_sensitivity_pairs": passing(
            "simultaneous_certified", "H_fiber_sensitivity"),
        "simultaneous_certified_H_fiber_eq_pairs": passing(
            "simultaneous_certified", "H_fiber_eq"),
        "observed_H_fiber_vec_pairs": passing("observed", "H_fiber_vec"),
        "certified_H_fiber_vec_pairs": passing("certified", "H_fiber_vec"),
        "simultaneous_certified_H_fiber_vec_pairs": passing(
            "simultaneous_certified", "H_fiber_vec"),
        "simultaneous_certified_H_fiber_vec_eq_pairs": passing(
            "simultaneous_certified", "H_fiber_vec_eq"),
        "primary_confirmatory_membership": (
            "simultaneous_certified_H_fiber_pairs; both arms must pass the union-family "
            "content-specific H_J grade and the pairwise mutual-rank lower bound. This is "
            "ordinal-policy concordance, not full numerical policy equality."
        ),
        "strict_vector_secondary": (
            "H_fiber_vec additionally requires mutual form-quotient MAE, threshold-flip, and "
            "absolute-bias upper confidence bounds inside the frozen margins. It remains a "
            "quotient-policy claim, not matched-form or semantic equality."
        ),
    }


def _analysis_implementation() -> dict:
    root = Path(__file__).resolve().parent
    repo_root = root.parents[2]
    files = [
        root / "run_policy_isomorphism.py",
        root / "score_fresh_name_arms.py",
        root / "policy_isomorphism.py",
        root / "policy_data.py",
        root / "build_fresh_item_partitions.py",
        root / "target_articulation_frontier.py",
        root.parent / "grid_auc_report.py",
        root / "common_target_ladder.py",
        repo_root / "methods/metric_implementer/manifest.py",
        repo_root / "methods/metric_implementer/artifact.py",
        repo_root / "methods/metric_implementer/config.py",
        repo_root / "methods/metric_implementer/vinfo.py",
    ]
    return {
        "semantics": (
            "complete fixed-target/direct-endpoint certification, paired item inference, "
            "ordinal and quotient-vector equal-but-different fibers, and common-target ladder "
            "implementation"
        ),
        "files": [{
            "path": str(path.relative_to(repo_root)),
            "sha256": sha256_file(path),
        } for path in files],
    }


def write_calibration_release_artifact(
    report: dict,
    *,
    report_path: str | Path,
    execution_manifest_path: str | Path,
    selection_artifact_path: str | Path,
) -> dict:
    """Write the auditable production-calibration gate that permits later lockbox access."""
    manifest_path = Path(execution_manifest_path)
    manifest = json.loads(manifest_path.read_text())
    specification = manifest.get("lockbox_release", {})
    if not isinstance(specification, dict) or specification.get("required") is not True:
        raise ValueError("execution manifest omits its required lockbox-release specification")
    expected_report_path = _resolve_declared_path(
        specification.get("calibration_report_path", ""), manifest_path=manifest_path)
    report_path = Path(report_path).resolve()
    if report_path.resolve() != expected_report_path.resolve():
        raise ValueError("calibration report output path differs from frozen release path")
    if not report_path.is_file() or json.loads(report_path.read_text()) != report:
        raise ValueError("calibration report argument differs from its on-disk artifact")
    if report.get("partition_authorization", {}).get("phase") != "calibration":
        raise ValueError("only a hash-bound calibration report can release the lockbox")
    if report.get("frozen_invocation_validation", {}).get("valid") is not True:
        raise ValueError("calibration report did not use the frozen runner invocation")
    for cell in report.get("cells", []):
        for validation in (cell.get("score_provenance_validation") or {}).values():
            if validation.get("fake_backend") is not False:
                raise ValueError("fake score inputs can never release the lockbox")
    selection_path = Path(selection_artifact_path).resolve()
    if not selection_path.is_file():
        raise ValueError("calibration release requires its frozen selection artifact")
    selection_sha256 = sha256_file(selection_path)
    declared_selection_path = manifest.get("selection_artifact_path")
    declared_selection_sha256 = manifest.get("selection_artifact_sha256")
    if declared_selection_path is not None and (
            _resolve_declared_path(
                declared_selection_path, manifest_path=manifest_path).resolve()
            != selection_path
            or declared_selection_sha256 != selection_sha256):
        raise ValueError("lockbox manifest binds a different selection artifact")

    calibration_manifest = manifest
    calibration_manifest_path = manifest_path.resolve()
    calibration_manifest_sha256 = sha256_file(manifest_path)
    calibration_partitions = manifest.get("phases", {}).get("calibration", [])
    if not calibration_partitions:
        selection = json.loads(selection_path.read_text())
        if selection.get("schema") != "policy_articulation_selection/v1":
            raise ValueError(
                "two-manifest lockbox release requires a policy-articulation selection"
            )
        calibration_manifest_path = _resolve_declared_path(
            selection.get("search_execution_manifest_path", ""),
            manifest_path=selection_path,
        ).resolve()
        calibration_manifest_sha256 = selection.get(
            "search_execution_manifest_sha256")
        if (not calibration_manifest_path.is_file()
                or not isinstance(calibration_manifest_sha256, str)
                or sha256_file(calibration_manifest_path)
                != calibration_manifest_sha256):
            raise ValueError("selection-bound calibration manifest is missing or changed")
        selected_report_path = _resolve_declared_path(
            selection.get("search_report_path", ""), manifest_path=selection_path
        ).resolve()
        if (selected_report_path != report_path
                or selection.get("search_report_sha256") != sha256_file(report_path)):
            raise ValueError("selection binds a different calibration report")
        calibration_manifest = json.loads(calibration_manifest_path.read_text())
        calibration_partitions = calibration_manifest.get("phases", {}).get(
            "calibration", [])

    lockbox_partitions = manifest.get("phases", {}).get("lockbox", [])
    if len(calibration_partitions) != 1 or len(lockbox_partitions) != 1:
        raise ValueError(
            "calibration release requires one authenticated calibration and lockbox partition"
        )
    calibration_partition = calibration_partitions[0]
    lockbox_partition = lockbox_partitions[0]
    if (specification.get("calibration_partition", calibration_partition)
            != calibration_partition
            or specification.get("lockbox_partition", lockbox_partition)
            != lockbox_partition):
        raise ValueError("lockbox-release partitions differ from the frozen DAG")
    if (report.get("partition") != calibration_partition
            or report.get("partition_authorization", {}).get(
                "execution_manifest_sha256") != calibration_manifest_sha256):
        raise ValueError("calibration report is not bound to the calibration manifest")

    expected_release_path = _resolve_declared_path(
        specification.get("artifact_path", ""), manifest_path=manifest_path)
    release = {
        "schema": specification.get("schema"),
        "status": "calibration-complete-production-only-lockbox-release",
        "execution_manifest_path": str(manifest_path),
        "execution_manifest_sha256": sha256_file(manifest_path),
        "selection_artifact_sha256": selection_sha256,
        "calibration_partition": calibration_partition,
        "lockbox_partition": lockbox_partition,
        "calibration_report_path": specification["calibration_report_path"],
        "calibration_report_sha256": sha256_file(report_path),
        "fake_inputs": False,
    }
    if calibration_manifest_path != manifest_path.resolve():
        release["calibration_execution_manifest_sha256"] = (
            calibration_manifest_sha256)
    expected_release_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = expected_release_path.with_suffix(expected_release_path.suffix + ".tmp")
    temporary.write_text(json.dumps(release, indent=2) + "\n")
    temporary.replace(expected_release_path)
    return validate_lockbox_release(
        manifest,
        manifest_path=manifest_path,
        selection_sha256=selection_sha256,
        release_artifact_path=expected_release_path,
    )


def write_two_manifest_lockbox_release(
    *,
    search_execution_manifest_path: str | Path,
    search_report_path: str | Path,
    selection_artifact_path: str | Path,
    validation_execution_manifest_path: str | Path,
    release_artifact_path: str | Path,
) -> dict:
    """Authenticate the complete breadth DAG and emit only its lockbox release.

    This path deliberately performs no validation scoring or report analysis.  The search
    report has already been analyzed to produce ``selection_artifact_path``; here we replay the
    immutable provenance gates, require the exact sealed validation manifest, and emit the
    production-only capability that the scorer will later consume.
    """
    paths = {
        "search manifest": Path(search_execution_manifest_path).resolve(),
        "search report": Path(search_report_path).resolve(),
        "selection": Path(selection_artifact_path).resolve(),
        "validation manifest": Path(validation_execution_manifest_path).resolve(),
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(
            f"release-only inputs are missing: {', '.join(sorted(missing))}"
        )
    if paths["search manifest"] == paths["validation manifest"]:
        raise ValueError("release-only mode requires distinct search and validation manifests")

    selection = json.loads(paths["selection"].read_text())
    validation = json.loads(paths["validation manifest"].read_text())
    if (validation.get("schema") != "fresh_name_execution_manifest/v2"
            or not str(validation.get("status", "")).startswith("frozen-before-")):
        raise ValueError("release-only validation manifest is not frozen v2")
    phases = validation.get("phases")
    if (not isinstance(phases, dict) or set(phases) != {"lockbox"}
            or not isinstance(phases["lockbox"], list)
            or len(phases["lockbox"]) != 1
            or not isinstance(phases["lockbox"][0], str)
            or not phases["lockbox"][0]):
        raise ValueError(
            "release-only validation manifest must declare one lockbox partition"
        )
    lockbox_partition = phases["lockbox"][0]
    if (validation.get("phase_access") != {"lockbox": "sealed_confirmation"}
            or validation.get("selection_required_phases") != ["lockbox"]):
        raise ValueError(
            "release-only validation must retain the sealed selection-gated lockbox"
        )

    declared_selection = _resolve_declared_path(
        validation.get("selection_artifact_path", ""),
        manifest_path=paths["validation manifest"],
    ).resolve()
    selection_sha256 = sha256_file(paths["selection"])
    if (declared_selection != paths["selection"]
            or validation.get("selection_artifact_sha256") != selection_sha256):
        raise ValueError("release-only validation binds a different selection artifact")

    for label, field in (
            ("search manifest", "search_execution_manifest_path"),
            ("search report", "search_report_path")):
        declared = _resolve_declared_path(
            selection.get(field, ""), manifest_path=paths["selection"]
        ).resolve()
        if declared != paths[label]:
            raise ValueError(f"release-only selection binds a different {label}")

    provenance = validate_policy_articulation_selection_provenance(
        selection,
        selection_path=paths["selection"],
        execution_manifest=validation,
        execution_manifest_path=paths["validation manifest"],
    )
    if (provenance.get("search_execution_manifest_sha256")
            != sha256_file(paths["search manifest"])
            or provenance.get("search_report_sha256")
            != sha256_file(paths["search report"])):
        raise ValueError("release-only search provenance differs from the supplied artifacts")

    specification = validation.get("lockbox_release")
    if not isinstance(specification, dict) or specification.get("required") is not True:
        raise ValueError("release-only validation omits its required lockbox-release gate")
    expected_release_path = _resolve_declared_path(
        specification.get("artifact_path", ""),
        manifest_path=paths["validation manifest"],
    ).resolve()
    observed_release_path = Path(release_artifact_path).resolve()
    if observed_release_path != expected_release_path:
        raise ValueError("release-only output path differs from the frozen release path")

    report = json.loads(paths["search report"].read_text())
    prior_release = (
        observed_release_path.read_bytes() if observed_release_path.is_file() else None
    )
    try:
        release_validation = write_calibration_release_artifact(
            report,
            report_path=paths["search report"],
            execution_manifest_path=paths["validation manifest"],
            selection_artifact_path=paths["selection"],
        )
        authorization = authorize_policy_partition(
            lockbox_partition,
            operation="release-only lockbox authorization",
            execution_manifest_path=paths["validation manifest"],
            selection_artifact_path=paths["selection"],
            lockbox_release_artifact_path=observed_release_path,
        )
        if (authorization.get("phase") != "lockbox"
                or authorization.get("sealed_partition_authorized") is not True
                or authorization.get("lockbox_release_validation", {}).get("valid")
                is not True
                or authorization.get("selection_provenance_validation", {}).get("valid")
                is not True):
            raise ValueError(
                "release-only authorization did not close the sealed lockbox DAG"
            )
    except Exception:
        # A failed final authorization must never leave behind a newly minted capability.
        if prior_release is None:
            observed_release_path.unlink(missing_ok=True)
        else:
            rollback = observed_release_path.with_suffix(
                observed_release_path.suffix + ".rollback.tmp"
            )
            rollback.write_bytes(prior_release)
            rollback.replace(observed_release_path)
        raise
    return {
        "valid": True,
        "mode": "release_only",
        "artifact_path": release_validation["artifact_path"],
        "artifact_sha256": release_validation["artifact_sha256"],
        "search_selection_provenance": provenance,
        "lockbox_authorization": authorization,
    }


def _validate_frozen_runner_invocation(manifest: dict, provided: dict) -> dict:
    """Require the prospective runner call to equal its frozen numerical configuration."""
    expected = manifest.get("analysis", {}).get("runner")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("execution manifest omits frozen analysis.runner configuration")
    missing = sorted(set(provided) - set(expected))
    if missing:
        raise ValueError(f"frozen analysis.runner omits invocation keys: {missing}")
    mismatches = {
        key: {"expected": expected[key], "observed": value}
        for key, value in provided.items()
        if expected[key] != value
    }
    if mismatches:
        raise ValueError(f"runner invocation differs from frozen analysis config: {mismatches}")
    return {"valid": True, "runner": {key: expected[key] for key in provided}}


def _validate_frozen_score_bundle(
    bundle: dict,
    *,
    label: str,
    job_id: str,
    manifest: dict,
    execution_manifest_sha256: str,
    arm_bank_sha256: str,
    packet_manifest_sha256: str,
    allow_fake_inputs: bool,
) -> dict:
    """Bind averaged shards to the exact frozen scoring job and protocol."""
    jobs = {row["id"]: row for row in manifest.get("model_jobs", [])}
    if job_id not in jobs:
        raise ValueError(f"{label} job {job_id!r} is absent from execution manifest")
    expected_repetitions = sorted(jobs[job_id].get("required_repetitions", []))
    if bundle.get("repetitions") != expected_repetitions:
        raise ValueError(
            f"{label} repetitions differ from frozen job: "
            f"observed={bundle.get('repetitions')} expected={expected_repetitions}"
        )
    expected_role = jobs[job_id].get("role")
    if not expected_role or bundle.get("role") != expected_role:
        raise ValueError(
            f"{label} role differs from frozen job: "
            f"observed={bundle.get('role')!r} expected={expected_role!r}"
        )
    expected = {
        "execution_manifest_sha256": execution_manifest_sha256,
        "arm_bank_sha256": arm_bank_sha256,
        "packet_manifest_sha256": packet_manifest_sha256,
        "binary_readout": manifest.get("binary_readout"),
        "readout_template_sha256": manifest.get("readout_template_sha256"),
    }
    mismatches = {
        key: {"expected": value, "observed": bundle.get(key)}
        for key, value in expected.items()
        if value is None or bundle.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} score provenance differs from frozen manifest: {mismatches}")
    fake_backend = bundle.get("fake_backend")
    if fake_backend is None:
        raise ValueError(f"{label} score shards omit fake-backend provenance")
    if fake_backend and not allow_fake_inputs:
        raise ValueError(f"{label} score shards were produced by the fake backend")
    backend_class = bundle.get("backend_class")
    if not backend_class:
        raise ValueError(f"{label} score shards omit backend-class provenance")
    if fake_backend:
        if backend_class != "FakeVLLM":
            raise ValueError(f"{label} fake-backend flag/class are inconsistent")
    else:
        expected_backend = manifest.get("execution_environment", {}).get(
            "production_backend_class")
        if not expected_backend or backend_class != expected_backend:
            raise ValueError(
                f"{label} production backend differs from frozen manifest: "
                f"observed={backend_class!r} expected={expected_backend!r}"
            )
    return {
        "valid": True,
        "job_id": job_id,
        "repetitions": expected_repetitions,
        "execution_manifest_sha256": execution_manifest_sha256,
        "arm_bank_sha256": arm_bank_sha256,
        "packet_manifest_sha256": packet_manifest_sha256,
        "binary_readout": expected["binary_readout"],
        "readout_template_sha256": expected["readout_template_sha256"],
        "role": expected_role,
        "backend_class": backend_class,
        "fake_backend": fake_backend,
    }


def _functional_floor_capacity(certificate: dict) -> dict:
    """Largest absolute rank floor supported by one already-scored certificate.

    This is a threshold-free re-expression of the functional result.  It never promotes a
    candidate that fails target health, polarity, or the corresponding MAE-improvement gate.
    """
    functional = certificate["functional"]
    gates = functional["gates"]
    observed_eligible = all(gates.get(key, False) for key in (
        "target_identity_valid",
        "positive_polarity",
        "mae_point_improves_over_small_sparse",
    ))
    certified_eligible = all(gates.get(key, False) for key in (
        "target_identity_valid",
        "positive_polarity",
        "mae_CI_improves_over_small_sparse",
    ))
    rho_ci = functional.get("adverse_rho_CI")
    quotient_rho = functional.get("quotient_rho_point")
    quotient_rho_ci = functional.get("quotient_rho_CI")
    candidate_rho = functional.get("adverse_rho_point")
    observed_joint_rho = (
        min(candidate_rho, quotient_rho)
        if candidate_rho is not None and quotient_rho is not None else None
    )
    certified_joint_rho = (
        min(rho_ci[0], quotient_rho_ci[0])
        if rho_ci and quotient_rho_ci else None
    )
    adverse_mae = (certificate.get("point", {}).get("candidate_robust", {})
                   .get("mae_tvd"))
    sparse_robust = (certificate.get("small_sparse_point", {})
                     .get("candidate_robust", {}))
    sparse_rho = sparse_robust.get("spearman")
    sparse_rho_ci = functional.get("small_sparse_adverse_rho_CI")
    sparse_mae = sparse_robust.get("mae_tvd")
    target_robust = (certificate.get("point", {}).get("target_self_robust", {}))
    target_rho = target_robust.get("spearman")
    target_mae = target_robust.get("mae_tvd")
    rank_gap = (None if sparse_rho is None or target_rho is None
                else target_rho - sparse_rho)
    mae_gap = (None if sparse_mae is None or target_mae is None
               else sparse_mae - target_mae)
    mae_gain = (certificate.get("differences", {}).get(
        "mae_gain_over_small_sparse", {}).get("point"))
    return {
        "adverse_rho_point": candidate_rho,
        "adverse_rho_CI": rho_ci,
        "quotient_rho_point": quotient_rho,
        "quotient_rho_CI": quotient_rho_ci,
        "joint_adverse_quotient_rho_point": observed_joint_rho,
        "joint_adverse_quotient_rho_lower_CI": certified_joint_rho,
        "adverse_mae_tvd": adverse_mae,
        "mae_gain_over_small_sparse": mae_gain,
        "small_sparse_adverse_rho": sparse_rho,
        "small_sparse_adverse_rho_CI": sparse_rho_ci,
        "small_sparse_adverse_mae_tvd": sparse_mae,
        "rho_gain_over_small_sparse": (
            None if sparse_rho is None or candidate_rho is None
            else float(candidate_rho - sparse_rho)
        ),
        "fraction_rank_scale_gap_closed": (
            None if rank_gap is None or rank_gap <= 0.0
            or candidate_rho is None
            else float((candidate_rho - sparse_rho) / rank_gap)
        ),
        "fraction_mae_scale_gap_closed": (
            None if mae_gap is None or mae_gap <= 0.0 or adverse_mae is None
            else float((sparse_mae - adverse_mae) / mae_gap)
        ),
        "observed_max_rho_floor": (
            observed_joint_rho if observed_eligible else None
        ),
        "observed_min_rho_floor_exclusive": (
            sparse_rho if observed_eligible else None
        ),
        "certified_max_rho_floor": (
            certified_joint_rho if certified_eligible else None
        ),
        "certified_min_rho_floor_exclusive": (
            sparse_rho_ci[1]
            if certified_eligible and sparse_rho_ci else None
        ),
        "observed_base_gates_pass": observed_eligible,
        "certified_base_gates_pass": certified_eligible,
    }


def _concatenate_fold_orbits(orbits: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not orbits:
        raise ValueError("cannot pool an empty orbit list")
    forms = set(orbits[0])
    if not forms or any(set(orbit) != forms for orbit in orbits[1:]):
        raise ValueError("pooled fold orbits must expose the same nonempty form set")
    return {
        form: np.concatenate([np.asarray(orbit[form], float) for orbit in orbits])
        for form in sorted(forms)
    }


_POLICY_REPORT_SCHEMA = "policy_isomorphism_experiment/v5"
_READABLE_POLICY_REPORT_SCHEMAS = frozenset({
    "policy_isomorphism_experiment/v4",
    _POLICY_REPORT_SCHEMA,
})
_POLICY_FAMILY_CONFIG_KEYS = (
    "small_job", "big_job", "target_arm_id", "mae_margin", "rho_margin",
    "flip_margin", "bias_margin", "functional_rho_floor", "confidence", "n_boot",
)
_POLICY_FAMILY_OPTIONAL_CONFIG_KEYS = (
    "scale_comparator_job", "scale_comparator_arm_id", "scale_comparator_use_target",
)


def _validate_policy_report_family(reports: list[dict], *, operation: str) -> None:
    """Bind a fold family to one executor, target, bank, readout, and cell panel."""
    if not reports:
        raise ValueError(f"{operation} requires at least one report")
    schemas = {report.get("schema") for report in reports}
    if len(schemas) != 1 or not schemas <= _READABLE_POLICY_REPORT_SCHEMAS:
        raise ValueError(f"{operation} received unexpected report schemas: {schemas}")
    if len({report.get("partition") for report in reports}) != len(reports):
        raise ValueError(f"{operation} requires distinct partitions")
    banks = {report.get("arm_bank_sha256") for report in reports}
    if len(banks) != 1 or None in banks:
        raise ValueError(f"{operation} reports use different or missing arm banks")
    for key in _POLICY_FAMILY_CONFIG_KEYS:
        values = {report.get("config", {}).get(key) for report in reports}
        if len(values) != 1 or None in values:
            raise ValueError(f"{operation} reports disagree on config {key!r}")
    for key in _POLICY_FAMILY_OPTIONAL_CONFIG_KEYS:
        values = {report.get("config", {}).get(key) for report in reports}
        if len(values) != 1:
            raise ValueError(f"{operation} reports disagree on config {key!r}")
    comparator_configs = {
        json.dumps(report.get("scale_comparator", {"enabled": False}), sort_keys=True)
        for report in reports
    }
    if len(comparator_configs) != 1:
        raise ValueError(f"{operation} reports use different scale comparators")
    comparator_enabled = bool(
        reports[0].get("scale_comparator", {}).get("enabled", False)
    )
    implementations = [report.get("analysis_implementation") for report in reports]
    if any(value is not None for value in implementations):
        if not all(value is not None for value in implementations):
            raise ValueError(f"{operation} mixes hashed and unhashed implementations")
        encoded = {json.dumps(value, sort_keys=True) for value in implementations}
        if len(encoded) != 1:
            raise ValueError(f"{operation} reports use different analysis implementations")

    source_group_modes = {
        bool(report.get("source_group_inference", {}).get("enabled"))
        for report in reports
    }
    if len(source_group_modes) != 1:
        raise ValueError(f"{operation} mixes item and source-group bootstrap reports")
    if next(iter(source_group_modes)):
        packet_hashes = {
            report["source_group_inference"].get("packet_manifest_sha256")
            for report in reports
        }
        if len(packet_hashes) != 1 or None in packet_hashes:
            raise ValueError(
                f"{operation} changes or omits the source-group packet manifest"
            )

    for index, report in enumerate(reports):
        validate_policy_cell_panel(
            report.get("cells", []), context=f"{operation} report {index}"
        )
    cell_sets = [{cell.get("cell_id") for cell in report.get("cells", [])}
                 for report in reports]
    if not cell_sets or any(values != cell_sets[0] for values in cell_sets[1:]):
        raise ValueError(f"{operation} reports use different cell panels")
    for cell_id in sorted(cell_sets[0]):
        cells = [
            {cell["cell_id"]: cell for cell in report["cells"]}[cell_id]
            for report in reports
        ]
        try:
            require_same_policy_cell_identity(
                cells, context=f"{operation}/{cell_id}"
            )
        except ValueError as exc:
            raise ValueError(
                f"{operation} changes domain/construct or hierarchy identity for "
                f"{cell_id}: {exc}"
            ) from exc
        if not all(cell.get("executor_prompt_bank_validation", {}).get("valid")
                   for cell in cells):
            raise ValueError(f"{operation} lacks prompt-bank validation for {cell_id}")
        readouts = {
            value for cell in cells for value in (
                cell.get("small_readout_template_sha256"),
                cell.get("target_readout_template_sha256"),
            )
        }
        if None in readouts or len(readouts) != 1:
            raise ValueError(f"{operation} changes or omits readout identity for {cell_id}")
        binary_readouts = {
            value for cell in cells for value in (
                cell.get("small_binary_readout"),
                cell.get("target_binary_readout"),
            )
        }
        if len(binary_readouts) != 1:
            raise ValueError(f"{operation} changes binary readout protocol for {cell_id}")
        for role in ("small", "target"):
            execution_hashes = {
                cell.get(f"{role}_score_execution_manifest_sha256") for cell in cells
            }
            if len(execution_hashes) != 1:
                raise ValueError(
                    f"{operation} changes {role} score execution manifest for {cell_id}"
                )
        if comparator_enabled:
            comparator_readouts = {
                cell.get("scale_comparator_readout_template_sha256") for cell in cells
            }
            if (None in comparator_readouts or len(comparator_readouts) != 1
                    or comparator_readouts != readouts):
                raise ValueError(
                    f"{operation} changes or omits scale-comparator readout identity "
                    f"for {cell_id}"
                )
            comparator_binary_readouts = {
                cell.get("scale_comparator_binary_readout") for cell in cells
            }
            if (len(comparator_binary_readouts) != 1
                    or comparator_binary_readouts != binary_readouts):
                raise ValueError(
                    f"{operation} changes scale-comparator binary readout protocol "
                    f"for {cell_id}"
                )
            comparator_execution_hashes = {
                cell.get("scale_comparator_score_execution_manifest_sha256")
                for cell in cells
            }
            if len(comparator_execution_hashes) != 1:
                raise ValueError(
                    f"{operation} changes scale-comparator score execution manifest "
                    f"for {cell_id}"
                )
            if not all(cell.get("scale_comparator_validation", {}).get("valid")
                       for cell in cells):
                raise ValueError(
                    f"{operation} lacks scale-comparator validation for {cell_id}"
                )
        source_group_validations = [cell.get("source_group_validation") for cell in cells]
        if next(iter(source_group_modes)) and not all(
            value and value.get("valid") for value in source_group_validations
        ):
            raise ValueError(f"{operation} lacks source-group validation for {cell_id}")


def pool_crossfold_policy_reports(
        report_paths: list[str], *, n_boot: int = 10_000, seed: int = 1217,
        confidence: float = 0.95, functional_rho_floor: float | None = None,
        packet_root: str | None = None,
        packet_manifest_path: str | None = None,
        use_bootstrap_cache: bool = True) -> dict:
    """Pool disjoint public folds for a stratified, explicitly exploratory precision audit.

    The same already-inspected candidates are rescored on the union of their saved item policies;
    this does not create a new holdout.  Both nominal per-arm intervals and Bonferroni intervals
    over the complete pooled arm set are reported so post-hoc selection cannot masquerade as
    simultaneous certification.
    """
    if len(report_paths) < 2:
        raise ValueError("pooled crossfold analysis requires at least two reports")
    if not isinstance(use_bootstrap_cache, bool):
        raise TypeError("use_bootstrap_cache must be boolean")
    reports = [json.loads(Path(path).read_text()) for path in report_paths]
    _validate_policy_report_family(reports, operation="pooled analysis")
    if bool(packet_root) != bool(packet_manifest_path):
        raise ValueError("pooled source-group inference requires both packet root and manifest")
    source_floor = {report["config"]["functional_rho_floor"] for report in reports}
    if functional_rho_floor is None:
        functional_rho_floor = float(next(iter(source_floor)))

    loaded = []
    for report in reports:
        partition = require_partition(
            report["partition"],
            allowed=PUBLIC_DEVELOPMENT_PARTITIONS,
            operation="pooled public-fold precision analysis",
        )
        report_source_groups = report.get("source_group_inference", {})
        local_packet_root = packet_root or report_source_groups.get("packet_root")
        local_packet_manifest = (
            packet_manifest_path or report_source_groups.get("packet_manifest_path")
        )
        if bool(local_packet_root) != bool(local_packet_manifest):
            raise ValueError(
                "source report provides only one of the source-group packet root/manifest"
            )
        if local_packet_manifest and report_source_groups.get("enabled"):
            observed_packet_sha = sha256_file(local_packet_manifest)
            if observed_packet_sha != report_source_groups.get("packet_manifest_sha256"):
                raise ValueError(
                    f"source-group packet manifest changed for {report['partition']}"
                )
        loaded.append({
            "report": report,
            "executor_index": load_public_index(report["executor_shard_root"], partition),
            "target_index": load_public_index(report["target_shard_root"], partition),
            "cells": {cell["cell_id"]: cell for cell in report["cells"]},
            "packet_root": local_packet_root,
            "packet_manifest_path": local_packet_manifest,
        })

    cell_ids = sorted(set.intersection(*(set(row["cells"]) for row in loaded)))
    cells = []
    for cell_index, cell_id in enumerate(cell_ids):
        fold_rows = []
        seen_hashes: set[str] = set()
        for row in loaded:
            report, cell = row["report"], row["cells"][cell_id]
            config = report["config"]
            domain = cell["domain"]
            small = _average_repetitions(
                row["executor_index"][(config["small_job"], domain)])
            big = _average_repetitions(
                row["target_index"][(config["big_job"], domain)])
            if small["shard_sha256"] != cell.get("small_shards"):
                raise ValueError(
                    f"pooled raw executor shards changed for {report['partition']}/{cell_id}")
            if big["shard_sha256"] != cell.get("target_shards"):
                raise ValueError(
                    f"pooled raw target shards changed for {report['partition']}/{cell_id}")
            if len(big["hashes"]) != cell.get("n_items"):
                raise ValueError(
                    f"pooled raw item count changed for {report['partition']}/{cell_id}")
            expected_readout = cell["small_readout_template_sha256"]
            if (small.get("readout_template_sha256") != expected_readout
                    or big.get("readout_template_sha256") != expected_readout):
                raise ValueError(
                    f"pooled raw readout changed for {report['partition']}/{cell_id}")
            expected_binary_readout = cell.get("small_binary_readout")
            if (small.get("binary_readout") != expected_binary_readout
                    or big.get("binary_readout") != expected_binary_readout):
                raise ValueError(
                    "pooled raw binary readout changed for "
                    f"{report['partition']}/{cell_id}"
                )
            if (small.get("execution_manifest_sha256")
                    != cell.get("small_score_execution_manifest_sha256")
                    or big.get("execution_manifest_sha256")
                    != cell.get("target_score_execution_manifest_sha256")):
                raise ValueError(
                    "pooled raw score execution manifest changed for "
                    f"{report['partition']}/{cell_id}"
                )
            overlap = seen_hashes.intersection(big["hashes"])
            if overlap:
                raise ValueError(
                    f"pooled folds overlap for {cell_id}; example hash={sorted(overlap)[0]}"
                )
            seen_hashes.update(big["hashes"])
            small_orbits = _orbits(small["scores"], small["meta"], cell_id=cell_id)
            big_orbits = _orbits(big["scores"], big["meta"], cell_id=cell_id)
            target = big_orbits[config["target_arm_id"]]
            source_group_data = None
            if row["packet_root"]:
                source_group_data = load_partition_source_groups(
                    row["packet_root"],
                    row["packet_manifest_path"],
                    domain=domain,
                    partition=report["partition"],
                    item_hashes=big["hashes"],
                )
            aligned = {
                arm: _align_orbit(orbit, small["hashes"], big["hashes"])
                for arm, orbit in small_orbits.items()
            }
            report_arm_rows = {item["arm_id"]: item for item in cell["rows"]}
            report_arms = {
                arm_id for arm_id, item in report_arm_rows.items()
                if not _is_control_row(item)
            }
            fold_rows.append({
                "partition": report["partition"],
                "domain": domain,
                "construct": cell["construct"],
                "cell_identity": _identity_payload(
                    cell,
                    context=f"pooled policy report/{report['partition']}/{cell_id}",
                ),
                "hashes": big["hashes"],
                "target": target,
                "aligned": aligned,
                "report_arms": report_arms,
                "report_arm_rows": report_arm_rows,
                "target_shards": big["shard_sha256"],
                "small_shards": small["shard_sha256"],
                "source_groups": (
                    source_group_data["source_groups"] if source_group_data else None
                ),
                "source_group_validation": (
                    source_group_data["validation"] if source_group_data else None
                ),
            })
        if len({row["domain"] for row in fold_rows}) != 1:
            raise ValueError(f"pooled cell {cell_id} changes domain across folds")
        common_arms = sorted(set.intersection(*(row["report_arms"] for row in fold_rows)))
        if not common_arms:
            continue
        for arm_id in common_arms:
            identities = {
                (
                    row["report_arm_rows"][arm_id].get("channel"),
                    row["report_arm_rows"][arm_id].get("provenance"),
                    tuple(row["report_arm_rows"][arm_id].get("components", [])),
                )
                for row in fold_rows
            }
            if len(identities) != 1:
                raise ValueError(
                    f"pooled arm {arm_id!r} changes channel/provenance/components across folds"
                )
        if any("name" not in row["aligned"] for row in fold_rows):
            raise ValueError(f"pooled cell {cell_id} lacks the small name-only baseline")
        target = _concatenate_fold_orbits([row["target"] for row in fold_rows])
        sparse = _concatenate_fold_orbits([row["aligned"]["name"] for row in fold_rows])
        strata = [
            row["partition"] for row in fold_rows for _ in range(len(row["hashes"]))
        ]
        source_groups = (
            [value for row in fold_rows for value in row["source_groups"]]
            if all(row["source_groups"] is not None for row in fold_rows)
            else None
        )
        if any(row["source_groups"] is not None for row in fold_rows) and source_groups is None:
            raise ValueError(f"pooled cell {cell_id} mixes cluster and item inference folds")
        simultaneous_confidence = 1.0 - (1.0 - confidence) / len(common_arms)
        arm_rows = []
        for arm_index, arm_id in enumerate(common_arms):
            if any(arm_id not in row["aligned"] for row in fold_rows):
                raise ValueError(f"pooled raw shards lack reported arm {arm_id!r}")
            candidate = _concatenate_fold_orbits([
                row["aligned"][arm_id] for row in fold_rows
            ])
            kwargs = {
                "sparse_orbit": sparse,
                "bootstrap_strata": strata,
                "bootstrap_clusters": source_groups,
                "mae_margin": reports[0]["config"]["mae_margin"],
                "rho_margin": reports[0]["config"]["rho_margin"],
                "flip_margin": reports[0]["config"]["flip_margin"],
                "bias_margin": reports[0]["config"]["bias_margin"],
                "functional_rho_floor": functional_rho_floor,
                "n_boot": n_boot,
                "seed": seed + cell_index * 1000 + arm_index,
            }
            if use_bootstrap_cache:
                kwargs["bootstrap_context"] = PolicyBootstrapContext(
                    n_items=len(seen_hashes),
                    n_boot=n_boot,
                    seed=kwargs["seed"],
                    bootstrap_strata=strata,
                    bootstrap_clusters=source_groups,
                )
            nominal = certify_policy_isomorphism(
                target, candidate, confidence=confidence, **kwargs)
            simultaneous = certify_policy_isomorphism(
                target, candidate, confidence=simultaneous_confidence, **kwargs)
            arm_rows.append({
                "arm_id": arm_id,
                "channel": fold_rows[0]["report_arm_rows"][arm_id].get("channel"),
                "provenance": fold_rows[0]["report_arm_rows"][arm_id].get("provenance"),
                "components": fold_rows[0]["report_arm_rows"][arm_id].get(
                    "components", []),
                "nominal_certificate": nominal,
                "simultaneous_certificate": simultaneous,
            })
        cells.append({
            **fold_rows[0]["cell_identity"],
            "n_items": len(seen_hashes),
            "n_folds": len(fold_rows),
            "n_arms": len(common_arms),
            "folds": [{
                "partition": row["partition"],
                "n_items": len(row["hashes"]),
                "target_shards": row["target_shards"],
                "small_shards": row["small_shards"],
                "source_group_validation": row["source_group_validation"],
            } for row in fold_rows],
            "rows": arm_rows,
        })

    def count(key: str, certificate_key: str) -> int:
        return sum(
            row[certificate_key]["functional"][key]
            for cell in cells for row in cell["rows"]
        )

    return {
        "schema": "pooled_crossfold_policy_isomorphism/v3",
        "status": "retrospective public-fold precision analysis; not a new holdout",
        "reports": [{"path": path, "sha256": sha256_file(path)} for path in report_paths],
        "arm_bank_sha256": reports[0]["arm_bank_sha256"],
        "functional_rho_floor": functional_rho_floor,
        "bootstrap": {
            "n": n_boot,
            "seed": seed,
            "sampling": (
                "fold-stratified paired source-group cluster bootstrap"
                if any(
                    fold.get("source_group_validation") is not None
                    for cell in cells for fold in cell["folds"]
                )
                else "paired item bootstrap stratified by source fold"
            ),
            "nominal_confidence": confidence,
            "familywise_method": (
                "Bonferroni over all eligible non-control pooled articulation arms within each cell"
            ),
            "point_estimand": "item-weighted policy metrics on the pooled scored panel",
        },
        "cells": cells,
        "summary": {
            "n_cells": len(cells),
            "n_items": sum(cell["n_items"] for cell in cells),
            "n_arms": sum(cell["n_arms"] for cell in cells),
            "n_nominal_observed_functional_substitutions": count(
                "observed_functional_policy_substitution", "nominal_certificate"),
            "n_nominal_certified_functional_substitutions": count(
                "certified_functional_policy_substitution", "nominal_certificate"),
            "n_simultaneous_certified_functional_substitutions": count(
                "certified_functional_policy_substitution", "simultaneous_certificate"),
        },
        "claim_boundary": (
            "Pooling increases precision but reuses inspected public items. Nominal intervals "
            "diagnose sample-size uncertainty only; simultaneous intervals account for the arm "
            "search but still do not create confirmatory evidence. This pooled report reloads "
            "only the fixed-target candidate and sparse policies; it emits no pooled scale-step, "
            "two-sided endpoint-equivalence, or direct larger-endpoint inference."
        ),
    }


def summarize_crossfold_fibers(
        report_paths: list[str], *,
        rho_floors: tuple[float, ...] = DEFAULT_FUNCTIONAL_FLOOR_PROFILE) -> dict:
    """Intersect functional and near-identity fibers across fixed-policy fold reports."""
    if len(report_paths) < 2:
        raise ValueError("crossfold fiber summary requires at least two reports")
    reports = [json.loads(Path(path).read_text()) for path in report_paths]
    _validate_policy_report_family(reports, operation="crossfold fiber summary")
    floors = {report["config"].get("functional_rho_floor") for report in reports}
    if len(floors) != 1 or None in floors:
        raise ValueError("crossfold fiber reports use different or missing functional floors")
    rho_floors = tuple(sorted(set(float(value) for value in rho_floors)))
    if not rho_floors or any(value < -1.0 or value > 1.0 for value in rho_floors):
        raise ValueError("rho_floors must be nonempty and lie in [-1, 1]")
    cells_by_report = [
        {cell["cell_id"]: cell for cell in report["cells"]} for report in reports
    ]
    cell_ids = sorted(set.intersection(*(set(values) for values in cells_by_report)))
    cells = []
    for cell_id in cell_ids:
        fold_cells = [values[cell_id] for values in cells_by_report]
        rows_by_fold = [
            {row["arm_id"]: row for row in cell["rows"]} for cell in fold_cells
        ]
        all_common_arms = sorted(
            set.intersection(*(set(values) for values in rows_by_fold))
        )
        common_arms = [
            arm_id for arm_id in all_common_arms
            if all(not _is_control_row(values[arm_id]) for values in rows_by_fold)
        ]
        control_arms = sorted(set(all_common_arms) - set(common_arms))
        observed_members = [
            arm_id for arm_id in common_arms
            if all(values[arm_id]["certificate"]["functional"][
                "observed_functional_policy_substitution"] for values in rows_by_fold)
        ]
        certified_members = [
            arm_id for arm_id in common_arms
            if all(values[arm_id]["certificate"]["functional"][
                "certified_functional_policy_substitution"] for values in rows_by_fold)
        ]
        near_identity_members = [
            arm_id for arm_id in common_arms
            if all(values[arm_id]["certificate"]["policy_isomorphic"]
                   for values in rows_by_fold)
        ]

        all_capacity_by_arm = []
        for arm_id in all_common_arms:
            component_values = [
                tuple(values[arm_id].get("components", (arm_id,)))
                for values in rows_by_fold
            ]
            if len(set(component_values)) != 1:
                raise ValueError(f"arm {arm_id!r} changes components across folds")
            fold_capacity = [
                _functional_floor_capacity(values[arm_id]["certificate"])
                for values in rows_by_fold
            ]

            def stable_capacity(key: str) -> float | None:
                values = [row[key] for row in fold_capacity]
                return None if any(value is None for value in values) else float(min(values))

            def stable_floor_threshold(key: str) -> float | None:
                values = [row[key] for row in fold_capacity]
                return None if any(value is None for value in values) else float(max(values))

            all_capacity_by_arm.append({
                "arm_id": arm_id,
                "components": list(component_values[0]),
                "channel": rows_by_fold[0][arm_id].get("channel"),
                "provenance": rows_by_fold[0][arm_id].get("provenance"),
                "control_for": rows_by_fold[0][arm_id].get("control_for"),
                "composition_degree": rows_by_fold[0][arm_id].get("composition_degree"),
                "semantic_content_word_count": rows_by_fold[0][arm_id].get(
                    "semantic_content_word_count"),
                "stable_observed_max_rho_floor": stable_capacity(
                    "observed_max_rho_floor"),
                "stable_observed_min_rho_floor_exclusive": stable_floor_threshold(
                    "observed_min_rho_floor_exclusive"),
                "stable_certified_max_rho_floor": stable_capacity(
                    "certified_max_rho_floor"),
                "stable_certified_min_rho_floor_exclusive": stable_floor_threshold(
                    "certified_min_rho_floor_exclusive"),
                "folds": [{
                    "partition": report["partition"],
                    **capacity,
                } for report, capacity in zip(reports, fold_capacity)],
            })
        capacity_by_arm = [
            row for row in all_capacity_by_arm if row["arm_id"] in set(common_arms)
        ]

        def component_minimal(member_ids: list[str]) -> list[str]:
            component_sets = {
                row["arm_id"]: set(row["components"])
                for row in capacity_by_arm if row["arm_id"] in member_ids
            }
            return sorted(
                arm_id for arm_id, components in component_sets.items()
                if not any(
                    other_components < components
                    for other_id, other_components in component_sets.items()
                    if other_id != arm_id
                )
            )

        floor_profile = []
        for rho_floor in rho_floors:
            observed = [
                row["arm_id"] for row in capacity_by_arm
                if row["stable_observed_max_rho_floor"] is not None
                and row["stable_observed_max_rho_floor"] >= rho_floor
                and row["stable_observed_min_rho_floor_exclusive"] is not None
                and row["stable_observed_min_rho_floor_exclusive"] < rho_floor
            ]
            certified = [
                row["arm_id"] for row in capacity_by_arm
                if row["stable_certified_max_rho_floor"] is not None
                and row["stable_certified_max_rho_floor"] >= rho_floor
                and row["stable_certified_min_rho_floor_exclusive"] is not None
                and row["stable_certified_min_rho_floor_exclusive"] < rho_floor
            ]

            observed_minimal = component_minimal(observed)
            certified_minimal = component_minimal(certified)
            floor_profile.append({
                "rho_floor": rho_floor,
                "epsilon_rank_loss": float(1.0 - rho_floor),
                "observed_functional_members": observed,
                "certified_functional_members": certified,
                "observed_component_minimal_members": observed_minimal,
                "certified_component_minimal_members": certified_minimal,
                "n_observed_functional_members": len(observed),
                "n_certified_functional_members": len(certified),
                "n_observed_component_minimal_members": len(observed_minimal),
                "n_certified_component_minimal_members": len(certified_minimal),
            })

        reported_observed = observed_members
        capacity_index = {row["arm_id"]: row for row in capacity_by_arm}
        component_topology = []
        for base_id in component_minimal(reported_observed):
            base = capacity_index[base_id]
            base_components = set(base["components"])
            supersets = [
                row for row in capacity_by_arm
                if base_components < set(row["components"])
            ]
            effects = []
            for superset in supersets:
                fold_effects = []
                for base_fold, superset_fold in zip(base["folds"], superset["folds"]):
                    mae_delta = None
                    if (base_fold["adverse_mae_tvd"] is not None
                            and superset_fold["adverse_mae_tvd"] is not None):
                        mae_delta = float(
                            superset_fold["adverse_mae_tvd"]
                            - base_fold["adverse_mae_tvd"])
                    fold_effects.append({
                        "partition": base_fold["partition"],
                        "adverse_rho_delta": float(
                            superset_fold["adverse_rho_point"]
                            - base_fold["adverse_rho_point"]),
                        "adverse_mae_delta": mae_delta,
                    })
                effects.append({
                    "superset_arm_id": superset["arm_id"],
                    "added_components": sorted(
                        set(superset["components"]) - base_components),
                    "folds": fold_effects,
                })
            component_topology.append({
                "component_minimal_arm_id": base_id,
                "components": base["components"],
                "n_strict_supersets": len(effects),
                "all_strict_supersets_nonimproving_on_rank_across_folds": bool(
                    effects and all(
                        fold["adverse_rho_delta"] <= 0.0
                        for effect in effects for fold in effect["folds"]
                    )
                ),
                "strict_superset_effects": effects,
            })

        matched_control_contrasts = []
        for source in capacity_by_arm:
            controls = [
                row for row in all_capacity_by_arm
                if row["control_for"] == source["arm_id"]
            ]
            if not controls:
                continue
            control_rows = []
            for control in controls:
                fold_effects = []
                for source_fold, control_fold in zip(source["folds"], control["folds"]):
                    fold_effects.append({
                        "partition": source_fold["partition"],
                        "rho_advantage_source_minus_control": float(
                            source_fold["adverse_rho_point"]
                            - control_fold["adverse_rho_point"]),
                        "mae_advantage_control_minus_source": float(
                            control_fold["adverse_mae_tvd"]
                            - source_fold["adverse_mae_tvd"]),
                    })
                control_rows.append({
                    "control_arm_id": control["arm_id"],
                    "control_provenance": control["provenance"],
                    "folds": fold_effects,
                    "source_rank_better_on_all_folds": all(
                        fold["rho_advantage_source_minus_control"] > 0.0
                        for fold in fold_effects),
                    "source_mae_better_on_all_folds": all(
                        fold["mae_advantage_control_minus_source"] > 0.0
                        for fold in fold_effects),
                })
            matched_control_contrasts.append({
                "source_arm_id": source["arm_id"],
                "controls": control_rows,
                "source_rank_better_than_all_controls_on_all_folds": all(
                    row["source_rank_better_on_all_folds"] for row in control_rows),
                "source_mae_better_than_all_controls_on_all_folds": all(
                    row["source_mae_better_on_all_folds"] for row in control_rows),
            })

        def fold_pairs(key: str) -> list[dict[tuple[str, str], dict]]:
            result = []
            for cell in fold_cells:
                pairs = cell["fiber"].get(key, [])
                result.append({
                    tuple(sorted((row["left"], row["right"]))): row for row in pairs
                })
            return result

        def stable_pairs(key: str) -> list[dict]:
            values = fold_pairs(key)
            common = sorted(set.intersection(*(set(rows) for rows in values)))
            return [{
                "left": pair[0],
                "right": pair[1],
                "articulation_surface_distance": values[0][pair][
                    "articulation_surface_distance"],
                "folds": [{
                    "partition": report["partition"],
                    "behavior": rows[pair]["behavior"],
                    "behavior_rho_floor": rows[pair].get("behavior_rho_floor"),
                } for report, rows in zip(reports, values)],
            } for pair in common]

        observed_pairs = stable_pairs(
            "observed_functional_equal_but_different_pairs")
        certified_pairs = stable_pairs(
            "certified_functional_equal_but_different_pairs")
        strict_pairs = stable_pairs("equal_but_different_pairs")

        sensitivity_by_fold = []
        for cell in fold_cells:
            profiles = cell["fiber"].get(
                "pairwise_behavior_threshold_sensitivity", [])
            sensitivity_by_fold.append({row["rho_floor"]: row for row in profiles})
        common_pairwise_floors = sorted(set.intersection(*(
            set(values) for values in sensitivity_by_fold
        ))) if sensitivity_by_fold else []
        stable_pairwise_sensitivity = []
        for rho_floor in common_pairwise_floors:
            profile = {"rho_floor": rho_floor, "pairwise_gate_grade": "point_only"}
            for tier in ("near_identity", "observed_functional", "certified_functional"):
                pair_maps = []
                for fold_profile in sensitivity_by_fold:
                    pairs = fold_profile[rho_floor][tier]["pairs"]
                    pair_maps.append({
                        tuple(sorted((row["left"], row["right"]))): row
                        for row in pairs
                    })
                stable = sorted(set.intersection(*(set(rows) for rows in pair_maps)))
                profile[tier] = {
                    "n_pairs": len(stable),
                    "pairs": [{
                        "left": pair[0],
                        "right": pair[1],
                        "folds": [{
                            "partition": report["partition"],
                            "quotient_spearman": rows[pair]["quotient_spearman"],
                        } for report, rows in zip(reports, pair_maps)],
                    } for pair in stable],
                }
            stable_pairwise_sensitivity.append(profile)

        scale_step_arms = [
            arm_id for arm_id in common_arms
            if all(values[arm_id].get("scale_step_certificate") is not None
                   for values in rows_by_fold)
        ]

        def stable_scale_members(
                evidence_grade: str, result_key: str, *,
                certificate_key: str = "scale_step_certificate") -> list[str]:
            return [
                arm_id for arm_id in scale_step_arms
                if all(
                    values[arm_id].get(certificate_key) is not None
                    and values[arm_id][certificate_key]["evidence"][evidence_grade][
                        result_key]
                    for values in rows_by_fold
                )
            ]

        scale_step_by_arm = [{
            "arm_id": arm_id,
            "folds": [{
                "partition": report["partition"],
                "observed": values[arm_id]["scale_step_certificate"]["evidence"][
                    "observed"],
                "certified": values[arm_id]["scale_step_certificate"]["evidence"][
                    "certified"],
                "simultaneous_certified": (
                    (values[arm_id].get("scale_step_simultaneous_certificate") or {})
                    .get("evidence", {}).get("certified")),
                "target_fidelity": values[arm_id]["scale_step_certificate"][
                    "target_fidelity"],
                "direct_endpoint_isomorphism": values[arm_id][
                    "scale_step_certificate"]["direct_endpoint_isomorphism"],
                "descriptive_step_closure": values[arm_id][
                    "scale_step_certificate"]["descriptive_step_closure"],
            } for report, values in zip(reports, rows_by_fold)],
        } for arm_id in scale_step_arms]

        observed_local_scale_members = stable_scale_members(
            "observed", "local_primary_scale_substitution")
        certified_local_scale_members = stable_scale_members(
            "certified", "local_primary_scale_substitution")
        simultaneous_certified_local_scale_members = stable_scale_members(
            "certified", "local_primary_scale_substitution",
            certificate_key="scale_step_simultaneous_certificate")
        observed_two_sided_equivalence_recovery_members = stable_scale_members(
            "observed", "local_primary_two_sided_equivalence_recovery")
        certified_two_sided_equivalence_recovery_members = stable_scale_members(
            "certified", "local_primary_two_sided_equivalence_recovery")
        simultaneous_certified_two_sided_equivalence_recovery_members = (
            stable_scale_members(
                "certified",
                "local_primary_two_sided_equivalence_recovery",
                certificate_key="scale_step_simultaneous_certificate",
            )
        )
        observed_functional_scale_members = stable_scale_members(
            "observed", "functional_target_scale_substitution")
        certified_functional_scale_members = stable_scale_members(
            "certified", "functional_target_scale_substitution")
        simultaneous_certified_functional_scale_members = stable_scale_members(
            "certified", "functional_target_scale_substitution",
            certificate_key="scale_step_simultaneous_certificate")
        observed_endpoint_isomorphic_scale_members = stable_scale_members(
            "observed", "local_functional_endpoint_isomorphic_scale_substitution")
        certified_endpoint_isomorphic_scale_members = stable_scale_members(
            "certified", "local_functional_endpoint_isomorphic_scale_substitution")
        simultaneous_certified_endpoint_isomorphic_scale_members = stable_scale_members(
            "certified",
            "local_functional_endpoint_isomorphic_scale_substitution",
            certificate_key="scale_step_simultaneous_certificate",
        )
        observed_endpoint_equivalent_scale_members = stable_scale_members(
            "observed", "local_functional_endpoint_equivalent_scale_substitution")
        certified_endpoint_equivalent_scale_members = stable_scale_members(
            "certified", "local_functional_endpoint_equivalent_scale_substitution")
        simultaneous_certified_endpoint_equivalent_scale_members = stable_scale_members(
            "certified",
            "local_functional_endpoint_equivalent_scale_substitution",
            certificate_key="scale_step_simultaneous_certificate",
        )
        observed_near_identity_endpoint_scale_members = stable_scale_members(
            "observed", "local_near_identity_isomorphic_scale_substitution")
        certified_near_identity_endpoint_scale_members = stable_scale_members(
            "certified", "local_near_identity_isomorphic_scale_substitution")
        simultaneous_certified_near_identity_endpoint_scale_members = stable_scale_members(
            "certified",
            "local_near_identity_isomorphic_scale_substitution",
            certificate_key="scale_step_simultaneous_certificate",
        )
        observed_joint_endpoint_isomorphic_scale_members = stable_scale_members(
            "observed",
            "joint_fixed_target_and_endpoint_functional_isomorphic_scale_substitution",
        )
        certified_joint_endpoint_isomorphic_scale_members = stable_scale_members(
            "certified",
            "joint_fixed_target_and_endpoint_functional_isomorphic_scale_substitution",
        )
        simultaneous_certified_joint_endpoint_isomorphic_scale_members = (
            stable_scale_members(
                "certified",
                "joint_fixed_target_and_endpoint_functional_isomorphic_scale_substitution",
                certificate_key="scale_step_simultaneous_certificate",
            )
        )
        observed_joint_endpoint_equivalent_scale_members = stable_scale_members(
            "observed",
            "joint_fixed_target_and_endpoint_functional_equivalent_scale_substitution",
        )
        certified_joint_endpoint_equivalent_scale_members = stable_scale_members(
            "certified",
            "joint_fixed_target_and_endpoint_functional_equivalent_scale_substitution",
        )
        simultaneous_certified_joint_endpoint_equivalent_scale_members = (
            stable_scale_members(
                "certified",
                "joint_fixed_target_and_endpoint_functional_equivalent_scale_substitution",
                certificate_key="scale_step_simultaneous_certificate",
            )
        )
        matched_by_fold = []
        for cell in fold_cells:
            matched_by_fold.append({
                (row["source_arm_id"], row["control_arm_id"]): row
                for row in cell.get("matched_control_certificates", [])
            })
        stable_matched = []
        if matched_by_fold and all(matched_by_fold):
            common_matched = sorted(set.intersection(
                *(set(rows) for rows in matched_by_fold)))
            for source_id, control_id in common_matched:
                certificates = [rows[(source_id, control_id)]["certificate"]
                                for rows in matched_by_fold]
                simultaneous_certificates = [
                    rows[(source_id, control_id)].get("simultaneous_certificate")
                    for rows in matched_by_fold
                ]
                stable_matched.append({
                    "source_arm_id": source_id,
                    "control_arm_id": control_id,
                    "control_provenance": matched_by_fold[0][
                        (source_id, control_id)]["control_provenance"],
                    "source_better_rank_point_all_folds": all(
                        certificate["gates"]["source_rank_better_point"]
                        for certificate in certificates),
                    "source_better_rank_CI_all_folds": all(
                        certificate["gates"]["source_rank_better_CI"]
                        for certificate in certificates),
                    "source_better_rank_simultaneous_CI_all_folds": all(
                        certificate is not None
                        and certificate["gates"]["source_rank_better_CI"]
                        for certificate in simultaneous_certificates),
                    "source_better_mae_point_all_folds": all(
                        certificate["gates"]["source_mae_better_point"]
                        for certificate in certificates),
                    "source_better_mae_CI_all_folds": all(
                        certificate["gates"]["source_mae_better_CI"]
                        for certificate in certificates),
                    "source_better_mae_simultaneous_CI_all_folds": all(
                        certificate is not None
                        and certificate["gates"]["source_mae_better_CI"]
                        for certificate in simultaneous_certificates),
                    "simultaneous_inference_available_all_folds": all(
                        certificate is not None
                        for certificate in simultaneous_certificates),
                    "folds": [{
                        "partition": report["partition"],
                        "certificate": certificate,
                        "simultaneous_certificate": simultaneous_certificate,
                    } for report, certificate, simultaneous_certificate in zip(
                        reports, certificates, simultaneous_certificates)],
                })
        content_specific_by_fold = [
            _content_specific_scale_memberships(
                cell["rows"], cell.get("matched_control_certificates", []))
            for cell in fold_cells
        ]
        content_specific_membership_keys = (
            "observed_joint_fixed_target_endpoint_isomorphic_members",
            "certified_joint_fixed_target_endpoint_isomorphic_members",
            "simultaneous_certified_joint_fixed_target_endpoint_isomorphic_members",
            "observed_joint_fixed_target_endpoint_equivalent_members",
            "certified_joint_fixed_target_endpoint_equivalent_members",
            "simultaneous_certified_joint_fixed_target_endpoint_equivalent_members",
        )
        stable_content_specific = {
            key: sorted(set.intersection(*(
                set(profile[key]) for profile in content_specific_by_fold
            )))
            for key in content_specific_membership_keys
        }
        content_specific_crossfold = {
            "required_control_provenances": list(
                REQUIRED_SPECIFICITY_CONTROL_PROVENANCES),
            "stability_rule": (
                "membership at the same observed, certified, or union-family simultaneous "
                "grade on every joined fold"
            ),
            **stable_content_specific,
            "folds": [{
                "partition": report["partition"],
                **{key: profile[key] for key in content_specific_membership_keys},
                "arm_grades": profile["arm_grades"],
            } for report, profile in zip(reports, content_specific_by_fold)],
        }
        cells.append({
            **_identity_payload(
                fold_cells[0], context=f"crossfold policy report/{cell_id}"
            ),
            "common_arms": common_arms,
            "control_arms_excluded_from_membership": control_arms,
            "observed_functional_members": observed_members,
            "certified_functional_members": certified_members,
            "near_identity_members": near_identity_members,
            "functional_capacity_by_arm": capacity_by_arm,
            "functional_floor_profile": floor_profile,
            "component_topology_at_reported_floor": component_topology,
            "matched_control_contrasts": matched_control_contrasts,
            "matched_control_certificates": stable_matched,
            "content_specific_scale_step": content_specific_crossfold,
            "observed_functional_equal_but_different_pairs": observed_pairs,
            "certified_functional_equal_but_different_pairs": certified_pairs,
            "near_identity_equal_but_different_pairs": strict_pairs,
            "pairwise_behavior_threshold_sensitivity": stable_pairwise_sensitivity,
            "scale_step": {
                "enabled": bool(scale_step_arms),
                "arms": scale_step_by_arm,
                "observed_local_primary_members": observed_local_scale_members,
                "certified_local_primary_members": certified_local_scale_members,
                "simultaneous_certified_local_primary_members": (
                    simultaneous_certified_local_scale_members),
                "observed_one_sided_noninferiority_recovery_members": (
                    observed_local_scale_members),
                "certified_one_sided_noninferiority_recovery_members": (
                    certified_local_scale_members),
                "simultaneous_certified_one_sided_noninferiority_recovery_members": (
                    simultaneous_certified_local_scale_members),
                "observed_two_sided_equivalence_recovery_members": (
                    observed_two_sided_equivalence_recovery_members),
                "certified_two_sided_equivalence_recovery_members": (
                    certified_two_sided_equivalence_recovery_members),
                "simultaneous_certified_two_sided_equivalence_recovery_members": (
                    simultaneous_certified_two_sided_equivalence_recovery_members),
                "observed_functional_target_members": (
                    observed_functional_scale_members),
                "certified_functional_target_members": (
                    certified_functional_scale_members),
                "simultaneous_certified_functional_target_members": (
                    simultaneous_certified_functional_scale_members),
                "observed_functional_endpoint_isomorphic_members": (
                    observed_endpoint_isomorphic_scale_members),
                "certified_functional_endpoint_isomorphic_members": (
                    certified_endpoint_isomorphic_scale_members),
                "simultaneous_certified_functional_endpoint_isomorphic_members": (
                    simultaneous_certified_endpoint_isomorphic_scale_members),
                "observed_functional_endpoint_equivalent_members": (
                    observed_endpoint_equivalent_scale_members),
                "certified_functional_endpoint_equivalent_members": (
                    certified_endpoint_equivalent_scale_members),
                "simultaneous_certified_functional_endpoint_equivalent_members": (
                    simultaneous_certified_endpoint_equivalent_scale_members),
                "observed_near_identity_endpoint_isomorphic_members": (
                    observed_near_identity_endpoint_scale_members),
                "certified_near_identity_endpoint_isomorphic_members": (
                    certified_near_identity_endpoint_scale_members),
                "simultaneous_certified_near_identity_endpoint_isomorphic_members": (
                    simultaneous_certified_near_identity_endpoint_scale_members),
                "observed_joint_fixed_target_endpoint_isomorphic_members": (
                    observed_joint_endpoint_isomorphic_scale_members),
                "certified_joint_fixed_target_endpoint_isomorphic_members": (
                    certified_joint_endpoint_isomorphic_scale_members),
                "simultaneous_certified_joint_fixed_target_endpoint_isomorphic_members": (
                    simultaneous_certified_joint_endpoint_isomorphic_scale_members),
                "observed_joint_fixed_target_endpoint_equivalent_members": (
                    observed_joint_endpoint_equivalent_scale_members),
                "certified_joint_fixed_target_endpoint_equivalent_members": (
                    certified_joint_endpoint_equivalent_scale_members),
                "simultaneous_certified_joint_fixed_target_endpoint_equivalent_members": (
                    simultaneous_certified_joint_endpoint_equivalent_scale_members),
            },
        })
    return {
        "schema": "crossfold_policy_isomorphism_fibers/v5",
        "status": "inherits source report claim grade; no new confirmation",
        "reports": [{"path": path, "sha256": sha256_file(path),
                     "partition": report["partition"]}
                    for path, report in zip(report_paths, reports)],
        "arm_bank_sha256": reports[0]["arm_bank_sha256"],
        "functional_rho_floor": next(iter(floors)),
        "functional_rho_floor_profile": list(rho_floors),
        "cells": cells,
        "summary": {
            "n_cells": len(cells),
            "n_observed_functional_members": sum(
                len(cell["observed_functional_members"]) for cell in cells),
            "n_certified_functional_members": sum(
                len(cell["certified_functional_members"]) for cell in cells),
            "n_near_identity_members": sum(
                len(cell["near_identity_members"]) for cell in cells),
            "n_observed_local_primary_scale_step_members": sum(
                len(cell["scale_step"]["observed_local_primary_members"])
                for cell in cells),
            "n_certified_local_primary_scale_step_members": sum(
                len(cell["scale_step"]["certified_local_primary_members"])
                for cell in cells),
            "n_simultaneous_certified_local_primary_scale_step_members": sum(
                len(cell["scale_step"][
                    "simultaneous_certified_local_primary_members"])
                for cell in cells),
            "n_observed_one_sided_noninferiority_recovery_members": sum(
                len(cell["scale_step"][
                    "observed_one_sided_noninferiority_recovery_members"])
                for cell in cells),
            "n_certified_one_sided_noninferiority_recovery_members": sum(
                len(cell["scale_step"][
                    "certified_one_sided_noninferiority_recovery_members"])
                for cell in cells),
            "n_simultaneous_certified_one_sided_noninferiority_recovery_members": sum(
                len(cell["scale_step"][
                    "simultaneous_certified_one_sided_noninferiority_recovery_members"])
                for cell in cells),
            "n_observed_two_sided_equivalence_recovery_members": sum(
                len(cell["scale_step"][
                    "observed_two_sided_equivalence_recovery_members"])
                for cell in cells),
            "n_certified_two_sided_equivalence_recovery_members": sum(
                len(cell["scale_step"][
                    "certified_two_sided_equivalence_recovery_members"])
                for cell in cells),
            "n_simultaneous_certified_two_sided_equivalence_recovery_members": sum(
                len(cell["scale_step"][
                    "simultaneous_certified_two_sided_equivalence_recovery_members"])
                for cell in cells),
            "n_observed_functional_target_scale_step_members": sum(
                len(cell["scale_step"]["observed_functional_target_members"])
                for cell in cells),
            "n_certified_functional_target_scale_step_members": sum(
                len(cell["scale_step"]["certified_functional_target_members"])
                for cell in cells),
            "n_simultaneous_certified_functional_target_scale_step_members": sum(
                len(cell["scale_step"][
                    "simultaneous_certified_functional_target_members"])
                for cell in cells),
            "n_observed_functional_endpoint_isomorphic_scale_step_members": sum(
                len(cell["scale_step"][
                    "observed_functional_endpoint_isomorphic_members"])
                for cell in cells),
            "n_certified_functional_endpoint_isomorphic_scale_step_members": sum(
                len(cell["scale_step"][
                    "certified_functional_endpoint_isomorphic_members"])
                for cell in cells),
            "n_simultaneous_certified_functional_endpoint_isomorphic_scale_step_members": sum(
                len(cell["scale_step"][
                    "simultaneous_certified_functional_endpoint_isomorphic_members"])
                for cell in cells),
            "n_observed_functional_endpoint_equivalent_scale_step_members": sum(
                len(cell["scale_step"][
                    "observed_functional_endpoint_equivalent_members"])
                for cell in cells),
            "n_certified_functional_endpoint_equivalent_scale_step_members": sum(
                len(cell["scale_step"][
                    "certified_functional_endpoint_equivalent_members"])
                for cell in cells),
            "n_simultaneous_certified_functional_endpoint_equivalent_scale_step_members": sum(
                len(cell["scale_step"][
                    "simultaneous_certified_functional_endpoint_equivalent_members"])
                for cell in cells),
            "n_observed_near_identity_endpoint_isomorphic_scale_step_members": sum(
                len(cell["scale_step"][
                    "observed_near_identity_endpoint_isomorphic_members"])
                for cell in cells),
            "n_certified_near_identity_endpoint_isomorphic_scale_step_members": sum(
                len(cell["scale_step"][
                    "certified_near_identity_endpoint_isomorphic_members"])
                for cell in cells),
            "n_simultaneous_certified_near_identity_endpoint_isomorphic_scale_step_members": sum(
                len(cell["scale_step"][
                    "simultaneous_certified_near_identity_endpoint_isomorphic_members"])
                for cell in cells),
            "n_observed_joint_fixed_target_endpoint_isomorphic_scale_step_members": sum(
                len(cell["scale_step"][
                    "observed_joint_fixed_target_endpoint_isomorphic_members"])
                for cell in cells),
            "n_certified_joint_fixed_target_endpoint_isomorphic_scale_step_members": sum(
                len(cell["scale_step"][
                    "certified_joint_fixed_target_endpoint_isomorphic_members"])
                for cell in cells),
            "n_simultaneous_certified_joint_fixed_target_endpoint_isomorphic_scale_step_members": (
                sum(
                    len(cell["scale_step"][
                        "simultaneous_certified_joint_fixed_target_endpoint_"
                        "isomorphic_members"])
                    for cell in cells
                )
            ),
            "n_observed_joint_fixed_target_endpoint_equivalent_scale_step_members": sum(
                len(cell["scale_step"][
                    "observed_joint_fixed_target_endpoint_equivalent_members"])
                for cell in cells),
            "n_certified_joint_fixed_target_endpoint_equivalent_scale_step_members": sum(
                len(cell["scale_step"][
                    "certified_joint_fixed_target_endpoint_equivalent_members"])
                for cell in cells),
            "n_simultaneous_certified_joint_fixed_target_endpoint_equivalent_scale_step_members": (
                sum(
                    len(cell["scale_step"][
                        "simultaneous_certified_joint_fixed_target_endpoint_"
                        "equivalent_members"])
                    for cell in cells
                )
            ),
            "n_observed_content_specific_joint_fixed_target_endpoint_isomorphic_"
            "scale_step_members": sum(
                len(cell["content_specific_scale_step"][
                    "observed_joint_fixed_target_endpoint_isomorphic_members"])
                for cell in cells),
            "n_certified_content_specific_joint_fixed_target_endpoint_isomorphic_"
            "scale_step_members": sum(
                len(cell["content_specific_scale_step"][
                    "certified_joint_fixed_target_endpoint_isomorphic_members"])
                for cell in cells),
            "n_simultaneous_certified_content_specific_joint_fixed_target_endpoint_"
            "isomorphic_scale_step_members": sum(
                len(cell["content_specific_scale_step"][
                    "simultaneous_certified_joint_fixed_target_endpoint_"
                    "isomorphic_members"])
                for cell in cells),
            "n_observed_content_specific_joint_fixed_target_endpoint_equivalent_"
            "scale_step_members": sum(
                len(cell["content_specific_scale_step"][
                    "observed_joint_fixed_target_endpoint_equivalent_members"])
                for cell in cells),
            "n_certified_content_specific_joint_fixed_target_endpoint_equivalent_"
            "scale_step_members": sum(
                len(cell["content_specific_scale_step"][
                    "certified_joint_fixed_target_endpoint_equivalent_members"])
                for cell in cells),
            "n_simultaneous_certified_content_specific_joint_fixed_target_endpoint_"
            "equivalent_scale_step_members": sum(
                len(cell["content_specific_scale_step"][
                    "simultaneous_certified_joint_fixed_target_endpoint_"
                    "equivalent_members"])
                for cell in cells),
            "n_observed_functional_equal_but_different_pairs": sum(
                len(cell["observed_functional_equal_but_different_pairs"])
                for cell in cells),
            "n_certified_functional_equal_but_different_pairs": sum(
                len(cell["certified_functional_equal_but_different_pairs"])
                for cell in cells),
            "max_stable_observed_rho_floor": max(
                (row["stable_observed_max_rho_floor"]
                 for cell in cells for row in cell["functional_capacity_by_arm"]
                 if row["stable_observed_max_rho_floor"] is not None),
                default=None,
            ),
            "max_stable_certified_rho_floor": max(
                (row["stable_certified_max_rho_floor"]
                 for cell in cells for row in cell["functional_capacity_by_arm"]
                 if row["stable_certified_max_rho_floor"] is not None),
                default=None,
            ),
        },
        "claim_boundary": (
            "Intersection across reports strengthens public stability only. Observed functional "
            "members use a point rank floor; certified members require the lower confidence "
            "bound. The floor profile is a retrospective sensitivity analysis, not a family of "
            "new confirmatory thresholds. Legacy local-primary/vector scale-step members are "
            "one-sided fixed-target noninferiority recoveries, not endpoint-isomorphism claims. "
            "Direct larger-endpoint functional isomorphism, the stricter two-sided endpoint-"
            "equivalent tier, and direct near-identity are intersected separately. Content-"
            "specific joint tiers additionally require both matched control types and rank-and-"
            "MAE superiority on every fold; their simultaneous inputs come from the source "
            "reports' union-family adjustment. Controls are excluded from every scale-step "
            "membership family."
        ),
    }


def run(*, executor_shard_root: str, arm_bank_path: str, partition: str,
        target_shard_root: str | None = None,
        scale_comparator_shard_root: str | None = None,
        scale_comparator_job: str | None = None,
        scale_comparator_arm_id: str = "name",
        scale_comparator_use_target: bool = False,
        packet_root: str | None = None,
        packet_manifest_path: str | None = None,
        small_job: str = "llama3_small", big_job: str = "llama8_big_sparse",
        target_arm_id: str = "name",
        n_boot: int = 2000, seed: int = 1207, mae_margin: float = 0.02,
        rho_margin: float = 0.05, flip_margin: float = 0.02,
        bias_margin: float = 0.02, functional_rho_floor: float = 0.70,
        include_controls: bool = False,
        crossfit_only: bool = False, confidence: float = 0.95,
        fiber_mutual_rho_floor: float = 0.90,
        fiber_mutual_rho_sensitivity_floor: float = 0.85,
        fiber_min_rank_valid_fraction: float = 0.99,
        fiber_distinctness_floor: float = 0.35,
        cell_ids: tuple[str, ...] | None = None,
        execution_manifest_path: str | None = None,
        selection_artifact_path: str | None = None,
        lockbox_release_artifact_path: str | None = None,
        allow_fake_inputs: bool = False,
        use_bootstrap_cache: bool = True) -> dict:
    partition_authorization = authorize_policy_partition(
        partition,
        operation="direct policy-isomorphism analysis",
        execution_manifest_path=execution_manifest_path,
        selection_artifact_path=selection_artifact_path,
        lockbox_release_artifact_path=lockbox_release_artifact_path,
    )
    if bool(packet_root) != bool(packet_manifest_path):
        raise ValueError("source-group inference requires both packet root and manifest")
    if not isinstance(use_bootstrap_cache, bool):
        raise TypeError("use_bootstrap_cache must be boolean")
    for name, value in (
            ("fiber_mutual_rho_floor", fiber_mutual_rho_floor),
            ("fiber_mutual_rho_sensitivity_floor",
             fiber_mutual_rho_sensitivity_floor)):
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [-1, 1]")
    if (not np.isfinite(fiber_min_rank_valid_fraction)
            or not 0.0 < fiber_min_rank_valid_fraction <= 1.0):
        raise ValueError("fiber_min_rank_valid_fraction must lie in (0, 1]")
    if not np.isfinite(fiber_distinctness_floor) or fiber_distinctness_floor < 0.0:
        raise ValueError("fiber_distinctness_floor must be finite and nonnegative")
    if scale_comparator_use_target and (
            scale_comparator_shard_root or scale_comparator_job):
        raise ValueError(
            "scale comparator must be either the fixed target or an explicit shard root/job"
        )
    if not scale_comparator_use_target and (
            bool(scale_comparator_shard_root) != bool(scale_comparator_job)):
        raise ValueError(
            "explicit scale comparator requires both shard root and job"
        )
    frozen_manifest = None
    frozen_invocation_validation = None
    frozen_selection_by_cell = None
    additional_artifact_validation = None
    if execution_manifest_path is not None:
        frozen_manifest = json.loads(Path(execution_manifest_path).read_text())
        additional_artifact_validation = validate_additional_artifacts(
            frozen_manifest, manifest_path=execution_manifest_path)
        validate_frozen_implementation(
            frozen_manifest,
            manifest_path=execution_manifest_path,
            section="analysis",
        )
        if not allow_fake_inputs:
            validate_frozen_environment(frozen_manifest)
        frozen_invocation_validation = _validate_frozen_runner_invocation(
            frozen_manifest,
            {
                "small_job": small_job,
                "big_job": big_job,
                "target_arm_id": target_arm_id,
                "scale_comparator_job": scale_comparator_job,
                "scale_comparator_arm_id": scale_comparator_arm_id,
                "scale_comparator_use_target": scale_comparator_use_target,
                "n_boot": n_boot,
                "seed": seed,
                "mae_margin": mae_margin,
                "rho_margin": rho_margin,
                "flip_margin": flip_margin,
                "bias_margin": bias_margin,
                "functional_rho_floor": functional_rho_floor,
                "confidence": confidence,
                "fiber_mutual_rho_floor": fiber_mutual_rho_floor,
                "fiber_mutual_rho_sensitivity_floor": (
                    fiber_mutual_rho_sensitivity_floor),
                "fiber_min_rank_valid_fraction": fiber_min_rank_valid_fraction,
                "fiber_distinctness_floor": fiber_distinctness_floor,
                "include_controls": include_controls,
                "crossfit_only": crossfit_only,
                "cell_ids": list(cell_ids) if cell_ids is not None else None,
                "source_group_inference": bool(packet_root),
                "allow_fake_inputs": allow_fake_inputs,
            },
        )
        if sha256_file(arm_bank_path) != frozen_manifest.get("arm_bank_sha256"):
            raise ValueError("runner arm bank differs from frozen execution manifest")
        if (not packet_manifest_path
                or sha256_file(packet_manifest_path)
                != frozen_manifest.get("packet_manifest_sha256")):
            raise ValueError("runner packet manifest differs from frozen execution manifest")
        phase = partition_authorization["phase"]
        if selection_required_for_phase(frozen_manifest, phase):
            if selection_artifact_path is None:
                raise ValueError(
                    f"frozen runner phase {phase!r} requires its exact selection artifact"
                )
    index = load_public_index(executor_shard_root, partition)
    target_index = (load_public_index(target_shard_root, partition)
                    if target_shard_root else index)
    scale_comparator_index = (
        load_public_index(scale_comparator_shard_root, partition)
        if scale_comparator_shard_root else None
    )
    bank = json.loads(Path(arm_bank_path).read_text())
    frozen_jobs = (
        {row.get("id"): row for row in frozen_manifest.get("model_jobs", [])}
        if frozen_manifest is not None else {}
    )
    if frozen_manifest is not None and (
            small_job not in frozen_jobs or big_job not in frozen_jobs):
        raise ValueError("runner jobs are absent from the frozen execution manifest")
    if selection_artifact_path is not None:
        selection_scope = selection_scope_for_manifest(
            frozen_manifest,
            phase=partition_authorization["phase"],
            selection_path=selection_artifact_path,
        )
        frozen_selection_by_cell = load_lockbox_selection(
            selection_artifact_path,
            arm_bank_sha256=sha256_file(arm_bank_path),
            packet_manifest_sha256=sha256_file(packet_manifest_path),
            expected_partition=selection_scope[0],
            expected_phase=selection_scope[1],
            arm_bank=bank,
        )
    panel_identity_validation = validate_policy_cell_panel(
        bank.get("cells", []), context="policy-isomorphism arm bank"
    )
    available_cell_ids = {cell["id"] for cell in bank["cells"]}
    if cell_ids is not None:
        missing_cell_ids = sorted(set(cell_ids) - available_cell_ids)
        if missing_cell_ids:
            raise ValueError(f"requested cell ids are absent from arm bank: {missing_cell_ids}")
        selected_cells = [cell for cell in bank["cells"] if cell["id"] in set(cell_ids)]
    else:
        selected_cells = bank["cells"]
    if frozen_selection_by_cell is not None:
        selected_ids = {cell["id"] for cell in selected_cells}
        if selected_ids != set(frozen_selection_by_cell):
            raise ValueError(
                "runner cell panel differs from the frozen selection artifact"
            )
    cells = []
    for cell_index, cell in enumerate(selected_cells):
        domain, cell_id = cell["domain"], cell["id"]
        small = _average_repetitions(index[(small_job, domain)])
        big = _average_repetitions(target_index[(big_job, domain)])
        score_provenance_validation = None
        if frozen_manifest is not None:
            expected_execution_sha256 = partition_authorization[
                "execution_manifest_sha256"]
            expected_bank_sha256 = frozen_manifest["arm_bank_sha256"]
            expected_packet_sha256 = frozen_manifest["packet_manifest_sha256"]
            score_provenance_validation = {
                "small": _validate_frozen_score_bundle(
                    small,
                    label="small executor",
                    job_id=small_job,
                    manifest=frozen_manifest,
                    execution_manifest_sha256=expected_execution_sha256,
                    arm_bank_sha256=expected_bank_sha256,
                    packet_manifest_sha256=expected_packet_sha256,
                    allow_fake_inputs=allow_fake_inputs,
                ),
                "target": _validate_frozen_score_bundle(
                    big,
                    label="fixed target",
                    job_id=big_job,
                    manifest=frozen_manifest,
                    execution_manifest_sha256=expected_execution_sha256,
                    arm_bank_sha256=expected_bank_sha256,
                    packet_manifest_sha256=expected_packet_sha256,
                    allow_fake_inputs=allow_fake_inputs,
                ),
            }
        elif not allow_fake_inputs and (
                small.get("fake_backend") is True or big.get("fake_backend") is True):
            raise ValueError("policy analysis rejects fake-backend score shards")
        small_readout = small.get("readout_template_sha256")
        target_readout = big.get("readout_template_sha256")
        small_binary_readout = small.get("binary_readout")
        target_binary_readout = big.get("binary_readout")
        if (not small_readout or not target_readout
                or small_readout != target_readout):
            raise ValueError(
                f"executor/target readout identity mismatch for {cell_id}: "
                f"executor={small_readout!r} target={target_readout!r}"
            )
        if small_binary_readout != target_binary_readout:
            raise ValueError(
                f"executor/target binary readout mismatch for {cell_id}: "
                f"executor={small_binary_readout!r} target={target_binary_readout!r}"
            )
        small_orbits = _orbits(small["scores"], small["meta"], cell_id=cell_id)
        big_orbits = _orbits(big["scores"], big["meta"], cell_id=cell_id)
        arm_specs = {arm["id"]: arm for arm in cell["arms"]}
        score_cell_identity_validation = {
            "small": _validate_scored_breadth_identity(
                small["meta"], cell, label="small executor"
            ),
            "target": _validate_scored_breadth_identity(
                big["meta"], cell, label="fixed target"
            ),
        }
        scored_arm_panel_validation = None
        if frozen_manifest is not None:
            scored_arm_panel_validation = {
                "small": _validate_frozen_scored_arm_panel(
                    observed_arm_ids=set(small_orbits),
                    cell=cell,
                    model_job=frozen_jobs[small_job],
                    label=f"small executor/{cell_id}",
                    selected_arm_ids=(
                        frozen_selection_by_cell[cell_id]
                        if frozen_selection_by_cell is not None else None
                    ),
                ),
                "target": _validate_frozen_scored_arm_panel(
                    observed_arm_ids=set(big_orbits),
                    cell=cell,
                    model_job=frozen_jobs[big_job],
                    label=f"fixed target/{cell_id}",
                ),
            }
        prompt_bank_validation = validate_executor_prompt_arms(
            small["meta"], cell, arm_ids=set(small_orbits))
        prompt_bank_validation["selection_scope"] = (
            "authenticated scored-arm panel; frozen all/name-only policy or exact selection "
            "completeness is enforced before analysis"
        )
        target_hashes = big["hashes"]
        target_prompt_bank_validation = (
            validate_executor_prompt_arms(
                big["meta"], cell, arm_ids=set(big_orbits))
            if frozen_manifest is not None else {
                "valid": None,
                "scope": (
                    "legacy external target prompt manifest; not represented in the executor "
                    "arm bank"
                ),
            }
        )
        target = big_orbits[target_arm_id]
        source_group_data = (
            load_partition_source_groups(
                packet_root,
                packet_manifest_path,
                domain=domain,
                partition=partition,
                item_hashes=target_hashes,
            )
            if packet_root and packet_manifest_path else None
        )
        source_groups = (
            source_group_data["source_groups"] if source_group_data else None
        )
        cell_bootstrap_seed = seed + cell_index
        cell_bootstrap_context = (
            PolicyBootstrapContext(
                n_items=len(target_hashes),
                n_boot=n_boot,
                seed=cell_bootstrap_seed,
                bootstrap_clusters=source_groups,
            )
            if use_bootstrap_cache else None
        )
        aligned = {arm: _align_orbit(orbit, small["hashes"], target_hashes)
                   for arm, orbit in small_orbits.items()}
        larger_sparse = None
        scale_comparator_validation = None
        scale_comparator_shards = None
        scale_comparator_readout = None
        scale_comparator_binary_readout = None
        scale_comparator_execution_manifest = None
        if scale_comparator_use_target:
            larger_sparse = target
            scale_comparator_shards = big["shard_sha256"]
            scale_comparator_readout = target_readout
            scale_comparator_binary_readout = target_binary_readout
            scale_comparator_execution_manifest = big.get(
                "execution_manifest_sha256")
            scale_comparator_validation = {
                "valid": True,
                "source": "fixed_target_orbit",
                "job": big_job,
                "arm_id": target_arm_id,
                "item_hashes_identical": True,
                "readout_identical": True,
            }
        elif scale_comparator_index is not None:
            comparator = _average_repetitions(
                scale_comparator_index[(scale_comparator_job, domain)])
            scale_comparator_readout = comparator.get("readout_template_sha256")
            scale_comparator_binary_readout = comparator.get("binary_readout")
            scale_comparator_execution_manifest = comparator.get(
                "execution_manifest_sha256")
            if (not scale_comparator_readout
                    or scale_comparator_readout != target_readout):
                raise ValueError(
                    f"scale-comparator/target readout identity mismatch for {cell_id}: "
                    f"comparator={scale_comparator_readout!r} target={target_readout!r}"
                )
            if scale_comparator_binary_readout != target_binary_readout:
                raise ValueError(
                    f"scale-comparator/target binary readout mismatch for {cell_id}: "
                    f"comparator={scale_comparator_binary_readout!r} "
                    f"target={target_binary_readout!r}"
                )
            scale_comparator_validation = validate_executor_prompt_arms(
                comparator["meta"], cell, arm_ids={scale_comparator_arm_id})
            comparator_orbits = _orbits(
                comparator["scores"], comparator["meta"], cell_id=cell_id)
            score_cell_identity_validation["scale_comparator"] = (
                _validate_scored_breadth_identity(
                    comparator["meta"], cell, label="scale comparator"
                )
            )
            if scale_comparator_arm_id not in comparator_orbits:
                raise ValueError(
                    f"scale comparator lacks arm {scale_comparator_arm_id!r} for {cell_id}"
                )
            larger_sparse = _align_orbit(
                comparator_orbits[scale_comparator_arm_id],
                comparator["hashes"],
                target_hashes,
            )
            scale_comparator_shards = comparator["shard_sha256"]
        allowed = {"source_telling", "source_composed", "residual_teaching",
                   "ostensive_teaching", "formative_sequence", "composed",
                   "fitted_optimizer", "target_self_articulation",
                   "target_behavior_articulation", "target_residual_revision"}
        allowed |= {"target_hierarchical_articulation", "target_rank_articulation"}
        allowed |= SOURCE_HIERARCHY_PROVENANCES
        if include_controls:
            allowed |= {"wrong_construct_control", "inert_length_control"}
        eligible_arm_ids = [
            arm_id for arm_id in sorted(aligned)
            if arm_id != "name"
            and arm_specs[arm_id]["provenance"] in allowed
            and not (
                crossfit_only
                and arm_specs[arm_id].get("source_partition") == partition
            )
        ]
        scale_candidate_arm_ids = [
            arm_id for arm_id in eligible_arm_ids
            if not _is_control_row(arm_specs[arm_id])
        ]
        matched_pairs = []
        if include_controls:
            matched_pairs = [
                (control_id, control_spec.get("control_for"), control_spec)
                for control_id, control_spec in sorted(arm_specs.items())
                if control_id in eligible_arm_ids
                and control_spec.get("control_for") in scale_candidate_arm_ids
                and control_id in aligned
            ]
        scale_simultaneous_confidence = (
            1.0 - (1.0 - confidence) / len(scale_candidate_arm_ids)
            if scale_candidate_arm_ids else confidence
        )
        specificity_family_size = len(scale_candidate_arm_ids) + len(matched_pairs)
        specificity_simultaneous_confidence = (
            1.0 - (1.0 - confidence) / specificity_family_size
            if specificity_family_size else confidence
        )
        rows = []
        for arm_id in eligible_arm_ids:
            certificate = certify_policy_isomorphism(
                target, aligned[arm_id], sparse_orbit=aligned["name"],
                mae_margin=mae_margin, rho_margin=rho_margin,
                flip_margin=flip_margin, bias_margin=bias_margin,
                functional_rho_floor=functional_rho_floor,
                bootstrap_clusters=source_groups,
                n_boot=n_boot, seed=cell_bootstrap_seed, confidence=confidence,
                bootstrap_context=cell_bootstrap_context)
            scale_step_certificate = None
            scale_step_simultaneous_certificate = None
            scale_step_specificity_simultaneous_certificate = None
            if larger_sparse is not None and arm_id in scale_candidate_arm_ids:
                scale_kwargs = {
                    "bootstrap_clusters": source_groups,
                    "endpoint_mae_margin": mae_margin,
                    "endpoint_rho_margin": rho_margin,
                    "endpoint_flip_margin": flip_margin,
                    "endpoint_bias_margin": bias_margin,
                    "functional_rho_floor": functional_rho_floor,
                    "n_boot": n_boot,
                    "seed": cell_bootstrap_seed,
                    "bootstrap_context": cell_bootstrap_context,
                }
                scale_step_certificate = certify_scale_step_substitution(
                    target,
                    aligned["name"],
                    aligned[arm_id],
                    larger_sparse,
                    confidence=confidence,
                    **scale_kwargs,
                )
                scale_step_simultaneous_certificate = certify_scale_step_substitution(
                    target,
                    aligned["name"],
                    aligned[arm_id],
                    larger_sparse,
                    confidence=scale_simultaneous_confidence,
                    **scale_kwargs,
                )
                scale_step_specificity_simultaneous_certificate = (
                    certify_scale_step_substitution(
                        target,
                        aligned["name"],
                        aligned[arm_id],
                        larger_sparse,
                        confidence=specificity_simultaneous_confidence,
                        **scale_kwargs,
                    )
                )
            rows.append({
                "arm_id": arm_id,
                "channel": arm_specs[arm_id]["channel"],
                "provenance": arm_specs[arm_id]["provenance"],
                "control_for": arm_specs[arm_id].get("control_for"),
                "components": arm_specs[arm_id].get("components", []),
                "composition_degree": arm_specs[arm_id].get("composition_degree"),
                "n_address_units": arm_specs[arm_id].get("n_address_units"),
                "added_content_word_count": arm_specs[arm_id].get(
                    "added_content_word_count"),
                "semantic_content_word_count": arm_specs[arm_id][
                    "semantic_content_word_count"],
                "certificate": certificate,
                "scale_step_certificate": scale_step_certificate,
                "scale_step_simultaneous_certificate": (
                    scale_step_simultaneous_certificate),
                "scale_step_specificity_simultaneous_certificate": (
                    scale_step_specificity_simultaneous_certificate),
                "scale_step_multiplicity": (
                    {
                        "method": "Bonferroni over eligible non-control arms within cell",
                        "family_size": len(scale_candidate_arm_ids),
                        "familywise_confidence": scale_simultaneous_confidence,
                    }
                    if scale_step_certificate is not None else None
                ),
            })
        matched_control_certificates = []
        if include_controls:
            simultaneous_confidence = (
                1.0 - (1.0 - confidence) / len(matched_pairs)
                if matched_pairs else confidence
            )
            for control_id, source_id, control_spec in matched_pairs:
                pair_seed = seed + cell_index * 100 + len(matched_control_certificates)
                pair_context = (
                    PolicyBootstrapContext(
                        n_items=len(target_hashes),
                        n_boot=n_boot,
                        seed=pair_seed,
                        bootstrap_clusters=source_groups,
                    )
                    if use_bootstrap_cache else None
                )
                certificate = compare_articulation_to_matched_control(
                    target,
                    aligned[source_id],
                    aligned[control_id],
                    bootstrap_clusters=source_groups,
                    n_boot=n_boot,
                    seed=pair_seed,
                    confidence=confidence,
                    bootstrap_context=pair_context,
                )
                simultaneous_certificate = compare_articulation_to_matched_control(
                    target,
                    aligned[source_id],
                    aligned[control_id],
                    bootstrap_clusters=source_groups,
                    n_boot=n_boot,
                    seed=pair_seed,
                    confidence=simultaneous_confidence,
                    bootstrap_context=pair_context,
                )
                specificity_simultaneous_certificate = (
                    compare_articulation_to_matched_control(
                        target,
                        aligned[source_id],
                        aligned[control_id],
                        bootstrap_clusters=source_groups,
                        n_boot=n_boot,
                        seed=pair_seed,
                        confidence=specificity_simultaneous_confidence,
                        bootstrap_context=pair_context,
                    )
                )
                matched_control_certificates.append({
                    "source_arm_id": source_id,
                    "control_arm_id": control_id,
                    "control_provenance": control_spec["provenance"],
                    "certificate": certificate,
                    "simultaneous_certificate": simultaneous_certificate,
                    "specificity_simultaneous_certificate": (
                        specificity_simultaneous_certificate),
                    "multiplicity": {
                        "method": "Bonferroni within cell over matched source-control pairs",
                        "family_size": len(matched_pairs),
                        "familywise_confidence": simultaneous_confidence,
                    },
                })
        content_specific_scale_step = _content_specific_scale_memberships(
            rows, matched_control_certificates)
        content_specific_scale_step["multiplicity"] = {
            "method": (
                "Bonferroni within cell over the union of eligible non-control scale "
                "candidates and matched source-control pairs"
            ),
            "n_scale_candidates": len(scale_candidate_arm_ids),
            "n_matched_source_control_pairs": len(matched_pairs),
            "family_size": specificity_family_size,
            "familywise_confidence": specificity_simultaneous_confidence,
        }
        content_specific_joint_fiber = _content_specific_joint_fiber(
            candidate_arm_ids=scale_candidate_arm_ids,
            content_specific_membership=content_specific_scale_step,
            arm_specs=arm_specs,
            arm_orbits=aligned,
            bootstrap_clusters=source_groups,
            n_boot=n_boot,
            seed=seed + 700_000 + cell_index * 10_000,
            confidence=confidence,
            mutual_rho_floor=fiber_mutual_rho_floor,
            mutual_rho_sensitivity_floor=fiber_mutual_rho_sensitivity_floor,
            min_rank_valid_fraction=fiber_min_rank_valid_fraction,
            mutual_mae_margin=mae_margin,
            mutual_flip_margin=flip_margin,
            mutual_bias_margin=bias_margin,
            distinctness_floor=fiber_distinctness_floor,
            use_bootstrap_cache=use_bootstrap_cache,
        )
        fiber = summarize_isomorphism_fiber(rows, arm_specs, aligned)
        cells.append({**_identity_payload(
                          cell, context=f"policy-isomorphism report/{cell_id}"),
                      "target_job": big_job,
                      "small_job": small_job, "n_items": len(target_hashes),
                      "score_cell_identity_validation": (
                          score_cell_identity_validation),
                      "scored_arm_panel_validation": scored_arm_panel_validation,
                      "executor_prompt_bank_validation": prompt_bank_validation,
                      "target_prompt_bank_validation": target_prompt_bank_validation,
                      "score_provenance_validation": score_provenance_validation,
                      "scale_comparator_validation": scale_comparator_validation,
                      "source_group_validation": (
                          source_group_data["validation"] if source_group_data else None
                      ),
                      "rows": rows, "fiber": fiber,
                      "matched_control_certificates": matched_control_certificates,
                      "content_specific_scale_step": content_specific_scale_step,
                      "content_specific_joint_fiber": content_specific_joint_fiber,
                      "target_shards": big["shard_sha256"],
                      "small_shards": small["shard_sha256"],
                      "scale_comparator_shards": scale_comparator_shards,
                      "target_readout_template_sha256": big[
                          "readout_template_sha256"],
                      "small_readout_template_sha256": small[
                          "readout_template_sha256"],
                      "scale_comparator_readout_template_sha256": (
                          scale_comparator_readout),
                      "target_binary_readout": target_binary_readout,
                      "small_binary_readout": small_binary_readout,
                      "scale_comparator_binary_readout": (
                          scale_comparator_binary_readout),
                      "target_score_execution_manifest_sha256": big.get(
                          "execution_manifest_sha256"),
                      "small_score_execution_manifest_sha256": small.get(
                          "execution_manifest_sha256"),
                      "scale_comparator_score_execution_manifest_sha256": (
                          scale_comparator_execution_manifest)})
    return {
        "schema": _POLICY_REPORT_SCHEMA,
        "estimand": (
            "fixed-target small-executor policy reconstruction plus optional direct replication "
            "of a separately declared larger sparse scale endpoint"
        ),
        "partition": partition, "arm_bank_sha256": sha256_file(arm_bank_path),
        "cell_panel_identity_validation": panel_identity_validation,
        "partition_authorization": partition_authorization,
        "additional_artifact_validation": additional_artifact_validation,
        "frozen_invocation_validation": frozen_invocation_validation,
        "executor_shard_root": executor_shard_root,
        "target_shard_root": target_shard_root or executor_shard_root,
        "scale_comparator_shard_root": scale_comparator_shard_root,
        "scale_comparator": {
            "enabled": bool(scale_comparator_use_target or scale_comparator_shard_root),
            "use_fixed_target_orbit": scale_comparator_use_target,
            "job": big_job if scale_comparator_use_target else scale_comparator_job,
            "arm_id": (
                target_arm_id if scale_comparator_use_target
                else scale_comparator_arm_id
            ),
        },
        "analysis_implementation": _analysis_implementation(),
        "source_group_inference": {
            "enabled": bool(packet_root),
            "packet_root": packet_root,
            "packet_manifest_path": packet_manifest_path,
            "packet_manifest_sha256": (
                sha256_file(packet_manifest_path) if packet_manifest_path else None
            ),
            "resampling_unit": (
                "source_group_with_all_member_items_retained"
                if packet_root else "item"
            ),
            "point_estimand": "item-weighted policy metrics on the scored panel",
        },
        "config": {"small_job": small_job, "big_job": big_job, "n_boot": n_boot,
                   "target_arm_id": target_arm_id,
                   "scale_comparator_job": scale_comparator_job,
                   "scale_comparator_arm_id": scale_comparator_arm_id,
                   "scale_comparator_use_target": scale_comparator_use_target,
                   "seed": seed, "mae_margin": mae_margin, "rho_margin": rho_margin,
                   "flip_margin": flip_margin, "bias_margin": bias_margin,
                   "functional_rho_floor": functional_rho_floor,
                   "confidence": confidence,
                   "fiber_mutual_rho_floor": fiber_mutual_rho_floor,
                   "fiber_mutual_rho_sensitivity_floor": (
                       fiber_mutual_rho_sensitivity_floor),
                   "fiber_min_rank_valid_fraction": fiber_min_rank_valid_fraction,
                   "fiber_distinctness_floor": fiber_distinctness_floor,
                   "include_controls": include_controls, "crossfit_only": crossfit_only,
                   "cell_ids": list(cell_ids) if cell_ids is not None else None},
        "cells": cells,
        "summary": {
            "n_cells": len(cells),
            "n_arms": sum(len(cell["rows"]) for cell in cells),
            "n_isomorphic": sum(cell["fiber"]["n_isomorphic"] for cell in cells),
            "n_equal_but_different_pairs": sum(
                cell["fiber"]["n_equal_but_different_pairs"] for cell in cells),
            "n_observed_functional_equal_but_different_pairs": sum(
                cell["fiber"]["n_observed_functional_equal_but_different_pairs"]
                for cell in cells),
            "n_certified_functional_equal_but_different_pairs": sum(
                cell["fiber"]["n_certified_functional_equal_but_different_pairs"]
                for cell in cells),
            "n_rescues": sum(row["certificate"]["articulation_rescue"]
                             for cell in cells for row in cell["rows"]),
            "n_observed_functional_ordinal": sum(
                row["certificate"]["functional"][
                    "observed_functional_ordinal_isomorphism"]
                for cell in cells for row in cell["rows"]),
            "n_certified_functional_ordinal": sum(
                row["certificate"]["functional"][
                    "certified_functional_ordinal_isomorphism"]
                for cell in cells for row in cell["rows"]),
            "n_observed_functional_substitutions": sum(
                row["certificate"]["functional"][
                    "observed_functional_policy_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_certified_functional_substitutions": sum(
                row["certificate"]["functional"][
                    "certified_functional_policy_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_observed_local_primary_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["observed"][
                    "local_primary_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_certified_local_primary_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["certified"][
                    "local_primary_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_simultaneous_certified_local_primary_scale_substitutions": sum(
                bool(row.get("scale_step_simultaneous_certificate"))
                and row["scale_step_simultaneous_certificate"]["evidence"]["certified"][
                    "local_primary_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_observed_one_sided_noninferiority_recoveries": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["observed"][
                    "local_primary_one_sided_noninferiority_recovery"]
                for cell in cells for row in cell["rows"]),
            "n_certified_one_sided_noninferiority_recoveries": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["certified"][
                    "local_primary_one_sided_noninferiority_recovery"]
                for cell in cells for row in cell["rows"]),
            "n_simultaneous_certified_one_sided_noninferiority_recoveries": sum(
                bool(row.get("scale_step_simultaneous_certificate"))
                and row["scale_step_simultaneous_certificate"]["evidence"]["certified"][
                    "local_primary_one_sided_noninferiority_recovery"]
                for cell in cells for row in cell["rows"]),
            "n_observed_two_sided_equivalence_recoveries": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["observed"][
                    "local_primary_two_sided_equivalence_recovery"]
                for cell in cells for row in cell["rows"]),
            "n_certified_two_sided_equivalence_recoveries": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["certified"][
                    "local_primary_two_sided_equivalence_recovery"]
                for cell in cells for row in cell["rows"]),
            "n_simultaneous_certified_two_sided_equivalence_recoveries": sum(
                bool(row.get("scale_step_simultaneous_certificate"))
                and row["scale_step_simultaneous_certificate"]["evidence"]["certified"][
                    "local_primary_two_sided_equivalence_recovery"]
                for cell in cells for row in cell["rows"]),
            "n_observed_functional_target_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["observed"][
                    "functional_target_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_certified_functional_target_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["certified"][
                    "functional_target_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_simultaneous_certified_functional_target_scale_substitutions": sum(
                bool(row.get("scale_step_simultaneous_certificate"))
                and row["scale_step_simultaneous_certificate"]["evidence"]["certified"][
                    "functional_target_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_observed_functional_endpoint_isomorphic_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["observed"][
                    "local_functional_endpoint_isomorphic_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_certified_functional_endpoint_isomorphic_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["certified"][
                    "local_functional_endpoint_isomorphic_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_simultaneous_certified_functional_endpoint_isomorphic_scale_substitutions": sum(
                bool(row.get("scale_step_simultaneous_certificate"))
                and row["scale_step_simultaneous_certificate"]["evidence"]["certified"][
                    "local_functional_endpoint_isomorphic_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_observed_functional_endpoint_equivalent_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["observed"][
                    "local_functional_endpoint_equivalent_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_certified_functional_endpoint_equivalent_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["certified"][
                    "local_functional_endpoint_equivalent_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_simultaneous_certified_functional_endpoint_equivalent_scale_substitutions": sum(
                bool(row.get("scale_step_simultaneous_certificate"))
                and row["scale_step_simultaneous_certificate"]["evidence"]["certified"][
                    "local_functional_endpoint_equivalent_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_observed_near_identity_endpoint_isomorphic_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["observed"][
                    "local_near_identity_isomorphic_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_certified_near_identity_endpoint_isomorphic_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["certified"][
                    "local_near_identity_isomorphic_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_simultaneous_certified_near_identity_endpoint_isomorphic_scale_substitutions": sum(
                bool(row.get("scale_step_simultaneous_certificate"))
                and row["scale_step_simultaneous_certificate"]["evidence"]["certified"][
                    "local_near_identity_isomorphic_scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_observed_joint_fixed_target_endpoint_isomorphic_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["observed"][
                    "joint_fixed_target_and_endpoint_functional_isomorphic_"
                    "scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_certified_joint_fixed_target_endpoint_isomorphic_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["certified"][
                    "joint_fixed_target_and_endpoint_functional_isomorphic_"
                    "scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_simultaneous_certified_joint_fixed_target_endpoint_isomorphic_"
            "scale_substitutions": sum(
                bool(row.get("scale_step_simultaneous_certificate"))
                and row["scale_step_simultaneous_certificate"]["evidence"]["certified"][
                    "joint_fixed_target_and_endpoint_functional_isomorphic_"
                    "scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_observed_joint_fixed_target_endpoint_equivalent_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["observed"][
                    "joint_fixed_target_and_endpoint_functional_equivalent_"
                    "scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_certified_joint_fixed_target_endpoint_equivalent_scale_substitutions": sum(
                bool(row.get("scale_step_certificate"))
                and row["scale_step_certificate"]["evidence"]["certified"][
                    "joint_fixed_target_and_endpoint_functional_equivalent_"
                    "scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_simultaneous_certified_joint_fixed_target_endpoint_equivalent_"
            "scale_substitutions": sum(
                bool(row.get("scale_step_simultaneous_certificate"))
                and row["scale_step_simultaneous_certificate"]["evidence"]["certified"][
                    "joint_fixed_target_and_endpoint_functional_equivalent_"
                    "scale_substitution"]
                for cell in cells for row in cell["rows"]),
            "n_observed_content_specific_joint_fixed_target_endpoint_isomorphic_"
            "scale_substitutions": sum(
                len(cell["content_specific_scale_step"][
                    "observed_joint_fixed_target_endpoint_isomorphic_members"])
                for cell in cells),
            "n_certified_content_specific_joint_fixed_target_endpoint_isomorphic_"
            "scale_substitutions": sum(
                len(cell["content_specific_scale_step"][
                    "certified_joint_fixed_target_endpoint_isomorphic_members"])
                for cell in cells),
            "n_simultaneous_certified_content_specific_joint_fixed_target_endpoint_"
            "isomorphic_scale_substitutions": sum(
                len(cell["content_specific_scale_step"][
                    "simultaneous_certified_joint_fixed_target_endpoint_"
                    "isomorphic_members"])
                for cell in cells),
            "n_observed_content_specific_joint_fixed_target_endpoint_equivalent_"
            "scale_substitutions": sum(
                len(cell["content_specific_scale_step"][
                    "observed_joint_fixed_target_endpoint_equivalent_members"])
                for cell in cells),
            "n_certified_content_specific_joint_fixed_target_endpoint_equivalent_"
            "scale_substitutions": sum(
                len(cell["content_specific_scale_step"][
                    "certified_joint_fixed_target_endpoint_equivalent_members"])
                for cell in cells),
            "n_simultaneous_certified_content_specific_joint_fixed_target_endpoint_"
            "equivalent_scale_substitutions": sum(
                len(cell["content_specific_scale_step"][
                    "simultaneous_certified_joint_fixed_target_endpoint_"
                    "equivalent_members"])
                for cell in cells),
            "n_observed_content_specific_H_fiber_pairs": sum(
                len(cell["content_specific_joint_fiber"]["observed_H_fiber_pairs"])
                for cell in cells),
            "n_certified_content_specific_H_fiber_pairs": sum(
                len(cell["content_specific_joint_fiber"]["certified_H_fiber_pairs"])
                for cell in cells),
            "n_simultaneous_certified_content_specific_H_fiber_pairs": sum(
                len(cell["content_specific_joint_fiber"][
                    "simultaneous_certified_H_fiber_pairs"])
                for cell in cells),
            "n_simultaneous_certified_content_specific_H_fiber_eq_pairs": sum(
                len(cell["content_specific_joint_fiber"][
                    "simultaneous_certified_H_fiber_eq_pairs"])
                for cell in cells),
            "n_observed_content_specific_H_fiber_vec_pairs": sum(
                len(cell["content_specific_joint_fiber"]["observed_H_fiber_vec_pairs"])
                for cell in cells),
            "n_certified_content_specific_H_fiber_vec_pairs": sum(
                len(cell["content_specific_joint_fiber"]["certified_H_fiber_vec_pairs"])
                for cell in cells),
            "n_simultaneous_certified_content_specific_H_fiber_vec_pairs": sum(
                len(cell["content_specific_joint_fiber"][
                    "simultaneous_certified_H_fiber_vec_pairs"])
                for cell in cells),
            "n_simultaneous_certified_content_specific_H_fiber_vec_eq_pairs": sum(
                len(cell["content_specific_joint_fiber"][
                    "simultaneous_certified_H_fiber_vec_eq_pairs"])
                for cell in cells),
        },
        "claim_boundary": ("Public partitions support search/validation only. A fiber member is a "
                           "behavioral certificate under declared margins, not proof that its text "
                           "is a faithful articulation until provenance/semantic review passes. "
                           "Functional ordinal status is an absolute-rank approximation tier; "
                           "policy_isomorphic remains the stricter target-self-band tier. When a "
                           "scale comparator is supplied, the legacy local-primary/vector keys "
                           "are explicitly one-sided target-relative noninferiority recovery, "
                           "not isomorphism. Direct larger-endpoint functional isomorphism, its "
                           "stricter two-sided endpoint-equivalent tier, fixed-target functional "
                           "fidelity, and direct target-self-band identity remain separately "
                           "reported, with Bonferroni intervals over eligible non-control "
                           "articulation arms. Content-specific joint grades additionally "
                           "require rank-and-MAE superiority to every matched inert-length and "
                           "wrong-construct control, with both types present; their simultaneous "
                           "tier uses one union-family Bonferroni adjustment over scale arms and "
                           "matched contrasts. The content-specific H_fiber additionally "
                           "requires two frozen surface-distinct H_J routes and an interval-"
                           "certified mutual quotient-rank floor with at least the frozen valid-"
                           "draw fraction; point-only mutual rank is never labelled confirmed. "
                           "H_fiber is convergent ordinal reconstruction, not numerical policy "
                           "equality. The nested H_fiber^vec additionally requires mutual "
                           "form-quotient MAE, threshold-flip, and absolute-bias equivalence; it "
                           "still does not imply matched-form or semantic equality."),
    }


_BREADTH_DECOMPOSITION_SCHEMA = "tacit_breadth_decomposition_report/v3"
_BREADTH_LEVELS = ("R1", "R2", "R3")
_BREADTH_BINARY_OUTCOMES = (
    "observed_content_specific_joint_functional_substitution",
    "certified_content_specific_joint_functional_substitution",
    "simultaneous_certified_content_specific_joint_functional_substitution",
    "observed_content_specific_joint_equivalent_substitution",
    "certified_content_specific_joint_equivalent_substitution",
    "simultaneous_certified_content_specific_joint_equivalent_substitution",
    "observed_functional_equal_but_different_fiber",
    "certified_functional_equal_but_different_fiber",
    "simultaneous_certified_functional_equal_but_different_fiber",
    "observed_vector_equal_but_different_fiber",
    "certified_vector_equal_but_different_fiber",
    "simultaneous_certified_vector_equal_but_different_fiber",
    "observed_matched_control_improvement",
    "certified_matched_control_improvement",
    "simultaneous_certified_matched_control_improvement",
)
_BREADTH_COORDINATES = (
    "best_raw_joint_adverse_quotient_rho_floor",
    "best_observed_functional_rho_capacity",
    "best_certified_functional_rho_capacity",
    "best_adverse_rho_point",
    "best_quotient_rho_point",
    "lowest_adverse_mae_tvd",
    "best_fraction_rank_target_self_gap_closed",
    "best_fraction_mae_target_self_gap_closed",
    "best_fraction_flip_target_self_gap_closed",
    "best_fraction_absolute_bias_target_self_gap_closed",
    "best_native_scale_step_rank_closure",
    "best_native_scale_step_mae_closure",
    "best_native_scale_step_flip_closure",
    "best_native_scale_step_absolute_bias_closure",
)


def _load_breadth_json(value: dict | str | Path, *, label: str) -> tuple[dict, dict]:
    """Load a report/panel while retaining a portable, content-bound input record."""
    if isinstance(value, dict):
        payload = value
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return payload, {
            "source": "in_memory",
            "canonical_content_sha256": sha256_bytes(canonical),
        }
    path = Path(value)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload, {
        "source": "file",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def _validate_breadth_decomposition_inputs(
        report: dict, panel: dict) -> list[tuple[dict, dict]]:
    """Require the complete frozen 11 x 3 x 30 panel in its original order."""
    if panel.get("schema") != "tacit_breadth_metric_panel/v3":
        raise ValueError("breadth decomposition requires a v3 frozen metric panel")
    declared_hash = panel.get("panel_content_sha256")
    panel_core = {key: value for key, value in panel.items()
                  if key != "panel_content_sha256"}
    observed_hash = sha256_bytes(json.dumps(
        panel_core, sort_keys=True, separators=(",", ":")
    ).encode())
    if not declared_hash or declared_hash != observed_hash:
        raise ValueError("metric panel content hash does not recompute")
    if report.get("schema") != _POLICY_REPORT_SCHEMA:
        raise ValueError("breadth decomposition requires the exact v5 policy report schema")
    if report.get("scale_comparator", {}).get("enabled") is not True:
        raise ValueError(
            "breadth scale-articulation decomposition requires a declared scale comparator"
        )
    if report.get("config", {}).get("include_controls") is not True:
        raise ValueError(
            "breadth content-specific decomposition requires frozen matched controls"
        )
    tasks = panel.get("tasks")
    levels = panel.get("levels")
    if (not isinstance(tasks, list) or len(tasks) != 11
            or any(not isinstance(task, str) or not task for task in tasks)
            or len(tasks) != len(set(tasks))):
        raise ValueError("frozen breadth panel must declare exactly 11 unique tasks")
    if levels != list(_BREADTH_LEVELS):
        raise ValueError("frozen breadth panel must declare R1/R2/R3 in order")
    if panel.get("n_per_task_level") != 30:
        raise ValueError("frozen breadth decomposition requires exactly 30 metrics per cell")
    panel_cells = panel.get("cells")
    report_cells = report.get("cells")
    if not isinstance(panel_cells, list) or not isinstance(report_cells, list):
        raise ValueError("breadth panel/report cells must be lists")
    if (panel.get("n_cells") != 990 or len(panel_cells) != 990
            or len(report_cells) != 990):
        raise ValueError("breadth decomposition requires the complete 990-cell panel/report")
    validate_policy_cell_panel(panel_cells, context="breadth metric panel")
    validate_policy_cell_panel(report_cells, context="breadth policy report")
    panel_ids = [cell.get("id") for cell in panel_cells]
    report_ids = [cell.get("cell_id") for cell in report_cells]
    if panel_ids != report_ids:
        raise ValueError(
            "breadth report must preserve the frozen metric-panel cell identities and order"
        )
    expected_pairs = [(task, level) for task in tasks for level in _BREADTH_LEVELS]
    observed_pairs = [(cell.get("task"), cell.get("level")) for cell in panel_cells]
    for task, level in expected_pairs:
        indexes = [index for index, pair in enumerate(observed_pairs)
                   if pair == (task, level)]
        if len(indexes) != 30:
            raise ValueError(f"{task}/{level}: expected exactly 30 frozen metrics")
        if indexes != list(range(indexes[0], indexes[0] + 30)):
            raise ValueError(f"{task}/{level}: panel metrics are not contiguous")
    if [pair for pair in expected_pairs for _ in range(30)] != observed_pairs:
        raise ValueError("breadth panel task/level order differs from its frozen declaration")

    # This readout is the prospective breadth result, not an exploratory summary of the
    # all-arm search fold.  Recheck the self-contained closure records written by the frozen
    # runner before treating any cell certificate as validation evidence.
    authorization = report.get("partition_authorization")
    invocation = report.get("frozen_invocation_validation")
    runner = invocation.get("runner") if isinstance(invocation, dict) else None
    if (not isinstance(authorization, dict)
            or authorization.get("phase") != "lockbox"
            or report.get("partition") != "tacit_breadth_validation"
            or authorization.get("sealed_partition_authorized") is not True
            or authorization.get("lockbox_release_validation", {}).get("valid") is not True):
        raise ValueError(
            "breadth decomposition requires the released frozen lockbox phase/partition"
        )
    if (not isinstance(invocation, dict) or invocation.get("valid") is not True
            or not isinstance(runner, dict)):
        raise ValueError("breadth report lacks frozen-runner invocation closure")
    if (runner.get("cell_ids") != panel_ids
            or runner.get("include_controls") is not True
            or runner.get("source_group_inference") is not True
            or runner.get("allow_fake_inputs") is not False
            or runner.get("scale_comparator_use_target") is not True
            or runner.get("functional_rho_floor") != 0.70):
        raise ValueError("breadth report runner differs from the frozen validation design")
    config = report.get("config")
    if not isinstance(config, dict):
        raise ValueError("breadth report config is missing")
    config_keys = (
        "small_job", "big_job", "target_arm_id", "scale_comparator_job",
        "scale_comparator_arm_id", "scale_comparator_use_target", "n_boot", "seed",
        "mae_margin", "rho_margin", "flip_margin", "bias_margin",
        "functional_rho_floor", "confidence", "fiber_mutual_rho_floor",
        "fiber_mutual_rho_sensitivity_floor", "fiber_min_rank_valid_fraction",
        "fiber_distinctness_floor", "include_controls", "crossfit_only", "cell_ids",
    )
    changed_config = sorted(
        key for key in config_keys if key not in runner or config.get(key) != runner[key]
    )
    if changed_config:
        raise ValueError(
            f"breadth report config differs from frozen invocation: {changed_config}"
        )
    if (report.get("cell_panel_identity_validation", {}).get("valid") is not True
            or report.get("source_group_inference", {}).get("enabled") is not True):
        raise ValueError("breadth report lacks cell-panel or source-group closure")

    required_design_fields = set(BREADTH_DESIGN_FIELDS)
    for panel_cell, report_cell in zip(panel_cells, report_cells):
        panel_identity = policy_cell_identity(
            panel_cell, context="breadth metric panel cell")
        report_identity = policy_cell_identity(
            report_cell, context="breadth policy report cell")
        if panel_identity != report_identity:
            raise ValueError(
                f"breadth identity changed for {panel_cell.get('id')!r}"
            )
        missing_panel = sorted(required_design_fields - set(panel_cell))
        missing_report = sorted(required_design_fields - set(report_cell))
        if missing_panel or missing_report:
            raise ValueError(
                f"{panel_cell['id']}: missing frozen breadth design fields; "
                f"panel={missing_panel} report={missing_report}"
            )
        changed = sorted(
            key for key in required_design_fields
            if panel_cell[key] != report_cell[key]
        )
        if changed:
            raise ValueError(
                f"{panel_cell['id']}: report changes frozen design fields {changed}"
            )
        for key in ("nominal_poststratification_weight", "stratum_coverage_fraction"):
            value = panel_cell[key]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{panel_cell['id']}: invalid {key}")
        for key in ("stratum_population_n", "stratum_selected_n", "leaf_support_count"):
            value = panel_cell[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{panel_cell['id']}: invalid {key}")
        if panel_cell["stratum_selected_n"] > panel_cell["stratum_population_n"]:
            raise ValueError(f"{panel_cell['id']}: selected stratum exceeds its population")
        expected_probability = (
            panel_cell["stratum_selected_n"] / panel_cell["stratum_population_n"]
        )
        if (not np.isclose(panel_cell["stratum_coverage_fraction"], expected_probability)
                or not np.isclose(
                    panel_cell["nominal_poststratification_weight"],
                    1.0 / expected_probability)):
            raise ValueError(
                f"{panel_cell['id']}: nominal post-stratification factor is not the frozen "
                "inverse stratum-coverage fraction"
            )
        for key in (
                "dependency_component_size", "provenance_component_size",
                "task_raw_provenance_component_size",
                "source_assignment_multiplicity_max",
                "provenance_assignment_multiplicity_max",
                "task_raw_provenance_assignment_multiplicity_max"):
            value = panel_cell[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{panel_cell['id']}: invalid {key}")
        for key in (
                "dependency_degree", "provenance_overlap_degree",
                "task_raw_provenance_overlap_degree"):
            value = panel_cell[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{panel_cell['id']}: invalid {key}")
        if (not isinstance(panel_cell["source_index"], int)
                or isinstance(panel_cell["source_index"], bool)
                or panel_cell["source_index"] < 0):
            raise ValueError(f"{panel_cell['id']}: invalid source_index")
        for key in (
                "dependency_component_id", "provenance_component_id",
                "task_raw_provenance_component_id", "source_kind",
                "source_path", "source_sha256", "leaf_support_sha256",
                "breadth_stratum"):
            if not isinstance(panel_cell[key], str) or not panel_cell[key]:
                raise ValueError(f"{panel_cell['id']}: invalid {key}")

        closures = (
            report_cell.get("executor_prompt_bank_validation"),
            report_cell.get("target_prompt_bank_validation"),
            report_cell.get("source_group_validation"),
            report_cell.get("scale_comparator_validation"),
        )
        arm_panel = report_cell.get("scored_arm_panel_validation")
        identity = report_cell.get("score_cell_identity_validation")
        provenance = report_cell.get("score_provenance_validation")
        if (any(not isinstance(value, dict) or value.get("valid") is not True
                for value in closures)
                or not isinstance(arm_panel, dict)
                or set(arm_panel) != {"small", "target"}
                or arm_panel["small"].get("valid") is not True
                or arm_panel["small"].get("arm_policy") != "frozen_selection"
                or arm_panel["target"].get("valid") is not True
                or arm_panel["target"].get("arm_policy") != "name_only"
                or not isinstance(identity, dict)
                or set(identity) != {"small", "target"}
                or any(value.get("valid") is not True for value in identity.values())
                or not isinstance(provenance, dict)
                or set(provenance) != {"small", "target"}
                or any(value.get("valid") is not True
                       or value.get("fake_backend") is not False
                       for value in provenance.values())):
            raise ValueError(
                f"{panel_cell['id']}: report lacks authenticated validation-score closure"
            )

    for field, size_field, scope_fields in (
            ("dependency_component_id", "dependency_component_size",
             ("task", "level", "bucket")),
            ("provenance_component_id", "provenance_component_size",
             ("task", "level", "bucket")),
            ("task_raw_provenance_component_id",
             "task_raw_provenance_component_size", ("task", "bucket"))):
        groups: dict[tuple[str, ...], list[dict]] = {}
        for cell in panel_cells:
            key = (*[cell[value] for value in scope_fields], cell[field])
            groups.setdefault(key, []).append(cell)
        for key, members in groups.items():
            declared = {member[size_field] for member in members}
            if (len(declared) != 1 or len(members) > next(iter(declared))
                    or field == "task_raw_provenance_component_id"
                    and len(members) != next(iter(declared))):
                raise ValueError(
                    f"breadth component {key!r} has inconsistent or impossible {size_field}"
                )

    terminal = panel.get("terminal_frontier_sensitivities")
    if not isinstance(terminal, list):
        raise ValueError("metric panel omits terminal-frontier sensitivity audits")
    terminal_pairs = [(row.get("task"), row.get("level"))
                      for row in terminal if isinstance(row, dict)]
    if (len(terminal_pairs) != 33
            or terminal_pairs != expected_pairs
            or len(terminal_pairs) != len(set(terminal_pairs))):
        raise ValueError(
            "metric panel terminal-frontier audits must cover the exact task/level panel"
        )
    for audit in terminal:
        available = audit.get("available")
        if not isinstance(available, bool):
            raise ValueError("terminal-frontier availability must be boolean")
        if audit["level"] == "R1" and available:
            raise ValueError("R1 terminal-frontier sensitivity must remain unavailable")
        if audit["level"] in {"R2", "R3"} and not available:
            raise ValueError("R2/R3 terminal-frontier sensitivity unexpectedly unavailable")
        if available:
            n_frontier = audit.get("n_frontier_nodes")
            n_eligible = audit.get("n_eligible_nodes")
            if (not isinstance(n_frontier, int) or isinstance(n_frontier, bool)
                    or not isinstance(n_eligible, int) or isinstance(n_eligible, bool)
                    or n_frontier < 1 or not 30 <= n_eligible <= n_frontier
                    or audit.get("global_partition_claim") is not False):
                raise ValueError("available terminal-frontier audit has invalid counts/claim")
        elif not isinstance(audit.get("reason"), str) or not audit["reason"]:
            raise ValueError("unavailable terminal-frontier audit lacks its reason")
    return list(zip(panel_cells, report_cells))


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _error_gap_closure(*, sparse: float | None, candidate: float | None,
                       endpoint: float | None) -> float | None:
    values = tuple(_finite_or_none(value) for value in (sparse, candidate, endpoint))
    if any(value is None for value in values):
        return None
    sparse_value, candidate_value, endpoint_value = values
    gap = sparse_value - endpoint_value
    if gap <= 0.0:
        return None
    return float((sparse_value - candidate_value) / gap)


def _rank_gap_closure(*, sparse: float | None, candidate: float | None,
                      endpoint: float | None) -> float | None:
    values = tuple(_finite_or_none(value) for value in (sparse, candidate, endpoint))
    if any(value is None for value in values):
        return None
    sparse_value, candidate_value, endpoint_value = values
    gap = endpoint_value - sparse_value
    if gap <= 0.0:
        return None
    return float((candidate_value - sparse_value) / gap)


def _scale_step_coordinate_closures(scale_certificate: dict | None) -> dict:
    result = {
        "rank": None, "mae": None, "flip": None, "absolute_bias": None,
    }
    if not scale_certificate:
        return result
    point = scale_certificate.get("point", {})
    small = point.get("small_sparse", {})
    candidate = point.get("candidate", {})
    larger = point.get("larger_sparse", {})
    result["rank"] = _rank_gap_closure(
        sparse=small.get("spearman"), candidate=candidate.get("spearman"),
        endpoint=larger.get("spearman"),
    )
    for output_key, metric in (
            ("mae", "mae_tvd"),
            ("flip", "binary_flip_rate"),
            ("absolute_bias", "absolute_bias")):
        result[output_key] = _error_gap_closure(
            sparse=small.get(metric), candidate=candidate.get(metric),
            endpoint=larger.get(metric),
        )
    declared = scale_certificate.get("descriptive_step_closure", {})
    for key in ("rank", "mae"):
        declared_value = (declared.get(key, {})
                          .get("chi_articulation_gain_over_native_gap"))
        if declared_value is not None and not np.isclose(
                declared_value, result[key], equal_nan=True):
            raise ValueError(f"scale certificate changes its {key} closure identity")
    return result


def _certificate_coordinate_closures(certificate: dict) -> dict:
    capacity = _functional_floor_capacity(certificate)
    point = certificate.get("point", {})
    candidate = point.get("candidate_robust", {})
    endpoint = point.get("target_self_robust", {})
    sparse = certificate.get("small_sparse_point", {}).get("candidate_robust", {})
    return {
        **capacity,
        "fraction_flip_target_self_gap_closed": _error_gap_closure(
            sparse=sparse.get("binary_flip_rate"),
            candidate=candidate.get("binary_flip_rate"),
            endpoint=endpoint.get("binary_flip_rate"),
        ),
        "fraction_absolute_bias_target_self_gap_closed": _error_gap_closure(
            sparse=sparse.get("absolute_bias"),
            candidate=candidate.get("absolute_bias"),
            endpoint=endpoint.get("absolute_bias"),
        ),
    }


def _membership_list(payload: dict, key: str, *, cell_id: str,
                     candidate_ids: set[str]) -> list[str]:
    value = payload.get(key)
    if (not isinstance(value, list)
            or any(not isinstance(arm_id, str) or arm_id not in candidate_ids
                   for arm_id in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{cell_id}: invalid or unknown members in {key}")
    return value


def _pair_members(payload: dict, key: str, *, cell_id: str,
                  candidate_ids: set[str]) -> list[tuple[str, str]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{cell_id}: invalid fiber pair list {key}")
    pairs = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"{cell_id}: malformed fiber pair in {key}")
        left, right = row.get("left"), row.get("right")
        if (not isinstance(left, str) or not isinstance(right, str)
                or left == right or left not in candidate_ids or right not in candidate_ids):
            raise ValueError(f"{cell_id}: unknown fiber endpoints in {key}")
        pairs.append(tuple(sorted((left, right))))
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"{cell_id}: duplicate fiber pairs in {key}")
    return pairs


def _fiber_topology(pairs: list[tuple[str, str]], *, rows_by_id: dict[str, dict],
                    certificates_by_pair: dict[tuple[str, str], dict]) -> dict:
    nodes = sorted({value for pair in pairs for value in pair})
    adjacency = {node: set() for node in nodes}
    channel_pair_counts: dict[str, int] = {}
    structural_basis_counts: dict[str, int] = {}
    distances = []
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
        channels = sorted((str(rows_by_id[left].get("channel")),
                           str(rows_by_id[right].get("channel"))))
        channel_key = " <-> ".join(channels)
        channel_pair_counts[channel_key] = channel_pair_counts.get(channel_key, 0) + 1
        certificate = certificates_by_pair.get((left, right), {})
        basis = str(certificate.get("structural_basis", "unknown"))
        structural_basis_counts[basis] = structural_basis_counts.get(basis, 0) + 1
        distance = _finite_or_none(certificate.get("articulation_surface_distance"))
        if distance is not None:
            distances.append(distance)
    components = []
    unseen = set(nodes)
    while unseen:
        seed_node = min(unseen)
        stack, component = [seed_node], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component, reverse=True))
        unseen -= component
        components.append(sorted(component))
    return {
        "n_pairs": len(pairs),
        "n_distinct_articulations": len(nodes),
        "n_connected_components": len(components),
        "largest_connected_component_n": max(map(len, components), default=0),
        "connected_components": components,
        "channel_pair_counts": dict(sorted(channel_pair_counts.items())),
        "structural_basis_counts": dict(sorted(structural_basis_counts.items())),
        "surface_distance": {
            "n_defined": len(distances),
            "minimum": min(distances) if distances else None,
            "mean": float(np.mean(distances)) if distances else None,
            "maximum": max(distances) if distances else None,
        },
    }


def _best_candidate(candidates: list[dict], key: str, *, maximize: bool = True,
                    eligible_ids: set[str] | None = None) -> dict | None:
    defined = [row for row in candidates
               if (eligible_ids is None or row["arm_id"] in eligible_ids)
               and _finite_or_none(row.get(key)) is not None]
    if not defined:
        return None
    ordered = sorted(
        defined,
        key=lambda row: (
            -float(row[key]) if maximize else float(row[key]),
            int(row.get("added_content_word_count") or 0),
            row["arm_id"],
        ),
    )
    row = ordered[0]
    return {
        "arm_id": row["arm_id"],
        "channel": row.get("channel"),
        "provenance": row.get("provenance"),
        "n_address_units": row.get("n_address_units"),
        "added_content_word_count": row.get("added_content_word_count"),
        "value": float(row[key]),
    }


def _breadth_cell_decomposition(panel_cell: dict, report_cell: dict) -> dict:
    cell_id = panel_cell["id"]
    rows = report_cell.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{cell_id}: breadth report has no articulation rows")
    if any(not isinstance(row, dict)
           or not isinstance(row.get("arm_id"), str) or not row["arm_id"]
           for row in rows):
        raise ValueError(f"{cell_id}: articulation rows have invalid arm ids")
    rows_by_id = {row.get("arm_id"): row for row in rows}
    if None in rows_by_id or len(rows_by_id) != len(rows):
        raise ValueError(f"{cell_id}: missing or duplicate articulation row ids")
    candidates = [row for row in rows if not _is_control_row(row)]
    control_rows = [row for row in rows if _is_control_row(row)]
    candidate_ids = {row["arm_id"] for row in candidates}
    if not candidates:
        raise ValueError(f"{cell_id}: no explicit-content candidate rows")
    if any(
            row["arm_id"] == "name"
            or row.get("control_for") is not None
            or row.get("provenance") not in SOURCE_HIERARCHY_PROVENANCES
            for row in candidates):
        raise ValueError(
            f"{cell_id}: candidate panel contains a name, control, or non-source arm"
        )
    if any(
            row.get("provenance") not in REQUIRED_SPECIFICITY_CONTROL_PROVENANCES
            or row.get("control_for") not in candidate_ids
            or row.get("components") != []
            for row in control_rows):
        raise ValueError(f"{cell_id}: specificity-control row shape is invalid")
    contrasts = report_cell.get("matched_control_certificates")
    if not isinstance(contrasts, list):
        raise ValueError(f"{cell_id}: matched-control certificates are missing")
    controls_by_source: dict[str, set[str]] = {}
    contrast_control_ids = []
    for contrast in contrasts:
        if not isinstance(contrast, dict):
            raise ValueError(f"{cell_id}: malformed matched-control certificate")
        source_id = contrast.get("source_arm_id")
        control_id = contrast.get("control_arm_id")
        provenance = contrast.get("control_provenance")
        source_row = rows_by_id.get(source_id, {})
        control_row = rows_by_id.get(control_id, {})
        if (source_id not in candidate_ids or control_id not in rows_by_id
                or not _is_control_row(control_row)
                or control_row.get("control_for") != source_id
                or control_row.get("provenance") != provenance
                or provenance not in REQUIRED_SPECIFICITY_CONTROL_PROVENANCES):
            raise ValueError(f"{cell_id}: unmatched or unknown control contrast")
        for key in ("semantic_content_word_count", "added_content_word_count"):
            if source_row.get(key) != control_row.get(key):
                raise ValueError(
                    f"{cell_id}/{source_id}: matched control changes {key}"
                )
        controls_by_source.setdefault(source_id, set()).add(provenance)
        contrast_control_ids.append(control_id)
    required_controls = set(REQUIRED_SPECIFICITY_CONTROL_PROVENANCES)
    scored_control_ids = {
        row["arm_id"] for row in rows if _is_control_row(row)
    }
    if (len(contrast_control_ids) != 2 * len(candidate_ids)
            or len(contrast_control_ids) != len(set(contrast_control_ids))
            or set(contrast_control_ids) != scored_control_ids
            or any(controls_by_source.get(arm_id) != required_controls
                   for arm_id in candidate_ids)):
        raise ValueError(
            f"{cell_id}: every candidate must have exactly both frozen specificity controls"
        )
    scale = report_cell.get("content_specific_scale_step")
    fiber = report_cell.get("content_specific_joint_fiber")
    if not isinstance(scale, dict) or not isinstance(fiber, dict):
        raise ValueError(f"{cell_id}: missing content-specific scale/fiber analyses")
    replayed_scale = _content_specific_scale_memberships(rows, contrasts)
    changed_scale = sorted(
        key for key, value in replayed_scale.items() if scale.get(key) != value
    )
    if changed_scale:
        raise ValueError(
            f"{cell_id}: content-specific scale memberships fail deterministic replay: "
            f"{changed_scale}"
        )
    membership_keys = {
        "observed_functional": (
            "observed_joint_fixed_target_endpoint_isomorphic_members"),
        "certified_functional": (
            "certified_joint_fixed_target_endpoint_isomorphic_members"),
        "simultaneous_functional": (
            "simultaneous_certified_joint_fixed_target_endpoint_isomorphic_members"),
        "observed_equivalent": (
            "observed_joint_fixed_target_endpoint_equivalent_members"),
        "certified_equivalent": (
            "certified_joint_fixed_target_endpoint_equivalent_members"),
        "simultaneous_equivalent": (
            "simultaneous_certified_joint_fixed_target_endpoint_equivalent_members"),
    }
    memberships = {
        grade: _membership_list(
            scale, key, cell_id=cell_id, candidate_ids=candidate_ids)
        for grade, key in membership_keys.items()
    }
    for family in ("functional", "equivalent"):
        certified = set(memberships[f"certified_{family}"])
        simultaneous = set(memberships[f"simultaneous_{family}"])
        if not simultaneous <= certified:
            raise ValueError(
                f"{cell_id}: simultaneous {family} grade is not a nominal-certified subset"
            )
    for grade in ("observed", "certified", "simultaneous"):
        if not set(memberships[f"{grade}_equivalent"]) <= set(
                memberships[f"{grade}_functional"]):
            raise ValueError(
                f"{cell_id}: endpoint-equivalent membership is not a functional subset"
            )
    grades_by_arm = {}
    arm_grades = scale.get("arm_grades")
    if not isinstance(arm_grades, list):
        raise ValueError(f"{cell_id}: content-specific scale arm grades are missing")
    for row in arm_grades:
        arm_id = row.get("arm_id") if isinstance(row, dict) else None
        if arm_id not in candidate_ids or arm_id in grades_by_arm:
            raise ValueError(f"{cell_id}: invalid content-specific arm grade")
        grades = row.get("grades")
        if not isinstance(grades, dict):
            raise ValueError(f"{cell_id}/{arm_id}: grade map is missing")
        for grade in ("observed", "certified", "simultaneous_certified"):
            if not isinstance(
                    grades.get(grade, {}).get(
                        "better_than_every_required_control_on_rank_and_mae"), bool):
                raise ValueError(f"{cell_id}/{arm_id}: control grade {grade} is missing")
        grades_by_arm[arm_id] = row
    if set(grades_by_arm) != candidate_ids:
        raise ValueError(f"{cell_id}: arm-grade coverage differs from candidate panel")

    candidate_coordinates = []
    for row in candidates:
        components = row.get("components")
        semantic_words = row.get("semantic_content_word_count")
        if (not isinstance(row.get("channel"), str) or not row["channel"]
                or not isinstance(components, list)
                or any(not isinstance(value, str) or not value for value in components)
                or len(components) != len(set(components))
                or not isinstance(semantic_words, int) or isinstance(semantic_words, bool)
                or semantic_words < 0):
            raise ValueError(f"{cell_id}/{row['arm_id']}: invalid source-arm metadata")
        certificate = row.get("certificate")
        if not isinstance(certificate, dict):
            raise ValueError(f"{cell_id}/{row['arm_id']}: policy certificate is missing")
        for key in (
                "scale_step_certificate", "scale_step_simultaneous_certificate",
                "scale_step_specificity_simultaneous_certificate"):
            if not isinstance(row.get(key), dict):
                raise ValueError(f"{cell_id}/{row['arm_id']}: {key} is missing")
        closures = _certificate_coordinate_closures(certificate)
        scale_closures = _scale_step_coordinate_closures(
            row.get("scale_step_certificate"))
        added_words = row.get("added_content_word_count")
        if (added_words is not None
                and (not isinstance(added_words, int) or isinstance(added_words, bool)
                     or added_words < 0)):
            raise ValueError(f"{cell_id}/{row['arm_id']}: invalid added word count")
        n_units = row.get("n_address_units")
        if row.get("channel") == "address_dose":
            if not isinstance(n_units, int) or isinstance(n_units, bool) or n_units < 1:
                raise ValueError(f"{cell_id}/{row['arm_id']}: invalid address-unit dose")
        elif n_units is not None:
            raise ValueError(f"{cell_id}/{row['arm_id']}: non-dose arm declares units")
        grade = grades_by_arm[row["arm_id"]]["grades"]
        candidate_coordinates.append({
            "arm_id": row["arm_id"],
            "channel": row.get("channel"),
            "provenance": row.get("provenance"),
            "n_address_units": n_units,
            "added_content_word_count": added_words,
            "raw_joint_adverse_quotient_rho_floor": closures[
                "joint_adverse_quotient_rho_point"],
            "observed_functional_rho_capacity": closures["observed_max_rho_floor"],
            "certified_functional_rho_capacity": closures["certified_max_rho_floor"],
            "adverse_rho_point": closures["adverse_rho_point"],
            "quotient_rho_point": closures["quotient_rho_point"],
            "adverse_mae_tvd": closures["adverse_mae_tvd"],
            "fraction_rank_target_self_gap_closed": closures[
                "fraction_rank_scale_gap_closed"],
            "fraction_mae_target_self_gap_closed": closures[
                "fraction_mae_scale_gap_closed"],
            "fraction_flip_target_self_gap_closed": closures[
                "fraction_flip_target_self_gap_closed"],
            "fraction_absolute_bias_target_self_gap_closed": closures[
                "fraction_absolute_bias_target_self_gap_closed"],
            "native_scale_step_rank_closure": scale_closures["rank"],
            "native_scale_step_mae_closure": scale_closures["mae"],
            "native_scale_step_flip_closure": scale_closures["flip"],
            "native_scale_step_absolute_bias_closure": scale_closures[
                "absolute_bias"],
            "matched_control_superiority": {
                name: bool(grade[name][
                    "better_than_every_required_control_on_rank_and_mae"])
                for name in ("observed", "certified", "simultaneous_certified")
            },
        })

    frontier_specs = {
        "best_raw_joint_adverse_quotient_rho_floor": (
            "raw_joint_adverse_quotient_rho_floor", True),
        "best_observed_functional_rho_capacity": (
            "observed_functional_rho_capacity", True),
        "best_certified_functional_rho_capacity": (
            "certified_functional_rho_capacity", True),
        "best_adverse_rho_point": ("adverse_rho_point", True),
        "best_quotient_rho_point": ("quotient_rho_point", True),
        "lowest_adverse_mae_tvd": ("adverse_mae_tvd", False),
        "best_fraction_rank_target_self_gap_closed": (
            "fraction_rank_target_self_gap_closed", True),
        "best_fraction_mae_target_self_gap_closed": (
            "fraction_mae_target_self_gap_closed", True),
        "best_fraction_flip_target_self_gap_closed": (
            "fraction_flip_target_self_gap_closed", True),
        "best_fraction_absolute_bias_target_self_gap_closed": (
            "fraction_absolute_bias_target_self_gap_closed", True),
        "best_native_scale_step_rank_closure": (
            "native_scale_step_rank_closure", True),
        "best_native_scale_step_mae_closure": (
            "native_scale_step_mae_closure", True),
        "best_native_scale_step_flip_closure": (
            "native_scale_step_flip_closure", True),
        "best_native_scale_step_absolute_bias_closure": (
            "native_scale_step_absolute_bias_closure", True),
    }
    coordinate_frontiers = {
        output_key: _best_candidate(
            candidate_coordinates, candidate_key, maximize=maximize)
        for output_key, (candidate_key, maximize) in frontier_specs.items()
    }
    coordinates = {
        key: (frontier["value"] if frontier else None)
        for key, frontier in coordinate_frontiers.items()
    }
    observed_control_ids = {
        arm_id for arm_id, grade_row in grades_by_arm.items()
        if grade_row["grades"]["observed"][
            "better_than_every_required_control_on_rank_and_mae"]
    }
    best_content_specific_route = _best_candidate(
        candidate_coordinates,
        "raw_joint_adverse_quotient_rho_floor",
        eligible_ids=set(memberships["observed_functional"]),
    )

    pair_specs = {
        "observed_H_fiber_pairs": ("observed", "H_fiber"),
        "certified_H_fiber_pairs": ("certified", "H_fiber"),
        "simultaneous_certified_H_fiber_pairs": (
            "simultaneous_certified", "H_fiber"),
        "simultaneous_certified_H_fiber_sensitivity_pairs": (
            "simultaneous_certified", "H_fiber_sensitivity"),
        "simultaneous_certified_H_fiber_eq_pairs": (
            "simultaneous_certified", "H_fiber_eq"),
        "observed_H_fiber_vec_pairs": ("observed", "H_fiber_vec"),
        "certified_H_fiber_vec_pairs": ("certified", "H_fiber_vec"),
        "simultaneous_certified_H_fiber_vec_pairs": (
            "simultaneous_certified", "H_fiber_vec"),
        "simultaneous_certified_H_fiber_vec_eq_pairs": (
            "simultaneous_certified", "H_fiber_vec_eq"),
    }
    parsed_pair_lists = {
        key: _pair_members(
            fiber, key, cell_id=cell_id, candidate_ids=candidate_ids)
        for key in pair_specs
    }
    fiber_pairs = {
        "observed": parsed_pair_lists["observed_H_fiber_pairs"],
        "certified": parsed_pair_lists["certified_H_fiber_pairs"],
        "simultaneous_certified": parsed_pair_lists[
            "simultaneous_certified_H_fiber_pairs"],
        "observed_vector": parsed_pair_lists["observed_H_fiber_vec_pairs"],
        "certified_vector": parsed_pair_lists["certified_H_fiber_vec_pairs"],
        "simultaneous_certified_vector": parsed_pair_lists[
            "simultaneous_certified_H_fiber_vec_pairs"],
    }
    if fiber.get("candidate_arm_ids") != sorted(candidate_ids):
        raise ValueError(f"{cell_id}: fiber candidate panel differs from report rows")
    certificates_by_pair = {}
    pair_certificates = fiber.get("pair_certificates")
    if not isinstance(pair_certificates, list):
        raise ValueError(f"{cell_id}: fiber pair certificates are missing")
    for row in pair_certificates:
        if not isinstance(row, dict):
            raise ValueError(f"{cell_id}: malformed fiber pair certificate")
        left, right = row.get("left"), row.get("right")
        if (not isinstance(left, str) or not isinstance(right, str)
                or left == right):
            raise ValueError(f"{cell_id}: invalid fiber pair certificate endpoints")
        pair = tuple(sorted((left, right)))
        if pair in certificates_by_pair or not set(pair) <= candidate_ids:
            raise ValueError(f"{cell_id}: invalid fiber pair certificate endpoints")
        certificates_by_pair[pair] = row
    expected_certificate_pairs = {
        tuple(sorted((left, right)))
        for left_index, left in enumerate(sorted(candidate_ids))
        for right in sorted(candidate_ids)[left_index + 1:]
    }
    if set(certificates_by_pair) != expected_certificate_pairs:
        raise ValueError(f"{cell_id}: fiber certificates do not cover every candidate pair")
    component_sets = {
        arm_id: set(rows_by_id[arm_id].get("components") or [])
        for arm_id in candidate_ids
    }
    component_minimal = {
        arm_id for arm_id, components in component_sets.items()
        if components and not any(
            other_components < components
            for other_id, other_components in component_sets.items()
            if other_id != arm_id and other_components
        )
    }
    membership_sets = {
        "observed": set(memberships["observed_functional"]),
        "certified": set(memberships["certified_functional"]),
        "simultaneous_certified": set(memberships["simultaneous_functional"]),
    }
    equivalent_sets = {
        "observed": set(memberships["observed_equivalent"]),
        "certified": set(memberships["certified_equivalent"]),
        "simultaneous_certified": set(memberships["simultaneous_equivalent"]),
    }
    for pair, certificate in certificates_by_pair.items():
        structural_gate = certificate.get("structural_gate")
        grades = certificate.get("grades")
        mutual_gates = certificate.get("mutual_policy_certificate", {}).get("gates")
        if (not isinstance(structural_gate, bool) or not isinstance(grades, dict)
                or not isinstance(mutual_gates, dict)):
            raise ValueError(f"{cell_id}: fiber certificate omits structural/grade gates")
        required_mutual_gates = {
            "point_at_least_primary_floor", "lower_CI_at_least_primary_floor",
            "point_at_least_sensitivity_floor", "lower_CI_at_least_sensitivity_floor",
            "point_vector_equivalent", "certified_vector_equivalent",
        }
        if (not required_mutual_gates <= set(mutual_gates)
                or any(not isinstance(mutual_gates[key], bool)
                       for key in required_mutual_gates)):
            raise ValueError(f"{cell_id}: fiber mutual-policy gate panel is incomplete")
        left, right = pair
        both_have_components = bool(component_sets[left] and component_sets[right])
        both_atomic = not component_sets[left] and not component_sets[right]
        components_incomparable = (
            not (
                component_sets[left] <= component_sets[right]
                or component_sets[right] <= component_sets[left]
            ) if both_have_components else None
        )
        channels_distinct = (
            rows_by_id[left].get("channel") != rows_by_id[right].get("channel")
        )
        if both_have_components:
            structural_basis = "declared_component_topology"
            component_minimal_pair = left in component_minimal and right in component_minimal
            route_gate = bool(component_minimal_pair and components_incomparable)
        elif both_atomic:
            structural_basis = "frozen_atomic_routes_with_distinct_channels"
            component_minimal_pair = None
            route_gate = channels_distinct
        else:
            structural_basis = "incompatible_component_metadata"
            component_minimal_pair = None
            route_gate = False
        distance = _finite_or_none(certificate.get("articulation_surface_distance"))
        distance_floor = _finite_or_none(certificate.get("distinctness_floor"))
        if (distance is None or not 0.0 <= distance <= 1.0
                or distance_floor is None or not 0.0 <= distance_floor <= 1.0
                or certificate.get("structural_basis") != structural_basis
                or certificate.get("component_minimal") is not component_minimal_pair
                or certificate.get("components_incomparable") is not components_incomparable
                or certificate.get("both_frozen_atomic_routes") is not both_atomic
                or certificate.get("channels_distinct") is not channels_distinct
                or structural_gate is not bool(route_gate and distance >= distance_floor)):
            raise ValueError(f"{cell_id}: fiber structural topology fails replay")
        for grade in ("observed", "certified", "simultaneous_certified"):
            grade_row = grades.get(grade)
            if not isinstance(grade_row, dict):
                raise ValueError(f"{cell_id}: fiber certificate omits {grade} grades")
            both_hj = set(pair) <= membership_sets[grade]
            both_equivalent = set(pair) <= equivalent_sets[grade]
            mutual_gate = grade_row.get("mutual_rank_gate")
            sensitivity_gate = grade_row.get("mutual_rank_sensitivity_gate")
            vector_gate = grade_row.get("mutual_quotient_vector_equivalence_gate")
            required_booleans = (
                mutual_gate, sensitivity_gate, vector_gate,
                grade_row.get("both_members_pass_content_specific_H_J"),
                grade_row.get("both_members_also_pass_H_J_eq"),
            )
            if any(not isinstance(value, bool) for value in required_booleans):
                raise ValueError(f"{cell_id}: fiber {grade} grade has non-boolean gates")
            expected_mutual = {
                "mutual_rank_gate": mutual_gates[
                    "point_at_least_primary_floor" if grade == "observed"
                    else "lower_CI_at_least_primary_floor"],
                "mutual_rank_sensitivity_gate": mutual_gates[
                    "point_at_least_sensitivity_floor" if grade == "observed"
                    else "lower_CI_at_least_sensitivity_floor"],
                "mutual_quotient_vector_equivalence_gate": mutual_gates[
                    "point_vector_equivalent" if grade == "observed"
                    else "certified_vector_equivalent"],
            }
            if any(grade_row.get(key) is not value
                   for key, value in expected_mutual.items()):
                raise ValueError(
                    f"{cell_id}: fiber {grade} mutual gates differ from their certificate"
                )
            expected_grades = {
                "both_members_pass_content_specific_H_J": both_hj,
                "both_members_also_pass_H_J_eq": both_equivalent,
                "H_fiber": bool(structural_gate and both_hj and mutual_gate),
                "H_fiber_sensitivity": bool(
                    structural_gate and both_hj and sensitivity_gate),
                "H_fiber_eq": bool(
                    structural_gate and both_equivalent and mutual_gate),
                "H_fiber_vec": bool(structural_gate and both_hj and vector_gate),
                "H_fiber_vec_eq": bool(
                    structural_gate and both_equivalent and vector_gate),
            }
            if any(grade_row.get(key) is not value
                   for key, value in expected_grades.items()):
                raise ValueError(
                    f"{cell_id}: fiber {grade} grade is inconsistent with its gates"
                )
    for key, (grade, result_key) in pair_specs.items():
        expected_pairs = {
            pair for pair, certificate in certificates_by_pair.items()
            if certificate["grades"][grade][result_key]
        }
        if set(parsed_pair_lists[key]) != expected_pairs:
            raise ValueError(f"{cell_id}: fiber membership list {key} fails grade replay")
    fiber_topology = {
        grade: _fiber_topology(
            pairs, rows_by_id=rows_by_id,
            certificates_by_pair=certificates_by_pair)
        for grade, pairs in fiber_pairs.items()
    }

    outcomes = {
        "observed_content_specific_joint_functional_substitution": bool(
            memberships["observed_functional"]),
        "certified_content_specific_joint_functional_substitution": bool(
            memberships["certified_functional"]),
        "simultaneous_certified_content_specific_joint_functional_substitution": bool(
            memberships["simultaneous_functional"]),
        "observed_content_specific_joint_equivalent_substitution": bool(
            memberships["observed_equivalent"]),
        "certified_content_specific_joint_equivalent_substitution": bool(
            memberships["certified_equivalent"]),
        "simultaneous_certified_content_specific_joint_equivalent_substitution": bool(
            memberships["simultaneous_equivalent"]),
        "observed_functional_equal_but_different_fiber": bool(
            fiber_pairs["observed"]),
        "certified_functional_equal_but_different_fiber": bool(
            fiber_pairs["certified"]),
        "simultaneous_certified_functional_equal_but_different_fiber": bool(
            fiber_pairs["simultaneous_certified"]),
        "observed_vector_equal_but_different_fiber": bool(
            fiber_pairs["observed_vector"]),
        "certified_vector_equal_but_different_fiber": bool(
            fiber_pairs["certified_vector"]),
        "simultaneous_certified_vector_equal_but_different_fiber": bool(
            fiber_pairs["simultaneous_certified_vector"]),
        "observed_matched_control_improvement": bool(observed_control_ids),
        "certified_matched_control_improvement": any(
            row["grades"]["certified"][
                "better_than_every_required_control_on_rank_and_mae"]
            for row in grades_by_arm.values()),
        "simultaneous_certified_matched_control_improvement": any(
            row["grades"]["simultaneous_certified"][
                "better_than_every_required_control_on_rank_and_mae"]
            for row in grades_by_arm.values()),
    }
    if set(outcomes) != set(_BREADTH_BINARY_OUTCOMES):
        raise AssertionError("breadth binary outcome inventory drifted")

    dose_rows = sorted(
        (row for row in candidate_coordinates if row["channel"] == "address_dose"),
        key=lambda row: (row["n_address_units"], row["added_content_word_count"] or 0,
                         row["arm_id"]),
    )
    dose_curve = []
    for row in dose_rows:
        arm_id = row["arm_id"]
        dose_curve.append({
            "arm_id": arm_id,
            "n_address_units": row["n_address_units"],
            "added_content_word_count": row["added_content_word_count"],
            "raw_joint_adverse_quotient_rho_floor": row[
                "raw_joint_adverse_quotient_rho_floor"],
            "fraction_rank_target_self_gap_closed": row[
                "fraction_rank_target_self_gap_closed"],
            "fraction_mae_target_self_gap_closed": row[
                "fraction_mae_target_self_gap_closed"],
            "observed_content_specific_functional_member": (
                arm_id in memberships["observed_functional"]),
            "certified_content_specific_functional_member": (
                arm_id in memberships["certified_functional"]),
            "simultaneous_certified_content_specific_functional_member": (
                arm_id in memberships["simultaneous_functional"]),
        })

    dose_panel_complete = bool(
        report_cell.get("scored_arm_panel_validation", {}).get("small", {}).get(
            "arm_policy") == "all"
    )

    def onset(grade: str) -> dict | None:
        member_ids = set(memberships[f"{grade}_functional"])
        passing = [row for row in dose_curve if row["arm_id"] in member_ids]
        if not passing:
            return None
        row = min(passing, key=lambda value: (
            value["n_address_units"], value["added_content_word_count"] or 0,
            value["arm_id"],
        ))
        return {
            "arm_id": row["arm_id"],
            "n_address_units": row["n_address_units"],
            "added_content_word_count": row["added_content_word_count"],
            "authenticated_complete_frozen_dose_panel": dose_panel_complete,
            "estimand": (
                "minimum passing dose in the complete frozen address candidate panel"
                if dose_panel_complete else
                "minimum passing tested dose among the frozen selected candidates"
            ),
        }

    return {
        "cell_id": cell_id,
        "task": panel_cell["task"],
        "level": panel_cell["level"],
        "bucket": panel_cell["bucket"],
        "node_id": panel_cell["node_id"],
        **{key: panel_cell[key] for key in BREADTH_DESIGN_FIELDS},
        "outcomes": outcomes,
        "coordinates": coordinates,
        "coordinate_frontiers": coordinate_frontiers,
        "best_content_specific_route": best_content_specific_route,
        "candidate_coordinate_rows": candidate_coordinates,
        "content_specific_memberships": memberships,
        "address_dose": {
            "n_routes": len(dose_curve),
            "authenticated_complete_frozen_dose_panel": dose_panel_complete,
            "curve": dose_curve,
            "observed_functional_onset": onset("observed"),
            "certified_functional_onset": onset("certified"),
            "simultaneous_certified_functional_onset": onset("simultaneous"),
            "unit_boundary": (
                "source-address prefixes are deterministic CUF dose instruments; this report "
                "does not promote them to form-robust certified semantic units. A validation "
                "selection that scores only the frozen best-dose role identifies a minimum "
                "passing tested dose, not the full-bank onset."
            ),
        },
        "fiber_pair_counts": {
            grade: len(pairs) for grade, pairs in fiber_pairs.items()
        },
        "fiber_topology": fiber_topology,
    }


def _derived_bootstrap_seed(seed: int, label: str) -> int:
    digest = sha256_bytes(label.encode())
    return int((int(seed) + int(digest[:12], 16)) % (2**63 - 1))


def _bootstrap_metric_estimates(
        records: list[dict], *, weights: np.ndarray, group_ids: list[str],
        n_boot: int, seed: int, confidence: float) -> dict:
    """Bootstrap weighted ratios over the blocks where each coordinate is defined.

    Binary cell grades are defined for every cell.  Gap-closure coordinates legitimately are
    not: a sparse-to-reference gap may be absent.  Resampling the full block panel and then
    discarding zero-denominator draws conditions the interval on a random event, especially
    when only a few blocks define a coordinate.  Instead, each coordinate's complete-case
    estimand resamples its own observed defined-block panel and never treats nulls as zeros.
    """
    if not records:
        return {"status": "no_cells", "prevalence": {}, "coordinate_means": {}}
    metric_names = [*_BREADTH_BINARY_OUTCOMES, *_BREADTH_COORDINATES]
    values = np.full((len(records), len(metric_names)), np.nan, dtype=float)
    for row_index, record in enumerate(records):
        for column, name in enumerate(_BREADTH_BINARY_OUTCOMES):
            values[row_index, column] = float(record["outcomes"][name])
        for offset, name in enumerate(_BREADTH_COORDINATES, len(_BREADTH_BINARY_OUTCOMES)):
            value = _finite_or_none(record["coordinates"].get(name))
            if value is not None:
                values[row_index, offset] = value
    if (weights.shape != (len(records),) or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)):
        raise ValueError("breadth bootstrap weights must be finite and positive")
    if len(group_ids) != len(records) or any(not value for value in group_ids):
        raise ValueError("breadth bootstrap groups must be complete")
    groups = sorted(set(group_ids))
    group_index = {value: index for index, value in enumerate(groups)}
    numerator = np.zeros((len(groups), len(metric_names)), dtype=float)
    denominator = np.zeros_like(numerator)
    group_mass = np.zeros(len(groups), dtype=float)
    group_n = np.zeros(len(groups), dtype=int)
    for row_index, group_id in enumerate(group_ids):
        index = group_index[group_id]
        valid = np.isfinite(values[row_index])
        numerator[index, valid] += weights[row_index] * values[row_index, valid]
        denominator[index, valid] += weights[row_index]
        group_mass[index] += weights[row_index]
        group_n[index] += 1
    total_denominator = denominator.sum(axis=0)
    points = np.divide(
        numerator.sum(axis=0), total_denominator,
        out=np.full(len(metric_names), np.nan), where=total_denominator > 0.0,
    )
    alpha = 1.0 - confidence
    defined_supports: dict[bytes, list[int]] = {}
    for index in range(len(metric_names)):
        support = (denominator[:, index] > 0.0).tobytes()
        defined_supports.setdefault(support, []).append(index)
    draws = np.full((n_boot, len(metric_names)), np.nan, dtype=float)
    bootstrap_seeds: list[int] = [0] * len(metric_names)
    for support, columns in sorted(defined_supports.items(), key=lambda row: row[0]):
        defined_groups = np.frombuffer(support, dtype=np.bool_)
        group_indexes = np.flatnonzero(defined_groups)
        support_seed = _derived_bootstrap_seed(
            seed, f"defined-block-support:{sha256_bytes(support)}")
        for column in columns:
            bootstrap_seeds[column] = support_seed
        if len(group_indexes) < 2:
            continue
        rng = np.random.default_rng(support_seed)
        chunk_size = min(128, n_boot)
        support_numerator = numerator[:, columns]
        support_denominator = denominator[:, columns]
        for start in range(0, n_boot, chunk_size):
            stop = min(start + chunk_size, n_boot)
            sampled_local = rng.integers(
                0, len(group_indexes),
                size=(stop - start, len(group_indexes)),
            )
            sampled = group_indexes[sampled_local]
            draw_num = np.take(support_numerator, sampled, axis=0).sum(axis=1)
            draw_den = np.take(support_denominator, sampled, axis=0).sum(axis=1)
            draws[start:stop, columns] = draw_num / draw_den

    def result(name: str, index: int, *, binary: bool) -> dict:
        observed_values = values[:, index]
        valid_rows = np.isfinite(observed_values)
        defined_groups = denominator[:, index] > 0.0
        group_denominator = denominator[defined_groups, index]
        n_defined_blocks = len(group_denominator)
        valid_draws = (
            draws[:, index][np.isfinite(draws[:, index])]
            if n_defined_blocks >= 2 else np.array([])
        )
        ci_status = (
            "estimated" if n_defined_blocks >= 2
            else "insufficient_defined_blocks" if n_defined_blocks == 1
            else "undefined"
        )
        weighted_denominator = float(group_denominator.sum())
        kish_defined = (
            float(weighted_denominator**2 / np.square(group_denominator).sum())
            if n_defined_blocks else None
        )
        return {
            "point": float(points[index]) if np.isfinite(points[index]) else None,
            "CI": (
                [float(np.quantile(valid_draws, alpha / 2.0)),
                 float(np.quantile(valid_draws, 1.0 - alpha / 2.0))]
                if n_defined_blocks >= 2 else None
            ),
            "CI_status": ci_status,
            "confidence": confidence,
            "n_defined_cells": int(valid_rows.sum()),
            "n_defined_blocks": n_defined_blocks,
            "weighted_defined_denominator": weighted_denominator,
            "kish_effective_defined_block_count": kish_defined,
            "n_positive_cells": (
                int(observed_values[valid_rows].sum()) if binary else None),
            "n_valid_bootstrap_draws": len(valid_draws),
            "bootstrap_seed": bootstrap_seeds[index],
        }

    prevalence = {
        name: result(name, index, binary=True)
        for index, name in enumerate(_BREADTH_BINARY_OUTCOMES)
    }
    coordinate_means = {
        name: result(name, index, binary=False)
        for index, name in enumerate(
            _BREADTH_COORDINATES, len(_BREADTH_BINARY_OUTCOMES))
    }
    return {
        "status": "estimated",
        "n_blocks": len(groups),
        "largest_observed_block_n": int(group_n.max()),
        "mean_observed_block_n": float(group_n.mean()),
        "kish_effective_block_count": float(
            group_mass.sum() ** 2 / np.square(group_mass).sum()),
        "n_boot": n_boot,
        "seed": int(seed),
        "n_unique_defined_block_panels": len(defined_supports),
        "prevalence": prevalence,
        "coordinate_means": coordinate_means,
    }


def _scope_breadth_summary(records: list[dict], *, label: str, n_boot: int,
                           seed: int, confidence: float) -> dict:
    if not records:
        return {"scope": label, "n_cells": 0, "status": "no_cells"}
    group_specs = {
        "independent_cell": {
            "description": "diagnostic bootstrap treating each selected action node as a unit",
            "field": "cell_id", "declared_size_field": None, "scope_fields": (),
        },
        "immediate_dependency_component": {
            "description": (
                "assign one resampling multiplicity to all selected members of each frozen "
                "immediate-source dependency component; ids are scoped by task, bucket, and "
                "round"
            ),
            "field": "dependency_component_id",
            "declared_size_field": "dependency_component_size",
            "scope_fields": ("task", "level", "bucket"),
        },
        "inherited_raw_provenance_component": {
            "description": (
                "assign one resampling multiplicity to all selected members of each inherited "
                "raw-provenance overlap component; this is a dependence sensitivity, not a "
                "raw-rubric prevalence estimand; ids are scoped by task, bucket, and round"
            ),
            "field": "provenance_component_id",
            "declared_size_field": "provenance_component_size",
            "scope_fields": ("task", "level", "bucket"),
        },
        "task_raw_provenance_component": {
            "description": (
                "conservative task-wide raw-provenance block: one resampling multiplicity "
                "covers every selected R1/R2/R3 action node connected through inherited raw "
                "rubric reuse; ids are scoped only by task and bucket"
            ),
            "field": "task_raw_provenance_component_id",
            "declared_size_field": "task_raw_provenance_component_size",
            "scope_fields": ("task", "bucket"),
        },
    }
    weights_by_estimand = {
        "balanced_selected_action_node_panel": np.ones(len(records), dtype=float),
        "native_action_node_nominal_poststratified_sensitivity": np.asarray(
            [record["nominal_poststratification_weight"] for record in records],
            dtype=float,
        ),
    }
    inference = {}
    for estimand, weights in weights_by_estimand.items():
        bootstrap_designs = {}
        for design, spec in group_specs.items():
            field = spec["field"]
            scope_fields = spec["scope_fields"]
            group_ids = [
                record["cell_id"] if field == "cell_id" else
                "::".join([*(str(record[key]) for key in scope_fields), record[field]])
                for record in records
            ]
            estimate = _bootstrap_metric_estimates(
                records,
                weights=weights,
                group_ids=group_ids,
                n_boot=n_boot,
                seed=_derived_bootstrap_seed(seed, f"{label}|{estimand}|{design}"),
                confidence=confidence,
            )
            declared_field = spec["declared_size_field"]
            estimate["bootstrap_unit"] = design
            estimate["description"] = spec["description"]
            estimate["largest_declared_component_n"] = (
                max(record[declared_field] for record in records)
                if declared_field else 1
            )
            estimate["subset_may_truncate_frozen_components"] = bool(
                declared_field and any(
                    sum(
                        all(other[key] == record[key] for key in scope_fields)
                        and other[field] == record[field]
                        for other in records
                    ) < record[declared_field]
                    for record in records
                )
            )
            bootstrap_designs[design] = estimate
        inference[estimand] = {
            "weighting": (
                "one vote per frozen selected action node"
                if estimand == "balanced_selected_action_node_panel" else
                "descriptive inverse source_kind x breadth-tertile coverage factor; the "
                "diversity-constrained deterministic selector has no known inclusion "
                "probabilities"
            ),
            "bootstrap_designs": bootstrap_designs,
        }

    def categorical_counts(field: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for record in records:
            value = str(record[field])
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))

    route_channels: dict[str, int] = {}
    route_provenances: dict[str, int] = {}
    dose_onsets = {
        "observed": [], "certified": [], "simultaneous_certified": [],
    }
    authenticated_dose_onsets = {
        "observed": 0, "certified": 0, "simultaneous_certified": 0,
    }
    for record in records:
        route = record["best_content_specific_route"]
        if route:
            channel = str(route.get("channel"))
            provenance = str(route.get("provenance"))
            route_channels[channel] = route_channels.get(channel, 0) + 1
            route_provenances[provenance] = route_provenances.get(provenance, 0) + 1
        for grade, key in (
                ("observed", "observed_functional_onset"),
                ("certified", "certified_functional_onset"),
                ("simultaneous_certified", "simultaneous_certified_functional_onset")):
            onset = record["address_dose"][key]
            if onset:
                dose_onsets[grade].append(onset["n_address_units"])
                authenticated_dose_onsets[grade] += bool(
                    onset["authenticated_complete_frozen_dose_panel"])
    fiber_pair_totals = {
        grade: sum(record["fiber_pair_counts"][grade] for record in records)
        for grade in next(iter(records))["fiber_pair_counts"]
    }
    fiber_topology_summary = {}
    for grade in next(iter(records))["fiber_topology"]:
        channel_counts: dict[str, int] = {}
        basis_counts: dict[str, int] = {}
        topologies = [record["fiber_topology"][grade] for record in records]
        for topology in topologies:
            for key, value in topology["channel_pair_counts"].items():
                channel_counts[key] = channel_counts.get(key, 0) + value
            for key, value in topology["structural_basis_counts"].items():
                basis_counts[key] = basis_counts.get(key, 0) + value
        fiber_topology_summary[grade] = {
            "n_cells_with_at_least_one_pair": sum(
                topology["n_pairs"] > 0 for topology in topologies),
            "n_pairs": sum(topology["n_pairs"] for topology in topologies),
            "sum_cell_distinct_articulations": sum(
                topology["n_distinct_articulations"] for topology in topologies),
            "sum_cell_connected_components": sum(
                topology["n_connected_components"] for topology in topologies),
            "largest_within_cell_connected_component_n": max(
                (topology["largest_connected_component_n"] for topology in topologies),
                default=0,
            ),
            "channel_pair_counts": dict(sorted(channel_counts.items())),
            "structural_basis_counts": dict(sorted(basis_counts.items())),
        }
    return {
        "scope": label,
        "status": "estimated",
        "n_cells": len(records),
        "tasks": sorted({record["task"] for record in records}),
        "levels": sorted({record["level"] for record in records}),
        "composition": {
            "source_kind_counts": categorical_counts("source_kind"),
            "breadth_stratum_counts": categorical_counts("breadth_stratum"),
            "n_immediate_dependency_disjoint": sum(
                record["dependency_degree"] == 0 for record in records),
            "n_inherited_raw_provenance_disjoint": sum(
                record["provenance_overlap_degree"] == 0 for record in records),
            "n_task_raw_provenance_disjoint": sum(
                record["task_raw_provenance_overlap_degree"] == 0
                for record in records),
            "n_jointly_immediate_and_level_provenance_disjoint": sum(
                record["dependency_degree"] == 0
                and record["provenance_overlap_degree"] == 0
                for record in records),
            "n_jointly_immediate_and_task_provenance_disjoint": sum(
                record["dependency_degree"] == 0
                and record["task_raw_provenance_overlap_degree"] == 0
                for record in records),
        },
        "inference": inference,
        "best_content_specific_route_distribution": {
            "n_cells_with_route": sum(route_channels.values()),
            "channel_counts": dict(sorted(route_channels.items())),
            "provenance_counts": dict(sorted(route_provenances.items())),
        },
        "address_dose_minimum_passing_tested_dose_distribution": {
            grade: {
                "n_cells_with_finite_onset": len(values),
                "n_cells_with_authenticated_complete_panel_onset": (
                    authenticated_dose_onsets[grade]),
                "unit_counts": {
                    str(unit): values.count(unit) for unit in sorted(set(values))
                },
                "median_units": float(np.median(values)) if values else None,
            }
            for grade, values in dose_onsets.items()
        },
        "fiber_pair_totals": fiber_pair_totals,
        "fiber_topology_summary": fiber_topology_summary,
    }


def _terminal_frontier_sensitivity(
        panel: dict, records: list[dict], *, n_boot: int, seed: int,
        confidence: float) -> list[dict]:
    records_by_node = {
        (record["task"], record["level"], record["node_id"]): record
        for record in records
    }
    if len(records_by_node) != len(records):
        raise ValueError("primary panel repeats a task/level/node identity")
    result = []
    for audit in panel.get("terminal_frontier_sensitivities", []):
        if not isinstance(audit, dict):
            raise ValueError("metric panel has a malformed terminal-frontier audit")
        base = {
            "task": audit.get("task"),
            "level": audit.get("level"),
            "design_available": bool(audit.get("available")),
            "n_frontier_nodes": audit.get("n_frontier_nodes"),
            "n_eligible_nodes": audit.get("n_eligible_nodes"),
            "global_partition_claim": audit.get("global_partition_claim"),
        }
        if not audit.get("available"):
            result.append({
                **base,
                "outcome_status": "design_unavailable",
                "reason": audit.get("reason"),
                "exact_primary_node_coverage": None,
                "estimates": None,
            })
            continue
        identity_keys = (
            "frontier_node_ids", "retained_node_ids", "node_ids",
        )
        identities = next(
            (audit[key] for key in identity_keys if isinstance(audit.get(key), list)),
            None,
        )
        if identities is None:
            result.append({
                **base,
                "outcome_status": "not_identifiable_from_frozen_panel",
                "reason": (
                    "the panel freezes terminal-frontier counts and a content hash but not the "
                    "frontier node identities; no primary-node outcome is imputed to carried or "
                    "otherwise unscored frontier nodes"
                ),
                "exact_primary_node_coverage": None,
                "n_exactly_matched_scored_primary_nodes": None,
                "estimates": None,
            })
            continue
        if (not identities
                or any(not isinstance(value, str) or not value for value in identities)
                or len(identities) != len(set(identities))):
            raise ValueError("terminal-frontier node identities are missing or duplicate")
        if len(identities) != audit.get("n_frontier_nodes"):
            raise ValueError("terminal-frontier identity count differs from its frozen audit")
        audit_task, audit_level = audit.get("task"), audit.get("level")
        matched = [
            records_by_node[(audit_task, audit_level, node_id)]
            for node_id in identities
            if (audit_task, audit_level, node_id) in records_by_node
        ]
        denominator = len(identities)
        result.append({
            **base,
            "outcome_status": (
                "exact_matched_primary_subset_estimated" if matched
                else "no_exactly_matched_scored_primary_nodes"),
            "exact_primary_node_coverage": (
                len(matched) / denominator if denominator else None),
            "n_exactly_matched_scored_primary_nodes": len(matched),
            "n_frozen_frontier_identities": denominator,
            "estimates": (
                _scope_breadth_summary(
                    matched,
                    label=(f"terminal-frontier/{audit.get('task')}/"
                           f"{audit.get('level')}/exact-primary-overlap"),
                    n_boot=n_boot,
                    seed=seed,
                    confidence=confidence,
                ) if matched else None
            ),
        })
    return result


def summarize_breadth_decomposition(
        report: dict | str | Path, metric_panel: dict | str | Path, *,
        n_boot: int = 5000, seed: int = 7319, confidence: float = 0.95) -> dict:
    """Decompose complete 990-cell unsupervised scale/articulation reconstruction results."""
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("breadth decomposition n_boot must be a positive integer")
    if (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            or not 0.0 < confidence < 1.0):
        raise ValueError("breadth decomposition seed/confidence are invalid")
    report_payload, report_binding = _load_breadth_json(report, label="policy report")
    panel_payload, panel_binding = _load_breadth_json(
        metric_panel, label="metric panel")
    paired_cells = _validate_breadth_decomposition_inputs(
        report_payload, panel_payload)
    records = [
        _breadth_cell_decomposition(panel_cell, report_cell)
        for panel_cell, report_cell in paired_cells
    ]
    tasks = panel_payload["tasks"]
    task_level = []
    for task in tasks:
        for level in _BREADTH_LEVELS:
            selected = [record for record in records
                        if record["task"] == task and record["level"] == level]
            task_level.append(_scope_breadth_summary(
                selected,
                label=f"task-level/{task}/{level}",
                n_boot=n_boot,
                seed=seed,
                confidence=confidence,
            ))
    aggregate = _scope_breadth_summary(
        records, label="aggregate/all-tasks/all-levels",
        n_boot=n_boot, seed=seed, confidence=confidence)

    sensitivity_specs: list[tuple[str, list[dict]]] = []
    source_kinds = sorted({record["source_kind"] for record in records})
    for source_kind in source_kinds:
        sensitivity_specs.append((
            f"source-kind/{source_kind}",
            [record for record in records if record["source_kind"] == source_kind],
        ))
    filters = {
        "merged-only": lambda record: record["source_kind"] in {
            "merged_group", "merged_tree"},
        "immediate-dependency-disjoint": lambda record: (
            record["dependency_degree"] == 0),
        "level-local-inherited-raw-provenance-disjoint": lambda record: (
            record["provenance_overlap_degree"] == 0),
        "task-global-raw-provenance-disjoint": lambda record: (
            record["task_raw_provenance_overlap_degree"] == 0),
        "joint-immediate-and-task-provenance-disjoint": lambda record: (
            record["dependency_degree"] == 0
            and record["task_raw_provenance_overlap_degree"] == 0),
    }
    for name, predicate in filters.items():
        sensitivity_specs.append((
            f"overlap-sensitivity/{name}",
            [record for record in records if predicate(record)],
        ))
        for level in _BREADTH_LEVELS:
            sensitivity_specs.append((
                f"overlap-sensitivity/{name}/{level}",
                [record for record in records
                 if record["level"] == level and predicate(record)],
            ))
    sensitivities = [
        _scope_breadth_summary(
            selected, label=label, n_boot=n_boot, seed=seed, confidence=confidence)
        for label, selected in sensitivity_specs
    ]
    terminal = _terminal_frontier_sensitivity(
        panel_payload, records, n_boot=n_boot, seed=seed, confidence=confidence)
    aggregate_prevalence = aggregate["inference"][
        "balanced_selected_action_node_panel"]["bootstrap_designs"][
            "task_raw_provenance_component"]["prevalence"]
    return {
        "schema": _BREADTH_DECOMPOSITION_SCHEMA,
        "status": "derived-from-complete-frozen-policy-report",
        "estimand": (
            "unsupervised reconstruction of each larger model policy by explicit source-only "
            "construct knowledge supplied to the smaller within-family executor"
        ),
        "input_bindings": {
            "policy_report": report_binding,
            "metric_panel": panel_binding,
            "metric_panel_content_sha256": panel_payload["panel_content_sha256"],
            "arm_bank_sha256": report_payload.get("arm_bank_sha256"),
            "partition": report_payload.get("partition"),
        },
        "source_policy_analysis_implementation": report_payload.get(
            "analysis_implementation"),
        "breadth_readout_implementation": _analysis_implementation(),
        "scale_comparator": report_payload.get("scale_comparator"),
        "analysis": {
            "n_boot": n_boot,
            "seed": seed,
            "confidence": confidence,
            "functional_rho_floor": report_payload.get("config", {}).get(
                "functional_rho_floor"),
            "binary_outcomes": list(_BREADTH_BINARY_OUTCOMES),
            "continuous_coordinates": list(_BREADTH_COORDINATES),
            "coordinate_meanings": {
                "rank": "adverse-form and form-quotient Spearman ordering fidelity",
                "mae_tvd": (
                    "absolute policy-level/calibration displacement from the target"
                ),
                "binary_flip_rate": "0.5-threshold decision disagreement",
                "absolute_bias": (
                    "absolute signed-mean policy bias; the available calibration-bias "
                    "coordinate, not a general proper-score calibration diagnostic"
                ),
                "gap_closure": (
                    "point articulation gain divided by a positive sparse-to-endpoint gap; "
                    "undefined when that native gap is absent and descriptive even above 1"
                ),
            },
            "bootstrap_contract": (
                "coordinate-specific complete-case percentile intervals resample the declared "
                "unit: selected cell, or all selected members carrying one frozen immediate-"
                "source/raw-provenance component id. Immediate and level-local provenance ids "
                "are scoped by task, bucket, and legacy round; the conservative task-global raw-"
                "provenance id joins inherited rubric reuse across R1/R2/R3 and is the headline "
                "aggregate block design. Null gap-closure coordinates are excluded from that "
                "coordinate's numerator, denominator, and resampled block panel, never coded as "
                "zero or handled by conditioning on positive-denominator draws"
            ),
            "two_stage_contract": (
                "metric-level bootstraps resample the already classified cell-grade events and "
                "coordinate point estimates. They do not rerun or continuously propagate the "
                "within-cell item/source-group bootstrap."
            ),
        },
        "panel_validation": {
            "valid": True,
            "n_tasks": len(tasks),
            "levels": list(_BREADTH_LEVELS),
            "n_metrics_per_task_level": 30,
            "n_cells": len(records),
            "report_preserves_exact_cell_identity_order": True,
            "report_preserves_frozen_design_fields": True,
        },
        "aggregate": aggregate,
        "task_level": task_level,
        "sensitivities": {
            "source_kind_and_overlap_subsets": sensitivities,
            "terminal_frontier": terminal,
        },
        "cells": records,
        "summary": {
            "n_cells": len(records),
            "n_task_level_scopes": len(task_level),
            **{
                f"balanced_task_raw_provenance_block_{name}_prevalence": row["point"]
                for name, row in aggregate_prevalence.items()
            },
        },
        "claim_boundaries": {
            "target": (
                "The target is the frozen larger-model policy itself. No supervised labels, "
                "community outcomes, compiler verdicts, or external ground truth enter this "
                "reconstruction estimand."
            ),
            "functional_substitution": (
                "A content-specific joint functional substitution is inherited unchanged from "
                "the cell certificate: fixed-target fidelity, direct larger-endpoint fidelity, "
                "scale-gap exclusion, and superiority to both matched inert-length and wrong-"
                "construct controls. Observed, nominally certified, and union-family "
                "simultaneously certified grades remain separate."
            ),
            "prevalence": (
                "Balanced prevalence describes the deliberately balanced 990 selected native "
                "action nodes. Nominal post-stratified values use inverse source-kind x breadth-"
                "tertile coverage factors as a descriptive sensitivity over the legacy native "
                "action-node inventory. Because the deterministic selector explicitly prefers "
                "new dependency/provenance components, those factors are not inclusion "
                "probabilities and the result is not a Horvitz-Thompson or randomization-based "
                "survey estimate. Neither quantity is prevalence over raw human rubrics, social "
                "norms, all possible constructs, or tasks in general."
            ),
            "hierarchy": (
                "R1/R2/R3 are operational rounds of a legacy expanded-source action-node DAG, "
                "not a certified partition, paired ancestry ladder, or modern semantic taxonomy."
            ),
            "dependence": (
                "Independent-cell and round-local raw-provenance intervals are diagnostic. The "
                "task-global raw-provenance bootstrap is conservative for cross-round inherited "
                "rubric reuse and is the headline aggregate design. Every selected member of a "
                "frozen component receives the same resampling multiplicity; subset analyses "
                "disclose when they represent only part of a source-population component."
            ),
            "metric_level_inference": (
                "These metric-level intervals condition on the cell-level certificate labels "
                "and are unadjusted across the many outcomes, task-level scopes, and declared "
                "sensitivities. Because metric selection is deterministic, they are empirical "
                "block-resampling stability intervals over the frozen panel, not design-based "
                "confidence intervals for the action-node inventory. They quantify breadth "
                "uncertainty, not a second confirmatory familywise test of every contrast."
            ),
            "fibers": (
                "Equal-but-different fibers are behavioral ordinal/vector concordance among "
                "surface-distinct explicit routes. They do not prove semantic equivalence, "
                "uniqueness, or faithful natural-language explanation."
            ),
            "dose": (
                "Address-prefix counts are deterministic articulation-dose instruments. They "
                "remain uncertified CUF units and cannot support a form-invariant units-for-"
                "scale law by themselves. A true frozen-panel onset is emitted only when the "
                "score audit authenticates the complete all-arm candidate panel; otherwise the "
                "reported quantity is explicitly the minimum passing tested dose."
            ),
            "candidate_search": (
                "Every per-cell 'best' coordinate and route is best only among the frozen scored "
                "candidate panel. It is not a global optimum over possible natural-language "
                "articulations."
            ),
            "terminal_frontier": (
                "Terminal-frontier outcomes are reported only for exact node identities that "
                "also occur in the scored primary panel. Counts or hashes never authorize "
                "imputation to carried or otherwise unscored frontier nodes."
            ),
            "universality": (
                "Even positive 990-cell prevalence estimates establish breadth within this "
                "frozen task/frame/model family; they do not by themselves prove a universal "
                "scale-articulation substitution law."
            ),
        },
    }


def _resolve_scoring_cell_ids(
    explicit_cell_ids: list[str] | None,
    *,
    execution_manifest_path: str | Path | None,
) -> tuple[str, ...] | None:
    """Resolve an omitted frozen invocation to its exact manifest-declared cell panel."""
    if explicit_cell_ids:
        # Preserve the existing explicit CLI path byte-for-byte, including order.  The frozen
        # runner validator remains responsible for accepting or rejecting that invocation.
        return tuple(explicit_cell_ids)
    if execution_manifest_path is None:
        return None
    manifest_path = Path(execution_manifest_path)
    if not manifest_path.is_file():
        raise ValueError("scoring execution manifest does not exist")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "fresh_name_execution_manifest/v2"
            or not str(manifest.get("status", "")).startswith("frozen-before-")):
        raise ValueError("scoring execution manifest is not frozen v2")
    values = manifest.get("analysis", {}).get("runner", {}).get("cell_ids")
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))):
        raise ValueError(
            "frozen analysis.runner.cell_ids must be a nonempty unique string list"
        )
    return tuple(values)


def _recalibration_panel(
        *, executor_root: Path, target_root: Path, partition: str, domain: str,
        cell_id: str, executor_job: str, target_job: str, target_arm_id: str,
        candidate_arm_ids: tuple[str, ...] | None, ladder: str,
        executor_model: str, target_model: str, n_boot: int, seed: int,
        confidence: float,
) -> dict:
    executor_data = _average_repetitions(
        load_public_index(executor_root, partition)[(executor_job, domain)])
    target_data = _average_repetitions(
        load_public_index(target_root, partition)[(target_job, domain)])
    target_orbits = _orbits(
        target_data["scores"], target_data["meta"], cell_id=cell_id)
    if target_arm_id not in target_orbits:
        raise ValueError(
            f"target shard omits {cell_id}/{target_arm_id} in {partition}")
    executor_orbits = _orbits(
        executor_data["scores"], executor_data["meta"], cell_id=cell_id)
    if candidate_arm_ids is not None:
        missing = sorted(set(candidate_arm_ids) - set(executor_orbits))
        if missing:
            raise ValueError(f"executor shard omits {cell_id} arms: {missing}")
        executor_orbits = {
            arm_id: executor_orbits[arm_id] for arm_id in candidate_arm_ids
        }
    candidates = {
        arm_id: _align_orbit(orbit, executor_data["hashes"], target_data["hashes"])
        for arm_id, orbit in executor_orbits.items()
    }
    diagnostic = oracle_mean_shift_diagnostic(
        target_orbits[target_arm_id], candidates,
        item_hashes=target_data["hashes"], n_boot=n_boot, seed=seed,
        confidence=confidence,
    )
    diagnostic.update({
        "panel_id": f"{ladder}:{partition}:{cell_id}",
        "ladder": ladder,
        "partition": partition,
        "domain": domain,
        "cell_id": cell_id,
        "executor_model": executor_model,
        "target_model": target_model,
        "target_arm_id": target_arm_id,
        "inputs": {
            "executor_root": str(executor_root),
            "target_root": str(target_root),
            "executor_shard_sha256": executor_data["shard_sha256"],
            "target_shard_sha256": target_data["shard_sha256"],
            "executor_repetitions": executor_data["repetitions"],
            "target_repetitions": target_data["repetitions"],
        },
    })
    return diagnostic


def run_recalibration_diagnostic(
        *, data_root: str | Path, n_boot: int = 5000, seed: int = 1207,
        confidence: float = 0.98,
) -> dict:
    """Reproduce the bounded five-panel oracle diagnostic from existing score shards."""
    data = Path(data_root)
    lockbox_root = data / "policy_isomorphism_lockbox_score_shards_v1"
    upper_root = data / "upper_scale_isomorphism_score_shards_v1"
    target70_root = data / "fresh_name_target_score_shards_v1"
    panels = []
    lockbox_cells = (
        ("humor", "N_humor_23"),
        ("humor", "N_humor_49"),
        ("pr", "N_pr_8"),
    )
    for index, (domain, cell_id) in enumerate(lockbox_cells):
        panels.append(_recalibration_panel(
            executor_root=lockbox_root, target_root=lockbox_root,
            partition="residual_lockbox", domain=domain, cell_id=cell_id,
            executor_job="llama3_lockbox_executor",
            target_job="llama8_lockbox_target", target_arm_id="name",
            candidate_arm_ids=None, ladder="Llama-3.2-3B_to_Llama-3.1-8B",
            executor_model="meta-llama/Llama-3.2-3B-Instruct",
            target_model="meta-llama/Llama-3.1-8B-Instruct",
            n_boot=n_boot, seed=seed + 1009 * index, confidence=confidence,
        ))
    for index, (partition, articulation_id) in enumerate((
            ("residual_prompt_selection", "iso_definition"),
            ("residual_unit_certification", "iso_rubric"),
    ), start=len(panels)):
        panels.append(_recalibration_panel(
            executor_root=upper_root, target_root=target70_root,
            partition=partition, domain="humor", cell_id="N_humor_49",
            executor_job="llama8_upper", target_job="llama70_n_target",
            target_arm_id="target",
            candidate_arm_ids=("name", articulation_id),
            ladder="Llama-3.1-8B_to_Llama-3.3-70B-FP8",
            executor_model="meta-llama/Llama-3.1-8B-Instruct",
            target_model="nvidia/Llama-3.3-70B-Instruct-FP8",
            n_boot=n_boot, seed=seed + 1009 * index, confidence=confidence,
        ))
    articulation_rows = [
        (panel, row)
        for panel in panels
        for row in panel["rows"]
        if row["arm_id"] not in {"name", "name_plus_crossfit_mean_shift"}
    ]
    return {
        "schema": "policy_isomorphism_oracle_recalibration_bundle/v1",
        "status": "retrospective-target-score-oracle-diagnostic",
        "claim_eligible_as_unsupervised_reconstruction": False,
        "scope": (
            "Five bounded comparisons requested to authenticate the audit: three historical "
            "3B-to-8B lockbox cells and the two historical 8B-to-70B public folds. The latter "
            "use a Llama-3.3 FP8 target and are not the unscored same-version H49 confirmation."
        ),
        "panels": panels,
        "summary": {
            "n_panels": len(panels),
            "n_articulation_rows": len(articulation_rows),
            "n_oracle_dominated_on_all_four_point_coordinates": sum(
                bool(row.get(
                    "oracle_mean_shift_point_dominates_all_four_coordinates"))
                for _panel, row in articulation_rows
            ),
            "n_rank_gains_over_oracle_with_ci_above_zero": sum(
                bool(row["improvement_over_oracle_mean_shift"]["spearman"]["CI"]
                     and row["improvement_over_oracle_mean_shift"]["spearman"]["CI"][0]
                     > 0.0)
                for _panel, row in articulation_rows
            ),
            "n_rank_losses_vs_oracle_with_ci_below_zero": sum(
                bool(row["improvement_over_oracle_mean_shift"]["spearman"]["CI"]
                     and row["improvement_over_oracle_mean_shift"]["spearman"]["CI"][1]
                     < 0.0)
                for _panel, row in articulation_rows
            ),
        },
    }


def recalibration_markdown_table(report: dict) -> str:
    lines = [
        "# Cross-fitted one-scalar recalibration diagnostic",
        "",
        ("This target-score oracle is a supervision price line, not an admissible "
         "unsupervised articulation arm."),
        "",
        "| Ladder / fold / cell | Arm | n | MAE | |bias| | rho | flip | Delta rho vs oracle (CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for panel in report["panels"]:
        for row in panel["rows"]:
            metrics = row["heldout_robust"]
            delta = row.get("improvement_over_oracle_mean_shift")
            if delta is None:
                delta_text = "reference"
            else:
                value = delta["spearman"]
                ci = value["CI"]
                delta_text = f"{value['point']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
            lines.append(
                f"| {panel['ladder']} / {panel['partition']} / {panel['cell_id']} "
                f"| {row['arm_id']} | {panel['split']['n_evaluation']} "
                f"| {metrics['mae_tvd']:.3f} | {metrics['absolute_bias']:.3f} "
                f"| {metrics['spearman']:.3f} | {metrics['binary_flip_rate']:.3f} "
                f"| {delta_text} |"
            )
    summary = report["summary"]
    lines.extend([
        "",
        (f"Pointwise, the one-scalar oracle dominates "
         f"{summary['n_oracle_dominated_on_all_four_point_coordinates']}/"
         f"{summary['n_articulation_rows']} articulation rows on all four coordinates. "
         f"Articulation has {summary['n_rank_gains_over_oracle_with_ci_above_zero']} "
         "rank gains and "
         f"{summary['n_rank_losses_vs_oracle_with_ci_below_zero']} rank losses whose paired "
         "interval excludes zero."),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--build-selection", action="store_true",
        help="compile the frozen breadth search-to-validation arm selection",
    )
    parser.add_argument(
        "--release-only", action="store_true",
        help=(
            "authenticate the frozen two-manifest DAG and emit only its production "
            "lockbox-release artifact"
        ),
    )
    parser.add_argument(
        "--summarize-breadth", action="store_true",
        help="summarize the complete frozen 11 x 3 x 30 breadth policy report",
    )
    parser.add_argument(
        "--recalibration-diagnostic", action="store_true",
        help="run the bounded five-panel target-score-oracle diagnostic on saved shards",
    )
    parser.add_argument(
        "--data-root", default="notebooks/data/two_faces_20260702",
        help="data root for --recalibration-diagnostic",
    )
    parser.add_argument(
        "--recalibration-table", default=None,
        help="optional Markdown table path for --recalibration-diagnostic",
    )
    parser.add_argument(
        "--breadth-report", default=None,
        help="complete generic policy-isomorphism report for breadth summary mode",
    )
    parser.add_argument("--breadth-n-boot", type=int, default=5000)
    parser.add_argument("--breadth-seed", type=int, default=7319)
    parser.add_argument("--search-execution-manifest", default=None)
    parser.add_argument("--search-report", default=None)
    parser.add_argument("--metric-panel", default=None)
    parser.add_argument(
        "--additional-artifact", action="append", default=[],
        help="non-panel artifact path already bound by the search manifest; repeat as needed",
    )
    parser.add_argument("--selected-phase", default=None)
    parser.add_argument("--selected-partition", default=None)
    parser.add_argument("--join-report", action="append",
                        help="join two or more existing fold reports instead of rescoring")
    parser.add_argument("--pool-fold-items", action="store_true",
                        help="also pool disjoint public items for a stratified precision audit")
    parser.add_argument("--pooled-n-boot", type=int, default=10000)
    parser.add_argument("--pooled-seed", type=int, default=1217)
    parser.add_argument("--executor-shard-root")
    parser.add_argument("--target-shard-root", default=None)
    parser.add_argument("--scale-comparator-shard-root", default=None,
                        help="larger-executor shards for a paired fixed-target scale-step test")
    parser.add_argument("--scale-comparator-job", default=None)
    parser.add_argument("--scale-comparator-arm-id", default="name")
    parser.add_argument(
        "--scale-comparator-use-target", action="store_true",
        help="use the fixed target orbit itself as the larger sparse scale comparator",
    )
    parser.add_argument("--arm-bank")
    parser.add_argument("--partition")
    parser.add_argument("--packet-root", default=None,
                        help=("local fresh-item packet root; with --packet-manifest, enables "
                              "authenticated source-group cluster inference"))
    parser.add_argument("--packet-manifest", default=None,
                        help="fresh-item packet manifest that authenticates --packet-root")
    parser.add_argument(
        "--execution-manifest",
        default=None,
        help="frozen v2 execution manifest; required to authorize a sealed partition",
    )
    parser.add_argument(
        "--selection-artifact",
        default=None,
        help="exact selection artifact bound by --execution-manifest",
    )
    parser.add_argument(
        "--lockbox-release-artifact",
        default=None,
        help="production calibration-release artifact required for a frozen lockbox",
    )
    parser.add_argument(
        "--allow-fake-inputs",
        action="store_true",
        help="test-only: permit explicitly marked FakeVLLM shards",
    )
    parser.add_argument("--small-job", default="llama3_small")
    parser.add_argument("--big-job", default="llama8_big_sparse")
    parser.add_argument("--target-arm-id", default="name")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--mae-margin", type=float, default=0.02)
    parser.add_argument("--rho-margin", type=float, default=0.05)
    parser.add_argument("--flip-margin", type=float, default=0.02)
    parser.add_argument("--bias-margin", type=float, default=0.02)
    parser.add_argument("--functional-rho-floor", type=float, default=0.70)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--fiber-mutual-rho-floor", type=float, default=0.90)
    parser.add_argument(
        "--fiber-mutual-rho-sensitivity-floor", type=float, default=0.85)
    parser.add_argument("--fiber-min-rank-valid-fraction", type=float, default=0.99)
    parser.add_argument("--fiber-distinctness-floor", type=float, default=0.35)
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--crossfit-only", action="store_true",
                        help="exclude arms whose teaching source is the evaluation partition")
    parser.add_argument("--cell-id", action="append",
                        help="score only the named bank cell; repeat for multiple cells")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--write-lockbox-release",
        action="store_true",
        help="after a production calibration run, write the frozen lockbox-release artifact",
    )
    args = parser.parse_args()
    special_modes = sum(bool(value) for value in (
        args.build_selection,
        args.release_only,
        args.summarize_breadth,
        args.recalibration_diagnostic,
        args.join_report,
    ))
    if special_modes > 1:
        parser.error("selection, release, breadth, recalibration, and join modes are exclusive")
    if args.release_only and (
            args.build_selection or args.summarize_breadth or args.join_report):
        parser.error(
            "--release-only is mutually exclusive with selection, summary, and join modes"
        )
    if args.release_only:
        required = {
            "--search-execution-manifest": args.search_execution_manifest,
            "--search-report": args.search_report,
            "--execution-manifest": args.execution_manifest,
            "--selection-artifact": args.selection_artifact,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(
                f"required for release-only mode: {', '.join(sorted(missing))}"
            )
        if args.allow_fake_inputs:
            parser.error("--release-only never permits --allow-fake-inputs")
        if args.write_lockbox_release or args.lockbox_release_artifact:
            parser.error(
                "--release-only emits --out directly; do not supply another release mode/path"
            )
        result = write_two_manifest_lockbox_release(
            search_execution_manifest_path=args.search_execution_manifest,
            search_report_path=args.search_report,
            selection_artifact_path=args.selection_artifact,
            validation_execution_manifest_path=args.execution_manifest,
            release_artifact_path=args.out,
        )
        print(json.dumps(result, indent=1))
        return
    if args.recalibration_diagnostic:
        report = run_recalibration_diagnostic(
            data_root=args.data_root,
            n_boot=args.n_boot,
            seed=args.seed,
            confidence=args.confidence,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1) + "\n")
        table = Path(args.recalibration_table) if args.recalibration_table else out.with_suffix(
            ".md")
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text(recalibration_markdown_table(report))
        print(json.dumps({
            "out": str(out),
            "table": str(table),
            **report["summary"],
        }, indent=1))
        return
    if args.summarize_breadth:
        required = {
            "--breadth-report": args.breadth_report,
            "--metric-panel": args.metric_panel,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(
                f"required for breadth summary mode: {', '.join(sorted(missing))}"
            )
        result = summarize_breadth_decomposition(
            args.breadth_report,
            args.metric_panel,
            n_boot=args.breadth_n_boot,
            seed=args.breadth_seed,
            confidence=args.confidence,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"out": str(out), **result["summary"]}, indent=1))
        return
    if args.build_selection:
        required = {
            "--search-execution-manifest": args.search_execution_manifest,
            "--search-report": args.search_report,
            "--arm-bank": args.arm_bank,
            "--packet-manifest": args.packet_manifest,
            "--metric-panel": args.metric_panel,
            "--selected-phase": args.selected_phase,
            "--selected-partition": args.selected_partition,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(
                f"required for selection mode: {', '.join(sorted(missing))}"
            )
        result = write_policy_articulation_selection(
            out_path=args.out,
            search_execution_manifest_path=args.search_execution_manifest,
            search_report_path=args.search_report,
            arm_bank_path=args.arm_bank,
            packet_manifest_path=args.packet_manifest,
            metric_panel_path=args.metric_panel,
            additional_artifact_paths=tuple(args.additional_artifact),
            selected_phase=args.selected_phase,
            selected_partition=args.selected_partition,
        )
        print(json.dumps(result, indent=1))
        return
    if args.join_report:
        result = summarize_crossfold_fibers(args.join_report)
        if args.pool_fold_items:
            pooled = pool_crossfold_policy_reports(
                args.join_report,
                n_boot=args.pooled_n_boot,
                seed=args.pooled_seed,
                confidence=args.confidence,
                functional_rho_floor=args.functional_rho_floor,
                packet_root=args.packet_root,
                packet_manifest_path=args.packet_manifest,
            )
            result["pooled_item_precision_analysis"] = pooled
            result["summary"]["pooled_item_precision_analysis"] = pooled["summary"]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        print(json.dumps({"out": str(out), **result["summary"]}, indent=1))
        return
    missing = [name for name, value in (
        ("--executor-shard-root", args.executor_shard_root),
        ("--arm-bank", args.arm_bank),
        ("--partition", args.partition),
    ) if not value]
    if missing:
        parser.error(f"required for scoring mode: {', '.join(missing)}")
    scoring_cell_ids = _resolve_scoring_cell_ids(
        args.cell_id,
        execution_manifest_path=args.execution_manifest,
    )
    report = run(
        executor_shard_root=args.executor_shard_root,
        target_shard_root=args.target_shard_root, arm_bank_path=args.arm_bank,
        scale_comparator_shard_root=args.scale_comparator_shard_root,
        scale_comparator_job=args.scale_comparator_job,
        scale_comparator_arm_id=args.scale_comparator_arm_id,
        scale_comparator_use_target=args.scale_comparator_use_target,
        packet_root=args.packet_root, packet_manifest_path=args.packet_manifest,
        partition=args.partition, small_job=args.small_job, big_job=args.big_job,
        target_arm_id=args.target_arm_id,
        n_boot=args.n_boot, seed=args.seed, mae_margin=args.mae_margin,
        rho_margin=args.rho_margin, flip_margin=args.flip_margin,
        bias_margin=args.bias_margin, functional_rho_floor=args.functional_rho_floor,
        fiber_mutual_rho_floor=args.fiber_mutual_rho_floor,
        fiber_mutual_rho_sensitivity_floor=args.fiber_mutual_rho_sensitivity_floor,
        fiber_min_rank_valid_fraction=args.fiber_min_rank_valid_fraction,
        fiber_distinctness_floor=args.fiber_distinctness_floor,
        include_controls=args.include_controls,
        crossfit_only=args.crossfit_only, confidence=args.confidence,
        cell_ids=scoring_cell_ids,
        execution_manifest_path=args.execution_manifest,
        selection_artifact_path=args.selection_artifact,
        lockbox_release_artifact_path=args.lockbox_release_artifact,
        allow_fake_inputs=args.allow_fake_inputs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    release = None
    if args.write_lockbox_release:
        if not args.execution_manifest or not args.selection_artifact:
            parser.error(
                "--write-lockbox-release requires --execution-manifest and --selection-artifact"
            )
        release = write_calibration_release_artifact(
            report,
            report_path=out,
            execution_manifest_path=args.execution_manifest,
            selection_artifact_path=args.selection_artifact,
        )
    print(json.dumps({"out": str(out), "lockbox_release": release,
                      **report["summary"]}, indent=1))


if __name__ == "__main__":
    main()
