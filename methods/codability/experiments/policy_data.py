"""Shared data and partition invariants for fixed-target policy experiments.

Experiment entry points should configure this layer rather than implementing their own shard
loading, repetition averaging, orbit construction, or hash alignment.  The generic loader can
read an explicitly supplied partition; authorization remains the responsibility of the calling
entry point through :func:`require_partition`.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Collection

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import (
    sha256_bytes,
    sha256_file,
    text_sha256,
)


# Canonical analysis-implementation closure.  This is the single source of truth for both the
# execution-manifest compiler and the report generator's self-recorded implementation: the two
# lists are compared for exact equality at every release/selection gate, so they must never be
# maintained separately (a 15-vs-12 drift between them blocked the concluding_policy_v1 lockbox
# on 2026-07-14).  Package __init__ files execute at import and therefore belong in the closure.
ANALYSIS_IMPLEMENTATION_PATHS = (
    "methods/codability/__init__.py",
    "methods/codability/experiments/__init__.py",
    "methods/codability/experiments/run_policy_isomorphism.py",
    "methods/codability/experiments/score_fresh_name_arms.py",
    "methods/codability/experiments/policy_isomorphism.py",
    "methods/codability/experiments/policy_data.py",
    "methods/codability/experiments/build_fresh_item_partitions.py",
    "methods/codability/experiments/target_articulation_frontier.py",
    "methods/codability/grid_auc_report.py",
    "methods/codability/experiments/common_target_ladder.py",
    "methods/metric_implementer/__init__.py",
    "methods/metric_implementer/manifest.py",
    "methods/metric_implementer/artifact.py",
    "methods/metric_implementer/config.py",
    "methods/metric_implementer/vinfo.py",
)

PUBLIC_DEVELOPMENT_PARTITIONS = (
    "residual_prompt_selection",
    "residual_unit_certification",
    "same_version_upper_calibration",
)
OPENED_CONFIRMATION_PARTITIONS = ("residual_lockbox",)
POLICY_ISOMORPHISM_PARTITIONS = (
    *PUBLIC_DEVELOPMENT_PARTITIONS,
    *OPENED_CONFIRMATION_PARTITIONS,
)


def atomic_savez_compressed(path: Path, **arrays) -> None:
    """Commit a complete NPZ with one same-filesystem rename."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial.npz")
    temporary.unlink(missing_ok=True)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Commit a UTF-8 text artifact without exposing a partial destination."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_declared_path(value: str, *, manifest_path: Path) -> Path:
    """Resolve a frozen repo-relative path without trusting the caller's cwd alone."""
    path = Path(value)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[3]
    candidates = (Path.cwd() / path, repo_root / path, manifest_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return repo_root / path


def validate_frozen_implementation(
    manifest: dict,
    *,
    manifest_path: str | Path,
    section: str,
) -> dict:
    """Authenticate a transitive scoring or analysis implementation file set."""
    manifest_path = Path(manifest_path)
    specification = manifest.get("implementation", {}).get(section)
    if not isinstance(specification, dict):
        raise ValueError(f"execution manifest omits implementation section {section!r}")
    files = specification.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"implementation section {section!r} has no files")
    declared_paths = [row.get("path") for row in files]
    if (None in declared_paths or len(set(declared_paths)) != len(declared_paths)):
        raise ValueError(f"implementation section {section!r} has invalid file paths")
    checked = []
    for row in files:
        path = _resolve_declared_path(row["path"], manifest_path=manifest_path)
        expected = row.get("sha256")
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(
                f"frozen {section} implementation changed: {row.get('path')!r}"
            )
        checked.append({"path": row["path"], "sha256": expected})
    return {
        "section": section,
        "semantics": specification.get("semantics"),
        "files": checked,
        "valid": True,
    }


def validate_additional_artifacts(
    manifest: dict,
    *,
    manifest_path: str | Path,
) -> dict:
    """Authenticate optional experiment inputs not covered by the core manifest fields.

    The breadth experiment uses this extension to bind its metric panel.  Keeping the field
    optional preserves every already-frozen execution manifest.
    """
    rows = manifest.get("additional_artifacts", [])
    if not isinstance(rows, list):
        raise ValueError("execution manifest additional_artifacts must be a list")
    manifest_path = Path(manifest_path)
    checked = []
    declared_paths = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"execution manifest additional_artifacts[{index}] must be an object"
            )
        declared_path = row.get("path")
        expected = row.get("sha256")
        if not isinstance(declared_path, str) or not declared_path or not expected:
            raise ValueError(
                f"execution manifest additional_artifacts[{index}] omits path/sha256"
            )
        declared_paths.append(declared_path)
        path = _resolve_declared_path(declared_path, manifest_path=manifest_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(
                f"execution manifest additional artifact changed: {declared_path!r}"
            )
        checked.append({"path": declared_path, "sha256": expected})
    if len(set(declared_paths)) != len(declared_paths):
        raise ValueError("execution manifest additional_artifacts contains duplicate paths")
    return {"valid": True, "files": checked}


def selection_required_for_phase(manifest: dict, phase: str) -> bool:
    """Return whether ``phase`` must consume the manifest-bound selection artifact."""
    phases = manifest.get("phases", {})
    default = ["lockbox"] if "lockbox" in phases else []
    declared = manifest.get("selection_required_phases", default)
    if (not isinstance(declared, list)
            or any(not isinstance(value, str) for value in declared)
            or len(set(declared)) != len(declared)):
        raise ValueError("execution manifest selection_required_phases is invalid")
    unexpected = sorted(set(declared) - set(phases))
    if unexpected:
        raise ValueError(
            f"selection_required_phases names undeclared phases: {unexpected}"
        )
    return phase in set(declared)


def _validate_unique_artifact_rows(rows, *, label: str) -> list[dict]:
    """Require unambiguous role/path/hash bindings before comparing artifact panels."""
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        role, path, digest = (row.get(key) for key in ("role", "path", "sha256"))
        if not all(isinstance(value, str) and value for value in (role, path, digest)):
            raise ValueError(f"{label}[{index}] omits role/path/sha256")
        normalized.append({"role": role, "path": path, "sha256": digest})
    for field in ("role", "path"):
        values = [row[field] for row in normalized]
        if len(values) != len(set(values)):
            raise ValueError(f"{label} contains duplicate {field} bindings")
    if len({json.dumps(row, sort_keys=True) for row in normalized}) != len(normalized):
        raise ValueError(f"{label} contains duplicate artifact rows")
    return normalized


def _single_manifest_phase(manifest: dict, *, label: str) -> tuple[str, str]:
    phases = manifest.get("phases")
    if not isinstance(phases, dict) or len(phases) != 1:
        raise ValueError(f"{label} must declare exactly one phase")
    phase, partitions = next(iter(phases.items()))
    if (not isinstance(phase, str) or not phase or not isinstance(partitions, list)
            or len(partitions) != 1 or not isinstance(partitions[0], str)
            or not partitions[0]):
        raise ValueError(f"{label} must declare exactly one phase/partition pair")
    return phase, partitions[0]


def _validate_search_validation_manifest_transition(
        source: dict, current: dict, *, selection: dict,
        selection_path: Path, current_manifest_path: Path) -> dict:
    """Prove that validation changes only the preregistered stage-specific fields.

    The two manifests are deliberately separate artifacts, so merely sharing the bank and
    packet hashes is insufficient: it would otherwise be possible to change code, models,
    readout semantics, numerical gates, or resource policy after inspecting search outcomes.
    """
    required_types = {
        "implementation": dict,
        "protocol_manifest_path": str,
        "protocol_manifest_sha256": str,
        "target_prompt_manifest_path": str,
        "target_prompt_manifest_sha256": str,
        "readout_template_sha256": str,
        "binary_readout": str,
        "label_support": list,
        "teacher_forced_label_validation": dict,
        "domains": list,
        "domain_tasks": dict,
        "item_text_max_chars_by_task": dict,
        "model_family": str,
        "model_jobs": list,
        "execution_environment": dict,
        "resource_policy": dict,
        "selection_policy": dict,
        "analysis": dict,
    }
    for field, expected_type in required_types.items():
        value = source.get(field)
        if not isinstance(value, expected_type) or isinstance(value, bool):
            raise ValueError(
                f"policy-articulation search manifest omits frozen {field} metadata"
            )
    domains = source["domains"]
    if (not domains or any(not isinstance(value, str) or not value for value in domains)
            or len(domains) != len(set(domains))):
        raise ValueError("policy-articulation search manifest has invalid domains")
    if (set(source["domain_tasks"]) != set(domains)
            or any(not isinstance(value, str) or not value
                   for value in source["domain_tasks"].values())):
        raise ValueError("policy-articulation search manifest has invalid domain_tasks")
    truncation = source["item_text_max_chars_by_task"]
    if (set(truncation) != set(domains)
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                   for value in truncation.values())):
        raise ValueError(
            "policy-articulation search manifest has invalid task truncation policy"
        )
    if (not source["implementation"] or not source["selection_policy"]
            or not source["execution_environment"] or not source["resource_policy"]):
        raise ValueError("policy-articulation search manifest has empty frozen gate metadata")
    source_runner = source["analysis"].get("runner")
    current_runner = current.get("analysis", {}).get("runner")
    if not isinstance(source_runner, dict) or not source_runner:
        raise ValueError("policy-articulation search manifest omits analysis.runner")
    if not isinstance(current_runner, dict) or not current_runner:
        raise ValueError("policy-articulation validation manifest omits analysis.runner")

    source_phase, source_partition = _single_manifest_phase(
        source, label="policy-articulation search manifest")
    current_phase, current_partition = _single_manifest_phase(
        current, label="policy-articulation validation manifest")
    if ((source_phase, source_partition) != (
            selection.get("search_phase"), selection.get("search_partition"))
            or (current_phase, current_partition) != (
                selection.get("selected_phase"), selection.get("selected_partition"))
            or source_phase == current_phase or source_partition == current_partition):
        raise ValueError("policy-articulation search/validation phase transition is invalid")
    source_access = source.get("phase_access")
    current_access = current.get("phase_access")
    expected_current_access = (
        "sealed_confirmation" if current_phase == "lockbox" else "open_development"
    )
    if (not isinstance(source_access, dict)
            or source_access != {source_phase: source_access.get(source_phase)}
            or not isinstance(current_access, dict)
            or current_access != {current_phase: current_access.get(current_phase)}
            or source_access[source_phase] != "open_development"
            or current_access[current_phase] != expected_current_access):
        raise ValueError(
            "policy-articulation phase access does not follow the frozen "
            "open-search/declared-confirmation transition"
        )
    if (source.get("selection_required_phases") != []
            or current.get("selection_required_phases") != [current_phase]):
        raise ValueError("policy-articulation selection gate is invalid across the DAG")
    if any(key in source for key in (
            "selection_artifact_path", "selection_artifact_sha256")):
        raise ValueError("policy-articulation search manifest binds a future selection")
    expected_selection_sha = sha256_file(selection_path)
    current_selection_path = current.get("selection_artifact_path")
    if (not isinstance(current_selection_path, str)
            or current.get("selection_artifact_sha256") != expected_selection_sha
            or _resolve_declared_path(
                current_selection_path, manifest_path=current_manifest_path
            ).resolve() != selection_path.resolve()):
        raise ValueError("policy-articulation validation selection binding is invalid")

    for label, manifest in (("search", source), ("validation", current)):
        if (manifest.get("schema") != "fresh_name_execution_manifest/v2"
                or not str(manifest.get("status", "")).startswith("frozen-before-")):
            raise ValueError(f"policy-articulation {label} manifest is not frozen v2")

    source_jobs = source["model_jobs"]
    current_jobs = current.get("model_jobs")
    if (not isinstance(current_jobs, list) or not source_jobs
            or any(not isinstance(row, dict) for row in source_jobs + current_jobs)):
        raise ValueError("policy-articulation model job panel is invalid")
    source_job_ids = [row.get("id") for row in source_jobs]
    current_job_ids = [row.get("id") for row in current_jobs]
    if (None in source_job_ids or len(source_job_ids) != len(set(source_job_ids))
            or source_job_ids != current_job_ids):
        raise ValueError("policy-articulation model job panel changed across the DAG")
    for source_job, current_job in zip(source_jobs, current_jobs):
        if (source_job.get("required_repetitions") != [0]
                or current_job.get("required_repetitions") != [0, 1]):
            raise ValueError(
                "policy-articulation repetitions must transition exactly [0] to [0, 1]"
            )

    for label, manifest, expected in (
            ("search", source, 2000), ("validation", current, 10000)):
        analysis = manifest.get("analysis", {})
        runner = analysis.get("runner", {})
        if (analysis.get("n_boot") != expected or runner.get("n_boot") != expected):
            raise ValueError(
                f"policy-articulation {label} n_boot must be exactly {expected}"
            )

    normalized = []
    for manifest in (source, current):
        value = deepcopy(manifest)
        for key in (
                "status", "phases", "phase_access", "selection_required_phases",
                "selection_artifact_path", "selection_artifact_sha256",
                # This validation-stage-only certificate is derived from, and rechecked
                # against, the immutable source artifacts below.  It is not a scientific
                # degree of freedom in the search-to-validation transition.
                "selection_provenance_validation_at_freeze"):
            value.pop(key, None)
        for job in value["model_jobs"]:
            job["required_repetitions"] = "__FROZEN_STAGE_REPETITIONS__"
        value["analysis"]["n_boot"] = "__FROZEN_STAGE_N_BOOT__"
        value["analysis"]["runner"]["n_boot"] = "__FROZEN_STAGE_N_BOOT__"
        normalized.append(value)
    if normalized[0] != normalized[1]:
        changed = sorted(
            key for key in set(normalized[0]) | set(normalized[1])
            if normalized[0].get(key) != normalized[1].get(key)
        )
        raise ValueError(
            "policy-articulation immutable search/validation fields differ: "
            f"{changed}"
        )
    return {
        "valid": True,
        "search_phase": source_phase,
        "validation_phase": current_phase,
        "normalized_runner": {
            **source_runner, "n_boot": "search=2000;validation=10000"
        },
    }


def _validate_selection_content_hash(selection: dict) -> str:
    observed = selection.get("selection_content_sha256")
    if not isinstance(observed, str) or not observed:
        raise ValueError("policy-articulation selection omits its content self-hash")
    payload = deepcopy(selection)
    payload.pop("selection_content_sha256", None)
    expected = sha256_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode())
    if observed != expected:
        raise ValueError("policy-articulation selection content self-hash is invalid")
    return expected


def _validate_report_production_closure(
        report: dict, *, source: dict, source_sha: str) -> None:
    """Recheck every report certificate used to authorize post-search selection."""
    invocation = report.get("frozen_invocation_validation")
    expected_runner = source.get("analysis", {}).get("runner")
    if (not isinstance(invocation, dict) or invocation.get("valid") is not True
            or invocation.get("runner") != expected_runner):
        raise ValueError(
            "policy-articulation search report runner differs from source analysis.runner"
        )
    expected_analysis = source.get("implementation", {}).get("analysis")
    if (not isinstance(expected_analysis, dict)
            or report.get("analysis_implementation") != expected_analysis):
        raise ValueError(
            "policy-articulation search analysis implementation is unauthenticated"
        )
    if report.get("cell_panel_identity_validation", {}).get("valid") is not True:
        raise ValueError("policy-articulation search report lacks cell-panel closure")

    jobs = {row.get("id"): row for row in source.get("model_jobs", [])}
    if None in jobs or len(jobs) != len(source.get("model_jobs", [])):
        raise ValueError("policy-articulation source model jobs are invalid")
    runner = source["analysis"]["runner"]
    expected_backend = source.get("execution_environment", {}).get(
        "production_backend_class")
    expected_by_role = {
        "small": runner.get("small_job"),
        "target": runner.get("big_job"),
    }
    if (not expected_backend
            or any(job_id not in jobs for job_id in expected_by_role.values())):
        raise ValueError("policy-articulation production job/backend metadata is incomplete")
    report_cells = report.get("cells")
    if (not isinstance(report_cells, list) or not report_cells
            or any(not isinstance(cell, dict) for cell in report_cells)):
        raise ValueError("policy-articulation search report has an invalid cell panel")
    for cell in report_cells:
        cell_id = cell.get("cell_id")
        prompt = (
            cell.get("executor_prompt_bank_validation", {}),
            cell.get("target_prompt_bank_validation", {}),
        )
        arm_panel = cell.get("scored_arm_panel_validation", {})
        identity = cell.get("score_cell_identity_validation", {})
        if (any(value.get("valid") is not True for value in prompt)
                or set(arm_panel) != {"small", "target"}
                or any(value.get("valid") is not True for value in arm_panel.values())
                or set(identity) != {"small", "target"}
                or any(value.get("valid") is not True for value in identity.values())):
            raise ValueError(
                f"policy-articulation search report lacks prompt/arm/cell closure for {cell_id}"
            )
        provenance = cell.get("score_provenance_validation")
        if not isinstance(provenance, dict) or set(provenance) != {"small", "target"}:
            raise ValueError(
                "policy-articulation search report lacks complete production score provenance"
            )
        for role, job_id in expected_by_role.items():
            job = jobs[job_id]
            expected = {
                "valid": True,
                "job_id": job_id,
                "repetitions": job.get("required_repetitions"),
                "execution_manifest_sha256": source_sha,
                "arm_bank_sha256": source.get("arm_bank_sha256"),
                "packet_manifest_sha256": source.get("packet_manifest_sha256"),
                "binary_readout": source.get("binary_readout"),
                "readout_template_sha256": source.get("readout_template_sha256"),
                "role": job.get("role"),
                "backend_class": expected_backend,
                "fake_backend": False,
            }
            validation = provenance[role]
            if (not isinstance(validation, dict)
                    or any(validation.get(key) != value for key, value in expected.items())):
                raise ValueError(
                    "policy-articulation search report lacks complete production score "
                    f"provenance for {cell_id}/{role}"
                )
        if runner.get("source_group_inference") is True and (
                cell.get("source_group_validation", {}).get("valid") is not True):
            raise ValueError(
                f"policy-articulation search report lacks source-group closure for {cell_id}"
            )


def _validate_selection_cell_metadata(
        selection: dict, *, bank: dict, report: dict, policy: dict,
        expected_cell_ids: list[str]) -> list[str]:
    """Validate the generic 1--4 candidate/role/control schema without scorer imports."""
    roles = policy.get("roles_in_order")
    if (not isinstance(roles, list) or not roles or len(roles) > 4
            or any(not isinstance(role, str) or not role for role in roles)
            or len(roles) != len(set(roles))):
        raise ValueError("policy-articulation selection policy has invalid roles")
    if (policy.get("minimum_candidates_per_cell") != 1
            or policy.get("maximum_candidates_per_cell") != 4):
        raise ValueError("policy-articulation selection policy must freeze 1--4 candidates")

    bank_rows = bank.get("cells", [])
    report_rows = report.get("cells", [])
    selection_rows = selection.get("cells")
    if (not isinstance(bank_rows, list) or not isinstance(report_rows, list)
            or not isinstance(selection_rows, list)
            or any(not isinstance(cell, dict)
                   for cell in bank_rows + report_rows + selection_rows)):
        raise ValueError("policy-articulation bank/report/selection cell panel is invalid")
    bank_ids = [cell.get("id") for cell in bank_rows]
    report_ids = [cell.get("cell_id") for cell in report_rows]
    selection_ids = (
        [cell.get("cell_id") for cell in selection_rows]
        if isinstance(selection_rows, list) else []
    )
    if (not isinstance(expected_cell_ids, list) or not expected_cell_ids
            or expected_cell_ids != bank_ids
            or not bank_ids or len(bank_ids) != len(set(bank_ids))
            or selection_ids != bank_ids or report_ids != bank_ids
            or selection.get("n_cells") != len(bank_ids)):
        raise ValueError(
            "policy-articulation selection must nonemptily cover every bank/report cell"
        )
    report_by_id = {cell["cell_id"]: cell for cell in report_rows}
    candidate_counts = []
    required_control_provenances = (
        "wrong_construct_control", "inert_length_control",
    )
    for bank_cell, selected_cell in zip(bank_rows, selection_rows):
        cell_id = bank_cell["id"]
        bank_arms = bank_cell.get("arms")
        if (not isinstance(bank_arms, list)
                or any(not isinstance(arm, dict) for arm in bank_arms)):
            raise ValueError(f"policy-articulation bank cell {cell_id} has invalid arms")
        arms = {arm.get("id"): arm for arm in bank_arms}
        if None in arms or len(arms) != len(bank_arms) or "name" not in arms:
            raise ValueError(f"policy-articulation bank cell {cell_id} has invalid arms")
        candidates = selected_cell.get("candidate_arm_ids")
        controls = selected_cell.get("control_ids")
        allowed = selected_cell.get("allowed_arm_ids")
        if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 4
                or len(candidates) != len(set(candidates))
                or any(candidate not in arms or candidate == "name"
                       or arms[candidate].get("control_for") is not None
                       for candidate in candidates)):
            raise ValueError(f"policy-articulation cell {cell_id} has invalid candidates")
        candidate_counts.append(len(candidates))
        expected_controls = []
        controls_by_candidate = {}
        for candidate in candidates:
            matched = [
                arm for arm in bank_cell["arms"] if arm.get("control_for") == candidate
            ]
            by_provenance = {arm.get("provenance"): arm for arm in matched}
            if (len(matched) != 2 or set(by_provenance) != set(
                    required_control_provenances)):
                raise ValueError(
                    f"policy-articulation cell {cell_id}/{candidate} lacks exact controls"
                )
            candidate_controls = [
                by_provenance[provenance]["id"]
                for provenance in required_control_provenances
            ]
            controls_by_candidate[candidate] = candidate_controls
            expected_controls.extend(candidate_controls)
        if (not isinstance(controls, list) or not isinstance(allowed, list)
                or controls != expected_controls or len(controls) != len(set(controls))
                or selected_cell.get("required_control_provenances")
                != list(required_control_provenances)
                or allowed != ["name", *candidates, *expected_controls]
                or len(allowed) != len(set(allowed))):
            raise ValueError(f"policy-articulation cell {cell_id} has invalid control metadata")

        assignments = selected_cell.get("role_assignments")
        if (not isinstance(assignments, list)
                or any(not isinstance(row, dict) for row in assignments)
                or [row.get("role") for row in assignments] != roles):
            raise ValueError(f"policy-articulation cell {cell_id} has invalid role assignments")
        assigned_roles = defaultdict(list)
        for assignment in assignments:
            role = assignment["role"]
            status = assignment.get("status")
            arm_id = assignment.get("arm_id")
            rank = assignment.get("role_rank")
            reason = assignment.get("selection_reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError(
                    f"policy-articulation cell {cell_id}/{role} omits a selection reason"
                )
            if status == "assigned":
                if (arm_id not in candidates or not isinstance(rank, int)
                        or isinstance(rank, bool) or rank <= 0):
                    raise ValueError(
                        f"policy-articulation cell {cell_id}/{role} has invalid assignment"
                    )
                assigned_roles[arm_id].append(assignment)
            elif status == "not_available":
                if arm_id is not None or rank is not None:
                    raise ValueError(
                        f"policy-articulation cell {cell_id}/{role} has invalid null assignment"
                    )
            else:
                raise ValueError(
                    f"policy-articulation cell {cell_id}/{role} has invalid role status"
                )

        selections = selected_cell.get("candidate_selections")
        if (not isinstance(selections, list)
                or any(not isinstance(row, dict) for row in selections)
                or [row.get("arm_id") for row in selections] != candidates):
            raise ValueError(
                f"policy-articulation cell {cell_id} has invalid candidate selections"
            )
        for candidate, candidate_row in zip(candidates, selections):
            candidate_assignments = assigned_roles.get(candidate, [])
            expected_roles = [row["role"] for row in candidate_assignments]
            expected_ranks = {
                row["role"]: row["role_rank"] for row in candidate_assignments
            }
            expected_reasons = [row["selection_reason"] for row in candidate_assignments]
            features = candidate_row.get("selection_features")
            bank_arm = arms[candidate]
            if (not expected_roles or candidate_row.get("roles") != expected_roles
                    or candidate_row.get("role_ranks") != expected_ranks
                    or candidate_row.get("selection_reasons") != expected_reasons
                    or candidate_row.get("matched_control_ids")
                    != controls_by_candidate[candidate]
                    or not isinstance(candidate_row.get("primary_rank"), int)
                    or isinstance(candidate_row.get("primary_rank"), bool)
                    or candidate_row["primary_rank"] <= 0
                    or not isinstance(candidate_row.get("vector_rank"), int)
                    or isinstance(candidate_row.get("vector_rank"), bool)
                    or candidate_row["vector_rank"] <= 0
                    or not isinstance(features, dict)
                    or features.get("arm_id") != candidate
                    or features.get("channel") != bank_arm.get("channel")
                    or (features.get("components") or [])
                    != (bank_arm.get("components") or [])
                    or features.get("n_address_units") != bank_arm.get("n_address_units")):
                raise ValueError(
                    f"policy-articulation cell {cell_id}/{candidate} metadata is invalid"
                )
        if not isinstance(selected_cell.get("selection_reason"), str) or not selected_cell[
                "selection_reason"]:
            raise ValueError(f"policy-articulation cell {cell_id} omits its selection reason")
        search_rows = report_by_id[cell_id].get("rows")
        if (not isinstance(search_rows, list)
                or any(not isinstance(row, dict) for row in search_rows)):
            raise ValueError(
                f"policy-articulation cell {cell_id} has invalid search-report rows"
            )
        report_arm_ids = {row.get("arm_id") for row in search_rows}
        if not set(candidates + expected_controls) <= report_arm_ids:
            raise ValueError(
                f"policy-articulation cell {cell_id} selects arms absent from its search report"
            )

    expected_range = [min(candidate_counts), max(candidate_counts)]
    if selection.get("candidate_count_range") != expected_range:
        raise ValueError("policy-articulation selection candidate_count_range is invalid")
    return bank_ids


def validate_policy_articulation_selection_provenance(
    selection: dict,
    *,
    selection_path: str | Path,
    execution_manifest: dict,
    execution_manifest_path: str | Path,
) -> dict:
    """Authenticate a generic selection's inspected search execution and full cell panel."""
    selection_path = Path(selection_path)
    execution_manifest_path = Path(execution_manifest_path)
    if (selection.get("schema") != "policy_articulation_selection/v1"
            or selection.get("status")
            != "frozen-after-search-before-validation-scoring"):
        raise ValueError("policy-articulation selection is not frozen v1")
    selection_content_sha = _validate_selection_content_hash(selection)
    source_path_value = selection.get("search_execution_manifest_path")
    report_path_value = selection.get("search_report_path")
    if not isinstance(source_path_value, str) or not isinstance(report_path_value, str):
        raise ValueError("policy-articulation selection omits source search paths")
    source_path = _resolve_declared_path(
        source_path_value, manifest_path=selection_path)
    report_path = _resolve_declared_path(
        report_path_value, manifest_path=selection_path)
    if not source_path.is_file() or not report_path.is_file():
        raise ValueError("policy-articulation source search manifest/report is missing")
    source_sha = sha256_file(source_path)
    report_sha = sha256_file(report_path)
    if (selection.get("search_execution_manifest_sha256") != source_sha
            or selection.get("search_report_sha256") != report_sha):
        raise ValueError("policy-articulation source search hashes do not match actual files")
    source = json.loads(source_path.read_text())
    report = json.loads(report_path.read_text())
    if (source.get("schema") != "fresh_name_execution_manifest/v2"
            or not str(source.get("status", "")).startswith("frozen-before-")):
        raise ValueError("policy-articulation source search manifest is not frozen v2")
    manifest_transition = _validate_search_validation_manifest_transition(
        source,
        execution_manifest,
        selection=selection,
        selection_path=selection_path,
        current_manifest_path=execution_manifest_path,
    )
    for key in ("arm_bank_sha256", "packet_manifest_sha256"):
        expected = execution_manifest.get(key)
        if (not expected or source.get(key) != expected
                or selection.get(key) != expected):
            raise ValueError(f"policy-articulation search/validation {key} mismatch")
    policy = selection.get("selection_policy")
    if (not isinstance(policy, dict) or source.get("selection_policy") != policy
            or execution_manifest.get("selection_policy") != policy):
        raise ValueError("policy-articulation selection policy differs across the DAG")
    policy_sha = sha256_bytes(json.dumps(
        policy, sort_keys=True, separators=(",", ":")).encode())
    if selection.get("selection_policy_sha256") != policy_sha:
        raise ValueError("policy-articulation selection policy hash is invalid")
    authorization = report.get("partition_authorization", {})
    source_group = report.get("source_group_inference", {})
    if (report.get("schema") != "policy_isomorphism_experiment/v5"
            or report.get("partition") != selection.get("search_partition")
            or authorization.get("phase") != selection.get("search_phase")
            or authorization.get("execution_manifest_sha256") != source_sha
            or authorization.get("selection_artifact_sha256") is not None
            or report.get("arm_bank_sha256") != selection.get("arm_bank_sha256")
            or source_group.get("enabled")
            is not bool(source.get("analysis", {}).get(
                "runner", {}).get("source_group_inference"))
            or source_group.get("packet_manifest_sha256")
            != selection.get("packet_manifest_sha256")):
        raise ValueError("policy-articulation search report provenance is invalid")
    _validate_report_production_closure(report, source=source, source_sha=source_sha)
    source_additional = _validate_unique_artifact_rows(
        source.get("additional_artifacts", []),
        label="policy-articulation source additional_artifacts",
    )
    current_additional = _validate_unique_artifact_rows(
        execution_manifest.get("additional_artifacts", []),
        label="policy-articulation validation additional_artifacts",
    )
    if source_additional != current_additional:
        raise ValueError("policy-articulation additional artifacts differ across the DAG")
    panel_rows = [
        row for row in source_additional if row.get("role") == "metric_panel"
    ]
    if (len(panel_rows) != 1
            or selection.get("metric_panel_sha256") != panel_rows[0].get("sha256")
            or selection.get("metric_panel_path") != panel_rows[0].get("path")):
        raise ValueError("policy-articulation metric-panel binding is invalid")
    selected_additional = _validate_unique_artifact_rows(
        selection.get("additional_artifacts"),
        label="policy-articulation selection additional_artifacts",
    )
    expected_selected_additional = [
        row for row in source_additional if row.get("role") != "metric_panel"
    ]
    if ({json.dumps(row, sort_keys=True) for row in selected_additional}
            != {json.dumps(row, sort_keys=True) for row in expected_selected_additional}):
        raise ValueError("policy-articulation selected additional-artifact bindings differ")
    report_additional = report.get("additional_artifact_validation", {})
    expected_report_files = [
        {"path": row.get("path"), "sha256": row.get("sha256")}
        for row in source_additional
    ]
    if (report_additional.get("valid") is not True
            or report_additional.get("files") != expected_report_files):
        raise ValueError("policy-articulation search report did not authenticate its panel")
    bank_path = _resolve_declared_path(
        execution_manifest.get("arm_bank_path", ""),
        manifest_path=execution_manifest_path,
    )
    selection_bank_value = selection.get("arm_bank_path")
    selection_packet_value = selection.get("packet_manifest_path")
    if (not isinstance(selection_bank_value, str) or not selection_bank_value
            or not isinstance(selection_packet_value, str) or not selection_packet_value):
        raise ValueError("policy-articulation selection omits bank/packet paths")
    selection_bank_path = _resolve_declared_path(
        selection_bank_value, manifest_path=selection_path)
    selection_packet_path = _resolve_declared_path(
        selection_packet_value, manifest_path=selection_path)
    execution_packet_path = _resolve_declared_path(
        execution_manifest.get("packet_manifest_path", ""),
        manifest_path=execution_manifest_path,
    )
    if (selection_bank_path.resolve() != bank_path.resolve()
            or selection_packet_path.resolve() != execution_packet_path.resolve()):
        raise ValueError("policy-articulation selected bank/packet paths differ")
    bank = json.loads(bank_path.read_text()) if bank_path.is_file() else {}
    bank_ids = _validate_selection_cell_metadata(
        selection,
        bank=bank,
        report=report,
        policy=policy,
        expected_cell_ids=source.get("analysis", {}).get(
            "runner", {}).get("cell_ids"),
    )
    expected_freeze_certificate = {
        "valid": True,
        "search_execution_manifest_sha256": source_sha,
        "search_report_sha256": report_sha,
        "n_cells": len(bank_ids),
    }
    observed_freeze_certificate = execution_manifest.get(
        "selection_provenance_validation_at_freeze")
    if (observed_freeze_certificate is not None
            and observed_freeze_certificate != expected_freeze_certificate):
        raise ValueError(
            "policy-articulation derived selection provenance freeze certificate is invalid"
        )
    return {
        "valid": True,
        "search_execution_manifest_path": str(source_path),
        "search_execution_manifest_sha256": source_sha,
        "search_report_path": str(report_path),
        "search_report_sha256": report_sha,
        "selection_content_sha256": selection_content_sha,
        "manifest_transition_validation": manifest_transition,
        "n_cells": len(bank_ids),
    }


def validate_frozen_environment(manifest: dict, *, require_accelerator: bool = False) -> dict:
    """Require the declared Python and package versions for bound-grade execution."""
    expected = manifest.get("execution_environment")
    if not isinstance(expected, dict):
        raise ValueError("execution manifest omits execution_environment")
    observed_python = platform.python_version()
    expected_python = expected.get("python_version")
    package_expectations = expected.get("packages")
    if not isinstance(package_expectations, dict) or not package_expectations:
        raise ValueError("execution manifest omits frozen package versions")
    observed_packages = {}
    for name in package_expectations:
        try:
            observed_packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed_packages[name] = None
    mismatches = {
        "python": {"expected": expected_python, "observed": observed_python}
        if observed_python != expected_python else None,
        "packages": {
            name: {"expected": version, "observed": observed_packages[name]}
            for name, version in package_expectations.items()
            if observed_packages[name] != version
        },
    }
    expected_hostname = expected.get("hostname")
    observed_hostname = platform.node()
    if not expected_hostname or observed_hostname != expected_hostname:
        mismatches["hostname"] = {
            "expected": expected_hostname, "observed": observed_hostname}
    expected_executable = expected.get("python_executable")
    if not expected_executable or sys.executable != expected_executable:
        mismatches["python_executable"] = {
            "expected": expected_executable, "observed": sys.executable}
    expected_overrides = expected.get("runtime_environment_overrides")
    if not isinstance(expected_overrides, dict):
        raise ValueError("execution manifest omits frozen runtime environment overrides")
    observed_overrides = {
        name: (os.environ.get(name) or None) for name in expected_overrides
    }
    override_mismatches = {
        name: {"expected": value, "observed": observed_overrides[name]}
        for name, value in expected_overrides.items()
        if observed_overrides[name] != value
    }
    if override_mismatches:
        mismatches["runtime_environment_overrides"] = override_mismatches
    accelerator = None
    if require_accelerator:
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"unable to authenticate frozen GPU environment: {exc}") from exc
        rows = [line.strip().rsplit(",", 1) for line in query.stdout.splitlines()
                if line.strip()]
        if not rows or any(len(row) != 2 for row in rows):
            raise ValueError("nvidia-smi returned no parseable GPU/driver rows")
        observed_gpu_models = sorted({row[0].strip() for row in rows})
        observed_drivers = sorted({row[1].strip() for row in rows})
        expected_gpu = expected.get("gpu_model")
        expected_driver = expected.get("driver_version")
        if observed_gpu_models != [expected_gpu] or observed_drivers != [expected_driver]:
            mismatches["accelerator"] = {
                "expected": {"gpu_model": expected_gpu, "driver_version": expected_driver},
                "observed": {
                    "gpu_models": observed_gpu_models, "driver_versions": observed_drivers},
            }
        accelerator = {
            "gpu_models": observed_gpu_models,
            "driver_versions": observed_drivers,
            "n_visible_gpus": len(rows),
        }
    if any(value for value in mismatches.values()):
        raise ValueError(f"runtime environment differs from frozen manifest: {mismatches}")
    return {
        "valid": True,
        "hostname": observed_hostname,
        "python_executable": sys.executable,
        "python_version": observed_python,
        "packages": observed_packages,
        "runtime_environment_overrides": observed_overrides,
        "accelerator": accelerator,
    }


def validate_lockbox_release(
    manifest: dict,
    *,
    manifest_path: str | Path,
    selection_sha256: str,
    release_artifact_path: str | Path | None,
) -> dict:
    """Authenticate the production calibration report required to open a frozen lockbox."""
    manifest_path = Path(manifest_path)
    specification = manifest.get("lockbox_release")
    if not isinstance(specification, dict) or specification.get("required") is not True:
        raise ValueError("execution manifest omits a required lockbox-release gate")
    if release_artifact_path is None:
        raise ValueError("lockbox phase requires a calibration-release artifact")
    expected_release_path = _resolve_declared_path(
        specification.get("artifact_path", ""), manifest_path=manifest_path)
    observed_release_path = Path(release_artifact_path).resolve()
    if observed_release_path != expected_release_path.resolve():
        raise ValueError("calibration-release path differs from frozen execution manifest")
    if not observed_release_path.is_file():
        raise ValueError("calibration-release artifact does not exist")
    release = json.loads(observed_release_path.read_text())
    expected_manifest_sha256 = sha256_file(manifest_path)

    # Legacy experiments place calibration and lockbox in one manifest.  The breadth search is
    # intentionally a two-manifest DAG: its validation manifest is frozen only after canonical
    # arm selection, but the search manifest/report are already authenticated by that selection.
    # Resolve the external calibration edge from the frozen selection instead of weakening the
    # held-out phase to ``open_development``.
    calibration_manifest = manifest
    calibration_manifest_path = manifest_path
    calibration_manifest_sha256 = expected_manifest_sha256
    calibration_partitions = manifest.get("phases", {}).get("calibration", [])
    if not calibration_partitions:
        selection_path_value = manifest.get("selection_artifact_path")
        if not isinstance(selection_path_value, str) or not selection_path_value:
            raise ValueError(
                "two-manifest lockbox release requires the frozen selection path"
            )
        selection_path = _resolve_declared_path(
            selection_path_value, manifest_path=manifest_path)
        if (not selection_path.is_file()
                or sha256_file(selection_path) != selection_sha256):
            raise ValueError("lockbox selection is missing or differs from its frozen digest")
        selection = json.loads(selection_path.read_text())
        if selection.get("schema") != "policy_articulation_selection/v1":
            raise ValueError(
                "two-manifest lockbox release requires a policy-articulation selection"
            )
        calibration_manifest_path = _resolve_declared_path(
            selection.get("search_execution_manifest_path", ""),
            manifest_path=selection_path,
        )
        calibration_manifest_sha256 = selection.get(
            "search_execution_manifest_sha256")
        if (not calibration_manifest_path.is_file()
                or not isinstance(calibration_manifest_sha256, str)
                or sha256_file(calibration_manifest_path)
                != calibration_manifest_sha256):
            raise ValueError(
                "selection-bound calibration execution manifest is missing or changed"
            )
        calibration_manifest = json.loads(calibration_manifest_path.read_text())
        calibration_partitions = calibration_manifest.get("phases", {}).get(
            "calibration", [])
        selected_report_path = _resolve_declared_path(
            selection.get("search_report_path", ""), manifest_path=selection_path)
        selected_report_sha256 = selection.get("search_report_sha256")
        if (not selected_report_path.is_file()
                or not isinstance(selected_report_sha256, str)
                or sha256_file(selected_report_path) != selected_report_sha256):
            raise ValueError("selection-bound calibration report is missing or changed")
    else:
        selected_report_path = None

    lockbox_partitions = manifest.get("phases", {}).get("lockbox", [])
    if len(calibration_partitions) != 1 or len(lockbox_partitions) != 1:
        raise ValueError(
            "lockbox release requires one authenticated calibration and lockbox partition"
        )
    calibration_partition = calibration_partitions[0]
    lockbox_partition = lockbox_partitions[0]
    for key, observed in (
            ("calibration_partition", calibration_partition),
            ("lockbox_partition", lockbox_partition)):
        declared = specification.get(key)
        if declared is not None and declared != observed:
            raise ValueError(f"lockbox-release {key} differs from the frozen DAG")

    exact = {
        "schema": specification.get("schema"),
        "status": "calibration-complete-production-only-lockbox-release",
        "execution_manifest_sha256": expected_manifest_sha256,
        "selection_artifact_sha256": selection_sha256,
        "calibration_partition": calibration_partition,
        "lockbox_partition": lockbox_partition,
        "fake_inputs": False,
    }
    if calibration_manifest_path.resolve() != manifest_path.resolve():
        exact["calibration_execution_manifest_sha256"] = (
            calibration_manifest_sha256)
    mismatches = {
        key: {"expected": value, "observed": release.get(key)}
        for key, value in exact.items() if value is None or release.get(key) != value
    }
    if mismatches:
        raise ValueError(f"calibration-release metadata mismatch: {mismatches}")
    expected_report_path = _resolve_declared_path(
        specification.get("calibration_report_path", ""), manifest_path=manifest_path)
    if (selected_report_path is not None
            and selected_report_path.resolve() != expected_report_path.resolve()):
        raise ValueError(
            "selection-bound calibration report differs from the frozen release path"
        )
    release_report_path = _resolve_declared_path(
        release.get("calibration_report_path", ""), manifest_path=manifest_path)
    if release_report_path.resolve() != expected_report_path.resolve():
        raise ValueError("calibration report path differs from frozen execution manifest")
    if (not release_report_path.is_file()
            or sha256_file(release_report_path) != release.get("calibration_report_sha256")):
        raise ValueError("calibration report is missing or differs from its release artifact")
    report = json.loads(release_report_path.read_text())
    if report.get("schema") != specification.get("calibration_report_schema"):
        raise ValueError("calibration release references the wrong report schema")
    authorization = report.get("partition_authorization", {})
    if (report.get("partition") != exact["calibration_partition"]
            or report.get("arm_bank_sha256") != manifest.get("arm_bank_sha256")
            or authorization.get("phase") != "calibration"
            or authorization.get("execution_manifest_sha256")
            != calibration_manifest_sha256
            or report.get("frozen_invocation_validation", {}).get("valid") is not True):
        raise ValueError("calibration report is not an authenticated frozen calibration run")
    expected_files = calibration_manifest.get(
        "implementation", {}).get("analysis", {}).get("files")
    if report.get("analysis_implementation", {}).get("files") != expected_files:
        raise ValueError("calibration report analysis implementation differs from manifest")
    expected_cells = calibration_manifest.get(
        "analysis", {}).get("runner", {}).get("cell_ids")
    cells = report.get("cells", [])
    if [cell.get("cell_id") for cell in cells] != expected_cells:
        raise ValueError("calibration report cell panel differs from frozen runner")
    for cell in cells:
        provenance = cell.get("score_provenance_validation")
        if not isinstance(provenance, dict) or set(provenance) != {"small", "target"}:
            raise ValueError("calibration report lacks complete score provenance validation")
        for role, validation in provenance.items():
            if (validation.get("valid") is not True
                    or validation.get("fake_backend") is not False
                    or validation.get("backend_class")
                    != manifest.get("execution_environment", {}).get(
                        "production_backend_class")):
                raise ValueError(
                    f"calibration report {role} scores are not authenticated production scores"
                )
    return {
        "valid": True,
        "artifact_path": str(observed_release_path),
        "artifact_sha256": sha256_file(observed_release_path),
        "calibration_report_path": str(release_report_path),
        "calibration_report_sha256": release["calibration_report_sha256"],
        "execution_manifest_sha256": expected_manifest_sha256,
        "calibration_execution_manifest_sha256": calibration_manifest_sha256,
    }


def authorize_policy_partition(
    partition: str,
    *,
    operation: str,
    execution_manifest_path: str | Path | None = None,
    selection_artifact_path: str | Path | None = None,
    lockbox_release_artifact_path: str | Path | None = None,
) -> dict:
    """Authorize an open partition or a hash-bound frozen confirmation partition.

    A sealed partition is intentionally absent from :data:`POLICY_ISOMORPHISM_PARTITIONS`.
    It becomes analyzable only through a frozen execution manifest whose declared lockbox and
    exact selection artifact authenticate that partition.  This lets the implementation be
    frozen before calibration without relying on a later source edit to open the final panel.
    """
    if execution_manifest_path is None:
        require_partition(
            partition,
            allowed=POLICY_ISOMORPHISM_PARTITIONS,
            operation=operation,
        )
        if selection_artifact_path is not None:
            raise ValueError(
                "a selection artifact cannot authorize analysis without an execution manifest"
            )
        if lockbox_release_artifact_path is not None:
            raise ValueError(
                "a lockbox-release artifact cannot authorize an unbound partition"
            )
        return {
            "partition": partition,
            "phase": None,
            "authorization": "hardcoded_open_partition_allowlist",
            "execution_manifest_path": None,
            "execution_manifest_sha256": None,
            "selection_artifact_path": None,
            "selection_artifact_sha256": None,
            "sealed_partition_authorized": False,
            "lockbox_release_validation": None,
        }

    manifest_path = Path(execution_manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "fresh_name_execution_manifest/v2":
        raise ValueError("unsupported policy-analysis execution manifest schema")
    if not str(manifest.get("status", "")).startswith("frozen-before-"):
        raise ValueError("policy-analysis execution manifest is not frozen")

    declared_phases = [
        phase
        for phase, partitions in manifest.get("phases", {}).items()
        if partition in partitions
    ]
    if len(declared_phases) != 1:
        raise ValueError(
            f"partition {partition!r} must occur in exactly one frozen phase; "
            f"found={declared_phases}"
        )
    phase = declared_phases[0]
    phase_access = manifest.get("phase_access")
    if phase_access is not None:
        if (not isinstance(phase_access, dict)
                or set(phase_access) != set(manifest.get("phases", {}))):
            raise ValueError("execution manifest phase_access must cover every phase exactly")
        invalid_access = {
            key: value for key, value in phase_access.items()
            if value not in {"open_development", "sealed_confirmation"}
        }
        if invalid_access:
            raise ValueError(f"execution manifest has invalid phase access: {invalid_access}")
        if ("lockbox" in manifest.get("phases", {})
                and phase_access.get("lockbox") != "sealed_confirmation"):
            raise ValueError("the lockbox phase must be declared sealed_confirmation")
        unexpected_sealed = [
            key for key, value in phase_access.items()
            if value == "sealed_confirmation" and key != "lockbox"
        ]
        if unexpected_sealed:
            raise ValueError(
                f"only the lockbox phase may be sealed: {unexpected_sealed}")
        is_open = phase_access[phase] == "open_development"
        authorization_mode = "manifest_declared_hash_bound_phase_access"
    else:
        # Backward-compatible path for already-frozen v2 manifests. New manifests must declare
        # phase_access so adding a public calibration partition never requires a source-code edit.
        is_open = partition in POLICY_ISOMORPHISM_PARTITIONS
        authorization_mode = "legacy_global_partition_allowlist"
    if not is_open and phase != "lockbox":
        raise ValueError(
            f"sealed partition {partition!r} is not the frozen lockbox phase"
        )

    artifact_fields = (
        ("protocol_manifest_path", "protocol_manifest_sha256"),
        ("packet_manifest_path", "packet_manifest_sha256"),
        ("partition_integrity_path", "partition_integrity_sha256"),
        ("arm_bank_path", "arm_bank_sha256"),
        ("target_prompt_manifest_path", "target_prompt_manifest_sha256"),
    )
    for path_key, digest_key in artifact_fields:
        declared_path = manifest.get(path_key)
        declared_digest = manifest.get(digest_key)
        if not declared_path or not declared_digest:
            raise ValueError(
                f"execution manifest omits frozen artifact binding {path_key}/{digest_key}"
            )
        resolved = _resolve_declared_path(declared_path, manifest_path=manifest_path)
        if not resolved.is_file() or sha256_file(resolved) != declared_digest:
            raise ValueError(f"execution manifest artifact changed: {path_key}")
    additional_artifact_validation = validate_additional_artifacts(
        manifest, manifest_path=manifest_path)

    target_prompt_path = _resolve_declared_path(
        manifest["target_prompt_manifest_path"], manifest_path=manifest_path)
    target_prompt = json.loads(target_prompt_path.read_text())
    if (text_sha256(target_prompt.get("readout_template", ""))
            != manifest.get("readout_template_sha256")):
        raise ValueError("execution manifest/readout template hash mismatch")
    if manifest.get("binary_readout") == "teacher_forced_declared_labels":
        if manifest.get("label_support") != ["YES", "NO"]:
            raise ValueError("teacher-forced execution must freeze YES/NO label support")
        label_validation = manifest.get("teacher_forced_label_validation", {})
        if not (label_validation.get("both_labels_single_token")
                and label_validation.get(
                    "contextual_continuation_ids_match_isolated_ids")):
            raise ValueError("teacher-forced label-token validation is absent or failed")

    integrity_path = _resolve_declared_path(
        manifest["partition_integrity_path"], manifest_path=manifest_path)
    integrity = json.loads(integrity_path.read_text())
    if (integrity.get("valid") is not True
            or integrity.get("packet_manifest_sha256")
            != manifest.get("packet_manifest_sha256")
            or integrity.get("protocol_manifest_sha256")
            != manifest.get("protocol_manifest_sha256")):
        raise ValueError("partition integrity certificate is invalid or misbound")
    if partition not in set(integrity.get("validated_partitions", [])):
        raise ValueError(
            f"partition integrity certificate does not cover {partition!r}")

    selection_required = selection_required_for_phase(manifest, phase)
    if selection_required and selection_artifact_path is None:
        raise ValueError(f"frozen phase {phase!r} requires its selection artifact")
    selection_digest = None
    selection_path_value = None
    selection_provenance_validation = None
    if selection_artifact_path is not None:
        selection_path = Path(selection_artifact_path)
        selection_digest = sha256_file(selection_path)
        selection_path_value = str(selection_path)
        expected_digest = manifest.get("selection_artifact_sha256")
        if not expected_digest or selection_digest != expected_digest:
            raise ValueError("selection artifact differs from frozen execution manifest")
        selection = json.loads(selection_path.read_text())
        schema = selection.get("schema")
        if schema == "policy_isomorphism_lockbox_selection/v2":
            if selection.get("status") != (
                    "frozen-before-declared-lockbox-target-or-executor-scoring"):
                raise ValueError("policy-isomorphism selection is not frozen")
            lockbox_partitions = manifest.get("phases", {}).get("lockbox", [])
            if (len(lockbox_partitions) != 1
                    or selection.get("lockbox_partition") != lockbox_partitions[0]):
                raise ValueError("selection/execution lockbox partition mismatch")
        elif schema == "policy_articulation_selection/v1":
            if selection.get("status") != (
                    "frozen-after-search-before-validation-scoring"):
                raise ValueError("policy-articulation selection is not frozen")
            phase_partitions = manifest.get("phases", {}).get(phase, [])
            if len(phase_partitions) != 1:
                raise ValueError(
                    "policy-articulation selection requires one partition in its phase"
                )
            if (selection.get("selected_phase") != phase
                    or selection.get("selected_partition") != partition
                    or selection.get("selected_partition") != phase_partitions[0]):
                raise ValueError("selection/execution selected phase or partition mismatch")
            selection_provenance_validation = (
                validate_policy_articulation_selection_provenance(
                    selection,
                    selection_path=selection_path,
                    execution_manifest=manifest,
                    execution_manifest_path=manifest_path,
                )
            )
        else:
            raise ValueError("unsupported frozen policy-articulation selection schema")
        for key in ("arm_bank_sha256", "packet_manifest_sha256"):
            if selection.get(key) != manifest.get(key):
                raise ValueError(f"selection/execution {key} mismatch")

    lockbox_release_validation = None
    if phase == "lockbox":
        lockbox_release_validation = validate_lockbox_release(
            manifest,
            manifest_path=manifest_path,
            selection_sha256=selection_digest,
            release_artifact_path=lockbox_release_artifact_path,
        )
    elif lockbox_release_artifact_path is not None:
        raise ValueError("an open phase must not consume a lockbox-release artifact")

    return {
        "partition": partition,
        "phase": phase,
        "authorization": authorization_mode,
        "execution_manifest_path": str(manifest_path),
        "execution_manifest_sha256": sha256_file(manifest_path),
        "selection_artifact_path": selection_path_value,
        "selection_artifact_sha256": selection_digest,
        "selection_provenance_validation": selection_provenance_validation,
        "sealed_partition_authorized": not is_open,
        "additional_artifact_validation": additional_artifact_validation,
        "lockbox_release_validation": lockbox_release_validation,
    }


def require_partition(partition: str, *, allowed: Collection[str], operation: str) -> str:
    """Reject undeclared partitions at the experiment boundary."""
    allowed_values = tuple(allowed)
    if partition not in allowed_values:
        raise ValueError(
            f"{operation} does not authorize partition {partition!r}; "
            f"allowed={sorted(allowed_values)}"
        )
    return partition


def load_partition_source_groups(
    packet_root: str | Path,
    packet_manifest_path: str | Path,
    *,
    domain: str,
    partition: str,
    item_hashes: list[str],
) -> dict:
    """Load and authenticate source groups, aligned to an already-scored item order.

    The manifest's recorded path is deliberately not trusted because packets are commonly copied
    between machines.  ``packet_root/domain/items/partition.jsonl`` is authenticated against the
    byte digest and ordered-set digest frozen in the packet manifest; every row's text is then
    authenticated against its own content hash before source groups are exposed to inference.
    """
    packet_root = Path(packet_root)
    packet_manifest_path = Path(packet_manifest_path)
    packet = json.loads(packet_manifest_path.read_text())
    if packet.get("schema") != "fresh_item_partitions/v1":
        raise ValueError(
            f"unsupported fresh-item packet schema {packet.get('schema')!r}"
        )

    domain_rows = [row for row in packet.get("domains", []) if row.get("domain") == domain]
    if len(domain_rows) != 1:
        raise ValueError(
            f"packet manifest must contain exactly one domain row for {domain!r}"
        )
    partition_rows = [
        row for row in domain_rows[0].get("partitions", []) if row.get("id") == partition
    ]
    if len(partition_rows) != 1:
        raise ValueError(
            f"packet manifest must contain exactly one {domain}/{partition} partition row"
        )
    partition_row = partition_rows[0]
    item_path = packet_root / domain / "items" / f"{partition}.jsonl"
    if not item_path.is_file():
        raise ValueError(f"local fresh-item partition is missing: {item_path}")
    observed_file_sha256 = sha256_file(item_path)
    if observed_file_sha256 != partition_row.get("items_sha256"):
        raise ValueError(
            f"fresh-item file SHA-256 mismatch for {domain}/{partition}: "
            f"observed={observed_file_sha256} "
            f"manifest={partition_row.get('items_sha256')!r}"
        )

    rows = [
        json.loads(line) for line in item_path.read_text().splitlines() if line.strip()
    ]
    expected_n = partition_row.get("n")
    if expected_n is None or len(rows) != int(expected_n):
        raise ValueError(
            f"fresh-item row count mismatch for {domain}/{partition}: "
            f"observed={len(rows)} manifest={expected_n!r}"
        )
    row_hashes = [str(row.get("text_sha256")) for row in rows]
    if len(row_hashes) != len(set(row_hashes)):
        raise ValueError(f"duplicate item hash in {domain}/{partition} fresh-item file")
    if any(text_sha256(row.get("text", "")) != row_hash for row, row_hash in zip(
        rows, row_hashes
    )):
        raise ValueError(f"item content hash mismatch in {domain}/{partition} fresh-item file")
    ordered_set_sha256 = sha256_bytes("\n".join(row_hashes).encode())
    if ordered_set_sha256 != partition_row.get("ordered_item_set_sha256"):
        raise ValueError(
            f"ordered item-set SHA-256 mismatch for {domain}/{partition}"
        )

    groups = [row.get("source_group") for row in rows]
    if any(not isinstance(value, str) or not value for value in groups):
        raise ValueError(f"missing source group in {domain}/{partition} fresh-item file")
    n_source_groups = len(set(groups))
    if n_source_groups != partition_row.get("n_source_groups"):
        raise ValueError(
            f"source-group count mismatch for {domain}/{partition}: "
            f"observed={n_source_groups} "
            f"manifest={partition_row.get('n_source_groups')!r}"
        )

    if len(item_hashes) != len(set(item_hashes)):
        raise ValueError("scored item hashes contain duplicates")
    if set(item_hashes) != set(row_hashes):
        missing = sorted(set(item_hashes) - set(row_hashes))
        extra = sorted(set(row_hashes) - set(item_hashes))
        raise ValueError(
            f"scored items differ from authenticated {domain}/{partition} packet items; "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    group_by_hash = dict(zip(row_hashes, groups))
    aligned_groups = [group_by_hash[value] for value in item_hashes]
    group_sizes: dict[str, int] = defaultdict(int)
    for value in aligned_groups:
        group_sizes[value] += 1
    sizes = list(group_sizes.values())
    return {
        "source_groups": aligned_groups,
        "validation": {
            "valid": True,
            "domain": domain,
            "partition": partition,
            "n_items": len(item_hashes),
            "n_source_groups": len(sizes),
            "n_singleton_source_groups": sum(size == 1 for size in sizes),
            "min_source_group_size": min(sizes),
            "max_source_group_size": max(sizes),
            "mean_source_group_size": float(np.mean(sizes)),
            "all_source_groups_singleton": all(size == 1 for size in sizes),
            "source_group_method": domain_rows[0].get("source_group_method"),
            "holdout_grade": domain_rows[0].get("holdout_grade"),
            "item_path": str(item_path),
            "items_sha256": observed_file_sha256,
            "ordered_item_set_sha256": ordered_set_sha256,
            "packet_manifest_path": str(packet_manifest_path),
            "packet_manifest_sha256": sha256_file(packet_manifest_path),
        },
    }


def load_score_shard(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        scores = np.asarray(z["scores"], float)
        meta = [json.loads(str(value)) for value in z["meta"]]
        hashes = [str(value) for value in z["probe_sha256"]]
        result = {
            "path": str(path),
            "sha256": sha256_file(path),
            "scores": scores,
            "meta": meta,
            "hashes": hashes,
            "model_job_id": str(z["model_job_id"]),
            "repetition": int(z["repetition"]),
            "source_artifact_sha256": str(z["source_artifact_sha256"]),
            "partition": str(z["isolated_partition"]),
        }
        if "role" in z.files:
            result["role"] = str(z["role"])
        if "readout_template_sha256" in z.files:
            result["readout_template_sha256"] = str(z["readout_template_sha256"])
        if "binary_readout" in z.files:
            result["binary_readout"] = str(z["binary_readout"])
        if "backend_class" in z.files:
            result["backend_class"] = str(z["backend_class"])
        if "fake_backend" in z.files:
            result["fake_backend"] = bool(z["fake_backend"])
        for key in (
                "execution_manifest_sha256", "arm_bank_sha256",
                "packet_manifest_sha256"):
            if key in z.files:
                result[key] = str(z[key])
    if scores.shape != (len(meta), len(hashes)):
        raise ValueError(f"unaligned shard {path}")
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"duplicate probe hash in shard {path}")
    if not meta:
        raise ValueError(f"empty score metadata in shard {path}")
    domains = {row["domain"] for row in meta}
    if len(domains) != 1:
        raise ValueError(f"mixed domains in shard {path}: {sorted(domains)}")
    result["domain"] = next(iter(domains))
    return result


def load_shard_index(root: str | Path, partition: str) -> dict[tuple, list[dict]]:
    """Load one explicit isolated partition and verify path/metadata agreement."""
    index: dict[tuple, list[dict]] = defaultdict(list)
    # v2 shards are nested under model_job_id; rglob also supports the original flat layout.
    paths = sorted((Path(root) / partition).rglob("*.npz"))
    if not paths:
        raise ValueError(f"no {partition} shards under {root}")
    for path in paths:
        shard = load_score_shard(path)
        if shard["partition"] != partition:
            raise ValueError(
                f"shard partition mismatch at {path}: metadata={shard['partition']!r}, "
                f"requested={partition!r}"
            )
        index[(shard["model_job_id"], shard["domain"])].append(shard)
    return dict(index)


def average_repetitions(shards: list[dict]) -> dict:
    if not shards:
        raise ValueError("cannot average an empty repetition set")
    shards = sorted(shards, key=lambda row: row["repetition"])
    repetitions = [row["repetition"] for row in shards]
    if len(repetitions) != len(set(repetitions)):
        raise ValueError(f"duplicate repetition ids: {repetitions}")
    first = shards[0]
    if any(
        row["hashes"] != first["hashes"] or row["meta"] != first["meta"]
        for row in shards[1:]
    ):
        raise ValueError("repetition shards are not aligned")
    readout_hashes = {row.get("readout_template_sha256") for row in shards}
    if len(readout_hashes) != 1:
        raise ValueError("repetition shards use different readout templates")
    binary_readouts = {row.get("binary_readout") for row in shards}
    if len(binary_readouts) != 1:
        raise ValueError("repetition shards use different binary readout protocols")
    provenance = {}
    for key in (
            "execution_manifest_sha256", "arm_bank_sha256",
            "packet_manifest_sha256"):
        values = {row.get(key) for row in shards}
        if len(values) != 1:
            raise ValueError(f"repetition shards use different {key} values")
        provenance[key] = next(iter(values))
    execution_metadata = {}
    for key in ("role", "backend_class", "fake_backend"):
        values = {row.get(key) for row in shards}
        if len(values) != 1:
            raise ValueError(f"repetition shards use different {key} values")
        execution_metadata[key] = next(iter(values))
    return {
        "scores": np.mean([row["scores"] for row in shards], axis=0),
        "meta": first["meta"],
        "hashes": first["hashes"],
        "source_artifact_sha256": [row["source_artifact_sha256"] for row in shards],
        "shard_sha256": [row["sha256"] for row in shards],
        "repetitions": repetitions,
        "readout_template_sha256": next(iter(readout_hashes)),
        "binary_readout": next(iter(binary_readouts)),
        **provenance,
        **execution_metadata,
    }


def score_orbits(
    scores: np.ndarray, meta: list[dict], *, cell_id: str
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for index, row in enumerate(meta):
        if row["cell_id"] != cell_id:
            continue
        arm_id = row.get("arm_id", "target")
        form = row["form"]
        if form in result[arm_id]:
            raise ValueError(f"duplicate orbit row for {cell_id}/{arm_id}/{form}")
        result[arm_id][form] = np.asarray(scores[index], float)
    return dict(result)


def align_orbit(
    orbit: dict[str, np.ndarray], source_hashes: list[str], target_hashes: list[str]
) -> dict[str, np.ndarray]:
    """Align an orbit by unique item hash; never silently choose a duplicate occurrence."""
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("source hashes contain duplicates")
    if len(target_hashes) != len(set(target_hashes)):
        raise ValueError("target hashes contain duplicates")
    locations = {value: index for index, value in enumerate(source_hashes)}
    missing = sorted(set(target_hashes) - set(locations))
    if missing:
        raise ValueError(
            f"executor and target shards do not contain the same probes; "
            f"missing={missing[:3]}"
        )
    order = [locations[value] for value in target_hashes]
    aligned = {}
    for form, values in orbit.items():
        values = np.asarray(values, float)
        if values.shape != (len(source_hashes),):
            raise ValueError(
                f"orbit form {form!r} has shape {values.shape}; "
                f"expected {(len(source_hashes),)}"
            )
        aligned[form] = values[order]
    return aligned


def validate_executor_prompt_arms(
        meta: list[dict], cell: dict, *, arm_ids: Collection[str] | None = None) -> dict:
    """Require selected scored prompts to be byte-identical to declared bank arms."""
    selected = (
        {arm["id"] for arm in cell["arms"]}
        if arm_ids is None else set(arm_ids)
    )
    available = {arm["id"] for arm in cell["arms"]}
    missing_arms = sorted(selected - available)
    if missing_arms:
        raise ValueError(
            f"requested prompt arms are absent from bank cell {cell['id']}: {missing_arms}"
        )
    if not selected:
        raise ValueError("at least one prompt arm must be selected for validation")
    expected = {}
    for arm in cell["arms"]:
        if arm["id"] not in selected:
            continue
        for form in arm["forms"]:
            key = (arm["id"], form["id"])
            if key in expected:
                raise ValueError(f"duplicate arm/form in bank for {cell['id']}: {key}")
            expected[key] = form["prompt_sha256"]
    actual = {}
    for row in meta:
        if (row.get("cell_id") != cell["id"]
                or row.get("arm_id") not in selected):
            continue
        key = (row.get("arm_id"), row.get("form"))
        if key in actual:
            raise ValueError(f"duplicate scored arm/form for {cell['id']}: {key}")
        actual[key] = row.get("prompt_sha256")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"executor prompt bank mismatch for {cell['id']}: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    changed = [key for key in expected if actual[key] != expected[key]]
    if changed:
        key = sorted(changed)[0]
        raise ValueError(
            f"executor prompt hash mismatch for {cell['id']}/{key[0]}/{key[1]}: "
            f"scored={actual[key]!r} bank={expected[key]!r}"
        )
    return {
        "valid": True,
        "cell_id": cell["id"],
        "arm_ids": sorted(selected),
        "n_arm_forms": len(expected),
        "prompt_hashes_identical": True,
    }


def validate_executor_prompt_bank(meta: list[dict], cell: dict) -> dict:
    """Require scored executor prompts to be byte-identical to the complete arm bank."""
    return validate_executor_prompt_arms(meta, cell)


# Compatibility aliases for the frozen experiment modules.  New code should use the public names.
_load_shard = load_score_shard
load_public_index = load_shard_index
_average_repetitions = average_repetitions
_orbits = score_orbits
_align_orbit = align_orbit
