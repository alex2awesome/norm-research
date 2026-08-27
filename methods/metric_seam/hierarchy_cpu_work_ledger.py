"""Build a resumable CPU work ledger for the frozen 990-cell hierarchy target.

This control-plane ledger validates source cells, compiler briefs, label-free item
splits, local registry paths, and explicit completion receipts.  It never imports or
executes candidate code, opens a prompt/reference score, reads an outcome, or calls a
model/API/GPU.  A compiler brief is recorded as an authoring input only.  Candidate and
artifact paths are declarations until local files and the completion receipt validate.

Re-running with ``--resume`` is idempotent when the derived state is unchanged.  When
the registry advances, the new snapshot receives a revision and retains the prior
summary in a compact history.  Operational history is not a scientific result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from methods.metric_seam.hierarchy_batch import (
    BRIEF_SCHEMA,
    READINESS_SCHEMA,
    build_readiness,
    compile_brief,
)
from methods.metric_seam.hierarchy_items import validate_task_items
from methods.metric_seam.hierarchy_panel_compat import validate_hierarchy_panel


SCHEMA = "metric-seam.hierarchy-cpu-work-ledger.v1"
CODE_REVIEW_CORRECTED_FUNNEL_SCHEMA = "metric-seam.code-review-corrected-funnel.v1"
MATH_STATIC_SCHEMA = "metric-seam.math-construct-fidelity-merged.v1"
MATH_EXECUTION_SCHEMA = "metric-seam.math-lclamp-execution.v1"
MATH_GATE_SCHEMA = "metric-seam.math-lclamp-train-profile-gate.v1"
MATH_PROMPT_SCHEMA = "metric-seam.math-prompt-articulability-batch.v1"
MATH_SYMBOLIC_SCHEMA = "metric-seam.math-symbolic-capability-construct-fidelity.v1"
SCIENCE_STATIC_SCHEMA = "metric-seam.hierarchy-science-claim-construct-fidelity.v1"
SCIENCE_BLOCKER_SCHEMA = "metric-seam.science-canonical-representation-blocker.v1"
SCIENCE_EXECUTION_SCHEMA = "metric-seam.science-fullarticle-execution.v1"
SCIENCE_GATE_SCHEMA = "metric-seam.science-fullarticle-train-gate.v1"
PATENT_STATIC_SCHEMA = "metric-seam.hierarchy-patent-claim-structure-fidelity.v1"
PATENT_EXECUTION_SCHEMA = "metric-seam.hierarchy-patent-claim-structure-execution.v3"
PATENT_PROGRAM_SCHEMA = "metric-seam.patent-claim-structure.v13"
PATENT_GATE_SCHEMA = "metric-seam.hierarchy-patent-claim-structure-train-gate.v1"
PATENT_OPERATIONAL_SCHEMA = (
    "metric-seam.hierarchy-patent-claim-structure-operational-summary.v1"
)
PATENT_PROMPT_SCHEMA = "metric-seam.patent-prompt-articulability-batch.v3"
PATENT_PROMPT_V1_AUDIT_SCHEMA = "metric-seam.patent-prompt-v1-cross-audit.v1"
PATENT_PROMPT_SUPERSESSION_SCHEMA = "metric-seam.patent-prompt-supersession.v2"
PATENT_PROMPT_VALIDATOR_SCHEMA = "metric-seam.patent-prompt-validator-freeze.v1"
ROOT = Path(__file__).resolve().parents[2]

MATH_ARTIFACT_KEYS = frozenset(
    {
        "canonical_static",
        "canonical_train_execution",
        "canonical_train_gate",
        "canonical_heldout_execution",
        "canonical_prompt_train",
        "canonical_prompt_heldout",
        "additive_symbolic_static",
    }
)
SCIENCE_ARTIFACT_KEYS = frozenset(
    {
        "canonical_static",
        "canonical_representation_blocker",
        "additive_fullarticle_train_execution",
        "additive_fullarticle_train_gate",
        "additive_fullarticle_heldout_execution",
    }
)
PATENT_ARTIFACT_KEYS = frozenset(
    {
        "canonical_static",
        "canonical_train_execution",
        "canonical_train_gate",
        "canonical_heldout_execution",
        "canonical_operational_summary",
        "canonical_prompt_train",
        "canonical_prompt_heldout",
        "prompt_v1_cross_audit",
        "prompt_supersession",
        "prompt_validator_freeze",
    }
)


class CpuWorkLedgerError(ValueError):
    """Raised when a work snapshot cannot be derived without guessing."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CpuWorkLedgerError(f"cannot load JSON input {path}: {exc}") from exc


def load_briefs(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CpuWorkLedgerError(
                        f"compiler brief line {line_number} is not a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CpuWorkLedgerError(f"cannot load compiler briefs {path}: {exc}") from exc
    return rows


def load_item_panels(items_root: Path, tasks: Sequence[str]) -> dict[str, tuple[dict, list, list]]:
    panels = {}
    for task in tasks:
        task_root = items_root / task
        manifest = _load_json(task_root / "manifest.json")
        train = _load_json(task_root / "compiler_train.json")
        heldout = _load_json(task_root / "sealed_heldout.json")
        if not isinstance(manifest, dict) or not isinstance(train, list) or not isinstance(
            heldout, list
        ):
            raise CpuWorkLedgerError(f"{task}: malformed shared-item artifacts")
        try:
            validate_task_items(manifest, train, heldout)
        except ValueError as exc:
            raise CpuWorkLedgerError(f"{task}: invalid shared-item artifacts: {exc}") from exc
        if manifest.get("task") != task:
            raise CpuWorkLedgerError(f"{task}: item manifest task identity drift")
        policy = manifest.get("policy", {})
        if (
            policy.get("external_supervision_used") is not False
            or policy.get("compiler_receives_heldout_text") is not False
        ):
            raise CpuWorkLedgerError(f"{task}: item policy is not compiler-safe")
        panels[task] = (manifest, train, heldout)
    return panels


def _validate_briefs(panel: Mapping, briefs: Sequence[Mapping]) -> dict[str, Mapping]:
    expected_cells = {str(cell["id"]): cell for cell in panel["cells"]}
    by_id: dict[str, Mapping] = {}
    for index, brief in enumerate(briefs):
        if not isinstance(brief, Mapping) or brief.get("schema") != BRIEF_SCHEMA:
            raise CpuWorkLedgerError(f"compiler brief {index} has the wrong schema")
        metric = brief.get("metric")
        cell_id = metric.get("cell_id") if isinstance(metric, Mapping) else None
        if not isinstance(cell_id, str) or cell_id in by_id:
            raise CpuWorkLedgerError(f"compiler brief {index} has an invalid/duplicate cell")
        by_id[cell_id] = brief
    if set(by_id) != set(expected_cells):
        raise CpuWorkLedgerError("compiler briefs do not cover the frozen panel exactly")
    for cell_id, cell in expected_cells.items():
        if by_id[cell_id] != compile_brief(cell):
            raise CpuWorkLedgerError(f"compiler brief content drift for {cell_id}")
    return by_id


def _code_review_scientific_stages(
    readiness: Mapping, corrected_funnel: Mapping | None
) -> dict[str, str]:
    rows = [row for row in readiness["rows"] if row["task"] == "code-review"]
    if corrected_funnel is None:
        return {row["cell_id"]: "not_integrated_na" for row in rows}
    if (
        corrected_funnel.get("schema") != CODE_REVIEW_CORRECTED_FUNNEL_SCHEMA
        or corrected_funnel.get("status")
        != "corrected_static_gate_propagated_without_reexecution"
        or corrected_funnel.get("task") != "code-review"
    ):
        raise CpuWorkLedgerError("unexpected code-review corrected-funnel artifact")
    sealed = corrected_funnel.get("sealed_inputs")
    if not isinstance(sealed, Mapping) or any(value is not False for value in sealed.values()):
        raise CpuWorkLedgerError("corrected funnel violates its sealed-input declaration")
    historical_static = {
        row["cell_id"] for row in rows if row["candidate_program_present"]
    }
    historical_train = {
        row["cell_id"] for row in rows if row["operational_relation_local_witness"]
    }
    historical_heldout = {
        row["cell_id"]
        for row in rows
        if row["heldout_confirmatory_reconstruction_ready"]
    }
    historical_prompt = {
        row["cell_id"] for row in rows if row["prompt_reference_compiled_unscored"]
    }
    removed = corrected_funnel.get("removed_mappings")
    if not isinstance(removed, Mapping):
        raise CpuWorkLedgerError("corrected funnel has no removal sets")

    def removed_ids(stage: str) -> set[str]:
        values = removed.get(stage)
        if not isinstance(values, list):
            raise CpuWorkLedgerError(f"corrected funnel has no {stage} removal list")
        ids = {str(value.get("cell_id")) for value in values if isinstance(value, Mapping)}
        if len(ids) != len(values):
            raise CpuWorkLedgerError(f"corrected funnel {stage} removals are malformed")
        return ids

    corrected_static = historical_static - removed_ids("static")
    corrected_train = historical_train - removed_ids("train_operational")
    corrected_heldout = historical_heldout - removed_ids("heldout_confirmatory")
    corrected_prompt = historical_prompt & corrected_heldout
    if not corrected_prompt <= corrected_heldout <= corrected_train <= corrected_static:
        raise CpuWorkLedgerError("corrected code-review stage sets are not nested")
    expected = corrected_funnel.get("corrected_readout", {}).get("stages", {})
    observed_counts = {
        "relation_local_static_fidelity": len(corrected_static),
        "train_operational_relation_witness": len(corrected_train),
        "heldout_confirmatory_reconstruction_evaluable": len(corrected_heldout),
    }
    for stage, count in observed_counts.items():
        if expected.get(stage, {}).get("balanced_panel", {}).get("n_positive") != count:
            raise CpuWorkLedgerError(f"corrected code-review count drift at {stage}")

    result = {}
    for row in rows:
        cell_id = row["cell_id"]
        if cell_id in corrected_prompt:
            stage = "corrected_prompt_ready_unscored"
        elif cell_id in corrected_heldout:
            stage = "corrected_heldout_ready_prompt_not_compiled"
        elif cell_id in corrected_train:
            stage = "corrected_train_operational_not_heldout_ready"
        elif cell_id in corrected_static:
            stage = "corrected_static_fidelity_not_train_operational"
        elif cell_id in historical_static:
            stage = "historical_candidate_rejected_by_corrected_construct_audit"
        else:
            stage = "no_construct_faithful_candidate_registered"
        result[cell_id] = stage
    return result


def _require_artifact(
    artifacts: Mapping[str, Mapping],
    key: str,
    *,
    schema: str,
    status: str,
    task: str | None = None,
) -> Mapping:
    artifact = artifacts.get(key)
    if not isinstance(artifact, Mapping):
        raise CpuWorkLedgerError(f"missing or malformed {key} stage artifact")
    if artifact.get("schema") != schema or artifact.get("status") != status:
        raise CpuWorkLedgerError(f"unexpected schema/status for {key} stage artifact")
    if task is not None and artifact.get("task") != task:
        raise CpuWorkLedgerError(f"unexpected task identity for {key} stage artifact")
    return artifact


def _rows_by_cell(
    artifact: Mapping, expected_cell_ids: set[str], *, label: str
) -> dict[str, Mapping]:
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise CpuWorkLedgerError(f"{label} has no row list")
    by_id: dict[str, Mapping] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("cell_id"), str):
            raise CpuWorkLedgerError(f"{label} has a malformed row")
        cell_id = row["cell_id"]
        if cell_id in by_id:
            raise CpuWorkLedgerError(f"{label} has duplicate cell {cell_id}")
        by_id[cell_id] = row
    if set(by_id) != expected_cell_ids:
        raise CpuWorkLedgerError(f"{label} does not cover its frozen task cells exactly")
    return by_id


def _relation_cell_ids_from_programs(artifact: Mapping, *, label: str) -> set[str]:
    programs = artifact.get("programs")
    if not isinstance(programs, list):
        raise CpuWorkLedgerError(f"{label} has no program list")
    ids: list[str] = []
    for program in programs:
        if not isinstance(program, Mapping):
            raise CpuWorkLedgerError(f"{label} has a malformed program")
        relations = program.get("relations")
        if not isinstance(relations, list):
            raise CpuWorkLedgerError(f"{label} program has no relation list")
        for relation in relations:
            if not isinstance(relation, Mapping) or not isinstance(
                relation.get("cell_id"), str
            ):
                raise CpuWorkLedgerError(f"{label} has a malformed relation mapping")
            ids.append(relation["cell_id"])
    if len(ids) != len(set(ids)):
        raise CpuWorkLedgerError(f"{label} has duplicate relation mappings")
    return set(ids)


def _relation_cell_ids_from_list(
    artifact: Mapping, key: str, *, label: str
) -> set[str]:
    mappings = artifact.get(key)
    if not isinstance(mappings, list):
        raise CpuWorkLedgerError(f"{label} has no {key} list")
    ids = []
    for mapping in mappings:
        if not isinstance(mapping, Mapping) or not isinstance(mapping.get("cell_id"), str):
            raise CpuWorkLedgerError(f"{label} has a malformed relation mapping")
        ids.append(mapping["cell_id"])
    if len(ids) != len(set(ids)):
        raise CpuWorkLedgerError(f"{label} has duplicate relation mappings")
    return set(ids)


def _require_all_false(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or not value or any(item is not False for item in value.values()):
        raise CpuWorkLedgerError(f"{label} must be a nonempty all-false declaration")


def _require_repo_file(path_value: Any, *, artifact_root: Path, label: str) -> None:
    if not isinstance(path_value, str) or not path_value:
        raise CpuWorkLedgerError(f"{label} has no declared local path")
    root = artifact_root.resolve()
    path = (root / path_value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CpuWorkLedgerError(f"{label} path escapes the artifact root") from exc
    if not path.is_file():
        raise CpuWorkLedgerError(f"{label} declared file is not locally present")


def _validate_math_execution(
    artifact: Mapping,
    *,
    phase: str,
    expected_ids: set[str],
    expected_items: int,
    label: str,
) -> None:
    expected_status = "conditional_slice_execution_complete"
    if (
        artifact.get("schema") != MATH_EXECUTION_SCHEMA
        or artifact.get("status") != expected_status
        or artifact.get("phase") != phase
        or artifact.get("n_items") != expected_items
    ):
        raise CpuWorkLedgerError(f"unexpected schema/status/phase for {label}")
    for key in (
        "reference_fields_passed_to_worker",
        "outcome_fields_passed_to_worker",
        "actual_llm_extractions_passed_to_worker",
        "models_or_apis_called_by_runner",
        "credentials_inherited_by_worker",
        "accelerators_visible_to_worker",
        "ops_corpus_or_retrieval_state_loaded",
        "original_hybrid_execution",
        "pure_code_rewrite_claimed",
        "whole_construct_fidelity_claimed",
    ):
        if artifact.get(key) is not False:
            raise CpuWorkLedgerError(f"{label} violates sealed-input/claim limit at {key}")
    if _relation_cell_ids_from_programs(artifact, label=label) != expected_ids:
        raise CpuWorkLedgerError(f"{label} relation cells drift from static fidelity")
    summary = artifact.get("summary")
    if not isinstance(summary, Mapping) or summary.get("n_relation_mappings") != len(
        expected_ids
    ):
        raise CpuWorkLedgerError(f"{label} relation count drift")


def _validate_math_prompt_manifest(
    artifact: Mapping,
    *,
    phase: str,
    expected_ids: set[str],
    artifact_root: Path,
    label: str,
) -> None:
    if (
        artifact.get("schema") != MATH_PROMPT_SCHEMA
        or artifact.get("status") != "compiled_unscored"
        or artifact.get("task") != "math-stackexchange"
        or artifact.get("phase") != phase
    ):
        raise CpuWorkLedgerError(f"unexpected schema/status/phase for {label}")
    _require_all_false(artifact.get("forbidden_inputs"), label=f"{label} forbidden_inputs")
    shared = artifact.get("shared_ctext_contract")
    if (
        not isinstance(shared, Mapping)
        or shared.get("same_ordered_item_rows_as_lclamp") is not True
        or shared.get("same_ctext_bytes_as_lclamp") is not True
        or shared.get("item_phase") != phase
    ):
        raise CpuWorkLedgerError(f"{label} does not preserve the shared-ctext contract")
    cells = artifact.get("cells")
    if not isinstance(cells, list):
        raise CpuWorkLedgerError(f"{label} has no cell list")
    ids = [cell.get("cell_id") for cell in cells if isinstance(cell, Mapping)]
    if len(ids) != len(cells) or len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise CpuWorkLedgerError(f"{label} cell set drift")
    summary = artifact.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("n_cells") != len(expected_ids)
        or summary.get("n_jobs", 0) <= 0
        or summary.get("n_prompt_responses") != 0
        or summary.get("n_reconstruction_estimates") != 0
        or summary.get("n_isomorphism_adjudications") != 0
    ):
        raise CpuWorkLedgerError(f"{label} summary overstates or drifts from compiled-unscored")
    jobs = artifact.get("jobs_artifact")
    if (
        not isinstance(jobs, Mapping)
        or jobs.get("model_or_api_calls_performed") is not False
        or jobs.get("n_jobs") != summary.get("n_jobs")
    ):
        raise CpuWorkLedgerError(f"{label} jobs receipt is malformed")
    _require_repo_file(jobs.get("path"), artifact_root=artifact_root, label=label)


def _math_scientific_overlay(
    readiness: Mapping,
    artifacts: Mapping[str, Mapping] | None,
    *,
    artifact_root: Path,
    train_items: int,
    heldout_items: int,
) -> tuple[dict[str, str], dict | None]:
    rows = [row for row in readiness["rows"] if row["task"] == "math-stackexchange"]
    if artifacts is None:
        return ({row["cell_id"]: "not_integrated_na" for row in rows}, None)
    if set(artifacts) != MATH_ARTIFACT_KEYS:
        raise CpuWorkLedgerError("math stage overlay requires its exact artifact set")
    expected_ids = {row["cell_id"] for row in rows}
    static = _require_artifact(
        artifacts,
        "canonical_static",
        schema=MATH_STATIC_SCHEMA,
        status="static_construct_fidelity_complete_pre_execution",
        task="math-stackexchange",
    )
    if static.get("panel_content_sha256") != readiness.get("panel_content_sha256"):
        raise CpuWorkLedgerError("math static artifact uses a different frozen panel")
    static_rows = _rows_by_cell(static, expected_ids, label="math canonical static")
    accepted = {
        cell_id
        for cell_id, row in static_rows.items()
        if row.get("eligible_for_relation_local_execution") is True
    }
    mismatched = {
        cell_id for cell_id, row in static_rows.items() if row.get("verdict") == "mismatch"
    }
    bounded = {
        cell_id
        for cell_id, row in static_rows.items()
        if row.get("verdict") == "no_candidate_bounded_non_discovery"
    }
    if (
        len(accepted) != 33
        or len(mismatched) != 14
        or len(bounded) != 43
        or accepted | mismatched | bounded != expected_ids
        or any(static_rows[cell_id].get("verdict") != "partial" for cell_id in accepted)
        or any(static_rows[cell_id].get("scope") != "subrelation_only" for cell_id in accepted)
    ):
        raise CpuWorkLedgerError("math canonical static verdict funnel drift")
    summary = static.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("eligible_for_relation_local_execution") != 33
        or summary.get("whole_construct_exact_count") != 0
    ):
        raise CpuWorkLedgerError("math canonical static summary drift")

    train_execution = _require_artifact(
        artifacts,
        "canonical_train_execution",
        schema=MATH_EXECUTION_SCHEMA,
        status="conditional_slice_execution_complete",
    )
    _validate_math_execution(
        train_execution,
        phase="compiler_train",
        expected_ids=accepted,
        expected_items=train_items,
        label="math canonical train execution",
    )
    gate = _require_artifact(
        artifacts,
        "canonical_train_gate",
        schema=MATH_GATE_SCHEMA,
        status="frozen_before_heldout_profile_execution",
    )
    for key in (
        "reference_values_used",
        "outcome_labels_used",
        "heldout_items_or_outputs_used",
        "prompt_or_llm_values_used",
        "score_direction_or_target_used",
    ):
        if gate.get(key) is not False:
            raise CpuWorkLedgerError(f"math train gate violates train-only policy at {key}")
    selected_profiles = gate.get("selected_program_profiles")
    if not isinstance(selected_profiles, list):
        raise CpuWorkLedgerError("math train gate has no selected profile list")
    selected_ids = {
        cell_id
        for profile in selected_profiles
        if isinstance(profile, Mapping)
        for cell_id in profile.get("cell_ids", [])
        if isinstance(cell_id, str)
    }
    gate_summary = gate.get("summary")
    if (
        selected_ids != accepted
        or not isinstance(gate_summary, Mapping)
        or gate_summary.get("n_selected_relation_mappings") != 33
        or gate_summary.get("n_static_relation_mappings") != 33
    ):
        raise CpuWorkLedgerError("math train gate relation set/count drift")

    heldout_execution = _require_artifact(
        artifacts,
        "canonical_heldout_execution",
        schema=MATH_EXECUTION_SCHEMA,
        status="conditional_slice_execution_complete",
    )
    _validate_math_execution(
        heldout_execution,
        phase="heldout_pre_reference",
        expected_ids=accepted,
        expected_items=heldout_items,
        label="math canonical heldout execution",
    )
    _validate_math_prompt_manifest(
        artifacts["canonical_prompt_train"],
        phase="compiler_train",
        expected_ids=accepted,
        artifact_root=artifact_root,
        label="math compiler-train prompt manifest",
    )
    _validate_math_prompt_manifest(
        artifacts["canonical_prompt_heldout"],
        phase="heldout_pre_reference",
        expected_ids=accepted,
        artifact_root=artifact_root,
        label="math heldout prompt manifest",
    )

    symbolic = _require_artifact(
        artifacts,
        "additive_symbolic_static",
        schema=MATH_SYMBOLIC_SCHEMA,
        status="static_five_dimension_adjudication_complete_pre_execution",
        task="math-stackexchange",
    )
    if (
        symbolic.get("panel_content_sha256") != readiness.get("panel_content_sha256")
        or symbolic.get("design_scope")
        != "additive_manual_capability_expansion_sensitivity"
        or symbolic.get("canonical_artifact_modified") is not False
    ):
        raise CpuWorkLedgerError("math symbolic artifact is not a noncanonical additive sensitivity")
    for key in (
        "programs_or_items_executed",
        "certificate_counts_loaded",
        "prompt_outputs_loaded",
        "reference_values_loaded",
        "outcome_labels_loaded",
        "correlations_or_reconstruction_loaded",
        "models_apis_or_gpus_used",
    ):
        if symbolic.get(key) is not False:
            raise CpuWorkLedgerError(f"math symbolic sensitivity overstates its stage at {key}")
    symbolic_rows = _rows_by_cell(symbolic, expected_ids, label="math symbolic static")
    symbolic_ids = {
        cell_id
        for cell_id, row in symbolic_rows.items()
        if row.get("symbolic_relation_local_static_fidelity") is True
    }
    symbolic_summary = symbolic.get("summary")
    if (
        len(symbolic_ids) != 7
        or len(symbolic_ids - accepted) != 5
        or len(symbolic_ids & accepted) != 2
        or len(symbolic_ids | accepted) != 38
        or not isinstance(symbolic_summary, Mapping)
        or symbolic_summary.get("canonical_relation_local_cells_unchanged") != 33
        or symbolic_summary.get("additive_sensitivity_union_cells") != 38
        or symbolic_summary.get("n_whole_construct_exact") != 0
    ):
        raise CpuWorkLedgerError("math symbolic additive union drift")

    stages = {}
    for cell_id in expected_ids:
        if cell_id in accepted:
            stage = "math_canonical_prompt_jobs_compiled_unscored"
        elif cell_id in mismatched:
            stage = "math_canonical_retrieved_candidate_relation_mismatch"
        else:
            stage = "math_canonical_no_candidate_bounded_non_discovery"
        stages[cell_id] = stage
    overlay = {
        "status": "canonical_task_overlay_with_noncanonical_additive_sensitivity",
        "source_schemas": sorted(
            {
                MATH_STATIC_SCHEMA,
                MATH_EXECUTION_SCHEMA,
                MATH_GATE_SCHEMA,
                MATH_PROMPT_SCHEMA,
                MATH_SYMBOLIC_SCHEMA,
            }
        ),
        "canonical_representation": "hierarchy items_v2 ctext",
        "canonical_stage_counts": {
            "relation_local_static_fidelity": 33,
            "train_operational_relation_witness": 33,
            "heldout_pre_reference_relation_witness": 33,
            "prompt_jobs_compiled_unscored": 33,
            "prompt_responses": 0,
            "reconstruction_estimates": 0,
            "isomorphism_adjudications": 0,
            "completed_deep_metric_seam_runs": 0,
        },
        "canonical_depth_counts": {"1": 10, "2": 23},
        "additive_symbolic_static_sensitivity": {
            "relation_local_static_matches": 7,
            "newly_covered_cells": 5,
            "overlapping_canonical_cells": 2,
            "static_union_cells": 38,
            "programs_or_items_executed": False,
            "promotes_canonical_stages": False,
        },
        "counts": dict(sorted(Counter(stages.values()).items())),
        "claim_limits": {
            "whole_construct_exact": 0,
            "articulability_established": False,
            "reconstruction_established": False,
            "isomorphism_established": False,
            "codability_established": False,
        },
    }
    return stages, overlay


def _validate_science_execution(
    artifact: Mapping,
    *,
    phase: str,
    expected_ids: set[str],
    expected_items: int,
    label: str,
) -> None:
    if (
        artifact.get("schema") != SCIENCE_EXECUTION_SCHEMA
        or artifact.get("status") != "execution_complete_pre_prompt_pre_reference"
        or artifact.get("task") != "peer-review"
        or artifact.get("phase") != phase
    ):
        raise CpuWorkLedgerError(f"unexpected schema/status/phase for {label}")
    representation = artifact.get("representation")
    if (
        not isinstance(representation, Mapping)
        or representation.get("canonical_hierarchy_items") is not False
        or representation.get("same_ctext_bytes_for_future_prompt_and_code") is not True
        or representation.get("complete_pdf_claimed") is not False
    ):
        raise CpuWorkLedgerError(f"{label} representation is not the declared additive frame")
    policy = artifact.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise CpuWorkLedgerError(f"{label} has no execution policy")
    for key in (
        "accelerators_used",
        "external_supervision_used",
        "models_or_apis_called",
        "outcome_values_loaded",
        "prompt_or_reconstruction_outputs_loaded",
        "reference_values_loaded",
    ):
        if policy.get(key) is not False:
            raise CpuWorkLedgerError(f"{label} violates its no-supervision policy at {key}")
    if _relation_cell_ids_from_list(artifact, "relation_mappings", label=label) != expected_ids:
        raise CpuWorkLedgerError(f"{label} relation mappings drift from canonical static")
    rows = artifact.get("rows")
    summary = artifact.get("summary")
    if (
        not isinstance(rows, list)
        or len(rows) != expected_items
        or not isinstance(summary, Mapping)
        or summary.get("n_unique_item_executions") != expected_items
        or summary.get("n_relation_mappings") != len(expected_ids)
        or summary.get("three_state_totals_unique_items", {}).get("failed") != 0
    ):
        raise CpuWorkLedgerError(f"{label} execution receipt drift")


def _science_scientific_overlay(
    readiness: Mapping,
    artifacts: Mapping[str, Mapping] | None,
    *,
    train_items: int,
    heldout_items: int,
) -> tuple[dict[str, str], dict | None]:
    rows = [row for row in readiness["rows"] if row["task"] == "peer-review"]
    if artifacts is None:
        return ({row["cell_id"]: "not_integrated_na" for row in rows}, None)
    if set(artifacts) != SCIENCE_ARTIFACT_KEYS:
        raise CpuWorkLedgerError("science stage overlay requires its exact artifact set")
    expected_ids = {row["cell_id"] for row in rows}
    static = _require_artifact(
        artifacts,
        "canonical_static",
        schema=SCIENCE_STATIC_SCHEMA,
        status="static-relation-local-adjudication-complete-pre-execution",
        task="peer-review",
    )
    if static.get("source_panel_content_sha256") != readiness.get("panel_content_sha256"):
        raise CpuWorkLedgerError("science static artifact uses a different frozen panel")
    static_rows = _rows_by_cell(static, expected_ids, label="science canonical static")
    accepted = {
        cell_id
        for cell_id, row in static_rows.items()
        if row.get("verdict") == "partial_relation_local"
    }
    mismatched = {
        cell_id
        for cell_id, row in static_rows.items()
        if row.get("verdict") == "relation_mismatch"
    }
    bounded = {
        cell_id for cell_id, row in static_rows.items() if row.get("verdict") == "no_candidate"
    }
    if (
        len(accepted) != 6
        or len(mismatched) != 3
        or len(bounded) != 81
        or accepted | mismatched | bounded != expected_ids
        or any(static_rows[cell_id].get("maximum_matching_relation_depth") != 3 for cell_id in accepted)
        or any(static_rows[cell_id].get("exact_whole_construct_fidelity") is not False for cell_id in accepted)
        or any(static_rows[cell_id].get("execution_witness_established") is not False for cell_id in accepted)
    ):
        raise CpuWorkLedgerError("science canonical static verdict funnel drift")
    static_summary = static.get("summary")
    if (
        not isinstance(static_summary, Mapping)
        or static_summary.get("n_partial_relation_local") != 6
        or static_summary.get("n_exact_whole_construct") != 0
        or static_summary.get("n_execution_witnesses") != 0
    ):
        raise CpuWorkLedgerError("science canonical static summary drift")

    blocker = _require_artifact(
        artifacts,
        "canonical_representation_blocker",
        schema=SCIENCE_BLOCKER_SCHEMA,
        status="canonical_execution_blocked_by_representation_mismatch",
        task="peer-review",
    )
    canonical_representation = blocker.get("canonical_representation")
    capability_requires = blocker.get("capability_requires")
    execution = blocker.get("execution")
    disposition = blocker.get("disposition")
    if (
        not isinstance(canonical_representation, Mapping)
        or canonical_representation.get("content") != "abstract only"
        or canonical_representation.get("same_bytes_required_for_prompt_and_code") is not True
        or not isinstance(capability_requires, Mapping)
        or capability_requires.get("distinct_fullpaper_body") is not True
        or not isinstance(execution, Mapping)
        or execution.get("performed") is not False
        or not isinstance(disposition, Mapping)
        or disposition.get("canonical_six_mappings_remain_static_only") is not True
        or disposition.get("forced_join_permitted") is not False
    ):
        raise CpuWorkLedgerError("science canonical representation blocker drift")
    _require_all_false(
        blocker.get("forbidden_inputs"), label="science canonical blocker forbidden_inputs"
    )

    train_execution = artifacts["additive_fullarticle_train_execution"]
    _validate_science_execution(
        train_execution,
        phase="compiler_train",
        expected_ids=accepted,
        expected_items=train_items,
        label="science additive full-article train execution",
    )
    gate = _require_artifact(
        artifacts,
        "additive_fullarticle_train_gate",
        schema=SCIENCE_GATE_SCHEMA,
        status="train_only_gate_frozen",
        task="peer-review",
    )
    _require_all_false(
        gate.get("forbidden_selection_inputs"),
        label="science additive train gate forbidden_selection_inputs",
    )
    if (
        gate.get("selected") is not True
        or _relation_cell_ids_from_list(
            gate, "selected_relation_mappings", label="science additive train gate"
        )
        != accepted
        or gate.get("summary", {}).get("n_selected_relation_mappings") != 6
    ):
        raise CpuWorkLedgerError("science additive train gate selection drift")
    heldout_execution = artifacts["additive_fullarticle_heldout_execution"]
    _validate_science_execution(
        heldout_execution,
        phase="heldout_pre_reference",
        expected_ids=accepted,
        expected_items=heldout_items,
        label="science additive full-article heldout execution",
    )
    if train_execution.get("representation") != heldout_execution.get("representation"):
        raise CpuWorkLedgerError("science additive train/heldout representations drift")
    if heldout_execution.get("train_gate", {}).get("selection_used_heldout") is not False:
        raise CpuWorkLedgerError("science additive heldout receipt violates train-only selection")

    stages = {}
    for cell_id in expected_ids:
        if cell_id in accepted:
            stage = "science_canonical_static_execution_blocked_representation_mismatch"
        elif cell_id in mismatched:
            stage = "science_canonical_retrieved_candidate_relation_mismatch"
        else:
            stage = "science_canonical_no_candidate_bounded_non_discovery"
        stages[cell_id] = stage
    overlay = {
        "status": "canonical_static_overlay_with_separate_representation_execution",
        "source_schemas": sorted(
            {
                SCIENCE_STATIC_SCHEMA,
                SCIENCE_BLOCKER_SCHEMA,
                SCIENCE_EXECUTION_SCHEMA,
                SCIENCE_GATE_SCHEMA,
            }
        ),
        "canonical_representation": "hierarchy items_v2 abstract-only ctext",
        "canonical_stage_counts": {
            "relation_local_static_fidelity": 6,
            "execution_blocked_by_representation_mismatch": 6,
            "train_operational_relation_witness": 0,
            "heldout_pre_reference_relation_witness": 0,
            "prompt_jobs_compiled_unscored": 0,
            "prompt_responses": 0,
            "reconstruction_estimates": 0,
            "isomorphism_adjudications": 0,
            "completed_deep_metric_seam_runs": 0,
        },
        "canonical_depth_counts": {"3": 6},
        "additive_fullarticle_representation": {
            "canonical_hierarchy_items": False,
            "train_operational_relation_mappings": 6,
            "heldout_pre_reference_relation_mappings": 6,
            "prompt_jobs_compiled_unscored": 0,
            "promotes_canonical_stages": False,
        },
        "counts": dict(sorted(Counter(stages.values()).items())),
        "claim_limits": {
            "whole_construct_exact": 0,
            "articulability_established": False,
            "reconstruction_established": False,
            "isomorphism_established": False,
            "codability_established": False,
        },
    }
    return stages, overlay


def _validate_patent_execution(
    artifact: Mapping,
    *,
    phase: str,
    expected_items: int,
    label: str,
) -> None:
    if (
        artifact.get("schema") != PATENT_EXECUTION_SCHEMA
        or artifact.get("program_schema") != PATENT_PROGRAM_SCHEMA
        or artifact.get("phase") != phase
    ):
        raise CpuWorkLedgerError(f"{label} has the wrong runner/program/phase")
    design = artifact.get("design")
    if not isinstance(design, Mapping) or design.get("input_fields") != [
        "item_key",
        "ctext",
    ]:
        raise CpuWorkLedgerError(f"{label} does not use exact shared ctext")
    for key in (
        "outcome_or_reference_values_loaded",
        "prompt_outputs_loaded",
        "prior_art_or_examiner_evidence_loaded",
        "external_supervision_used",
        "whole_patent_score_emitted",
        "absence_certificate_permitted",
    ):
        if design.get(key) is not False:
            raise CpuWorkLedgerError(f"{label} violates its sealed channel at {key}")
    summary = artifact.get("summary")
    rows = artifact.get("rows")
    if (
        not isinstance(summary, Mapping)
        or summary.get("n_items") != expected_items
        or summary.get("failure_types") != {}
        or not isinstance(rows, list)
        or len(rows) != expected_items
        or any(row.get("status") == "failed" for row in rows if isinstance(row, Mapping))
    ):
        raise CpuWorkLedgerError(f"{label} execution summary drift")


def _validate_patent_prompt_manifest(
    artifact: Mapping,
    *,
    phase: str,
    expected_ids: set[str],
    expected_jobs: int,
    artifact_root: Path,
    label: str,
) -> None:
    if (
        artifact.get("schema") != PATENT_PROMPT_SCHEMA
        or artifact.get("status") != "compiled_unscored"
        or artifact.get("task") != "patents"
        or artifact.get("phase") != phase
    ):
        raise CpuWorkLedgerError(f"unexpected {label} schema/status/task/phase")
    expected_role = (
        "train_only_articulation_and_relation_reconstruction"
        if phase == "compiler_train"
        else "fixed_after_train_gate_exploratory_pre_reference"
    )
    temporal = artifact.get("temporal_provenance")
    projection = artifact.get("model_input_projection_contract")
    if (
        artifact.get("batch_role") != expected_role
        or not isinstance(temporal, Mapping)
        or temporal.get("absence_of_human_influence_certified") is not False
        or temporal.get("mechanical_consumption_of_heldout_code_or_summary") is not False
        or not isinstance(projection, Mapping)
        or projection.get("post_code_schema_is_cap_specialized_per_job") is not True
        or projection.get("post_code_semantic_validator_required")
        != "validate_post_code_response.v3"
    ):
        raise CpuWorkLedgerError(f"{label} temporal/response-validation contract drift")
    if phase == "heldout_pre_reference" and (
        temporal.get("current_heldout_disposition")
        != "fixed-after-train-gate exploratory pre-reference replay"
        or temporal.get("fresh_confirmatory_split_required_for_temporal_preregistration")
        is not True
        or temporal.get("heldout_code_execution_existed_before_prompt_v1_v2_and_v3")
        is not True
    ):
        raise CpuWorkLedgerError(f"{label} overstates temporal predeclaration")
    forbidden = artifact.get("forbidden_inputs")
    if not isinstance(forbidden, Mapping) or set(forbidden.values()) != {False}:
        raise CpuWorkLedgerError(f"{label} crosses a forbidden input channel")
    cells = artifact.get("cells")
    if (
        not isinstance(cells, list)
        or {row.get("cell_id") for row in cells if isinstance(row, Mapping)}
        != expected_ids
    ):
        raise CpuWorkLedgerError(f"{label} cell set drift")
    summary = artifact.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("n_cells") != len(expected_ids)
        or summary.get("n_jobs") != expected_jobs
        or summary.get("n_prompt_responses") != 0
        or summary.get("n_reconstruction_estimates") != 0
        or summary.get("n_isomorphism_adjudications") != 0
    ):
        raise CpuWorkLedgerError(f"{label} summary overstates or drifts")
    jobs = artifact.get("jobs_artifact")
    if (
        not isinstance(jobs, Mapping)
        or jobs.get("n_jobs") != expected_jobs
        or jobs.get("model_api_or_gpu_calls_performed") is not False
    ):
        raise CpuWorkLedgerError(f"{label} jobs receipt drift")
    _require_repo_file(jobs.get("path"), artifact_root=artifact_root, label=label)


def _patent_scientific_overlay(
    readiness: Mapping,
    artifacts: Mapping[str, Mapping] | None,
    *,
    artifact_root: Path,
    train_items: int,
    heldout_items: int,
) -> tuple[dict[str, str], dict | None]:
    rows = [row for row in readiness["rows"] if row["task"] == "patents"]
    if artifacts is None:
        return ({row["cell_id"]: "not_integrated_pending_parallel_lane" for row in rows}, None)
    if set(artifacts) != PATENT_ARTIFACT_KEYS:
        raise CpuWorkLedgerError("patent stage overlay requires its exact artifact set")
    expected_ids = {row["cell_id"] for row in rows}
    static = _require_artifact(
        artifacts,
        "canonical_static",
        schema=PATENT_STATIC_SCHEMA,
        status="conservative-static-adjudication-complete",
        task="patents",
    )
    if static.get("source_panel_content_sha256") != readiness.get("panel_content_sha256"):
        raise CpuWorkLedgerError("patent static artifact uses a different frozen panel")
    static_rows = _rows_by_cell(static, expected_ids, label="patent canonical static")
    accepted = {
        cell_id
        for cell_id, row in static_rows.items()
        if row.get("verdict") == "partial_relation_local"
    }
    near_misses = {
        cell_id
        for cell_id, row in static_rows.items()
        if row.get("verdict") == "sensitivity_near_miss_not_accepted"
    }
    rejected = {
        cell_id
        for cell_id, row in static_rows.items()
        if row.get("verdict") == "no_faithful_relation"
    }
    if (
        len(accepted) != 8
        or len(near_misses) != 4
        or len(rejected) != 78
        or accepted | near_misses | rejected != expected_ids
        or any(
            static_rows[cell_id].get("exact_whole_construct_fidelity") is not False
            for cell_id in accepted
        )
    ):
        raise CpuWorkLedgerError("patent canonical static verdict funnel drift")
    static_only = set()
    train_operational = set()
    for cell_id in accepted:
        relations = static_rows[cell_id].get("matched_relations")
        if not isinstance(relations, list) or not relations:
            raise CpuWorkLedgerError("patent accepted static row has no matched relation")
        classes = {
            relation.get("train_operational_applicability", {}).get("classification")
            for relation in relations
            if isinstance(relation, Mapping)
        }
        if classes == {"measured_but_constant_non_operational"}:
            static_only.add(cell_id)
        elif "measured_but_constant_non_operational" not in classes:
            train_operational.add(cell_id)
        else:
            raise CpuWorkLedgerError("patent static row mixes constant/operational relations")
    if len(static_only) != 3 or len(train_operational) != 5:
        raise CpuWorkLedgerError("patent train applicability split drift")

    train_execution = artifacts["canonical_train_execution"]
    _validate_patent_execution(
        train_execution,
        phase="compiler_train",
        expected_items=train_items,
        label="patent compiler-train execution",
    )
    gate = _require_artifact(
        artifacts,
        "canonical_train_gate",
        schema=PATENT_GATE_SCHEMA,
        status="frozen-before-heldout-pre-reference-execution",
        task="patents",
    )
    boundaries = gate.get("channel_boundaries")
    if (
        not isinstance(boundaries, Mapping)
        or boundaries.get("input_fields") != ["item_key", "ctext"]
        or any(value is not False for key, value in boundaries.items() if key != "input_fields")
    ):
        raise CpuWorkLedgerError("patent gate violates its train-only channel")
    selected_rows = gate.get("selected_operational_cells")
    constant_rows = gate.get("static_only_cells")
    selected_ids = {
        row.get("cell_id") for row in selected_rows if isinstance(row, Mapping)
    } if isinstance(selected_rows, list) else set()
    constant_ids = {
        row.get("cell_id") for row in constant_rows if isinstance(row, Mapping)
    } if isinstance(constant_rows, list) else set()
    gate_summary = gate.get("summary", {})
    if (
        selected_ids != train_operational
        or constant_ids != static_only
        or gate_summary.get("n_selected_operational_cells") != 5
        or gate_summary.get("n_static_only_constant_cells") != 3
        or gate_summary.get("n_whole_construct_cells") != 0
        or gate_summary.get("prompt_scored_cells") != 0
        or gate_summary.get("reconstruction_evaluable_cells") != 0
        or gate_summary.get("isomorphism_evaluable_cells") != 0
    ):
        raise CpuWorkLedgerError("patent train gate selection/count drift")

    heldout_execution = artifacts["canonical_heldout_execution"]
    _validate_patent_execution(
        heldout_execution,
        phase="heldout_pre_reference",
        expected_items=heldout_items,
        label="patent heldout pre-reference execution",
    )
    operational = _require_artifact(
        artifacts,
        "canonical_operational_summary",
        schema=PATENT_OPERATIONAL_SCHEMA,
        status="heldout-relation-measurement-complete-pre-reference",
        task="patents",
    )
    operational_rows = operational.get("heldout_operational_cells")
    heldout_ids = {
        row.get("cell_id")
        for row in operational_rows
        if isinstance(row, Mapping) and row.get("heldout_relation_measurable") is True
    } if isinstance(operational_rows, list) else set()
    stage_summary = operational.get("stage_summary", {})
    if (
        heldout_ids != selected_ids
        or stage_summary.get("n_static_relation_local_cells") != 8
        or stage_summary.get("n_train_operational_cells") != 5
        or stage_summary.get("n_heldout_relation_measurable_cells") != 5
        or stage_summary.get("n_prompt_articulability_measured_cells") != 0
        or stage_summary.get("n_reference_reconstruction_measured_cells") != 0
        or stage_summary.get("n_prompt_code_isomorphism_evaluable_cells") != 0
        or stage_summary.get("n_whole_criterion_codability_established_cells") != 0
    ):
        raise CpuWorkLedgerError("patent heldout operational summary drift")
    operational_boundaries = operational.get("channel_boundaries", {})
    for key in (
        "reference_or_prompt_values_loaded",
        "outcomes_loaded",
        "prior_art_or_examiner_evidence_loaded",
        "external_supervision_loaded",
        "models_or_apis_called",
        "whole_patent_score_emitted",
    ):
        if operational_boundaries.get(key) is not False:
            raise CpuWorkLedgerError(f"patent operational summary violates {key}")

    _validate_patent_prompt_manifest(
        artifacts["canonical_prompt_train"],
        phase="compiler_train",
        expected_ids=selected_ids,
        expected_jobs=7_500,
        artifact_root=artifact_root,
        label="patent compiler-train prompt manifest",
    )
    _validate_patent_prompt_manifest(
        artifacts["canonical_prompt_heldout"],
        phase="heldout_pre_reference",
        expected_ids=selected_ids,
        expected_jobs=19_500,
        artifact_root=artifact_root,
        label="patent heldout prompt manifest",
    )
    prompt_audit = _require_artifact(
        artifacts,
        "prompt_v1_cross_audit",
        schema=PATENT_PROMPT_V1_AUDIT_SCHEMA,
        status="complete-v1-superseded-not-executable",
    )
    disposition = prompt_audit.get("disposition")
    findings = prompt_audit.get("findings")
    if (
        not isinstance(disposition, Mapping)
        or disposition.get("v1_prompt_manifests_and_jobs")
        != "superseded exploratory receipts; do not execute model calls from these packs"
        or disposition.get("code_train_gate")
        != "unaffected and remains frozen before heldout code execution"
        or disposition.get("code_heldout_execution")
        != "unaffected relation-local pre-reference code receipt"
        or disposition.get("temporal_predeclaration")
        != "not certifiable for the v1 prompt wording or response contracts"
        or not isinstance(findings, list)
        or {row.get("id") for row in findings if isinstance(row, Mapping)}
        != {"P1", "P2", "P3", "P4", "P5", "P6"}
    ):
        raise CpuWorkLedgerError("patent v1 prompt audit disposition drift")
    supersession = _require_artifact(
        artifacts,
        "prompt_supersession",
        schema=PATENT_PROMPT_SUPERSESSION_SCHEMA,
        status="v3-repaired-compiled-unscored",
    )
    temporal_disposition = supersession.get("temporal_disposition")
    execution_disposition = supersession.get("execution")
    if (
        supersession.get("v1_disposition")
        != "superseded exploratory receipt; do not execute model calls"
        or supersession.get("v2_disposition")
        != (
            "superseded unexecuted receipt; duplicate-certificate and cap-status "
            "semantic holes repaired in v3"
        )
        or not isinstance(temporal_disposition, Mapping)
        or temporal_disposition.get("temporally_predeclared_or_confirmatory") is not False
        or temporal_disposition.get("fresh_confirmatory_split_required") is not True
        or temporal_disposition.get("absence_of_human_influence_certified") is not False
        or not isinstance(execution_disposition, Mapping)
        or execution_disposition.get("prompt_responses") != 0
        or execution_disposition.get("reconstruction_estimates") != 0
        or execution_disposition.get("isomorphism_adjudications") != 0
        or execution_disposition.get("model_api_or_gpu_calls_performed") is not False
    ):
        raise CpuWorkLedgerError("patent v3 prompt supersession disposition drift")
    validator_freeze = _require_artifact(
        artifacts,
        "prompt_validator_freeze",
        schema=PATENT_PROMPT_VALIDATOR_SCHEMA,
        status="frozen-unscored-before-any-prompt-execution",
        task="patents",
    )
    validator_execution = validator_freeze.get("execution_contract")
    validator_temporal = validator_freeze.get("temporal_disposition")
    validator_tests = validator_freeze.get("tests")
    if (
        validator_freeze.get("validator_id") != "validate_post_code_response.v3"
        or not isinstance(validator_execution, Mapping)
        or validator_execution.get("v1_and_v2_packs_executable") is not False
        or validator_execution.get("v3_post_code_responses_must_pass_this_exact_validator")
        is not True
        or validator_execution.get("prompt_responses_observed_before_freeze") != 0
        or validator_execution.get("model_api_or_gpu_calls_performed") is not False
        or not isinstance(validator_temporal, Mapping)
        or validator_temporal.get("fresh_confirmatory_split_required") is not True
        or validator_temporal.get("absence_of_human_influence_certified") is not False
        or not isinstance(validator_tests, Mapping)
        or validator_tests.get("focused_passed") != 19
        or validator_tests.get("combined_patent_stack_passed") != 77
        or validator_tests.get("ruff_clean") is not True
    ):
        raise CpuWorkLedgerError("patent v3 prompt validator freeze drift")

    stages = {}
    for cell_id in expected_ids:
        if cell_id in selected_ids:
            stage = "patent_prompt_jobs_compiled_unscored"
        elif cell_id in static_only:
            stage = "patent_static_fidelity_train_and_heldout_constant"
        elif cell_id in near_misses:
            stage = "patent_sensitivity_near_miss_not_credited"
        else:
            stage = "patent_no_faithful_relation_bounded_non_discovery"
        stages[cell_id] = stage
    overlay = {
        "status": "canonical_shared_ctext_relation_local_pipeline_pre_reconstruction",
        "source_schemas": sorted(
            {
                PATENT_STATIC_SCHEMA,
                PATENT_EXECUTION_SCHEMA,
                PATENT_GATE_SCHEMA,
                PATENT_OPERATIONAL_SCHEMA,
                PATENT_PROMPT_SCHEMA,
                PATENT_PROMPT_V1_AUDIT_SCHEMA,
                PATENT_PROMPT_SUPERSESSION_SCHEMA,
                PATENT_PROMPT_VALIDATOR_SCHEMA,
            }
        ),
        "canonical_representation": "hierarchy items_v2 exact shared patent ctext",
        "canonical_stage_counts": {
            "relation_local_static_fidelity": 8,
            "train_operational_relation_witness": 5,
            "heldout_pre_reference_relation_witness": 5,
            "prompt_comparison_mappings_compiled_unscored": 5,
            "prompt_jobs_compiled_unscored": 27_000,
            "prompt_responses": 0,
            "reconstruction_estimates": 0,
            "isomorphism_adjudications": 0,
            "completed_deep_metric_seam_runs": 0,
        },
        "static_depth_counts": {"1": 7, "2": 1},
        "operational_depth_counts": {"1": 4, "2": 1},
        "prompt_temporal_status": {
            "v1_packs": "superseded_not_executable",
            "v2_packs": "superseded_unexecuted",
            "v3_train": "compiled_unscored",
            "v3_heldout": "fixed_after_train_gate_exploratory_pre_reference",
            "fresh_split_required_for_confirmatory_temporal_claim": True,
        },
        "counts": dict(sorted(Counter(stages.values()).items())),
        "claim_limits": {
            "whole_construct_exact": 0,
            "articulability_established": False,
            "relation_local_code_verifiability_established": True,
            "reconstruction_established": False,
            "isomorphism_established": False,
            "codability_established": False,
        },
    }
    return stages, overlay


def _next_action(row: Mapping, scientific_stage: str) -> tuple[str, bool]:
    evidence = row["artifact_evidence"]
    local = set(evidence["locally_present_fields"])
    if row["completed_deep_metric_seam_run"]:
        return "none_validated_complete", False
    if row["bounded_non_discovery_recorded"]:
        return "none_terminal_bounded_non_discovery_recorded", False
    if scientific_stage == "not_integrated_na":
        return "task_specific_progress_not_integrated_na", False
    if scientific_stage == "not_integrated_pending_parallel_lane":
        return "task_specific_progress_pending_parallel_lane_integration", False
    if scientific_stage == "math_canonical_prompt_jobs_compiled_unscored":
        return "prompt_reference_scoring_required_not_cpu_only", False
    if scientific_stage == "patent_prompt_jobs_compiled_unscored":
        return "prompt_reference_scoring_required_not_cpu_only", False
    if scientific_stage in {
        "math_canonical_retrieved_candidate_relation_mismatch",
        "math_canonical_no_candidate_bounded_non_discovery",
        "science_canonical_retrieved_candidate_relation_mismatch",
        "science_canonical_no_candidate_bounded_non_discovery",
        "patent_sensitivity_near_miss_not_credited",
        "patent_no_faithful_relation_bounded_non_discovery",
    }:
        return "none_terminal_static_bounded_result", False
    if scientific_stage == "science_canonical_static_execution_blocked_representation_mismatch":
        return "none_canonical_execution_blocked_representation_mismatch", False
    if scientific_stage == "patent_static_fidelity_train_and_heldout_constant":
        return "none_frozen_train_gate_did_not_select", False
    if not row["candidate_program_declared"]:
        return "candidate_authoring_or_bounded_non_discovery_required", False
    if not row["candidate_program_present"]:
        return "repair_missing_or_out_of_root_candidate_declaration", True
    if scientific_stage == "historical_candidate_rejected_by_corrected_construct_audit":
        return "none_historical_seed_rejected_bounded_result", False
    if scientific_stage == "corrected_static_fidelity_not_train_operational":
        return "none_frozen_train_gate_did_not_select", False
    if scientific_stage == "corrected_train_operational_not_heldout_ready":
        return "none_frozen_heldout_not_confirmatory", False
    if scientific_stage == "corrected_heldout_ready_prompt_not_compiled":
        return "compile_prompt_reference_jobs_not_score_them", True
    if scientific_stage == "corrected_prompt_ready_unscored":
        return "prompt_reference_scoring_required_not_cpu_only", False
    if not local:
        return "scientific_stage_unknown_stop_before_execution", False
    return "scientific_stage_unknown_stop_before_execution", False


def _summary(
    rows: Sequence[Mapping],
    item_panels: Mapping[str, tuple],
    task_overlays: Mapping[str, Mapping | None],
) -> dict:
    actions = Counter(str(row["next_action"]) for row in rows)
    return {
        "target_cells": len(rows),
        "validated_label_free_item_tasks": len(item_panels),
        "compiler_briefs_ready_input_only": len(rows),
        "candidate_programs_declared": sum(row["candidate_program_declared"] for row in rows),
        "candidate_sources_locally_present": sum(row["candidate_program_present"] for row in rows),
        "candidate_execution_artifacts_locally_present": sum(
            "candidate_execution_path" in row["artifact_evidence"]["locally_present_fields"]
            for row in rows
        ),
        "validated_completed_deep_runs": sum(
            row["completed_deep_metric_seam_run"] for row in rows
        ),
        "cpu_only_followups_available_without_new_scientific_judgment": sum(
            row["cpu_only_followup_available"] for row in rows
        ),
        "scientific_stage_by_task": {
            task: (
                dict(task_overlays[task])
                if task_overlays.get(task) is not None
                else (
                    {
                        "status": "not_integrated_pending_parallel_lane",
                        "counts": None,
                    }
                    if task == "patents"
                    else {"status": "not_integrated_na", "counts": None}
                )
            )
            for task in sorted(item_panels)
        },
        "truly_untouched_tasks_na": sorted(
            task
            for task in item_panels
            if task_overlays.get(task) is None and task != "patents"
        ),
        "next_action_counts": dict(sorted(actions.items())),
    }


def build_ledger(
    panel: Mapping,
    briefs: Sequence[Mapping],
    item_panels: Mapping[str, tuple[Mapping, Sequence[Mapping], Sequence[Mapping]]],
    *,
    program_registry: Mapping[str, Mapping] | None = None,
    code_review_corrected_funnel: Mapping | None = None,
    math_stage_artifacts: Mapping[str, Mapping] | None = None,
    science_stage_artifacts: Mapping[str, Mapping] | None = None,
    patent_stage_artifacts: Mapping[str, Mapping] | None = None,
    artifact_root: Path = ROOT,
    previous: Mapping | None = None,
) -> dict:
    errors = validate_hierarchy_panel(panel)
    if errors:
        raise CpuWorkLedgerError(f"invalid frozen hierarchy panel: {errors}")
    _validate_briefs(panel, briefs)
    expected_tasks = set(panel["tasks"])
    if set(item_panels) != expected_tasks:
        raise CpuWorkLedgerError("shared-item panels do not cover every frozen task")
    item_summary = {}
    for task, triple in item_panels.items():
        if not isinstance(triple, tuple) or len(triple) != 3:
            raise CpuWorkLedgerError(f"{task}: item panel must be (manifest, train, heldout)")
        manifest, train, heldout = triple
        try:
            validate_task_items(manifest, train, heldout)
        except ValueError as exc:
            raise CpuWorkLedgerError(f"{task}: invalid item panel: {exc}") from exc
        if manifest.get("task") != task:
            raise CpuWorkLedgerError(f"{task}: item manifest identity drift")
        policy = manifest.get("policy", {})
        if (
            policy.get("outcome_columns_emitted") is not False
            or policy.get("external_supervision_used") is not False
            or policy.get("compiler_receives_heldout_text") is not False
        ):
            raise CpuWorkLedgerError(f"{task}: item panel violates label-free compiler policy")
        item_summary[task] = {
            "compiler_train_n": len(train),
            "sealed_heldout_n": len(heldout),
            "candidate_received_heldout_text": False,
        }

    readiness = build_readiness(
        panel,
        program_registry=program_registry,
        artifact_root=artifact_root,
    )
    if readiness.get("schema") != READINESS_SCHEMA:
        raise CpuWorkLedgerError("derived readiness has an unexpected schema")
    code_review_stages = _code_review_scientific_stages(
        readiness, code_review_corrected_funnel
    )
    math_stages, math_overlay = _math_scientific_overlay(
        readiness,
        math_stage_artifacts,
        artifact_root=artifact_root,
        train_items=len(item_panels["math-stackexchange"][1]),
        heldout_items=len(item_panels["math-stackexchange"][2]),
    )
    science_stages, science_overlay = _science_scientific_overlay(
        readiness,
        science_stage_artifacts,
        train_items=150,
        heldout_items=150,
    )
    patent_stages, patent_overlay = _patent_scientific_overlay(
        readiness,
        patent_stage_artifacts,
        artifact_root=artifact_root,
        train_items=len(item_panels["patents"][1]),
        heldout_items=len(item_panels["patents"][2]),
    )
    code_review_overlay = None
    if any(stage != "not_integrated_na" for stage in code_review_stages.values()):
        code_review_overlay = {
            "status": "corrected_metadata_overlay",
            "counts": dict(sorted(Counter(code_review_stages.values()).items())),
        }
    task_overlays = {
        "code-review": code_review_overlay,
        "math-stackexchange": math_overlay,
        "peer-review": science_overlay,
        "patents": patent_overlay,
    }
    rows = []
    for source in readiness["rows"]:
        if source["task"] == "code-review":
            scientific_stage = code_review_stages[source["cell_id"]]
        elif source["task"] == "math-stackexchange":
            scientific_stage = math_stages[source["cell_id"]]
        elif source["task"] == "peer-review":
            scientific_stage = science_stages[source["cell_id"]]
        elif source["task"] == "patents":
            scientific_stage = patent_stages[source["cell_id"]]
        else:
            scientific_stage = "not_integrated_na"
        action, cpu_only = _next_action(source, scientific_stage)
        rows.append(
            {
                "queue_index": source["queue_index"],
                "cell_id": source["cell_id"],
                "task": source["task"],
                "level": source["level"],
                "compiler_brief_state": "compiled_authoring_input_not_a_run",
                "compiler_train_items_state": "validated_label_free_text_only",
                "candidate_program_declared": source["candidate_program_declared"],
                "candidate_program_present": source["candidate_program_present"],
                "artifact_evidence": source["artifact_evidence"],
                "completed_deep_metric_seam_run": source[
                    "completed_deep_metric_seam_run"
                ],
                "bounded_non_discovery_recorded": source[
                    "bounded_non_discovery_recorded"
                ],
                "task_specific_bounded_static_result": scientific_stage.endswith(
                    ("bounded_non_discovery", "relation_mismatch")
                )
                or scientific_stage
                == "historical_candidate_rejected_by_corrected_construct_audit",
                "scientific_stage": scientific_stage,
                "scientific_stage_source": (
                    CODE_REVIEW_CORRECTED_FUNNEL_SCHEMA
                    if source["task"] == "code-review"
                    and scientific_stage != "not_integrated_na"
                    else (
                        "validated_math_stage_artifact_set"
                        if source["task"] == "math-stackexchange"
                        and math_overlay is not None
                        else (
                            "validated_science_stage_artifact_set"
                            if source["task"] == "peer-review"
                            and science_overlay is not None
                            else (
                                "validated_patent_stage_artifact_set"
                                if source["task"] == "patents"
                                and patent_overlay is not None
                                else None
                            )
                        )
                    )
                ),
                "next_action": action,
                "cpu_only_followup_available": cpu_only,
            }
        )
    summary = _summary(rows, item_panels, task_overlays)
    base = {
        "schema": SCHEMA,
        "panel_content_sha256": panel["panel_content_sha256"],
        "goal": "30 validated deep metric-seam runs per task and R1/R2/R3 level",
        "scope": {
            "tasks": len(panel["tasks"]),
            "levels": list(panel["levels"]),
            "metrics_per_task_level": panel["n_per_task_level"],
            "total_metrics": len(rows),
        },
        "execution_policy": {
            "cpu_control_plane_only": True,
            "candidate_code_executed": False,
            "prompt_or_reference_scores_loaded": False,
            "outcome_values_loaded": False,
            "external_supervised_target_used": False,
            "model_api_or_gpu_used": False,
            "briefs_count_as_runs": False,
            "path_declarations_count_as_runs": False,
            "cross_task_scientific_stage_ledger_emitted": any(
                overlay is not None for overlay in task_overlays.values()
            ),
            "heldout_text_passed_to_candidate": False,
        },
        "item_panels": item_summary,
        "summary": summary,
        "rows": rows,
    }
    if previous is not None:
        if (
            previous.get("schema") != SCHEMA
            or previous.get("panel_content_sha256") != panel["panel_content_sha256"]
            or [row.get("cell_id") for row in previous.get("rows", [])]
            != [row["cell_id"] for row in rows]
        ):
            raise CpuWorkLedgerError("previous ledger is not resumable for this frozen panel")
        unchanged = all(
            previous.get(key) == base[key]
            for key in ("scope", "execution_policy", "item_panels", "summary", "rows")
        )
        if unchanged:
            return dict(previous)
        prior_revision = int(previous.get("revision", 1))
        history = list(previous.get("history", []))
        history.append({"revision": prior_revision, "summary": previous.get("summary")})
        return {**base, "revision": prior_revision + 1, "history": history}
    return {**base, "revision": 1, "history": []}


def _atomic_write(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_optional_artifact_group(paths: Mapping[str, Path | None], *, label: str) -> dict | None:
    provided = {key: path for key, path in paths.items() if path is not None}
    if not provided:
        return None
    if set(provided) != set(paths):
        missing = sorted(set(paths) - set(provided))
        raise CpuWorkLedgerError(f"{label} artifact group is partial; missing {missing}")
    loaded = {}
    for key, path in provided.items():
        value = _load_json(path)
        if not isinstance(value, Mapping):
            raise CpuWorkLedgerError(f"{label} {key} artifact must be a JSON object")
        loaded[key] = value
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--briefs", type=Path, required=True)
    parser.add_argument("--items-root", type=Path, required=True)
    parser.add_argument("--program-registry", type=Path)
    parser.add_argument("--code-review-corrected-funnel", type=Path)
    parser.add_argument("--math-canonical-static", type=Path)
    parser.add_argument("--math-canonical-train-execution", type=Path)
    parser.add_argument("--math-canonical-train-gate", type=Path)
    parser.add_argument("--math-canonical-heldout-execution", type=Path)
    parser.add_argument("--math-canonical-prompt-train", type=Path)
    parser.add_argument("--math-canonical-prompt-heldout", type=Path)
    parser.add_argument("--math-additive-symbolic-static", type=Path)
    parser.add_argument("--science-canonical-static", type=Path)
    parser.add_argument("--science-canonical-representation-blocker", type=Path)
    parser.add_argument("--science-additive-fullarticle-train-execution", type=Path)
    parser.add_argument("--science-additive-fullarticle-train-gate", type=Path)
    parser.add_argument("--science-additive-fullarticle-heldout-execution", type=Path)
    parser.add_argument("--patent-canonical-static", type=Path)
    parser.add_argument("--patent-canonical-train-execution", type=Path)
    parser.add_argument("--patent-canonical-train-gate", type=Path)
    parser.add_argument("--patent-canonical-heldout-execution", type=Path)
    parser.add_argument("--patent-canonical-operational-summary", type=Path)
    parser.add_argument("--patent-canonical-prompt-train", type=Path)
    parser.add_argument("--patent-canonical-prompt-heldout", type=Path)
    parser.add_argument("--patent-prompt-v1-cross-audit", type=Path)
    parser.add_argument("--patent-prompt-supersession", type=Path)
    parser.add_argument("--patent-prompt-validator-freeze", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --resume")
    panel = _load_json(args.panel)
    if not isinstance(panel, Mapping):
        raise CpuWorkLedgerError("panel must be a JSON object")
    briefs = load_briefs(args.briefs)
    registry = None
    if args.program_registry:
        raw_registry = _load_json(args.program_registry)
        if not isinstance(raw_registry, Mapping):
            raise CpuWorkLedgerError("program registry must be a JSON object")
        registry = raw_registry.get("registry", raw_registry)
        if not isinstance(registry, Mapping):
            raise CpuWorkLedgerError("program registry payload has no mapping registry")
    panels = load_item_panels(args.items_root, panel["tasks"])
    corrected_funnel = (
        _load_json(args.code_review_corrected_funnel)
        if args.code_review_corrected_funnel
        else None
    )
    if corrected_funnel is not None and not isinstance(corrected_funnel, Mapping):
        raise CpuWorkLedgerError("corrected funnel must be a JSON object")
    math_artifacts = _load_optional_artifact_group(
        {
            "canonical_static": args.math_canonical_static,
            "canonical_train_execution": args.math_canonical_train_execution,
            "canonical_train_gate": args.math_canonical_train_gate,
            "canonical_heldout_execution": args.math_canonical_heldout_execution,
            "canonical_prompt_train": args.math_canonical_prompt_train,
            "canonical_prompt_heldout": args.math_canonical_prompt_heldout,
            "additive_symbolic_static": args.math_additive_symbolic_static,
        },
        label="math",
    )
    science_artifacts = _load_optional_artifact_group(
        {
            "canonical_static": args.science_canonical_static,
            "canonical_representation_blocker": args.science_canonical_representation_blocker,
            "additive_fullarticle_train_execution": (
                args.science_additive_fullarticle_train_execution
            ),
            "additive_fullarticle_train_gate": args.science_additive_fullarticle_train_gate,
            "additive_fullarticle_heldout_execution": (
                args.science_additive_fullarticle_heldout_execution
            ),
        },
        label="science",
    )
    patent_artifacts = _load_optional_artifact_group(
        {
            "canonical_static": args.patent_canonical_static,
            "canonical_train_execution": args.patent_canonical_train_execution,
            "canonical_train_gate": args.patent_canonical_train_gate,
            "canonical_heldout_execution": args.patent_canonical_heldout_execution,
            "canonical_operational_summary": args.patent_canonical_operational_summary,
            "canonical_prompt_train": args.patent_canonical_prompt_train,
            "canonical_prompt_heldout": args.patent_canonical_prompt_heldout,
            "prompt_v1_cross_audit": args.patent_prompt_v1_cross_audit,
            "prompt_supersession": args.patent_prompt_supersession,
            "prompt_validator_freeze": args.patent_prompt_validator_freeze,
        },
        label="patent",
    )
    previous = _load_json(args.out) if args.resume and args.out.exists() else None
    payload = build_ledger(
        panel,
        briefs,
        panels,
        program_registry=registry,
        code_review_corrected_funnel=corrected_funnel,
        math_stage_artifacts=math_artifacts,
        science_stage_artifacts=science_artifacts,
        patent_stage_artifacts=patent_artifacts,
        artifact_root=args.artifact_root,
        previous=previous,
    )
    if previous != payload:
        _atomic_write(args.out, payload)
    print(json.dumps({"revision": payload["revision"], **payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
