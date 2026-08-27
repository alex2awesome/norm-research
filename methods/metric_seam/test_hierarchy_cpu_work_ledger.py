from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_cpu_work_ledger import (
    CpuWorkLedgerError,
    build_ledger,
    load_briefs,
    load_item_panels,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


@pytest.fixture(scope="module")
def inputs():
    panel = json.loads((BASE / "panel_v3.json").read_text(encoding="utf-8"))
    briefs = load_briefs(BASE / "compiler_briefs_v3.jsonl")
    item_panels = load_item_panels(BASE / "items_v2", panel["tasks"])
    registry_payload = json.loads(
        (BASE / "code_review_registry_v2.json").read_text(encoding="utf-8")
    )
    corrected_funnel = json.loads(
        (BASE / "code_review_corrected_funnel_v1.json").read_text(encoding="utf-8")
    )
    math_artifacts = {
        "canonical_static": "math_stackexchange_construct_fidelity_merged_v1.json",
        "canonical_train_execution": "math_stackexchange_lclamp_compiler_train_v1.json",
        "canonical_train_gate": "math_stackexchange_lclamp_train_profile_gate_v1.json",
        "canonical_heldout_execution": (
            "math_stackexchange_lclamp_heldout_pre_reference_v1.json"
        ),
        "canonical_prompt_train": "math_stackexchange_prompt_train_manifest_v1.json",
        "canonical_prompt_heldout": (
            "math_stackexchange_prompt_heldout_fixed_manifest_v1.json"
        ),
        "additive_symbolic_static": (
            "math_stackexchange_symbolic_capability_construct_fidelity_v1.json"
        ),
    }
    science_artifacts = {
        "canonical_static": "peer_review_science_claim_construct_fidelity_v1.json",
        "canonical_representation_blocker": (
            "peer_review_science_canonical_representation_blocker_v1.json"
        ),
        "additive_fullarticle_train_execution": (
            "peer_review_science_fullarticle_compiler_train_v1.json"
        ),
        "additive_fullarticle_train_gate": (
            "peer_review_science_fullarticle_train_gate_v1.json"
        ),
        "additive_fullarticle_heldout_execution": (
            "peer_review_science_fullarticle_heldout_pre_reference_v1.json"
        ),
    }
    patent_artifacts = {
        "canonical_static": "patents_claim_structure_construct_fidelity_v1.json",
        "canonical_train_execution": "patents_claim_structure_compiler_train_v14.json",
        "canonical_train_gate": "patents_claim_structure_train_gate_v1.json",
        "canonical_heldout_execution": (
            "patents_claim_structure_heldout_pre_reference_v1.json"
        ),
        "canonical_operational_summary": (
            "patents_claim_structure_operational_summary_v1.json"
        ),
        "canonical_prompt_train": "patents_prompt_train_manifest_v3.json",
        "canonical_prompt_heldout": "patents_prompt_heldout_fixed_manifest_v3.json",
        "prompt_v1_cross_audit": "patents_prompt_v1_cross_audit.json",
        "prompt_supersession": "patents_prompt_v3_supersession_receipt.json",
        "prompt_validator_freeze": "patents_prompt_v3_validator_freeze.json",
    }

    def load_group(paths):
        return {
            key: json.loads((BASE / path).read_text(encoding="utf-8"))
            for key, path in paths.items()
        }

    return (
        panel,
        briefs,
        item_panels,
        registry_payload["registry"],
        corrected_funnel,
        load_group(math_artifacts),
        load_group(science_artifacts),
        load_group(patent_artifacts),
    )


def test_real_cpu_ledger_keeps_inputs_declarations_and_runs_separate(inputs):
    (
        panel,
        briefs,
        item_panels,
        registry,
        corrected_funnel,
        math_artifacts,
        science_artifacts,
        patent_artifacts,
    ) = inputs
    ledger = build_ledger(
        panel,
        briefs,
        item_panels,
        program_registry=registry,
        code_review_corrected_funnel=corrected_funnel,
        math_stage_artifacts=math_artifacts,
        science_stage_artifacts=science_artifacts,
        patent_stage_artifacts=patent_artifacts,
    )
    summary = ledger["summary"]
    assert {
        key: summary[key]
        for key in (
            "target_cells",
            "validated_label_free_item_tasks",
            "compiler_briefs_ready_input_only",
            "candidate_programs_declared",
            "candidate_sources_locally_present",
            "candidate_execution_artifacts_locally_present",
            "validated_completed_deep_runs",
            "cpu_only_followups_available_without_new_scientific_judgment",
        )
    } == {
        "target_cells": 990,
        "validated_label_free_item_tasks": 11,
        "compiler_briefs_ready_input_only": 990,
        "candidate_programs_declared": 56,
        "candidate_sources_locally_present": 56,
        "candidate_execution_artifacts_locally_present": 56,
        "validated_completed_deep_runs": 0,
        "cpu_only_followups_available_without_new_scientific_judgment": 0,
    }
    assert summary["next_action_counts"] == {
        "candidate_authoring_or_bounded_non_discovery_required": 34,
        "none_canonical_execution_blocked_representation_mismatch": 6,
        "none_frozen_heldout_not_confirmatory": 9,
        "none_frozen_train_gate_did_not_select": 26,
        "none_historical_seed_rejected_bounded_result": 6,
        "none_terminal_static_bounded_result": 223,
        "prompt_reference_scoring_required_not_cpu_only": 56,
        "task_specific_progress_not_integrated_na": 630,
    }
    code_review_stage = summary["scientific_stage_by_task"]["code-review"]
    assert code_review_stage == {
        "status": "corrected_metadata_overlay",
        "counts": {
            "corrected_prompt_ready_unscored": 18,
            "corrected_static_fidelity_not_train_operational": 23,
            "corrected_train_operational_not_heldout_ready": 9,
            "historical_candidate_rejected_by_corrected_construct_audit": 6,
            "no_construct_faithful_candidate_registered": 34,
        },
    }
    math_stage = summary["scientific_stage_by_task"]["math-stackexchange"]
    assert math_stage["canonical_stage_counts"] == {
        "relation_local_static_fidelity": 33,
        "train_operational_relation_witness": 33,
        "heldout_pre_reference_relation_witness": 33,
        "prompt_jobs_compiled_unscored": 33,
        "prompt_responses": 0,
        "reconstruction_estimates": 0,
        "isomorphism_adjudications": 0,
        "completed_deep_metric_seam_runs": 0,
    }
    assert math_stage["additive_symbolic_static_sensitivity"] == {
        "relation_local_static_matches": 7,
        "newly_covered_cells": 5,
        "overlapping_canonical_cells": 2,
        "static_union_cells": 38,
        "programs_or_items_executed": False,
        "promotes_canonical_stages": False,
    }
    assert math_stage["claim_limits"] == {
        "whole_construct_exact": 0,
        "articulability_established": False,
        "reconstruction_established": False,
        "isomorphism_established": False,
        "codability_established": False,
    }
    science_stage = summary["scientific_stage_by_task"]["peer-review"]
    assert science_stage["canonical_stage_counts"] == {
        "relation_local_static_fidelity": 6,
        "execution_blocked_by_representation_mismatch": 6,
        "train_operational_relation_witness": 0,
        "heldout_pre_reference_relation_witness": 0,
        "prompt_jobs_compiled_unscored": 0,
        "prompt_responses": 0,
        "reconstruction_estimates": 0,
        "isomorphism_adjudications": 0,
        "completed_deep_metric_seam_runs": 0,
    }
    assert science_stage["additive_fullarticle_representation"] == {
        "canonical_hierarchy_items": False,
        "train_operational_relation_mappings": 6,
        "heldout_pre_reference_relation_mappings": 6,
        "prompt_jobs_compiled_unscored": 0,
        "promotes_canonical_stages": False,
    }
    patent_stage = summary["scientific_stage_by_task"]["patents"]
    assert patent_stage["canonical_stage_counts"] == {
        "relation_local_static_fidelity": 8,
        "train_operational_relation_witness": 5,
        "heldout_pre_reference_relation_witness": 5,
        "prompt_comparison_mappings_compiled_unscored": 5,
        "prompt_jobs_compiled_unscored": 27000,
        "prompt_responses": 0,
        "reconstruction_estimates": 0,
        "isomorphism_adjudications": 0,
        "completed_deep_metric_seam_runs": 0,
    }
    assert patent_stage["static_depth_counts"] == {"1": 7, "2": 1}
    assert patent_stage["operational_depth_counts"] == {"1": 4, "2": 1}
    assert patent_stage["prompt_temporal_status"] == {
        "v1_packs": "superseded_not_executable",
        "v2_packs": "superseded_unexecuted",
        "v3_train": "compiled_unscored",
        "v3_heldout": "fixed_after_train_gate_exploratory_pre_reference",
        "fresh_split_required_for_confirmatory_temporal_claim": True,
    }
    assert patent_stage["claim_limits"] == {
        "whole_construct_exact": 0,
        "articulability_established": False,
        "relation_local_code_verifiability_established": True,
        "reconstruction_established": False,
        "isomorphism_established": False,
        "codability_established": False,
    }
    assert summary["truly_untouched_tasks_na"] == [
        "creative-writing",
        "grant-funding",
        "humor",
        "legal-outcome-prediction",
        "news-homepages",
        "notice-and-comment",
        "press-releases",
    ]
    assert ledger["execution_policy"] == {
        "cpu_control_plane_only": True,
        "candidate_code_executed": False,
        "prompt_or_reference_scores_loaded": False,
        "outcome_values_loaded": False,
        "external_supervised_target_used": False,
        "model_api_or_gpu_used": False,
        "briefs_count_as_runs": False,
        "path_declarations_count_as_runs": False,
        "cross_task_scientific_stage_ledger_emitted": True,
        "heldout_text_passed_to_candidate": False,
    }
    assert not any(row["completed_deep_metric_seam_run"] for row in ledger["rows"])


def test_compiler_brief_drift_fails_closed(inputs):
    panel, briefs, item_panels, *_ = inputs
    drifted = copy.deepcopy(briefs)
    drifted[0]["metric"]["description"] = "changed"
    with pytest.raises(CpuWorkLedgerError, match="compiler brief content drift"):
        build_ledger(panel, drifted, item_panels)


def test_resume_is_idempotent_and_advances_only_when_registry_state_changes(
    inputs, tmp_path
):
    panel, briefs, item_panels, *_ = inputs
    first = build_ledger(
        panel,
        briefs,
        item_panels,
        artifact_root=tmp_path,
    )
    same = build_ledger(
        panel,
        briefs,
        item_panels,
        artifact_root=tmp_path,
        previous=first,
    )
    assert same == first
    assert same["revision"] == 1

    candidate = tmp_path / "candidate.py"
    candidate.write_text("def score(text, extracted, ops): return None\n", encoding="utf-8")
    cell_id = first["rows"][0]["cell_id"]
    advanced = build_ledger(
        panel,
        briefs,
        item_panels,
        program_registry={cell_id: {"candidate_path": candidate.name}},
        artifact_root=tmp_path,
        previous=first,
    )
    assert advanced["revision"] == 2
    assert advanced["history"] == [{"revision": 1, "summary": first["summary"]}]
    row = next(row for row in advanced["rows"] if row["cell_id"] == cell_id)
    assert row["candidate_program_present"] is True
    assert row["next_action"] == "task_specific_progress_not_integrated_na"
    assert row["completed_deep_metric_seam_run"] is False


def test_math_additive_symbolic_lane_cannot_promote_canonical_stage(inputs):
    panel, briefs, item_panels, registry, corrected, math_artifacts, science, patent = inputs
    ledger = build_ledger(
        panel,
        briefs,
        item_panels,
        program_registry=registry,
        code_review_corrected_funnel=corrected,
        math_stage_artifacts=math_artifacts,
        science_stage_artifacts=science,
        patent_stage_artifacts=patent,
    )
    newly_covered = {
        row["cell_id"]
        for row in math_artifacts["additive_symbolic_static"]["rows"]
        if row["sensitivity_effect"] == "newly_covered_in_additive_sensitivity"
    }
    indexed = {row["cell_id"]: row for row in ledger["rows"]}
    assert len(newly_covered) == 5
    assert all(
        indexed[cell_id]["scientific_stage"]
        != "math_canonical_prompt_jobs_compiled_unscored"
        for cell_id in newly_covered
    )
    assert not any(indexed[cell_id]["completed_deep_metric_seam_run"] for cell_id in newly_covered)


@pytest.mark.parametrize(
    ("group", "mutation", "message"),
    [
        (
            "math",
            lambda artifacts: artifacts["canonical_prompt_train"]["summary"].__setitem__(
                "n_prompt_responses", 1
            ),
            "summary overstates or drifts",
        ),
        (
            "math",
            lambda artifacts: artifacts["additive_symbolic_static"]["summary"].__setitem__(
                "additive_sensitivity_union_cells", 39
            ),
            "symbolic additive union drift",
        ),
        (
            "science",
            lambda artifacts: artifacts["canonical_representation_blocker"][
                "execution"
            ].__setitem__("performed", True),
            "representation blocker drift",
        ),
        (
            "science",
            lambda artifacts: artifacts["additive_fullarticle_train_execution"][
                "representation"
            ].__setitem__("canonical_hierarchy_items", True),
            "not the declared additive frame",
        ),
        (
            "patent",
            lambda artifacts: artifacts["canonical_prompt_heldout"]["summary"].__setitem__(
                "n_prompt_responses", 1
            ),
            "summary overstates or drifts",
        ),
    ],
)
def test_task_stage_overlays_fail_closed_on_overclaiming(inputs, group, mutation, message):
    panel, briefs, item_panels, registry, corrected, math_artifacts, science, patent = inputs
    math_copy = copy.deepcopy(math_artifacts)
    science_copy = copy.deepcopy(science)
    patent_copy = copy.deepcopy(patent)
    mutation(
        math_copy
        if group == "math"
        else science_copy
        if group == "science"
        else patent_copy
    )
    with pytest.raises(CpuWorkLedgerError, match=message):
        build_ledger(
            panel,
            briefs,
            item_panels,
            program_registry=registry,
            code_review_corrected_funnel=corrected,
            math_stage_artifacts=math_copy,
            science_stage_artifacts=science_copy,
            patent_stage_artifacts=patent_copy,
        )


def test_partial_task_stage_overlay_is_rejected(inputs):
    panel, briefs, item_panels, registry, corrected, math_artifacts, science, patent = inputs
    incomplete = dict(math_artifacts)
    incomplete.pop("canonical_heldout_execution")
    with pytest.raises(CpuWorkLedgerError, match="exact artifact set"):
        build_ledger(
            panel,
            briefs,
            item_panels,
            program_registry=registry,
            code_review_corrected_funnel=corrected,
            math_stage_artifacts=incomplete,
            science_stage_artifacts=science,
            patent_stage_artifacts=patent,
        )
