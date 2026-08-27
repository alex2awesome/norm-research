from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_hierarchy_manifest_is_internally_consistent_and_pre_reconstruction():
    current = _load(BASE / "CURRENT.json")
    panel = _load(ROOT / current["panel"])
    bank = _load(ROOT / current["prompt_arm_bank"])
    readiness = _load(ROOT / current["readiness"])
    registry = _load(ROOT / current["code_review"]["program_registry"])
    code_review_cross_audit = _load(
        ROOT / current["code_review"]["construct_fidelity_independent_cross_audit"]
    )
    code_review_corrected = _load(
        ROOT / current["code_review"]["corrected_funnel"]
    )
    prompt = _load(ROOT / current["code_review"]["prompt_manifest"])
    prevalence = _load(ROOT / current["code_review"]["witness_prevalence"])
    a104_representation = _load(
        ROOT
        / current["code_review"][
            "criterion_local_a104_representation_sensitivity"
        ]
    )
    a104_execution = _load(
        ROOT
        / current["code_review"]["criterion_local_a104_execution_augmentation"]
    )
    code_representation = _load(
        ROOT / current["code_review"]["representation_family_sensitivity"]
    )
    code_additive_fidelity = _load(
        ROOT / current["code_review"]["additive_unused_program_fidelity_gate"]
    )
    code_additive_gate = _load(
        ROOT / current["code_review"]["additive_unused_program_train_gate"]
    )
    code_additive_heldout = _load(
        ROOT
        / current["code_review"][
            "additive_unused_program_heldout_pre_reference"
        ]
    )
    code_additive_readiness = _load(
        ROOT / current["code_review"]["additive_unused_program_heldout_readiness"]
    )
    math_map = _load(ROOT / current["math_stackexchange"]["seed_map"])
    math_fidelity = _load(ROOT / current["math_stackexchange"]["construct_fidelity"])
    math_prevalence = _load(ROOT / current["math_stackexchange"]["witness_prevalence"])
    math_train = _load(ROOT / current["math_stackexchange"]["lclamp_compiler_train"])
    math_gate = _load(ROOT / current["math_stackexchange"]["lclamp_train_profile_gate"])
    math_heldout = _load(
        ROOT / current["math_stackexchange"]["lclamp_heldout_pre_reference"]
    )
    math_operational = _load(
        ROOT / current["math_stackexchange"]["lclamp_operational_prevalence"]
    )
    math_prompt_train = _load(
        ROOT / current["math_stackexchange"]["prompt_train_manifest"]
    )
    math_prompt_heldout = _load(
        ROOT / current["math_stackexchange"]["prompt_heldout_fixed_manifest"]
    )
    math_symbolic_fidelity = _load(
        ROOT
        / current["math_stackexchange"]["symbolic_capability_construct_fidelity"]
    )
    math_symbolic_prevalence = _load(
        ROOT
        / current["math_stackexchange"]["symbolic_capability_expansion_prevalence"]
    )
    patent_fidelity = _load(ROOT / current["patents"]["construct_fidelity"])
    patent_prevalence = _load(ROOT / current["patents"]["witness_prevalence"])
    patent_claim_graph_cross_audit = _load(
        ROOT
        / current["patents"]["claim_graph_additive_independent_cross_audit"]
    )
    science_seed = _load(ROOT / current["peer_review_science"]["seed_map"])
    science_fidelity = _load(
        ROOT / current["peer_review_science"]["construct_fidelity"]
    )
    science_prevalence = _load(
        ROOT / current["peer_review_science"]["witness_prevalence"]
    )
    science_blocker = _load(
        ROOT / current["peer_review_science"]["canonical_representation_blocker"]
    )
    science_fullarticle_manifest = _load(
        ROOT / current["peer_review_science"]["additive_fullarticle_items_manifest"]
    )
    science_fullarticle_train = _load(
        ROOT / current["peer_review_science"]["additive_compiler_train"]
    )
    science_fullarticle_heldout = _load(
        ROOT / current["peer_review_science"]["additive_heldout_pre_reference"]
    )
    science_fullarticle_operational = _load(
        ROOT / current["peer_review_science"]["additive_operational_prevalence"]
    )
    science_addressed_binding = _load(
        ROOT / current["peer_review_science"]["additive_addressed_subset_binding"]
    )
    science_exact_prompt = _load(
        ROOT
        / current["peer_review_science"]["additive_exact_ctext_prompt_manifest"]
    )
    science_exact_receipt = _load(
        ROOT
        / current["peer_review_science"][
            "additive_exact_ctext_prompt_audit_receipt"
        ]
    )
    science_exact_projection = _load(
        ROOT
        / current["peer_review_science"][
            "additive_numeric_comparative_projection"
        ]
    )

    assert panel["panel_content_sha256"] == bank["metric_panel_content_sha256"]
    assert panel["panel_content_sha256"] == readiness["panel_content_sha256"]
    assert panel["panel_content_sha256"] == prompt["panel_content_sha256"]
    assert panel["n_cells"] == readiness["n_cells"] == 990
    assert sum(len(cell["arms"]) for cell in bank["cells"]) == current["status"][
        "prompt_arms_compiled"
    ] == 28335

    status = current["status"]
    summary = registry["summary"]
    assert summary["n_relation_local_static_fidelity"] == 56
    assert summary["n_train_operational_relation_mappings"] == 30
    assert summary["n_heldout_confirmatory_reconstruction_ready"] == 21
    corrected_stages = code_review_corrected["corrected_readout"]["stages"]
    assert status["code_review_relation_local_static_fidelity"] == (
        code_review_cross_audit["after_summary"][
            "relation_local_static_fidelity_count"
        ]
    ) == corrected_stages["relation_local_static_fidelity"]["balanced_panel"][
        "n_positive"
    ] == 50
    assert status["code_review_train_operational_relation_mappings"] == (
        corrected_stages["train_operational_relation_witness"]["balanced_panel"][
            "n_positive"
        ]
    ) == 27
    assert status["code_review_heldout_reconstruction_evaluable"] == (
        corrected_stages["heldout_confirmatory_reconstruction_evaluable"][
            "balanced_panel"
        ]["n_positive"]
    ) == 18
    additive_static = code_additive_fidelity["summary"]
    assert status["code_review_additive_relation_local_static_fidelity"] == (
        additive_static["relation_local_static_fidelity_count"]
    ) == 59
    assert status["code_review_additive_new_static_mappings"] == 9
    assert status["code_review_additive_new_static_depth4_mappings"] == 5
    assert status["code_review_additive_train_selected_relation_mappings"] == (
        code_additive_gate["summary"]["n_selected_relation_mappings"]
    ) == 35
    assert status[
        "code_review_additive_heldout_nondegenerate_relation_mappings"
    ] == (
        code_additive_heldout["summary"][
            "n_relation_mappings_with_nondegenerate_measurement"
        ]
    ) == 35
    additive_ready = code_additive_readiness["summary"]
    assert status[
        "code_review_additive_heldout_confirmatory_relation_mappings"
    ] == additive_ready["n_confirmatory_relation_mappings"] == 19
    assert status[
        "code_review_additive_heldout_exploratory_sparse_mappings"
    ] == additive_ready["relation_readiness_counts"]["exploratory_sparse"] == 12
    assert status[
        "code_review_additive_heldout_insufficient_support_mappings"
    ] == additive_ready["relation_readiness_counts"][
        "insufficient_paired_support"
    ] == 4
    assert status["code_review_a104_representation_sensitivity_common_heldout"] == (
        a104_representation["heldout_readout"]["common_support_n"]
    ) == 93
    assert a104_representation["blindness_and_reference_order"][
        "direct_same_input_prefix_prompt_code_test"
    ] is False
    assert status["code_review_a104_execution_augmentation_overlap"] == (
        a104_execution["summary"]["exact_repository_pr_overlap"]
    ) == 32
    assert status[
        "code_review_a104_execution_augmentation_finite_certificates"
    ] == a104_execution["summary"]["finite_execution_certificates"] == 1
    assert a104_execution["representation_boundary"][
        "same_input_representation"
    ] is False
    assert status["code_review_representation_family_primary_unique_programs"] == (
        code_representation["population"]["primary_unique_programs"]
    ) == 10
    assert status["code_review_representation_family_typed_mappings"] == (
        code_representation["population"]["primary_relation_mappings"]
    ) == 18
    assert status["code_review_representation_family_prefix_replay_rows"] == (
        code_representation["P0_exact_frozen_replay"]["total_rows_exact"]
    ) == 4000
    assert status["code_review_representation_family_prefix_replay_mismatches"] == (
        code_representation["P0_exact_frozen_replay"]["total_mismatches"]
    ) == 0
    assert status["code_review_prompt_jobs_compiled_unscored"] == prompt["n_jobs"] == 13500
    assert prompt["n_unique_program_vectors"] == 10
    assert status["code_review_prompt_wrong_relation_controls_refrozen"] == len(
        prompt["analysis_preregistration"]["wrong_relation_control"][
            "reassignments_from_v2"
        ]
    ) == 4
    assert prompt["status"] == "compiled_unscored_static_cross_audit_filtered"
    assert prompt["external_ground_truth_used"] is False
    assert prompt["candidate_scores_read_or_embedded"] is False
    assert prompt["prompt_outputs_used"] is False
    assert prompt["outcome_labels_used"] is False
    assert prevalence["schema"] == "metric-seam.hierarchy-witness-prevalence.v2"
    assert prevalence["sampling_frame"]["n_complete_action_node_records"] == 1132
    assert prevalence["sampling_frame"]["n_eligible_action_node_records"] == 1128
    assert prevalence["sampling_frame"]["n_excluded_by_frozen_eligibility_rule"] == 4
    assert prevalence["sampling_frame"]["eligible_by_level"] == {
        "R1": 856,
        "R2": 217,
        "R3": 55,
    }
    assert status["code_review_prompt_references_scored"] == 0
    assert status["code_review_isomorphism_adjudications"] == 0
    assert status["completed_deep_metric_seam_runs"] == 0
    assert not any(row["completed_deep_metric_seam_run"] for row in readiness["rows"])
    assert status["math_retrospective_candidates_static_audited"] == math_map["summary"][
        "decision_counts"
    ]["candidate_seed_pending_independent_construct_fidelity_audit"] == 47
    assert math_fidelity["cross_audit"]["status"] == "complete"
    assert status["math_relation_local_static_fidelity"] == math_fidelity["summary"][
        "eligible_for_relation_local_execution"
    ] == 33
    assert status["math_whole_construct_exact"] == math_fidelity["summary"][
        "whole_construct_exact_count"
    ] == 0
    assert status["math_symbolic_relation_local_static_matches"] == (
        math_symbolic_fidelity["summary"]["n_relation_local_static_matches"]
    ) == 7
    assert status["math_symbolic_newly_covered_cells"] == (
        math_symbolic_fidelity["summary"]["n_newly_covered_cells"]
    ) == 5
    assert status["math_symbolic_additive_sensitivity_union_cells"] == (
        math_symbolic_fidelity["summary"]["additive_sensitivity_union_cells"]
    ) == 38
    assert math_symbolic_fidelity["programs_or_items_executed"] is False
    assert math_symbolic_prevalence["canonical_artifact_modified"] is False
    assert math_symbolic_prevalence["pooled_eligible_action_nodes"][
        "eligible_inventory_stratum_expansion"
    ]["additive_sensitivity_union_relation_local"]["rate"] == 0.376231
    assert math_fidelity["execution_performed"] is False
    assert math_prevalence["status"] == "static_descriptive_rates_cross_audited"
    assert math_prevalence["pooled_eligible_action_nodes"][
        "eligible_inventory_stratum_expansion"
    ]["relation_local_static_fidelity"]["rate"] == 0.361266
    assert status["math_lclamp_unique_programs_executed"] == math_train["summary"][
        "n_unique_programs"
    ] == math_heldout["summary"]["n_unique_programs"] == 16
    assert status["math_lclamp_train_measurable_relation_mappings"] == math_gate[
        "summary"
    ]["n_selected_relation_mappings"] == 33
    assert status["math_lclamp_heldout_measurable_relation_mappings"] == math_heldout[
        "summary"
    ]["n_relation_mappings"] == 33
    assert math_train["summary"]["three_state_totals"] == {
        "measured": 36000,
        "abstained": 0,
        "failed": 0,
    }
    assert math_heldout["summary"]["three_state_totals"] == {
        "measured": 2400,
        "abstained": 0,
        "failed": 0,
    }
    for execution in (math_train, math_heldout):
        assert execution["original_hybrid_execution"] is False
        assert execution["pure_code_rewrite_claimed"] is False
        assert execution["whole_construct_fidelity_claimed"] is False
        assert execution["reference_fields_passed_to_worker"] is False
        assert execution["outcome_fields_passed_to_worker"] is False
        assert execution["models_or_apis_called_by_runner"] is False
        assert execution["accelerators_visible_to_worker"] is False
    assert math_operational["status"] == (
        "complete_static_train_and_pre_reference_heldout_funnel"
    )
    assert math_operational["pooled_eligible_action_nodes"][
        "eligible_inventory_stratum_expansion"
    ]["heldout_measurable_constant_l_slice"]["rate"] == 0.361266
    assert math_operational["unsupervised_sentinel_sensitivity"][
        "used_for_heldout_decisions"
    ] is False
    assert status["math_prompt_train_jobs_compiled_unscored"] == math_prompt_train[
        "summary"
    ]["n_jobs"] == 295200
    assert status[
        "math_prompt_heldout_fixed_jobs_compiled_unscored"
    ] == math_prompt_heldout["summary"]["n_jobs"] == 128700
    assert status["math_prompt_responses_scored"] == math_prompt_train["summary"][
        "n_prompt_responses"
    ] == math_prompt_heldout["summary"]["n_prompt_responses"] == 0
    assert status["math_reconstruction_estimates"] == math_prompt_train["summary"][
        "n_reconstruction_estimates"
    ] == math_prompt_heldout["summary"]["n_reconstruction_estimates"] == 0
    assert status["math_isomorphism_adjudications"] == math_prompt_train["summary"][
        "n_isomorphism_adjudications"
    ] == math_prompt_heldout["summary"]["n_isomorphism_adjudications"] == 0
    for manifest in (math_prompt_train, math_prompt_heldout):
        assert manifest["status"] == "compiled_unscored"
        assert set(manifest["forbidden_inputs"].values()) == {False}
        assert manifest["jobs_artifact"]["model_or_api_calls_performed"] is False
    assert status["patent_retrospective_candidates_static_audited"] == patent_fidelity[
        "summary"
    ]["n_retrieved"] == 6
    assert status["patent_relation_local_static_fidelity"] == patent_fidelity["summary"][
        "n_partial_relation_local"
    ] == 6
    assert status["patent_whole_construct_exact"] == patent_fidelity["summary"][
        "n_exact_whole_construct"
    ] == 0
    assert status["patent_pure_code_witnesses"] == patent_fidelity["summary"][
        "n_pure_code_witnesses"
    ] == 0
    assert status["patent_candidate_programs_executed"] == 0
    assert patent_prevalence["status"] == "static_descriptive_rates_complete"
    patent_graph_summary = patent_claim_graph_cross_audit["summary"]
    assert status["patent_claim_graph_additive_original_relation_local_cells"] == (
        patent_graph_summary["n_original_additive_cells"]
    ) == 8
    assert status["patent_claim_graph_additive_original_relation_local_mappings"] == (
        patent_graph_summary["n_original_additive_mappings"]
    ) == 11
    assert status["patent_claim_graph_additive_certificate_safe_cells"] == (
        patent_graph_summary["n_current_executable_cells_after_cross_audit"]
    ) == 5
    assert status["patent_claim_graph_additive_certificate_safe_mappings"] == (
        patent_graph_summary["n_current_executable_mappings_after_cross_audit"]
    ) == 5
    assert status["patent_claim_graph_additive_quarantined_mappings"] == (
        patent_graph_summary["n_quarantined_mappings"]
    ) == 6
    assert status["patent_descriptive_three_lane_trusted_union_cells"] == (
        patent_claim_graph_cross_audit["descriptive_union_check"][
            "trusted_three_lane_union"
        ]
    ) == 19
    assert patent_claim_graph_cross_audit["claim_limits"] == {
        "codability_claim_permitted": False,
        "isomorphism_measured": False,
        "negative_result_establishes_tacitness": False,
        "prompt_articulability_measured": False,
        "reference_reconstruction_measured": False,
        "whole_construct_cells": 0,
    }
    assert status["science_retrospective_candidates_static_audited"] == science_seed[
        "summary"
    ]["decision_counts"][
        "candidate_seed_pending_independent_construct_fidelity_audit"
    ] == 9
    assert status["science_relation_local_static_fidelity"] == science_fidelity[
        "summary"
    ]["n_partial_relation_local"] == 6
    assert status["science_whole_construct_exact"] == science_fidelity["summary"][
        "n_exact_whole_construct"
    ] == 0
    assert status["science_candidate_programs_executed"] == 0
    assert science_fidelity["execution_performed"] is False
    assert science_prevalence["status"] == "static_descriptive_rates_complete_pre_execution"
    assert science_prevalence["pooled_eligible_action_nodes"][
        "eligible_inventory_stratum_expansion"
    ]["relation_local_static_fidelity"]["rate"] == 0.055407
    assert science_blocker["status"] == (
        "canonical_execution_blocked_by_representation_mismatch"
    )
    assert science_blocker["execution"]["performed"] is False
    assert science_blocker["coverage_audit"]["pooled"] == {
        "n_exact_abstract_joins": 12,
        "n_exact_joins_with_nonempty_body": 6,
        "n_items": 300,
    }
    assert science_fullarticle_manifest["comparability"][
        "canonical_hierarchy_items"
    ] is False
    assert science_fullarticle_manifest["representation"][
        "same_ctext_bytes_required_for_future_prompt_and_code"
    ] is True
    assert status["science_additive_unique_programs_executed"] == 1
    assert status["science_additive_train_operational_relation_mappings"] == (
        science_fullarticle_train["summary"]["n_relation_mappings"]
    ) == 6
    assert status["science_additive_heldout_measurable_relation_mappings"] == (
        science_fullarticle_heldout["summary"]["n_relation_mappings"]
    ) == 6
    assert status["science_additive_train_measured_items"] == (
        science_fullarticle_train["summary"]["three_state_totals_unique_items"][
            "measured"
        ]
    ) == 118
    assert status["science_additive_heldout_measured_items"] == (
        science_fullarticle_heldout["summary"]["three_state_totals_unique_items"][
            "measured"
        ]
    ) == 108
    assert status["science_additive_heldout_relation_certificates"] == (
        science_fullarticle_heldout["summary"]["n_relation_certificates"]
    ) == 10
    assert science_addressed_binding["status"] == (
        "cpu_only_subset_binding_complete_pre_prompt"
    )
    science_prompt = science_addressed_binding["prompt_plane"]
    assert status["science_additive_addressed_prepared_unscored_requests"] == (
        science_prompt["distinct_prepared_unscored_request_records"]
    ) == 235
    assert status["science_additive_addressed_structural_abstentions"] == (
        science_prompt["structural_abstentions_without_remote_call"]
    ) == 65
    assert status["science_additive_addressed_planned_two_pass_jobs"] == (
        science_prompt["planned_two_pass_prompt_jobs_if_executed"]
    ) == 470
    assert status["science_additive_addressed_prompt_responses"] == (
        science_prompt["prompt_responses"]
    ) == 0
    assert status["science_additive_addressed_v9_hierarchy_agreement_items"] == (
        science_addressed_binding["combined_summary"][
            "v9_hierarchy_item_field_agreement"
        ]["agree"]
    ) == 300
    assert science_addressed_binding["representation_contract"][
        "same_input_representation"
    ] is False
    assert science_addressed_binding["temporal_disposition"][
        "fresh_split_required_for_confirmatory_prompt_code_claim"
    ] is True
    exact_summary = science_exact_prompt["summary"]
    assert status[
        "science_additive_exact_ctext_prompt_pass_records_compiled_unscored"
    ] == (
        exact_summary["compiled_prompt_pass_records"]
    ) == 470
    assert status["science_additive_exact_ctext_structural_no_call_outcomes"] == (
        exact_summary["pass_expanded_structural_no_call_outcomes"]
    ) == 130
    assert status["science_additive_exact_ctext_payload_record_replays"] == (
        science_exact_receipt["validation"]["decoded_exact_payload_records"]
    ) == 470
    assert status["science_additive_exact_ctext_prompt_responses"] == (
        exact_summary["prompt_responses"]
    ) == 0
    projection_summary = science_exact_projection["summary"]
    assert status["science_additive_numeric_comparative_projection_items"] == (
        projection_summary["items"]
    ) == 300
    assert status["science_additive_numeric_comparative_selected_claims"] == (
        projection_summary["selected_claims"]
    ) == 158
    assert status["science_additive_numeric_comparative_supported_claims"] == (
        projection_summary["decision_counts"]["supported"]
    ) == 17
    assert status[
        "science_additive_numeric_comparative_insufficient_claims"
    ] == projection_summary["decision_counts"]["insufficient"] == 141
    assert status[
        "science_additive_numeric_comparative_evidence_link_decisions"
    ] == projection_summary["evidence_link_decisions"] == 0
    assert science_exact_prompt["representation_contract"][
        "same_frozen_ctext_payload_bytes_as_current_code"
    ] is True
    assert science_exact_prompt["representation_contract"][
        "raw_jsonl_or_provider_wire_byte_identity_claimed"
    ] is False
    assert science_exact_prompt["transport_control_inventory"][
        "compiled_prompt_pass_records"
    ] == 72
    assert science_exact_prompt["future_comparison_target"]["whole_frozen_code_vector"] is False
    assert science_exact_prompt["future_comparison_target"][
        "code_projection_compiled_and_replay_bound"
    ] is True
    assert science_exact_prompt["future_comparison_target"][
        "evidence_link_in_reconstruction_target"
    ] is False
    assert science_exact_prompt["chronology"][
        "fresh_split_required_for_confirmatory_reconstruction_or_isomorphism"
    ] is True
    assert science_fullarticle_operational["channel_contract"] == {
        "accelerators_used": False,
        "external_supervision_used": False,
        "item_text_loaded_by_summary": False,
        "models_or_apis_called": False,
        "outcome_values_loaded": False,
        "program_execution_outputs_read": True,
        "prompt_or_reconstruction_outputs_loaded": False,
        "reference_values_loaded": False,
    }


def test_every_current_artifact_path_exists_and_invalidated_train_v1_is_not_current():
    current = _load(BASE / "CURRENT.json")

    def paths(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from paths(child)
        elif isinstance(value, list):
            for child in value:
                yield from paths(child)
        elif isinstance(value, str) and value.startswith("outputs/"):
            yield value

    recorded = list(paths(current))
    assert recorded
    assert all((ROOT / path).exists() for path in recorded)
    assert not any(path.endswith("code_review_train_execution_v1.json") for path in recorded)


def test_metric_seam_notebook_has_a_saved_successful_hierarchy_cell():
    notebook = _load(
        ROOT / "notebooks/2026-07-02__metric-seam-certificates-and-overnight-report.ipynb"
    )
    cells = [
        cell for cell in notebook["cells"]
        if cell.get("id") == "seam-20260713-hierarchy-funnel"
    ]
    assert len(cells) == 1
    cell = cells[0]
    assert isinstance(cell.get("execution_count"), int)
    assert len(cell.get("outputs", [])) >= 2
    assert not any(output.get("output_type") == "error" for output in cell["outputs"])


def test_metric_seam_notebook_has_saved_successful_moved_survey_cell():
    notebook = _load(
        ROOT / "notebooks/2026-07-02__metric-seam-certificates-and-overnight-report.ipynb"
    )
    cells = [cell for cell in notebook["cells"] if cell.get("id") == "dde45571"]
    assert len(cells) == 1
    cell = cells[0]
    source = "".join(cell["source"])
    assert "survey_task_tables(OUTD)" in source
    assert 'tasks/{task}/seam_table.json' not in source
    assert isinstance(cell.get("execution_count"), int)
    assert len(cell.get("outputs", [])) >= 2
    assert not any(output.get("output_type") == "error" for output in cell["outputs"])
