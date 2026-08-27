from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.metric_seam.pilot import metric_seam_notebook_stats as stats


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_task_survey_loader_uses_archive_and_preserves_missingness(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "tasks/math/seam_table.json",
        {"table": [{"aspect": "m1", "rho": 0.2}]},
    )
    archived = tmp_path / "tasks/code_review/archive_pre_e2ladder/seam_table.json"
    _write_json(archived, {"table": [{"aspect": "c1", "rho": 0.3}]})

    result = stats.survey_task_tables(tmp_path)

    assert result["expected_task_count"] == 6
    assert result["available_task_count"] == 2
    assert result["unavailable_task_count"] == 4
    assert result["unavailable_tasks"] == [
        "patents",
        "code_review_diffs",
        "code_competition",
        "pr_exec",
    ]
    assert {(row["task"], row["aspect"]) for row in result["rows"]} == {
        ("math", "m1"),
        ("code_review", "c1"),
    }
    code_review = next(
        source for source in result["sources"] if source["task"] == "code_review"
    )
    assert code_review["status"] == "available"
    assert code_review["source_path"] == str(archived)
    assert code_review["row_count"] == 1
    missing = [
        source for source in result["sources"] if source["status"] != "available"
    ]
    assert all(source["row_count"] is None for source in missing)
    assert all(source["source_path"] is None for source in missing)


def test_repository_survey_loader_resolves_moved_code_review_table() -> None:
    """Regression for the notebook's former direct-open FileNotFoundError."""

    result = stats.survey_task_tables(stats.BASE)

    assert result["unavailable_tasks"] == []
    code_review = next(
        source for source in result["sources"] if source["task"] == "code_review"
    )
    assert code_review["status"] == "available"
    assert code_review["row_count"] == 40
    assert code_review["source_path"].endswith(
        "tasks/code_review/archive_pre_e2ladder/seam_table.json"
    )


def test_optional_seam_table_distinguishes_missing_empty_and_malformed(
    tmp_path: Path,
) -> None:
    missing = stats.optional_seam_table(tmp_path / "missing.json")
    assert missing["status"] == "unavailable_missing_artifact"
    assert missing["row_count"] is None
    assert missing["rows"] is None

    empty_path = tmp_path / "empty.json"
    _write_json(empty_path, {"table": []})
    empty = stats.optional_seam_table(empty_path)
    assert empty["status"] == "available"
    assert empty["row_count"] == 0
    assert empty["rows"] == []

    malformed_path = tmp_path / "malformed.json"
    _write_json(malformed_path, {"table": {"not": "a list"}})
    with pytest.raises(ValueError, match="malformed seam table"):
        stats.optional_seam_table(malformed_path)


def test_panel_codability_denominators_and_thresholds() -> None:
    rows = stats.panel_rows()
    assert len(rows) == 159

    overall = stats.codability_by_domain()[-1]
    assert overall["task"] == "ALL"
    assert overall["code_ge_30_n"] == 56
    assert overall["code_ge_50_n"] == 23
    assert overall["code_ge_80_n"] == 1
    assert overall["hybrid_ge_30_n"] == 118
    assert overall["hybrid_ge_50_n"] == 77
    assert overall["hybrid_ge_80_n"] == 13


def test_contract_channel_hypotheses_are_not_silently_imputed() -> None:
    summary = stats.channel_contract_summary()
    assert summary["panel_criteria"] == 159
    assert summary["fully_tagged_criteria"] == 142
    assert summary["legacy_untagged_criteria"] == 17
    assert summary["tagged_probes"] == 672
    assert summary["code_probes"] == 299
    assert summary["l_probes"] == 373
    assert summary["criterion_classes"] == {
        "mixed_CODE_L": 124,
        "all_L": 12,
        "all_CODE": 6,
    }
    association = summary["code_tag_share_vs_code_reconstruction"]
    assert association["criterion_level_n"] == 142
    assert association["criterion_level_spearman"] == 0.247
    assert association["domain_level_n"] == 7
    assert association["domain_level_spearman"] == 0.857
    assert summary["per_band"] == [
        {
            "band": "floor",
            "code_probes": 23,
            "typed_probes": 74,
            "untyped_probes": 11,
            "code_probe_pct": 31.1,
        },
        {
            "band": "mid",
            "code_probes": 204,
            "typed_probes": 473,
            "untyped_probes": 57,
            "code_probe_pct": 43.1,
        },
        {
            "band": "control",
            "code_probes": 72,
            "typed_probes": 125,
            "untyped_probes": 29,
            "code_probe_pct": 57.6,
        },
    ]


def test_census_progress_and_cw_multiplicity_correction() -> None:
    overall = stats.census_progress()[-1]
    assert overall["attempted"] == 43
    assert overall["panel_n"] == 159
    assert overall["train_contract_queue_n"] == 32

    outcomes = stats.census_outcome_summary()
    assert outcomes["attempted_cells"] == 43
    assert outcomes["final_contract_passes"] == 33
    assert outcomes["final_separations"] == 171
    assert outcomes["final_separation_opportunities"] == 212
    assert outcomes["nonqueued_pass_ids"] == ["creative_writing__a333"]

    heldout = stats.creative_writing_heldout_adjudication()
    assert heldout["candidate_count"] == 22
    assert heldout["unambiguous_count"] == 20
    assert heldout["exploratory_pairwise_ids"] == ["a135", "a144", "a207", "a90"]
    assert heldout["g1_and_pairwise_ids"] == ["a144", "a90"]
    assert heldout["bh_test_count"] == 20
    assert heldout["bh_survivor_count"] == 0


def test_ws4_depth_is_computed_from_executable_dags() -> None:
    summary = stats.ws4_depth_summary()
    assert summary["programs"] == 9
    assert summary["nodes"] == 145
    assert summary["median_nodes"] == 14
    assert summary["node_range"] == [12, 28]
    assert summary["l_nodes"] == 17
    assert summary["l_nodes_at_graph_root"] == 17
    assert summary["l_level_counts"] == {1: 15, 2: 2}
    assert summary["median_longest_path_edges"] == 5
    assert summary["longest_path_range"] == [2, 10]
    assert summary["retrieval_nodes"] == 4
    assert summary["evidence_nodes"] == 24
    assert summary["mean_l_frontier_to_output_longest_edges"] == 2.06
    assert summary["median_l_frontier_to_output_longest_edges"] == 2
    assert summary["l_frontier_to_output_range"] == [1, 3]


def test_active_code_depth_uses_the_full_retrospective_family() -> None:
    summary = stats.active_code_depth_retrospective()
    assert summary["active_criteria"] == 18
    assert summary["criteria_with_deep_program"] == 18
    assert summary["criteria_with_train_selected_shallow_comparator"] == 15
    assert summary["inferentially_eligible"] == 4
    assert summary["bh_family_size"] == 4
    assert summary["multiplicity_controlled_improvements"] == 0
    assert len(summary["rows"]) == 18

    a104 = next(row for row in summary["rows"] if row["criterion_id"] == "a104")
    assert a104["n_paired"] == 97
    assert round(a104["deep_rho"], 3) == 0.650
    assert round(a104["shallow_rho"], 3) == 0.509
    assert round(a104["delta_spearman"], 3) == 0.141
    assert round(a104["p_value"], 3) == 0.153
    assert round(a104["bh_q_value"], 3) == 0.460


def test_a104_supplements_keep_representation_and_augmentation_separate() -> None:
    result = stats.active_code_a104_supplemental()
    sensitivity = result["representation_sensitivity"]
    assert sensitivity["common_heldout_n"] == 93
    assert round(sensitivity["historical_head_tail_rho"], 3) == 0.645
    assert round(sensitivity["prefix4000_rho"], 3) == 0.514
    assert round(sensitivity["delta_prefix_minus_head_tail"], 3) == -0.131
    assert round(sensitivity["code_vector_rho"], 3) == 0.778
    assert sensitivity["applicability_status_changes_all_250"] == 12
    assert sensitivity["value_changes_on_common_scored_all_250"] == 118
    assert sensitivity["hierarchy_prefix_rows_at_cap"] == 205
    assert sensitivity["one_sided_not_same_input_prompt_code"] is True

    augmentation = result["execution_augmentation"]
    assert augmentation["exact_repository_pr_overlap"] == 32
    assert augmentation["finite_execution_certificates"] == 1
    assert augmentation["finite_certificate_rate_conditional_overlap"] == 0.03125
    assert augmentation["finite_certificate_rate_over_active_items"] == 0.004
    assert augmentation["relation_depth"] == 4
    assert augmentation["stored_prior_execution"] is True
    assert augmentation["same_input_representation"] is False
    assert augmentation["capability_augmentation_not_isomorphic_substitution"] is True


def test_code_representation_sensitivity_is_family_wide_and_program_macro() -> None:
    result = stats.code_review_representation_family_sensitivity()
    assert result["primary_unique_programs"] == 10
    assert result["primary_relation_mappings"] == 18
    assert result["secondary_unique_programs"] == 16
    assert result["P0_exact_replay_rows"] == 4000
    assert result["P0_exact_replay_mismatches"] == 0
    assert result["crosswalk_rows"] == 250
    rows = {row["comparison"]: row for row in result["primary_program_macro"]}
    prefix_head_tail = rows["P0_prefix4000 -> P1_head5000_tail2500"]
    head_tail_raw = rows["P1_head5000_tail2500 -> P2_raw_diff_capped300k"]
    prefix_raw = rows["P0_prefix4000 -> P2_raw_diff_capped300k"]
    assert prefix_head_tail["exact_row_agreement"] == 0.7068
    assert head_tail_raw["exact_row_agreement"] == 0.5556
    assert prefix_raw["exact_row_agreement"] == 0.4588
    assert prefix_head_tail["applicability_change_rate"] == 0.0708
    assert head_tail_raw["applicability_change_rate"] == 0.078
    assert prefix_raw["applicability_change_rate"] == 0.1488
    assert all(row["programs"] == 10 for row in rows.values())
    assert all(
        row["status_or_applicability_sensitive_programs"] == 10
        for row in rows.values()
    )
    assert result["primary_program_depth_counts"] == {1: 7, 2: 3}
    assert result["typed_mapping_level_counts"] == {"R3": 7, "R1": 7, "R2": 4}
    assert result["typed_mapping_depth_counts"] == {1: 12, 2: 6}
    depth = result["primary_program_depth_descriptive"]
    pair = "P0_prefix4000 -> P1_head5000_tail2500"
    assert round(depth[1]["comparisons"][pair]["mean_exact_row_agreement"], 4) == 0.7606
    assert round(depth[2]["comparisons"][pair]["mean_exact_row_agreement"], 4) == 0.5813
    assert result["axes"]["isomorphism"] == (
        "not_measured_code_code_projection_audit_only"
    )


def test_active_code_source_depth_is_a_separate_structural_descriptor() -> None:
    result = stats.active_code_source_structure()
    assert result["scope"]["deep_programs"] == 18
    assert result["scope"]["train_selected_shallow_programs"] == 15
    assert len(result["pairs"]) == 15
    assert len(result["programs"]) == 33

    summary = result["paired_summary"]
    assert summary["ast_nodes"]["deep_median"] == 1738.0
    assert summary["ast_nodes"]["shallow_median"] == 351.0
    assert summary["ast_nodes"]["deep_greater_count"] == 15
    assert summary["max_control_nesting"]["deep_median"] == 7.0
    assert summary["max_control_nesting"]["shallow_median"] == 2.0


def test_math_a12_coverage_generalizes_without_a_parent_scalar() -> None:
    result = stats.math_a12_relation_generalization()
    assert result["train"]["covered_rows"] == 42
    assert result["train"]["rows"] == 150
    assert result["train"]["coverage"] == 0.28
    assert result["heldout"]["covered_rows"] == 26
    assert result["heldout"]["rows"] == 100
    assert result["heldout"]["coverage"] == 0.26
    assert round(result["coverage_fisher_exact_two_sided_p"], 6) == 0.772968
    assert result["heldout"]["identity_classifications"] == 11
    assert result["heldout"]["nonidentity_classifications"] == 54
    assert result["prompt_reference"]["available_both_passes"] == 99
    assert round(result["prompt_reference"]["two_pass_spearman"], 3) == 0.835
    assert result["whole_criterion_reconstruction"] == "NOT_ESTIMATED"
    assert result["isomorphism"] == "NOT_ESTIMATED"


def test_math_a12_pair_projection_separates_attempts_from_evidence() -> None:
    result = stats.math_a12_pair_projection_depth()
    assert result["heldout_count"] == 100
    assert result["pair_certificate_count"] == 277
    assert result["pair_status_counts"] == {
        "exact_nonidentity_witness": 54,
        "parse_noncoverage": 212,
        "verified_rational_identity": 11,
    }
    assert result["row_category_counts"] == {
        "formal_parse_noncoverage_abstention": 39,
        "formal_positive_relation_evidence": 26,
        "parser_structure_only_no_pair_candidate": 35,
    }
    assert result["depth_views"]["deepest_attempted"]["histogram"] == {
        "1": 35,
        "3": 65,
    }
    assert result["depth_views"]["positive_relation_evidence"] == {
        "evidence_rows": 26,
        "histogram": {"3": 26},
        "no_positive_evidence_rows": 74,
        "semantics": "depth only for rows with at least one positive code witness",
    }
    assert result["formal_path_positive_evidence_rate"] == 0.4
    assert result["new_blind_result"] is False
    assert result["new_reconstruction_result"] is False
    assert result["new_isomorphism_result"] is False


def test_science_strict_witnesses_are_code_representation_robust() -> None:
    result = stats.science_relation_witness_summary()
    assert result["relation_witnesses"]["numeric"]["numerator"] == 68
    assert result["relation_witnesses"]["numeric"]["denominator"] == 561
    assert result["relation_witnesses"]["comparative"]["numerator"] == 32
    assert result["relation_witnesses"]["comparative"]["denominator"] == 634
    assert result["all_matched_relations"]["numerator"] == 100
    assert result["all_matched_relations"]["denominator"] == 4871
    assert result["supported_documents"]["numerator"] == 95
    assert result["supported_documents"]["denominator"] == 2400
    replay = result["representation_replay"]
    assert replay["strong_exact_text_intersection"] == 8
    assert replay["strong_witness_intersection"] == 100
    assert replay["strong_witness_continuous"] == 100
    assert replay["strong_witness_addressed"] == 100
    assert replay["supported_document_intersection"] == 95
    assert replay["supported_document_continuous"] == 95
    assert replay["supported_document_addressed"] == 95
    assert replay["paper_status_agreement"] == 2396
    assert replay["paper_status_total"] == 2400
    assert replay["weak_witness_intersection"] == 429
    assert replay["weak_witness_continuous"] == 434
    assert replay["weak_witness_addressed"] == 430
    assert result["prompt_articulability_status"] == "compiled_unscored_not_measured"
    prompt = result["prompt_batch"]
    assert prompt["corpus_records"] == 2400
    assert prompt["compiled_unscored_jobs"] == 1957
    assert prompt["structural_abstentions_without_remote_call"] == 443
    assert prompt["prompt_responses"] == 0
    assert prompt["same_evidence_content"] is True
    assert (
        prompt["same_input_representation_as_historical_continuous_code"] is False
    )
    assert prompt["semantic_prompt_code_comparison_measured"] is False
    assert prompt["fresh_split_required_for_confirmatory_prompt_code_claim"] is True


def test_patent_ws3_retrospective_keeps_family_and_precision_caveats() -> None:
    result = stats.patent_ws3_family_retrospective()
    assert result["summary"]["registered_criteria"] == 4
    assert result["summary"]["bh_family_size"] == 4
    assert result["summary"]["bh_fdr_rejections"] == 2
    assert result["summary"]["threshold_and_fdr_screen_ids"] == ["a34", "a35"]
    assert result["summary"]["effect_precision_characterized_ids"] == ["a35"]
    by_id = {row["criterion_id"]: row for row in result["criteria"]}
    assert round(by_id["a26"]["bh_q_value"], 4) == 0.0568
    assert by_id["a34"]["paired_bootstrap"]["interval"] is None
    assert by_id["a34"]["null_score_modal_fraction"] == 0.99
    assert round(by_id["a35"]["paired_bootstrap"]["interval"][0], 3) == 0.400


def test_technical_ledger_refuses_incompatible_pooling() -> None:
    result = stats.technical_evidence_ledger_summary()
    summary = result["summary"]
    assert summary["record_count"] == 39
    assert summary["by_stratum"] == {
        "criterion_scalar_reconstruction": 24,
        "program_structure_descriptor": 7,
        "relation_instance_verification": 8,
    }
    assert summary["explicitly_nonpoolable"] is True
    assert summary["cross_stratum_pooled_estimates_emitted"] == 0
    assert summary["domain_codability_estimates_emitted"] == 0
    patents = result["family_summaries"]["patent_historical_selected_family"]
    assert patents["bh_fdr_rejections"] == {"numerator": 2, "denominator": 4}
    assert patents["effect_precision_characterized"] == {
        "numerator": 1,
        "denominator": 4,
    }


def test_code_review_hierarchy_funnel_stops_before_reconstruction() -> None:
    result = stats.code_review_hierarchy_reconstruction_funnel()
    assert [(row["n"], row["denominator"]) for row in result["stages"]] == [
        (50, 90),
        (27, 90),
        (18, 90),
        (18, 90),
    ]
    assert result["whole_construct_exact"] == {
        "n": 0,
        "denominator": 90,
        "pct": 0.0,
    }
    assert [
        (
            row["level"],
            row["static_relation_local_n"],
            row["train_operational_n"],
            row["heldout_code_score_ready_n"],
        )
        for row in result["by_level"]
    ] == [
        ("R1", 14, 9, 7),
        ("R2", 15, 6, 4),
        ("R3", 21, 12, 7),
    ]
    assert [
        (
            row["depth"],
            row["static_relation_local_n"],
            row["train_operational_n"],
            row["heldout_code_score_ready_n"],
        )
        for row in result["by_depth"]
    ] == [(1, 25, 19, 12), (2, 25, 8, 6)]
    assert result["heldout_min_code_scores"] == 30
    assert result["prompt_manifest"] == {
        "status": "compiled_unscored_static_cross_audit_filtered",
        "cells": 18,
        "items_per_cell": 125,
        "channels": [
            "source_only_whole_construct",
            "source_only_subrelation",
            "implementation_disclosed",
        ],
        "passes": [1, 2],
        "jobs": 13500,
        "unique_program_vectors": 10,
        "scope_statements": result["prompt_manifest"]["scope_statements"],
        "external_ground_truth_used": False,
        "candidate_scores_read_or_embedded": False,
        "control_reassignments_after_scope_filter": 4,
        "old_batch_disposition": result["prompt_manifest"][
            "old_batch_disposition"
        ],
    }
    assert result["prompt_manifest"]["scope_statements"][
        "selected_construct_fidelity_verdict_counts"
    ] == {"partial": 18}
    assert result["axes"] == {
        "prompt_articulability": "not_measured_jobs_compiled_unscored",
        "code_verifiability": (
            "relation_local_candidate_measurements_available; "
            "whole_construct_verifiability_not_established"
        ),
        "reconstruction_agreement": "not_estimated",
        "isomorphism": "not_estimated",
        "codability": "not_estimated",
    }
    assert result["parser_incident"]["status"] == "closed_by_additive_rerun"
    assert result["parser_incident"]["invalidated_artifact"].endswith(
        "code_review_train_execution_v1.json"
    )
    assert result["parser_incident"]["canonical_replacement"].endswith(
        "code_review_train_execution_v2.json"
    )
    prevalence = result["prevalence"]
    assert prevalence["estimated_population_nodes"] == 1128
    assert prevalence["sampling_frame"] == {
        "n_complete_action_node_records": 1132,
        "n_eligible_action_node_records": 1128,
        "n_excluded_by_frozen_eligibility_rule": 4,
        "complete_by_level": {"R1": 860, "R2": 217, "R3": 55},
        "eligible_by_level": {"R1": 856, "R2": 217, "R3": 55},
        "n_sampling_strata": 18,
        "selected_per_stratum": [5],
        "eligibility_rule": "nonempty name, at least 8 description words, and at least 1 child",
    }
    assert prevalence["pooled"]["relation_local_static_fidelity"]["rate"] == 0.418085
    assert prevalence["pooled"]["train_operational_relation_witness"]["rate"] == 0.231028
    assert (
        prevalence["pooled"]["heldout_confirmatory_reconstruction_evaluable"]["rate"]
        == 0.160461
    )
    assert prevalence["by_level"] == {
        "R1": {
            "relation_local_static_fidelity": 0.373832,
            "train_operational_relation_witness": 0.229439,
            "heldout_confirmatory_reconstruction_evaluable": 0.164019,
        },
        "R2": {
            "relation_local_static_fidelity": 0.516129,
            "train_operational_relation_witness": 0.189862,
            "heldout_confirmatory_reconstruction_evaluable": 0.124424,
        },
        "R3": {
            "relation_local_static_fidelity": 0.72,
            "train_operational_relation_witness": 0.418182,
            "heldout_confirmatory_reconstruction_evaluable": 0.247273,
        },
    }
    assert prevalence["terminal_frontier"]["status"] == "not_yet_measured"
    diagnostics = prevalence["dependence_diagnostics"]
    assert diagnostics["cross_level_raw_support"]["n_components"] == 35
    assert diagnostics["cross_level_raw_support"]["largest_component"] == 33
    assert diagnostics["shared_candidate_program"]["n_components"] == 55
    assert diagnostics["joint_dependency_raw_program_union"]["n_components"] == 25
    assert diagnostics["joint_dependency_raw_program_union"]["largest_component"] == 49
    assert diagnostics["not_an_interval"] is True
    assert prevalence["corrected_outcome_perturbation_ranges_recomputed"] is False
    assert prevalence["supersedes"] == {
        "historical_funnel": "56 static -> 30 train -> 21 heldout",
        "corrected_funnel": "50 static -> 27 train -> 18 heldout",
        "historical_prompt_manifest": (
            "code_review_reconstruction_prompt_manifest_v2.json"
        ),
        "corrected_prompt_manifest": (
            "code_review_reconstruction_prompt_manifest_v3.json"
        ),
    }
    assert len(prevalence["outstanding_sensitivities"]) == 4
    assert result["construct_fidelity_cross_audit"] == {
        "retrieved_rows_reviewed": 68,
        "program_sources_reviewed": 33,
        "guarded_changes": 7,
        "complete": True,
    }
    assert result["corrected_gate_propagation"] == {
        "removed_static": 6,
        "removed_train_operational": 3,
        "removed_heldout_ready": 3,
        "depth_corrections": 1,
        "programs_reexecuted": False,
    }


def test_code_review_corrected_funnel_propagates_construct_audit_only() -> None:
    result = stats.code_review_hierarchy_corrected_funnel()

    assert result["panel_cells"] == 90
    assert result["stage_counts"] == {
        "retrieved_candidate": 68,
        "relation_local_static_fidelity": 50,
        "train_operational_relation_witness": 27,
        "heldout_confirmatory_reconstruction_evaluable": 18,
    }
    corrected = result["corrected_readout"]
    assert corrected["stages"]["relation_local_static_fidelity"][
        "conditional_eligible_inventory_expansion"
    ]["rate"] == 0.418085
    assert corrected["stages"]["train_operational_relation_witness"][
        "conditional_eligible_inventory_expansion"
    ]["rate"] == 0.231028
    assert corrected["stages"]["heldout_confirmatory_reconstruction_evaluable"][
        "conditional_eligible_inventory_expansion"
    ]["rate"] == 0.160461
    assert {
        stage: {
            depth: values["n_positive"]
            for depth, values in corrected["by_depth"][stage].items()
        }
        for stage in corrected["by_depth"]
    } == {
        "relation_local_static_fidelity": {"1": 25, "2": 25},
        "train_operational_relation_witness": {"1": 19, "2": 8},
        "heldout_confirmatory_reconstruction_evaluable": {"1": 12, "2": 6},
    }
    assert len(result["removed_mappings"]["static"]) == 6
    assert len(result["removed_mappings"]["train_operational"]) == 3
    assert len(result["removed_mappings"]["heldout_confirmatory"]) == 3
    assert result["depth_corrections"][0]["before_depth"] == 2
    assert result["depth_corrections"][0]["after_matched_relation_depth"] == 1
    assert result["cross_audit"] == {
        "retrieved_rows_reviewed": 68,
        "program_sources_reviewed": 33,
        "guarded_changes": 7,
        "complete": True,
    }
    assert result["axes"]["prompt_articulability"] == "not_measured"
    assert result["axes"]["reconstruction_agreement"] == "not_estimated"


def test_code_review_additive_unused_program_funnel_separates_execution_from_readiness() -> None:
    result = stats.code_review_additive_unused_program_funnel()

    assert result["canonical_corrected_static_unchanged"] == 50
    assert result["additive_static_union"] == 59
    assert result["additive_static_by_level"] == {"R1": 17, "R2": 19, "R3": 23}
    assert result["new_static_mappings"] == 9
    assert result["new_static_by_depth"] == {"2": 4, "4": 5}
    assert result["train_selected_mappings"] == 35
    assert result["train_selected_by_level"] == {"R1": 11, "R2": 10, "R3": 14}
    assert result["train_selected_by_depth"] == {"1": 18, "2": 12, "4": 5}
    assert result["heldout_nondegenerate_mappings"] == 35
    assert result["heldout_confirmatory_mappings"] == 19
    assert result["heldout_confirmatory_by_level"] == {
        "R1": 7,
        "R2": 5,
        "R3": 7,
    }
    assert result["heldout_confirmatory_by_depth"] == {"1": 12, "2": 7}
    assert result["heldout_readiness_counts"] == {
        "confirmatory_reconstruction_evaluable": 19,
        "exploratory_sparse": 12,
        "insufficient_paired_support": 4,
    }
    assert result["new_program_train_gate"]["a35"]["decision"] == (
        "insufficient_train_coverage"
    )
    assert result["new_program_heldout_readiness"]["a309"]["readiness"] == (
        "confirmatory_reconstruction_evaluable"
    )
    assert result["new_program_heldout_readiness"]["a72"]["readiness"] == (
        "exploratory_sparse"
    )
    assert result["axes"] == {
        "prompt_articulability": "not_measured",
        "code_verifiability": "relation_local_static_train_and_heldout_measured",
        "reconstruction": "not_measured",
        "isomorphism": "not_measured",
        "whole_construct_verifiability": 0,
        "external_supervised_anchor_used": False,
    }


def test_patent_hierarchy_static_funnel_preserves_evidence_provenance() -> None:
    result = stats.patent_hierarchy_static_funnel()
    assert result["historical_program_families"] == 4
    assert result["panel_cells"] == 90
    assert result["retrieved_candidates"] == 6
    assert result["relation_local_static_fidelity"] == 6
    assert result["whole_construct_exact"] == 0
    assert result["pure_code_witnesses"] == 0
    assert result["balanced_panel"]["relation_local_static_fidelity"]["rate"] == 0.066667
    conditional = result["conditional_eligible_inventory"]
    assert conditional["population_nodes"] == 1368
    assert conditional["relation_local_static_fidelity"]["rate"] == 0.051754
    assert conditional["depth3_evidence_relation"]["rate"] == 0.050877
    assert conditional["pure_code_witness"]["rate"] == 0.0
    assert result["maximum_matching_relation_depth_counts"] == {"1": 1, "3": 5}
    assert result["channel_provenance"]["autonomous_retrieval"] is False
    assert result["channel_provenance"]["pure_code"] is False
    assert result["axes"]["prompt_articulability"] == "not_measured"
    assert result["axes"]["reconstruction_agreement"] == "not_estimated"


def test_patent_claim_structure_static_funnel_separates_fidelity_and_variation() -> None:
    result = stats.patent_claim_structure_hierarchy_static_funnel()

    assert result["instrument"] == "pure_code_claim_structure_v13"
    assert result["panel_cells"] == 90
    assert result["relation_local_static_fidelity"] == 8
    assert result["train_operational_candidates_pre_frozen_gate"] == 5
    assert result["static_only_formatter_constant_cells"] == 3
    assert result["sensitivity_near_misses_not_credited"] == 4
    assert result["whole_construct_exact"] == 0
    assert result["maximum_matching_relation_depth_counts"] == {"1": 7, "2": 1}
    assert result["train"]["items"] == 150
    assert result["train"]["items_at_declared_character_cap"] == 119
    assert result["historical_comparison"] == {
        "manual_oracle_conditioned_cells": 6,
        "overlap": 0,
        "descriptive_union_cells": 14,
        "descriptive_union_depth_counts": {"1": 8, "2": 1, "3": 5},
        "provenance_warning": (
            "the union is descriptive only: the historical six are manual "
            "oracle-conditioned hybrids, whereas the current eight are pure-code "
            "static partial relation matches"
        ),
    }
    assert result["axes"]["prompt_articulability"] == "not_measured"
    assert result["axes"]["reconstruction_agreement"] == "not_estimated"
    assert result["axes"]["codability"] == "not_estimated"


def test_patent_claim_structure_operational_funnel_stops_before_reconstruction() -> None:
    result = stats.patent_claim_structure_hierarchy_operational_funnel()

    assert result["static_relation_local_cells"] == 8
    assert result["train_gate_selected_cells"] == 5
    assert result["heldout_relation_measurable_cells"] == 5
    assert result["static_only_formatter_constant_cells"] == 3
    assert result["whole_construct_exact"] == 0
    assert result["static_to_operational_fraction"] == 0.625
    assert result["selected_mean_maximum_depth"] == 1.2
    assert result["selected_cells_by_level"] == {"R2": 1, "R3": 4}
    assert result["selected_cells_by_maximum_depth"] == {"1": 4, "2": 1}
    heldout = result["heldout_pre_reference"]
    assert heldout["items"] == 150
    assert heldout["items_at_declared_character_cap"] == 123
    assert heldout["status_counts"] == {
        "measured": 27,
        "measured_with_possible_truncation": 122,
        "relation_abstained": 1,
    }
    assert heldout["finite_certificate_counts"] == {
        "claim_dependency_well_formedness": 1336,
        "statutory_category_surface_coverage": 168,
        "functional_limitation_incidence": 206,
    }
    assert result["prompt_batches_compiled_unscored"] == {
        "compiler_train": {
            "jobs": 7500,
            "prompt_specs": 25,
            "source_prompt_specs": 20,
            "post_code_structured_specs": 5,
        },
        "heldout_pre_reference": {
            "jobs": 19500,
            "prompt_specs": 65,
            "source_prompt_specs": 60,
            "post_code_structured_specs": 5,
        },
        "prompt_responses": 0,
        "reconstruction_estimates": 0,
        "isomorphism_adjudications": 0,
        "v1_packs": "superseded_not_executable",
        "v2_packs": "superseded_unexecuted",
        "v3_heldout_temporal_status": (
            "fixed_after_train_gate_exploratory_pre_reference"
        ),
        "semantic_validator": "validate_post_code_response.v3_frozen",
        "fresh_split_required_for_confirmatory_temporal_claim": True,
    }
    assert result["axes"] == {
        "prompt_articulability": "jobs_compiled_unscored_not_measured",
        "code_verifiability": "heldout_relation_local_outputs_executed",
        "reconstruction_agreement": "not_estimated",
        "isomorphism": "shared_ctext_confirmed_other_axes_not_estimated",
        "codability": "not_estimated",
    }


def test_patent_claim_graph_additive_cross_audit_quarantines_unsafe_programs() -> None:
    result = stats.patent_claim_graph_additive_cross_audited_funnel()

    assert result["instrument"] == "pure_code_claim_graph_additive_v1_cross_audited"
    assert result["original_relation_local_cells"] == 8
    assert result["original_relation_local_mappings"] == 11
    assert result["conceptual_relation_local_mappings_retained"] == 11
    assert result["current_certificate_safe_cells"] == 5
    assert result["current_certificate_safe_mappings"] == 5
    assert result["quarantined_mappings"] == 6
    assert result["quarantined_relations"] == [
        "formula_variable_definition_alignment",
        "numeric_constraint_definition_graph",
    ]
    assert result["certificate_safe_by_level"] == {"R2": 3, "R3": 2}
    assert result["certificate_safe_by_depth"] == {"2": 3, "3": 2}
    assert result["balanced_panel"]["current_certificate_safe"]["rate"] == 5 / 90
    assert result["heldout_markush_certificates"] == {
        "original": 25,
        "retained_after_truncation_filter": 23,
        "retained_items": 8,
    }
    assert result["descriptive_three_lane_union"]["original_union"] == 22
    assert result["descriptive_three_lane_union"]["trusted_current_union"] == 19
    assert result["axes"]["prompt_articulability"] == "not_measured"
    assert result["axes"]["reconstruction_agreement"] == "not_estimated"
    assert result["axes"]["isomorphism"] == "not_estimated"


def test_math_hierarchy_static_funnel_is_cross_audited_and_pre_execution() -> None:
    result = stats.math_hierarchy_static_funnel()

    assert result["historical_program_families"] == 16
    assert result["panel_cells"] == 90
    assert result["retrieved_candidates"] == 47
    assert result["relation_local_static_witnesses"] == 33
    assert result["whole_construct_exact"] == 0
    assert result["balanced_panel"]["relation_local_static_fidelity"]["rate"] == 0.366667
    conditional = result["eligible_inventory_stratum_expansion"]
    assert conditional["population_nodes"] == 1185
    assert conditional["relation_local_static_fidelity"]["rate"] == 0.361266
    assert result["witnesses_by_audited_depth"] == {1: 10, 2: 23}
    assert result["witnesses_by_level"] == {"R1": 12, "R2": 6, "R3": 15}
    assert result["cross_audit"] == {
        "status": "complete",
        "n_guarded_changes": 21,
        "provisional_until_complete": False,
    }
    assert result["axes"] == {
        "prompt_articulability": "not_measured",
        "code_verifiability": "not_measured_static_source_audit_only",
        "reconstruction_agreement": "not_estimated",
        "isomorphism": "not_estimated",
        "codability": "not_estimated",
        "hierarchy_trend": "not_estimated",
    }


def test_math_symbolic_capability_is_additive_static_sensitivity() -> None:
    result = stats.math_hierarchy_symbolic_capability_sensitivity()

    assert result["panel_cells"] == 90
    assert result["canonical_relation_local_cells"] == 33
    assert result["retrieved_candidates"] == 15
    assert result["formal_symbolic_relation_local_cells"] == 7
    assert result["newly_covered_cells"] == 5
    assert result["existing_cells_adding_formal_symbolic_relation"] == 2
    assert result["additive_union_cells"] == 38
    assert result["whole_construct_exact"] == 0
    balanced = result["balanced_panel"]
    assert balanced["canonical_relation_local_unchanged"]["rate"] == 0.366667
    assert balanced["formal_symbolic_relation_local"]["rate"] == 0.077778
    assert balanced["newly_covered_by_formal_symbolic_relation"]["rate"] == 0.055556
    assert balanced["additive_sensitivity_union_relation_local"]["rate"] == 0.422222
    expanded = result["eligible_inventory_stratum_expansion"]
    assert expanded["population_nodes"] == 1185
    assert expanded["canonical_relation_local_unchanged"]["rate"] == 0.361266
    assert expanded["additive_sensitivity_union_relation_local"]["rate"] == 0.376231
    assert result["matched_relation_depth"]["depth"] == 3
    assert result["matched_relation_depth"][
        "isolation_or_test_execution_adds_depth"
    ] is False
    assert result["axes"]["prompt_articulability"] == "not_measured"
    assert result["axes"]["reconstruction_agreement"] == "not_estimated"


def test_math_hierarchy_operational_funnel_keeps_constant_l_scope() -> None:
    result = stats.math_hierarchy_operational_funnel()

    assert result["panel_cells"] == 90
    assert result["unique_programs"] == 16
    assert result["stage_relation_mapping_counts"] == {
        "static_relation_local_witness": 33,
        "train_operational_constant_l_slice": 33,
        "heldout_measurable_constant_l_slice": 33,
    }
    assert result["balanced_panel"]["heldout_measurable_constant_l_slice"][
        "rate"
    ] == 0.366667
    expanded = result["eligible_inventory_stratum_expansion"]
    assert expanded["population_nodes"] == 1185
    assert expanded["heldout_measurable_constant_l_slice"]["rate"] == 0.361266
    assert result["stage_retention"] == {
        "train_given_static": {"numerator": 33, "denominator": 33, "fraction": 1.0},
        "heldout_given_train_operational": {
            "numerator": 33,
            "denominator": 33,
            "fraction": 1.0,
        },
    }
    assert result["compiler_train"]["three_state_totals"] == {
        "measured": 36000,
        "abstained": 0,
        "failed": 0,
    }
    assert result["heldout_pre_reference"]["three_state_totals"] == {
        "measured": 2400,
        "abstained": 0,
        "failed": 0,
    }
    sensitivity = result["sentinel_sensitivity"]["pooled_pair_weighted"]
    assert sensitivity["n_spearman_pairs"] == 2285
    assert sensitivity["spearman_median"] == 1.0
    assert sensitivity["spearman_min"] == 0.705609
    assert sensitivity["identical_vector_pair_rate"] == 0.419694
    assert result["prompt_batches"]["compiler_train"] == {
        "status": "compiled_unscored",
        "n_cells": 33,
        "n_cells_by_level": {"R1": 12, "R2": 6, "R3": 15},
        "n_cells_by_audited_depth": {"1": 10, "2": 23},
        "n_unique_program_vectors": 16,
        "n_items": 150,
        "n_passes": 2,
        "n_bank_arms_across_selected_cells": 951,
        "n_source_articulation_arms_across_selected_cells": 306,
        "n_control_arms_across_selected_cells": 612,
        "n_bank_prompt_specs_in_phase": 951,
        "n_post_code_disclosure_specs_in_phase": 33,
        "n_prompt_specs_in_phase": 984,
        "n_jobs": 295200,
        "n_prompt_responses": 0,
        "n_reconstruction_estimates": 0,
        "n_isomorphism_adjudications": 0,
    }
    assert result["prompt_batches"]["heldout_pre_reference"]["status"] == (
        "compiled_unscored"
    )
    assert result["prompt_batches"]["heldout_pre_reference"]["n_jobs"] == 128700
    assert result["prompt_batches"]["heldout_pre_reference"][
        "n_prompt_responses"
    ] == 0
    assert result["prompt_batches"]["raw_signed_heldout_primary"].startswith(
        "raw signed Spearman rho"
    )
    assert "abstain" in result["prompt_batches"]["isomorphism_polarity_gate"]
    assert result["axes"] == {
        "prompt_articulability": "not_measured_jobs_compiled_unscored",
        "code_verifiability": (
            "relation_local_constant_l_conditional_variation_established; "
            "original_hybrid_and_whole_construct_not_established"
        ),
        "reconstruction_agreement": "not_estimated",
        "isomorphism": "not_estimated",
        "codability": "not_estimated",
        "hierarchy_trend": "not_estimated",
    }


def test_science_hierarchy_static_funnel_is_full_article_and_pre_execution() -> None:
    result = stats.science_hierarchy_static_funnel()

    assert result["historical_program_families"] == 1
    assert result["panel_cells"] == 90
    assert result["retrieved_candidates"] == 9
    assert result["relation_local_static_witnesses"] == 6
    assert result["relation_mismatches"] == 3
    assert result["whole_construct_exact"] == 0
    assert result["balanced_panel"]["relation_local_static_fidelity"]["rate"] == 0.066667
    conditional = result["eligible_inventory_stratum_expansion"]
    assert conditional["population_nodes"] == 675
    assert conditional["relation_local_static_fidelity"]["rate"] == 0.055407
    assert result["witnesses_by_audited_depth"] == {3: 6}
    assert result["witnesses_by_level"] == {"R1": 2, "R2": 2, "R3": 2}
    assert result["channel_provenance"] == {
        "automatic_discovery": False,
        "certificate_scope": (
            "numeric/comparative document-internal consistency, not external scientific truth"
        ),
        "evidence_scope": "distinct body sentences within the same presented article",
        "historical_pipeline": "manually designed full-article pure-code verifier",
        "retrieval_scope": "document-local BM25; no corpus or external retrieval",
    }
    assert result["axes"] == {
        "prompt_articulability": "not_measured",
        "code_verifiability": "not_measured_static_source_audit_only",
        "reconstruction_agreement": "not_estimated",
        "isomorphism": "not_estimated",
        "codability": "not_estimated",
        "external_scientific_truth": "not_estimated_document_internal_only",
        "hierarchy_trend": "not_estimated",
    }


def test_science_fullarticle_execution_preserves_representation_boundary() -> None:
    result = stats.science_hierarchy_fullarticle_operational_funnel()

    assert result["canonical_representation_blocker"] == {
        "status": "canonical_execution_blocked_by_representation_mismatch",
        "canonical_items": 300,
        "exact_abstract_joins": 12,
        "exact_joins_with_nonempty_body": 6,
        "execution_performed": False,
        "reason": result["canonical_representation_blocker"]["reason"],
    }
    assert result["representation"]["canonical_hierarchy_items"] is False
    assert result["representation"][
        "direct_comparison_to_canonical_abstract_only_execution"
    ] is False
    assert result["representation"][
        "same_bytes_for_future_prompt_and_current_code"
    ] is True
    assert result["stage_relation_mapping_counts"] == {
        "static_relation_local_witness": 6,
        "train_operational_fullarticle_section_verifier": 6,
        "heldout_measurable_fullarticle_section_verifier": 6,
    }
    assert result["compiler_train"]["three_state_totals_unique_items"] == {
        "measured": 118,
        "abstained": 32,
        "failed": 0,
    }
    assert result["heldout_pre_reference"]["three_state_totals_unique_items"] == {
        "measured": 108,
        "abstained": 42,
        "failed": 0,
    }
    assert result["compiler_train"]["n_relation_certificates"] == 7
    assert result["heldout_pre_reference"]["n_relation_certificates"] == 10
    assert result["heldout_pre_reference"]["n_items_with_relation_certificate"] == 9
    assert result["balanced_panel"][
        "heldout_measurable_fullarticle_section_verifier"
    ]["rate"] == 0.066667
    assert result["eligible_inventory_stratum_expansion"]["population_nodes"] == 675
    assert result["eligible_inventory_stratum_expansion"][
        "heldout_measurable_fullarticle_section_verifier"
    ]["rate"] == 0.055407
    assert result["scientific_object"]["effective_code_depth"] == 3
    assert result["scientific_object"]["external_scientific_truth"] is False
    assert result["axes"]["prompt_articulability"] == "not_measured"
    assert result["axes"]["reconstruction_agreement"] == "not_estimated"
    overlay = result["additive_addressed_prompt_overlay"]
    assert overlay["prompt_plane"] == {
        "distinct_prepared_unscored_request_records": 235,
        "planned_stateless_passes": 2,
        "planned_two_pass_prompt_jobs_if_executed": 470,
        "prompt_articulability_measured": False,
        "prompt_code_reconstruction_measured": False,
        "prompt_responses": 0,
        "selected_items": 300,
        "six_relation_mappings_share_one_result_vector": True,
        "structural_abstentions_without_remote_call": 65,
        "two_pass_jobs_materialized_as_separate_requests": False,
    }
    assert overlay["split_prompt_transport"] == {
        "compiler_train": {
            "compiled_unscored_request": 124,
            "structural_abstention_no_remote_call": 26,
        },
        "sealed_heldout": {
            "compiled_unscored_request": 111,
            "structural_abstention_no_remote_call": 39,
        },
    }
    assert overlay["code_replay_agreement"]["agree"] == 300
    assert overlay["code_replay_agreement"]["total"] == 300
    assert overlay["code_aggregate_exact_for_both_splits"] is True
    assert overlay["representation_contract"]["same_evidence_content"] is True
    assert overlay["representation_contract"]["same_input_representation"] is False
    assert overlay["representation_contract"]["full_isomorphism_licensed"] is False
    exact = result["exact_ctext_prompt_instrument"]
    assert exact["summary"]["compiled_prompt_pass_records"] == 470
    assert exact["summary"]["pass_expanded_structural_no_call_outcomes"] == 130
    assert exact["summary"]["mapping_record_applications_if_executed"] == 2820
    assert exact["summary"]["prompt_responses"] == 0
    assert exact["by_phase"]["compiler_train"]["compiled_prompt_pass_records"] == 248
    heldout = exact["by_phase"]["current_heldout_post_code_exploratory"]
    assert heldout["compiled_prompt_pass_records"] == 222
    assert heldout["structural_abstention_unique_items"] == 39
    assert exact["validation"]["decoded_exact_payload_records"] == 470
    assert exact["validation"]["payload_mismatches"] == 0
    assert exact["validation"]["payload_multiple_occurrences"] == 0
    exact_representation = exact["representation_contract"]
    assert exact_representation["same_frozen_ctext_payload_bytes_as_current_code"] is True
    assert exact_representation["raw_jsonl_or_provider_wire_byte_identity_claimed"] is False
    assert exact_representation["full_semantic_isomorphism_licensed"] is False
    assert exact_representation["provider_transport_compatibility_tested"] is False
    assert exact["transport_control_inventory"]["eligible_unique_items"] == 36
    assert exact["transport_control_inventory"]["compiled_prompt_pass_records"] == 72
    target = exact["future_comparison_target"]
    assert target["whole_frozen_code_vector"] is False
    assert target["code_projection_compiled_and_replay_bound"] is True
    assert target["code_projection_summary"] == {
        "decision_counts": {"insufficient": 141, "supported": 17},
        "evidence_link_decisions": 0,
        "items": 300,
        "selected_claims": 158,
    }
    assert target["reconstruction_decisions"] == [
        "contradicted",
        "insufficient",
        "supported",
    ]
    assert target["evidence_link_in_reconstruction_target"] is False
    assert exact["claim_boundary"]["response_validation_checks_relation_truth"] is False
