from __future__ import annotations

from copy import deepcopy

import pytest

from methods.metric_seam.pilot import build_technical_evidence_ledger_v1 as ledger_v1


@pytest.fixture(scope="module")
def ledger() -> dict:
    return ledger_v1.build_ledger()


@pytest.fixture(scope="module")
def records(ledger: dict) -> dict[str, dict]:
    return {row["record_id"]: row for row in ledger["records"]}


def test_ledger_is_typed_nonpooling_union(ledger: dict) -> None:
    assert ledger["schema"] == ledger_v1.SCHEMA
    assert ledger["external_supervised_ground_truth_used"] is False
    assert ledger["summary"] == {
        "record_count": 39,
        "by_stratum": {
            "criterion_scalar_reconstruction": 24,
            "program_structure_descriptor": 7,
            "relation_instance_verification": 8,
        },
        "by_domain": {
            "code_review": 22,
            "math": 8,
            "patents": 7,
            "science": 2,
        },
        "domain_codability_estimates_emitted": 0,
        "cross_stratum_pooled_estimates_emitted": 0,
        "explicitly_nonpoolable": True,
    }
    assert all(
        row["claim_permissions"]["may_claim_domain_codability"] is False
        for row in ledger["records"]
    )


def test_bounded_family_denominators_are_exact(ledger: dict) -> None:
    families = ledger["family_summaries"]
    assert families["active_code_depth_family"][
        "bh_fdr_and_minimum_effect_improvements"
    ] == {"numerator": 0, "denominator": 4}
    assert families["blind_math_construct_family"]["construct_fidelity_passes"] == {
        "numerator": 0,
        "denominator": 2,
    }
    assert families["patent_historical_selected_family"][
        "reference_usable_for_historical_description"
    ] == {"numerator": 3, "denominator": 4}
    assert families["patent_historical_selected_family"]["bh_fdr_rejections"] == {
        "numerator": 2,
        "denominator": 4,
    }
    assert families["patent_historical_selected_family"][
        "effect_precision_characterized"
    ] == {"numerator": 1, "denominator": 4}


def test_math_a12_train_and_sealed_heldout_stay_relation_local(
    records: dict[str, dict],
) -> None:
    train = records["math.a12.train_symbolic_step"]
    heldout = records["math.a12.heldout_symbolic_step"]
    assert train["stratum"] == heldout["stratum"] == "relation_instance_verification"
    assert train["units"]["train_n"] == 150
    assert train["readouts"]["rows_with_executable_pair"]["numerator"] == 42
    assert train["readouts"]["parsed_pair_positive_witness_rate"] == {
        "metric": "fraction",
        "estimate": pytest.approx(115 / 116),
        "status": "observed",
        "numerator": 115,
        "denominator": 116,
        "support_n": None,
        "conditioning": "Parsed rational pairs; identity and exact nonidentity witnesses are disjoint pair outcomes.",
        "inference_status": "descriptive",
        "recomputable": True,
        "note": None,
    }

    assert heldout["units"]["heldout_n"] == 100
    assert heldout["readouts"]["rows_with_executable_pair"]["numerator"] == 26
    assert heldout["readouts"]["row_abstention_rate"]["numerator"] == 74
    assert heldout["readouts"]["verified_identity_pair_rate"]["numerator"] == 11
    assert heldout["readouts"]["exact_nonidentity_pair_rate"]["numerator"] == 54
    assert heldout["readouts"]["verified_identity_pair_rate"]["denominator"] == 65
    assert heldout["readouts"]["whole_criterion_reconstruction"]["status"] == (
        "unavailable"
    )
    assert heldout["fidelity"]["isomorphism"] == "not_estimated"
    assert heldout["claim_permissions"]["may_claim_code_verifiability"] is True
    assert heldout["claim_permissions"]["may_claim_confirmatory_isomorphism"] is False


def test_math_a12_projection_and_depth_views_remain_temporally_separate(
    records: dict[str, dict],
) -> None:
    projection = records["math.a12.post_reference_pair_projection"]
    assert projection["selection"]["mode"] == "post_reference_projection_replay"
    assert projection["fidelity"]["new_blind_result"] is False
    assert projection["fidelity"]["new_reconstruction_result"] is False
    assert (
        projection["readouts"]["verified_identity_share_of_pair_candidates"][
            "numerator"
        ]
        == 11
    )
    assert (
        projection["readouts"]["exact_nonidentity_share_of_pair_candidates"][
            "numerator"
        ]
        == 54
    )
    assert projection["readouts"]["parse_noncoverage_share_of_pair_candidates"] == {
        "metric": "fraction",
        "estimate": pytest.approx(212 / 277),
        "status": "observed",
        "numerator": 212,
        "denominator": 277,
        "support_n": None,
        "conditioning": "All projected pair candidates; noncoverage is abstention, not negative evidence.",
        "inference_status": "descriptive",
        "recomputable": True,
        "note": None,
    }

    depth = records["math.a12.relation_depth_multiview"]
    assert depth["fidelity"]["scale_id"] == "metric-seam.relation-depth.v1"
    assert depth["fidelity"]["supersedes_prior_dynamic_interpretation"] is True
    assert depth["readouts"]["attempted_depth_1_rate"]["numerator"] == 35
    assert depth["readouts"]["attempted_depth_3_rate"]["numerator"] == 65
    assert depth["readouts"]["decision_contributing_depth_3_rate"]["numerator"] == 65
    assert depth["readouts"]["positive_evidence_depth_3_rate"]["numerator"] == 26
    assert depth["readouts"]["formal_path_parse_noncoverage_rate"]["numerator"] == 39
    assert depth["readouts"]["formal_path_parse_noncoverage_rate"]["denominator"] == 65


def test_unavailable_and_not_run_channels_are_null(records: dict[str, dict]) -> None:
    a216 = records["math.a216.construct_adversary"]
    assert a216["readouts"]["heldout_reconstruction_spearman"]["status"] == ("unopened")
    assert a216["readouts"]["heldout_reconstruction_spearman"]["estimate"] is None

    science = records["science.v9.document_local_relations"]
    assert science["readouts"]["prompt_articulability_output_rate"]["status"] == (
        "not_run"
    )
    assert science["readouts"]["prompt_articulability_output_rate"]["estimate"] is None

    a407 = records["code.a407.structural_partial_historical"]
    assert a407["readouts"]["matched_raw_prompt_reconstruction"]["status"] == (
        "not_run"
    )
    assert a407["readouts"]["matched_hybrid_reconstruction"]["estimate"] is None


def test_science_strong_and_weak_tiers_have_relation_denominators(
    records: dict[str, dict],
) -> None:
    science = records["science.v9.document_local_relations"]["readouts"]
    expected = {
        "strong_numeric_witness_rate": (68, 561),
        "strong_comparative_witness_rate": (32, 634),
        "strong_witness_rate_all_matched": (100, 4871),
        "weak_theoretical_link_rate": (75, 276),
        "weak_empirical_link_rate": (211, 1772),
        "weak_qualitative_link_rate": (144, 1628),
    }
    for key, (numerator, denominator) in expected.items():
        assert science[key]["numerator"] == numerator
        assert science[key]["denominator"] == denominator
        assert science[key]["estimate"] == pytest.approx(numerator / denominator)


def test_science_representation_overlap_is_same_program_code_to_code_only(
    records: dict[str, dict],
) -> None:
    record = records["science.v9.document_local_relations"]
    readouts = record["readouts"]
    expected_fractions = {
        "same_program_strong_normalized_overlap_given_continuous": (100, 100),
        "same_program_strong_normalized_overlap_given_addressed": (100, 100),
        "same_program_supported_set_overlap_given_continuous": (95, 95),
        "same_program_supported_set_overlap_given_addressed": (95, 95),
        "same_program_paper_status_agreement": (2396, 2400),
        "same_program_weak_normalized_overlap_given_continuous": (429, 434),
        "same_program_weak_normalized_overlap_given_addressed": (429, 430),
    }
    for key, (numerator, denominator) in expected_fractions.items():
        assert readouts[key]["numerator"] == numerator
        assert readouts[key]["denominator"] == denominator
        assert readouts[key]["estimate"] == pytest.approx(numerator / denominator)
    assert (
        readouts["same_program_strong_normalized_intersection_count"]["estimate"] == 100
    )
    assert readouts["same_program_supported_set_intersection_count"]["estimate"] == 95
    assert (
        readouts["same_program_weak_normalized_intersection_count"]["estimate"] == 429
    )
    assert readouts["same_program_weak_continuous_only_count"]["estimate"] == 5
    assert readouts["same_program_weak_addressed_only_count"]["estimate"] == 1

    assert record["channels"]["representation_comparison_type"] == (
        "code_to_code_same_program_input_representation_robustness"
    )
    assert record["fidelity"]["prompt_to_code_isomorphism_from_overlap"] == (
        "not_licensed"
    )
    assert record["fidelity"]["domain_codability_from_overlap"] == "not_licensed"
    assert record["fidelity"]["archived_v9_manifest_source_bindings"] == "absent"
    source_paths = {source["path"] for source in record["sources"]}
    assert {
        "outputs/metric_seam_pilot/science_claims_v2_relation_strict_v23/results.json",
        "outputs/metric_seam_pilot/science_verifiability_v9_relation_strict_addressed/manifest.json",
        "methods/metric_seam/science_claims_v2/core_relation_strict.py",
        "methods/metric_seam/science_claims_v2/addressed_code_comparator_v8.py",
        "methods/metric_seam/science_claims_v2/addressed_code_comparator_v9.py",
    }.issubset(source_paths)


def test_active_code_full_family_and_a104(records: dict[str, dict]) -> None:
    active = [
        row
        for record_id, row in records.items()
        if record_id.startswith("code.active_depth.")
    ]
    assert len(active) == 18
    assert (
        sum(
            row["readouts"]["shallow_reconstruction_spearman"]["status"] == "observed"
            for row in active
        )
        == 15
    )
    assert (
        sum(row["readouts"]["bh_q_value"]["status"] == "observed" for row in active)
        == 4
    )

    a104 = records["code.active_depth.a104"]
    assert a104["readouts"]["deep_reconstruction_spearman"][
        "estimate"
    ] == pytest.approx(0.649794037210992)
    assert a104["readouts"]["shallow_reconstruction_spearman"][
        "estimate"
    ] == pytest.approx(0.5089068945741408)
    assert a104["readouts"]["deep_minus_shallow_delta"]["estimate"] == pytest.approx(
        0.1408871426368512
    )
    assert a104["readouts"]["bh_q_value"]["estimate"] == pytest.approx(
        0.46015398460153983
    )
    source_paths = {source["path"] for source in a104["sources"]}
    assert (
        "outputs/metric_seam_pilot/reconstruction_v2/code_depth_full_panel_retrospective_002/results.json"
        in source_paths
    )
    assert not any(
        "code_depth_full_panel_retrospective_001" in path for path in source_paths
    )


def test_active_code_source_structure_is_not_semantic_depth(
    records: dict[str, dict],
) -> None:
    structure = records["code.active_panel.entry_module_source_structure"]
    assert structure["fidelity"]["descriptor_kind"] == (
        "python_entry_module_source_structure"
    )
    assert structure["fidelity"]["semantic_relation_depth"] == "not_measured"
    assert structure["readouts"]["ast_nodes_deep_greater_rate"]["numerator"] == 15
    assert structure["readouts"]["ast_nodes_deep_greater_rate"]["denominator"] == 15
    assert structure["readouts"]["condensed_call_path_deep_greater_rate"] == {
        "metric": "fraction",
        "estimate": pytest.approx(13 / 15),
        "status": "observed",
        "numerator": 13,
        "denominator": 15,
        "support_n": None,
        "conditioning": "Scope-qualified lexical call graphs after SCC condensation; two ties.",
        "inference_status": "descriptive",
        "recomputable": True,
        "note": None,
    }
    assert structure["readouts"]["call_path_association_all_defined"]["estimate"] < 0
    assert (
        structure["readouts"]["call_path_association_comparison_eligible"]["estimate"]
        > 0
    )


def test_patent_full_family_multiplicity_and_degeneracy_guards(
    records: dict[str, dict],
) -> None:
    a26 = records["patents.ws3.a26.evidence_arm"]
    a34 = records["patents.ws3.a34.evidence_arm"]
    a35 = records["patents.ws3.a35.evidence_arm"]
    a60 = records["patents.ws3.a60.evidence_arm"]
    assert a26["readouts"]["bh_q_value"]["estimate"] == pytest.approx(
        0.0567943205679432
    )
    assert a26["readouts"]["bh_fdr_reject"]["estimate"] is False
    assert a34["readouts"]["bh_fdr_reject"]["estimate"] is True
    assert a35["readouts"]["bh_fdr_reject"]["estimate"] is True
    assert a60["readouts"]["bh_fdr_reject"]["estimate"] is False
    assert a34["readouts"]["paired_bootstrap_ci_lower"]["status"] == "unavailable"
    assert a34["readouts"]["paired_bootstrap_ci_lower"]["estimate"] is None
    assert a34["readouts"]["effect_precision_characterized"]["estimate"] is False
    assert a35["readouts"]["effect_precision_characterized"]["estimate"] is True
    for row in (a26, a34, a60):
        assert row["fidelity"]["null_rank_support_warning"] == (
            "near_degenerate_null_score_distribution"
        )
    assert a35["fidelity"]["null_rank_support_warning"] is None


def test_structure_descriptors_require_local_kind_and_scale(
    records: dict[str, dict],
) -> None:
    structures = [
        row
        for row in records.values()
        if row["stratum"] == "program_structure_descriptor"
    ]
    assert len(structures) == 7
    assert all(row["fidelity"]["descriptor_kind"] for row in structures)
    assert all(row["fidelity"]["scale_id"] for row in structures)

    science = records["science.v9.instance_evidence_graph"]
    assert science["fidelity"]["descriptor_kind"] == "instance_evidence_graph"
    assert science["fidelity"]["control_flow_depth_available"] is False

    patent_depths = {
        criterion: records[f"patents.ws4.{criterion}.typed_dag"]["readouts"][
            "deepest_output_path_edges"
        ]["estimate"]
        for criterion in ("a26", "a34", "a35")
    }
    assert patent_depths == {"a26": 3, "a34": 2, "a35": 2}


def test_validator_rejects_denominator_and_null_encoding_errors(ledger: dict) -> None:
    invalid = deepcopy(ledger)
    readout = invalid["records"][0]["readouts"]["candidate_coverage"]
    readout["denominator"] = 0
    with pytest.raises(ledger_v1.LedgerError, match="positive denominator"):
        ledger_v1.validate_ledger(invalid)

    invalid = deepcopy(ledger)
    a216 = next(
        row
        for row in invalid["records"]
        if row["record_id"] == "math.a216.construct_adversary"
    )
    a216["readouts"]["heldout_reconstruction_spearman"]["estimate"] = 0.0
    with pytest.raises(ledger_v1.LedgerError, match="null-status estimate"):
        ledger_v1.validate_ledger(invalid)


def test_validator_rejects_unscaled_structure_and_domain_claim(ledger: dict) -> None:
    invalid = deepcopy(ledger)
    structure = next(
        row
        for row in invalid["records"]
        if row["stratum"] == "program_structure_descriptor"
    )
    structure["fidelity"]["scale_id"] = None
    with pytest.raises(ledger_v1.LedgerError, match="needs kind and scale_id"):
        ledger_v1.validate_ledger(invalid)

    invalid = deepcopy(ledger)
    invalid["records"][0]["claim_permissions"]["may_claim_domain_codability"] = True
    with pytest.raises(ledger_v1.LedgerError, match="domain codability"):
        ledger_v1.validate_ledger(invalid)


def test_report_states_the_nonpooling_boundary(ledger: dict) -> None:
    report = ledger_v1.render_report(ledger)
    assert "No percentage below is a domain-wide codability estimate" in report
    assert "26/100 held-out rows contain an executable pair" in report
    assert "35/100 rows stop at depth 1 and 65/100 attempt" in report
    assert "a26 misses at q=.0568" in report
    assert "source syntax, not semantic relation depth" in report
    assert "same-program code-to-code representation audit" in report
    assert "Weak normalized overlap is 429/434" in report
    assert "not prompt-to-code isomorphism or codability" in report
    assert "archived v9 manifest lacks dependency bindings" in report
    assert "The ledger therefore emits zero domain-codability estimates" in report
