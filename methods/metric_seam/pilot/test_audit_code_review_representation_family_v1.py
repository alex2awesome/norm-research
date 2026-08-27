from __future__ import annotations

from collections import Counter

import pytest

from methods.metric_seam.pilot import audit_code_review_representation_family_v1 as audit


def _rows() -> list[dict]:
    return [
        {
            "item_key": f"item_{index:04d}",
            "applies": True,
            "score": index / 250,
            "status": "scored",
        }
        for index in range(250)
    ]


def test_frozen_contract_and_three_label_free_projections_crosswalk() -> None:
    bindings = audit.verify_frozen_inputs()
    arms, receipt = audit.build_representations()
    assert bindings[str(audit.CONTRACT.relative_to(audit.ROOT))] == (
        "dd9b264a5f8294f706ee8ffade8217f9f567ab4301f292aaa694549253421f08"
    )
    assert {name: len(rows) for name, rows in arms.items()} == {
        "P0_prefix4000": 250,
        "P1_head5000_tail2500": 250,
        "P2_raw_diff_capped300k": 250,
    }
    assert receipt["P0_P1_exact_prefix_crosswalk_n"] == 250
    assert receipt["P1_P2_exact_canonicalization_n"] == 250
    assert receipt["P2_local_path_crosswalk_n"] == 250
    assert receipt["outcome_bearing_items_json_loaded"] is False


def test_primary_population_is_ten_programs_and_eighteen_typed_mappings() -> None:
    primary, secondary, criteria, receipt = audit.load_populations()
    assert len(primary) == 10
    assert len(secondary) == 16
    assert sum(map(len, criteria.values())) == 18
    assert {row["aspect_id"] for row in primary} == {
        "a0", "a1", "a15", "a18", "a37", "a38", "a401", "a43", "a70", "a92",
    }
    assert receipt["selection_loaded_outcomes_or_references"] is False


def test_pairwise_readout_uses_predeclared_typed_classes() -> None:
    left = _rows()
    exact = audit.compare_rows(left, [dict(row) for row in left])
    assert exact["sensitivity_class"] == "exact_stable"
    assert exact["n_exact_rows"] == 250

    value_rows = [dict(row) for row in left]
    value_rows[0]["score"] = 0.5
    value = audit.compare_rows(left, value_rows)
    assert value["sensitivity_class"] == "value_sensitive_only"
    assert value["n_exact_values_on_common_scored"] == 249

    status_rows = [dict(row) for row in left]
    status_rows[0] = {
        "item_key": status_rows[0]["item_key"],
        "applies": False,
        "score": None,
        "status": "not_applicable",
    }
    status = audit.compare_rows(left, status_rows)
    assert status["sensitivity_class"] == "status_or_applicability_sensitive"
    assert status["n_applicability_changes"] == 1
    assert Counter(status["status_transition_counts"]) == Counter({
        "scored -> not_applicable": 1,
        "scored -> scored": 249,
    })


def test_checked_in_result_is_contract_bound_and_claim_limited() -> None:
    if not audit.DEFAULT_OUT.is_file():
        pytest.fail("family audit artifact has not been generated")
    result = audit.check()
    assert result["P0_exact_frozen_replay"]["total_rows_exact"] == 4000
    assert len(result["primary_program_results"]) == 10
    assert len(result["secondary_program_results"]) == 16
    assert result["typed_primary_criterion_join"]["n_relation_mappings"] == 18
    assert result["typed_primary_criterion_join"]["aggregation_performed"] is False
    assert result["interpretation"]["promotion_gate_defined"] is False
    assert "isomorphism" in result["interpretation"]["not_measured"]
    assert result["blindness_and_channel"] == {
        "outcome_bearing_items_json_loaded": False,
        "prompt_requests_used_as_input_serialization_only": True,
        "prompt_responses_loaded": False,
        "llm_judgments_loaded": False,
        "outcomes_loaded": False,
        "references_loaded": False,
        "reconstruction_results_or_correlations_loaded": False,
        "models_or_apis_called": False,
        "external_supervision_used": False,
        "gpu_or_accelerator_used": False,
    }
