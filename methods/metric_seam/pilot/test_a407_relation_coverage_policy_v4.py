"""Tests for the additive future a407 witness/absence coverage policy."""

import json
from pathlib import Path

import pytest

from methods.metric_seam.pilot.a407_relation_coverage_policy_v4 import (
    classify_event_relation,
    input_wide_absence_preconditions,
)


def _clean_counts() -> dict[str, int]:
    return {
        "supported_files_analyzed": 1,
        "truncated_input": 0,
        "orphan_fragments": 0,
        "parse_error_files": 0,
        "parse_error_nodes": 0,
        "parse_missing_nodes": 0,
        "unsupported_files_with_added_code": 0,
    }


def test_positive_placeholder_is_relation_local_even_under_partial_coverage():
    row = classify_event_relation(
        relation_id="placeholder_avoidance",
        detected_event_count=1,
        event_local_parse_valid=True,
        observation_universe_complete=False,
    )
    assert row["evidence_state"] == "positive_relation_witness"
    assert row["claim_scope"] == "detected_event_only"
    assert row["contextual_quality_established"] is False
    assert row["scalar_quality_score_emitted"] is False


def test_placeholder_occurrence_never_establishes_contextual_inappropriateness():
    row = classify_event_relation(
        relation_id="placeholder_avoidance",
        detected_event_count=2,
        event_local_parse_valid=True,
        observation_universe_complete=True,
    )
    assert "surface occurrence" in row["code_establishes"]
    assert row["quality_interpretation"] == "prompt_or_hybrid_frontier"
    assert row["contextual_quality_established"] is False


def test_collision_detection_does_not_establish_harmfulness():
    row = classify_event_relation(
        relation_id="collision_and_shadowing",
        detected_event_count=1,
        event_local_parse_valid=True,
        observation_universe_complete=False,
    )
    assert row["positive_relation_witness"] is True
    assert row["contextual_quality_established"] is False


def test_invalid_local_parse_cannot_certify_a_positive_detection():
    row = classify_event_relation(
        relation_id="collision_and_shadowing",
        detected_event_count=1,
        event_local_parse_valid=False,
        observation_universe_complete=False,
    )
    assert row["evidence_state"] == "uncertified_positive_detection"
    assert row["positive_relation_witness"] is False
    assert row["claim_scope"] == "none"


def test_zero_events_require_complete_observation_universe():
    partial = classify_event_relation(
        relation_id="placeholder_avoidance",
        detected_event_count=0,
        event_local_parse_valid=True,
        observation_universe_complete=False,
    )
    complete = classify_event_relation(
        relation_id="placeholder_avoidance",
        detected_event_count=0,
        event_local_parse_valid=False,
        observation_universe_complete=True,
    )
    assert partial["evidence_state"] == "partial_no_event_observed"
    assert partial["verified_visible_input_absence"] is False
    assert complete["evidence_state"] == "verified_visible_input_absence"
    assert complete["verified_visible_input_absence"] is True
    assert complete["contextual_quality_established"] is False


@pytest.mark.parametrize(
    "field",
    [
        "truncated_input",
        "orphan_fragments",
        "parse_error_files",
        "parse_error_nodes",
        "parse_missing_nodes",
        "unsupported_files_with_added_code",
    ],
)
def test_each_input_completeness_failure_blocks_negative_certificate(field: str):
    counts = _clean_counts()
    counts[field] = 1
    assert input_wide_absence_preconditions(
        counts, relation_observation_universe_complete=True
    ) is False


def test_clean_parse_still_requires_complete_relation_observation_universe():
    counts = _clean_counts()
    assert input_wide_absence_preconditions(
        counts, relation_observation_universe_complete=False
    ) is False
    assert input_wide_absence_preconditions(
        counts, relation_observation_universe_complete=True
    ) is True


def test_unsupported_relation_and_invalid_counts_fail_closed():
    with pytest.raises(ValueError):
        classify_event_relation(
            relation_id="semantic_context_fit",
            detected_event_count=0,
            event_local_parse_valid=True,
            observation_universe_complete=True,
        )
    with pytest.raises(ValueError):
        classify_event_relation(
            relation_id="placeholder_avoidance",
            detected_event_count=-1,
            event_local_parse_valid=True,
            observation_universe_complete=True,
        )


def test_future_design_separates_augmentation_from_substitution_and_controls_length():
    design_path = Path(__file__).with_name("a407_seam_placement_experiment_v4.json")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    current = design["current_full_graph_arms"]
    assert current["v1"]["causal_seam_placement_status"] == "launch_gated"
    assert current["matched_v2"]["label"] == "full_graph_augmentation"
    arms = {
        row["arm_id"]: row
        for row in design["augmentation_microexperiment"]["arms"]
    }
    assert set(arms) == {
        "null",
        "relation_matched",
        "relation_mismatched_length_matched",
    }
    control = arms["relation_mismatched_length_matched"]["matching"]
    assert "model-token count" in control["serialized_length"]
    substitution = design["substitution_microexperiment"]
    assert substitution["label"] == "offline_one_relation_code_substitution"
    eligibility = substitution["current_v3_scalar_eligibility"]
    assert eligibility["status"] == "unavailable"
    assert all(
        "ineligible" in eligibility["relation_status"][relation]
        for relation in ("placeholder_avoidance", "collision_and_shadowing")
    )
    assert "construct adversary" in eligibility["required_before_substitution"]
    assert "neutral 0.5 is never substituted" in substitution["noncoverage"]
