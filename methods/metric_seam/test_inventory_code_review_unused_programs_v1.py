from __future__ import annotations

import hashlib
import json

import pytest

from methods.metric_seam import inventory_code_review_unused_programs_v1 as inventory


@pytest.fixture(scope="module")
def payload() -> dict:
    return inventory.build_inventory()


def test_complete_corrected_gap_population_and_conservative_decisions(payload: dict) -> None:
    assert payload["schema"] == inventory.SCHEMA
    assert len(payload["rows"]) == 40
    assert payload["summary"]["corrected_gap_by_level"] == {
        "R1": 16,
        "R2": 15,
        "R3": 9,
    }
    assert payload["summary"]["decision_counts"] == {
        "bounded_non_discovery": 9,
        "near_match_reject": 16,
        "nonexecutable_catalog_only": 6,
        "propose_partial_mapping": 9,
    }
    assert len({row["cell_id"] for row in payload["rows"]}) == 40


def test_nine_proposals_are_unused_relation_local_and_most_are_depth_two(
    payload: dict,
) -> None:
    proposed = [
        row for row in payload["rows"]
        if row["decision"] == "propose_partial_mapping"
    ]
    observed = {
        (row["level"], row["metric_name"], row["candidate"]["aspect_id"])
        for row in proposed
    }
    assert observed == {
        ("R1", "Minimal, stable public API surface", "a35"),
        ("R1", "Adheres to project style guides and coding standards", "a72"),
        ("R1", "Consistency with conventions and existing code", "a72"),
        ("R2", "Performance engineering discipline and efficiency", "a400"),
        ("R2", "Defect prevention and verification techniques", "a181"),
        ("R2", "Contribution readiness and submission norms", "a309"),
        ("R2", "Conventions adherence (local and ecosystem)", "a72"),
        ("R3", "Simplicity, explicitness, and avoidance of over‑engineering", "a25"),
        ("R3", "Performance engineering discipline", "a400"),
    }
    assert payload["summary"]["n_unique_proposed_programs"] == 6
    assert payload["summary"]["proposed_matched_depth_counts"] == {"1": 2, "2": 7}
    assert all(row["proposed_scope"] == "subrelation_only" for row in proposed)
    assert all(row["eligible_for_future_independent_source_audit"] for row in proposed)
    assert all(row["candidate"]["declared_tool_tier"] >= 2 for row in proposed)


def test_rejected_and_catalog_rows_contribute_no_proposed_coverage(payload: dict) -> None:
    rejected = [
        row for row in payload["rows"]
        if row["decision"] != "propose_partial_mapping"
    ]
    assert len(rejected) == 31
    assert all(row["proposed_scope"] == "none" for row in rejected)
    assert all(not row["eligible_for_future_independent_source_audit"] for row in rejected)
    assert all(row["proposed_matched_relation_depth"] is None for row in rejected)
    catalog = [
        row for row in rejected if row["decision"] == "nonexecutable_catalog_only"
    ]
    assert len(catalog) == 6
    assert all(row["candidate"]["declared_tool_tier"] == 0 for row in catalog)
    assert all(row["candidate"]["declared_classification"] == "THICK" for row in catalog)


def test_projection_is_an_upper_bound_not_a_canonical_update(payload: dict) -> None:
    assert payload["summary"]["projected_static_upper_bound_if_all_survive"] == {
        "R1": {
            "current_corrected_static": 14,
            "proposed_pending_independent_audit": 3,
            "upper_bound_if_every_proposal_survives": 17,
            "remaining_to_30_under_that_upper_bound": 13,
        },
        "R2": {
            "current_corrected_static": 15,
            "proposed_pending_independent_audit": 4,
            "upper_bound_if_every_proposal_survives": 19,
            "remaining_to_30_under_that_upper_bound": 11,
        },
        "R3": {
            "current_corrected_static": 21,
            "proposed_pending_independent_audit": 2,
            "upper_bound_if_every_proposal_survives": 23,
            "remaining_to_30_under_that_upper_bound": 7,
        },
    }
    assert "does not modify the canonical 50/90" in payload["claim_limits"][0]


def test_sources_are_static_bound_and_no_forbidden_channel_was_loaded(payload: dict) -> None:
    assert payload["sealed_inputs"] == {
        "task_items_loaded": False,
        "candidate_programs_imported_or_executed": False,
        "program_outputs_loaded": False,
        "references_loaded": False,
        "outcomes_loaded": False,
        "prompt_responses_loaded": False,
        "models_or_apis_called": False,
        "gpu_used": False,
        "external_supervision_used": False,
    }
    candidates = [row["candidate"] for row in payload["rows"] if row["candidate"]]
    for candidate in candidates:
        path = inventory.ROOT / candidate["source_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == candidate["source_sha256"]
        assert candidate["model_or_network_import_detected"] is False
        assert candidate["execution_performed"] is False


def test_checked_artifact_rebuilds_byte_identically(payload: dict) -> None:
    checked = json.loads(inventory.DEFAULT_OUT.read_text(encoding="utf-8"))
    assert checked == payload
