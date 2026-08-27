from __future__ import annotations

import json
from pathlib import Path

from methods.metric_seam import propose_grant_structure_fidelity_v1 as proposal


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json"


def _built() -> dict:
    return proposal.build(json.loads(PANEL.read_text(encoding="utf-8")))


def test_proposal_covers_balanced_panel_but_credits_no_unaudited_mapping() -> None:
    result = _built()
    assert result["status"] == (
        "author_proposal_complete_pending_independent_construct_audit"
    )
    assert result["summary"] == {
        "panel_cells": 90,
        "mapped_pending_independent_audit": 52,
        "bounded_non_discovery_in_frozen_program_class": 38,
        "mapped_by_level": {"R1": 16, "R2": 18, "R3": 18},
        "mapped_by_relation": {
            "aim_hypothesis_experiment_graph": 5,
            "budget_sum_consistency": 4,
            "citation_claim_link": 4,
            "dissemination_output_channel_graph": 3,
            "document_outline_structure": 4,
            "evaluation_measurement_chain": 7,
            "front_matter_coverage": 3,
            "partner_role_graph": 3,
            "quantified_need_gap": 4,
            "resource_use_graph": 2,
            "risk_mitigation_graph": 3,
            "role_responsibility_graph": 6,
            "schedule_dependency_graph": 4,
        },
        "mapped_by_proposed_depth": {"1": 7, "2": 41, "3": 4},
        "whole_construct_exact": 0,
        "eligible_for_execution_before_independent_audit": 0,
    }
    assert len(result["rows"]) == 90
    assert len({row["cell_id"] for row in result["rows"]}) == 90
    assert all(row["eligible_for_execution"] is False for row in result["rows"])
    assert all(
        row["independent_construct_audit_complete"] is False
        for row in result["rows"]
    )


def test_proposal_preserves_code_prompt_reconstruction_axis_separation() -> None:
    result = _built()
    assert result["blindness"] == {
        "outcome_labels_used": False,
        "reference_values_used": False,
        "heldout_items_or_outputs_used": False,
        "external_supervised_anchor_used": False,
        "model_or_api_used": False,
        "accelerator_used": False,
    }
    assert result["claim_boundary"] == {
        "articulability_measured": False,
        "code_verifiability_established": False,
        "construct_fidelity_established": False,
        "reconstruction_measured": False,
        "isomorphism_measured": False,
        "codability_measured": False,
        "negative_result_policy": result["claim_boundary"]["negative_result_policy"],
    }


def test_rules_and_external_objects_are_not_silently_replaced_by_text_presence() -> None:
    by_name = {row["construct"]: row for row in _built()["rows"]}
    for construct in (
        "FOA alignment and topical responsiveness",
        "Compliance with funding and regulatory requirements",
        "Allowable costs compliance and categorization",
        "Program and mission fit",
        "Responsiveness to solicitation scope, priorities, instructions, and eligibility",
    ):
        assert by_name[construct]["proposal_status"] == "bounded_non_discovery"
        assert by_name[construct]["implemented_relation_id"] is None


def test_budget_mapping_is_narrow_arithmetic_not_budget_quality() -> None:
    rows = [
        row
        for row in _built()["rows"]
        if row["implemented_relation_id"] == "budget_sum_consistency"
    ]
    assert len(rows) == 4
    assert {row["proposed_depth"] for row in rows} == {3}
    assert all("last checkable stated currency total" in row["implemented_relation"] for row in rows)
    assert all(row["whole_construct_exact"] is False for row in rows)


def test_unmapped_cells_are_bounded_non_discovery_not_tacitness() -> None:
    result = _built()
    unmapped = [row for row in result["rows"] if row["proposal_status"] == "bounded_non_discovery"]
    assert len(unmapped) == 38
    assert all("frozen thirteen-relation program class" in row["scope_limit"] for row in unmapped)
    assert "never evidence of tacitness" in result["claim_boundary"]["negative_result_policy"]
