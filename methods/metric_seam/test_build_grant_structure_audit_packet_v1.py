from __future__ import annotations

import json
from pathlib import Path

from methods.metric_seam import build_grant_structure_audit_packet_v1 as packet


ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/grant_structure_static_proposal_v1.json"
)


def _build() -> dict:
    return packet.build(json.loads(PROPOSAL.read_text(encoding="utf-8")))


def test_packet_freezes_all_proposed_rows_without_prefilling_review_decisions() -> None:
    result = _build()
    assert result["status"] == "frozen_for_independent_source_audit"
    assert result["summary"] == {
        "panel_cells": 90,
        "rows_requiring_independent_decision": 52,
        "author_bounded_non_discovery_rows": 38,
        "author_withdrawals_before_freeze": 8,
        "independent_decisions_present": 0,
        "execution_eligible_rows": 0,
    }
    assert len(result["rows"]) == 52
    assert len({row["cell_id"] for row in result["rows"]}) == 52
    assert all(
        all(value is None for value in row["review_required"].values())
        for row in result["rows"]
    )


def test_packet_requires_relation_complete_audit_and_forbids_heldout_access() -> None:
    protocol = _build()["review_protocol"]
    assert protocol["acceptance_rule"] == (
        "partial_relation_local requires object, relation, polarity, applicability, "
        "aggregation, and depth match; otherwise mismatch"
    )
    assert protocol["whole_construct_credit_allowed"] is False
    assert protocol["heldout_access_allowed"] is False
    assert protocol["reference_or_outcome_access_allowed"] is False


def test_packet_records_author_withdrawals_instead_of_hiding_failed_mappings() -> None:
    withdrawals = {
        row["construct"]: row["reason"]
        for row in _build()["author_withdrawals_before_freeze"]
    }
    assert withdrawals == packet.AUTHOR_WITHDRAWALS
    assert "Internal consistency across application components" in withdrawals
    assert "Preliminary data to establish feasibility (when allowed)" in withdrawals
    assert "Inclusive design and stakeholder/patient engagement" in withdrawals

