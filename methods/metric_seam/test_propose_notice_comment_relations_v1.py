from __future__ import annotations

import json
from pathlib import Path

from methods.metric_seam import propose_notice_comment_relations_v1 as proposal


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json"


def _built() -> dict:
    return proposal.build(json.loads(PANEL.read_text(encoding="utf-8")))


def test_proposal_covers_all_ninety_cells_without_credit_before_cross_audit() -> None:
    result = _built()
    assert result["summary"] == {
        "panel_cells": 90,
        "mapped_cells_pending_independent_audit": 29,
        "relation_mappings_pending_independent_audit": 31,
        "mapped_cells_by_level": {"R1": 8, "R2": 11, "R3": 10},
        "mapping_depth_counts": {"2": 28, "3": 3},
        "bounded_non_discovery_cells": 61,
        "whole_construct_exact": 0,
        "execution_eligible_before_independent_audit": 0,
    }
    assert len(result["rows"]) == 90
    assert len({row["cell_id"] for row in result["rows"]}) == 90
    assert all(row["eligible_for_execution"] is False for row in result["rows"])


def test_representation_finding_blocks_full_rule_and_external_authority_claims() -> None:
    result = _built()
    assert result["representation_finding"] == {
        "compiler_train_items": 150,
        "compiler_train_median_characters": 110.5,
        "compiler_train_max_characters": 293,
        "implication": result["representation_finding"]["implication"],
    }
    assert result["blindness"] == {
        "outcomes_used": False,
        "reference_scores_used": False,
        "heldout_items_or_outputs_used": False,
        "external_authority_or_docket_loaded": False,
        "remote_model_or_api_used": False,
        "accelerator_used": False,
    }


def test_domain_rule_constructs_remain_bounded_non_discovery() -> None:
    by_name = {row["construct"]: row for row in _built()["rows"]}
    for construct in (
        "Clean Air Act source‑specific emission standards",
        "Title IX — grievance procedures and due process",
        "GDPR processor contractual obligations (Art. 28/32/33)",
        "NPRM/Federal Register notice content and publication compliance",
        "NEPA EIS—scope, process, content, and terminology compliance",
    ):
        row = by_name[construct]
        assert row["proposal_status"] == "bounded_non_discovery"
        assert row["relation_mappings"] == []


def test_composite_evidence_graph_is_depth_three_but_not_whole_comment_quality() -> None:
    rows = [
        row
        for row in _built()["rows"]
        if any(
            mapping["implemented_relation_id"] == "supported_actionable_target_graph"
            for mapping in row["relation_mappings"]
        )
    ]
    assert len(rows) == 3
    for row in rows:
        mapping = next(
            mapping
            for mapping in row["relation_mappings"]
            if mapping["implemented_relation_id"] == "supported_actionable_target_graph"
        )
        assert mapping["proposed_depth"] == 3
        assert mapping["whole_construct_exact"] is False

