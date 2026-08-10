from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam import run_notice_comment_train_v1 as runner


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _inputs() -> tuple[dict, dict, list[dict]]:
    return (
        json.loads(
            (BASE / "notice_comment_relations_static_proposal_v1.json").read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (BASE / "items_v2/notice-and-comment/manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (BASE / "items_v2/notice-and-comment/compiler_train.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def test_train_execution_is_label_free_pre_audit_and_never_opens_heldout() -> None:
    result = runner.run(*_inputs())
    assert result["status"] == (
        "compiler_train_exploratory_complete_pending_independent_construct_audit"
    )
    assert result["blindness"] == {
        "input_fields_passed_to_program": ["ctext"],
        "outcome_fields_passed_to_program": False,
        "reference_fields_passed_to_program": False,
        "heldout_items_or_outputs_loaded": False,
        "external_authority_or_docket_loaded": False,
        "remote_model_or_api_used": False,
        "accelerator_used": False,
    }
    assert result["summary"]["items"] == 150
    assert result["summary"]["relations_executed"] == 14
    assert result["summary"]["nondegenerate_relations"] == 10
    assert result["summary"]["hierarchy_mappings_promoted"] == 0
    assert result["summary"]["heldout_execution_authorized"] is False


def test_train_measurability_exposes_sparse_and_constant_relation_families() -> None:
    result = runner.run(*_inputs())
    assert result["by_relation"]["actionable_target_dependency"]["positive_items"] == 57
    assert result["by_relation"]["supported_actionable_target_graph"] == {
        "depth": 3,
        "status_counts": {"measured": 150},
        "measured": 150,
        "positive_items": 17,
        "unique_finite_scores": 3,
        "minimum": 0.0,
        "maximum": 1.0,
        "nondegenerate": True,
    }
    assert result["by_relation"]["burden_breakdown_relation"]["positive_items"] == 1
    # Unit-bearing quantities only: form/rule/part identifiers are excluded.
    assert result["by_relation"]["quantified_action_link"]["positive_items"] == 7
    assert result["by_relation"]["cost_comparison_relation"]["positive_items"] == 4
    for relation in (
        "identity_authenticity_action_link",
        "privacy_restriction_action_link",
        "time_value_relation",
        "uncertainty_bound_relation",
    ):
        assert result["by_relation"][relation]["nondegenerate"] is False
        assert result["by_relation"][relation]["positive_items"] == 0


def test_runner_rejects_contamination_and_wrong_split() -> None:
    proposal, manifest, rows = _inputs()
    contaminated = copy.deepcopy(rows)
    contaminated[0]["judgement"] = 1
    with pytest.raises(ValueError, match="only item_key and ctext"):
        runner.run(proposal, manifest, contaminated)
    wrong_split = copy.deepcopy(rows)
    wrong_split[0]["item_key"] = "heldout_0001"
    with pytest.raises(ValueError, match="split or representation"):
        runner.run(proposal, manifest, wrong_split)
