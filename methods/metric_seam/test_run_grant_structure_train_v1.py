from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam import run_grant_structure_train_v1 as runner


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _inputs() -> tuple[dict, dict, list[dict]]:
    proposal = json.loads(
        (BASE / "grant_structure_static_proposal_v1.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (BASE / "items_v2/grant-funding/manifest.json").read_text(encoding="utf-8")
    )
    rows = json.loads(
        (BASE / "items_v2/grant-funding/compiler_train.json").read_text(encoding="utf-8")
    )
    return proposal, manifest, rows


def test_train_execution_is_label_free_and_promotes_no_unaudited_mapping() -> None:
    result = runner.run(*_inputs())
    assert result["status"] == (
        "compiler_train_exploratory_complete_pending_independent_construct_audit"
    )
    assert result["phase"] == "compiler_train"
    assert result["blindness"] == {
        "input_fields_passed_to_program": ["ctext"],
        "reference_fields_passed_to_program": False,
        "outcome_fields_passed_to_program": False,
        "heldout_items_or_outputs_loaded": False,
        "external_supervised_anchor_used": False,
        "model_or_api_used": False,
        "accelerator_used": False,
        "credentials_required": False,
    }
    assert result["summary"] == {
        "items": 103,
        "relations_executed": 13,
        "nondegenerate_relations": 12,
        "nondegenerate_relation_ids": result["summary"][
            "nondegenerate_relation_ids"
        ],
        "hierarchy_mappings_promoted": 0,
        "heldout_execution_authorized": False,
        "prompt_articulability_measurements": 0,
        "reconstruction_measurements": 0,
        "isomorphism_measurements": 0,
    }
    assert "budget_sum_consistency" not in result["summary"][
        "nondegenerate_relation_ids"
    ]


def test_train_execution_reports_measurability_without_construct_credit() -> None:
    result = runner.run(*_inputs())
    assert result["by_relation"]["budget_sum_consistency"] == {
        "depth": 3,
        "status_counts": {"abstained": 102, "measured": 1},
        "measured": 1,
        "unique_finite_scores": 1,
        "minimum": 1.0,
        "maximum": 1.0,
        "nondegenerate": False,
    }
    assert result["by_relation"]["aim_hypothesis_experiment_graph"][
        "nondegenerate"
    ] is True
    assert result["by_relation"]["document_outline_structure"]["depth"] == 1
    assert result["by_relation"]["risk_mitigation_graph"]["depth"] == 2
    assert len(result["rows"]) == 103
    assert all(set(row) == {"item_key", "input_characters", "relations"} for row in result["rows"])


def test_runner_rejects_label_fields_and_wrong_split_rows() -> None:
    proposal, manifest, rows = _inputs()
    contaminated = copy.deepcopy(rows)
    contaminated[0]["outcome"] = 1
    with pytest.raises(ValueError, match="only item_key and ctext"):
        runner.run(proposal, manifest, contaminated)

    wrong_split = copy.deepcopy(rows)
    wrong_split[0]["item_key"] = "heldout_0001"
    with pytest.raises(ValueError, match="split or representation"):
        runner.run(proposal, manifest, wrong_split)


def test_program_ast_receipt_forbids_file_network_process_and_environment_access() -> None:
    result = runner.run(*_inputs())
    receipt = result["program"]["ast_restriction_receipt"]
    assert receipt == {
        "ast_parsed": True,
        "import_roots": ["__future__", "dataclasses", "decimal", "re", "typing"],
        "file_io_calls_allowed": False,
        "network_process_environment_imports_allowed": False,
    }

