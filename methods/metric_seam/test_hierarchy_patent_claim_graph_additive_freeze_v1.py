from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_patent_claim_graph_additive_freeze_v1 import (
    build_freeze,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "metric_seam_pilot" / "hierarchy_r123"
AUDIT = OUT / "patents_claim_graph_additive_construct_fidelity_v1.json"
TRAIN = OUT / "patents_claim_graph_additive_compiler_train_v2.json"
SUPERSEDED = OUT / "patents_claim_graph_additive_compiler_train_v1.json"
ARTIFACT = OUT / "patents_claim_graph_additive_train_freeze_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build() -> dict:
    return build_freeze(_load(AUDIT), _load(TRAIN), _load(SUPERSEDED))


def test_freeze_selects_all_source_audited_mappings_with_observed_train_witness() -> None:
    result = _build()
    assert result["summary"] == {
        "n_static_partial_cells": 8,
        "n_static_relation_mappings": 11,
        "n_selected_cells": 8,
        "n_selected_relation_mappings": 11,
        "selected_cells_by_level": {"R1": 2, "R2": 3, "R3": 3},
        "selected_cell_maximum_depth_counts": {"2": 3, "3": 5},
        "train_status_mapping_counts": {
            "train_observed": 2,
            "train_observed_dense": 2,
            "train_observed_sparse": 7,
        },
        "whole_construct_cells": 0,
    }
    assert len(result["mappings"]) == 11
    assert all(row["selected_for_heldout_pre_reference"] for row in result["mappings"])
    assert result["relation_train_status"]["formula_variable_definition_alignment"][
        "n_train_items_with_finite_certificates"
    ] == 1
    assert result["relation_train_status"]["numeric_constraint_definition_graph"][
        "status"
    ] == "train_observed_sparse"


def test_freeze_contract_has_no_heldout_or_reference_channel() -> None:
    design = _build()["design"]
    assert design["heldout_text_loaded"] is False
    assert design["outcome_or_reference_values_loaded"] is False
    assert design["prompt_outputs_loaded"] is False
    assert design["external_supervision_used"] is False
    assert design["threshold_or_weight_fitting_performed"] is False
    assert "heldout" not in " ".join(design["selection_inputs"])


def test_prefreeze_defect_is_preserved_and_never_reached_heldout() -> None:
    incident = _build()["prefreeze_supersession_incident"]
    assert incident["heldout_was_run_under_superseded_program"] is False
    assert "overwrote the certificate kind" in incident["defect"]
    assert "sole input to this freeze" in incident["disposition"]


def test_freeze_fails_closed_on_reference_or_execution_failure() -> None:
    train = deepcopy(_load(TRAIN))
    train["design"]["outcome_or_reference_values_loaded"] = True
    with pytest.raises(ValueError, match="blind pure-code"):
        build_freeze(_load(AUDIT), train, _load(SUPERSEDED))

    train = deepcopy(_load(TRAIN))
    train["summary"]["failure_types"] = {"RuntimeError": 1}
    with pytest.raises(ValueError, match="program failures"):
        build_freeze(_load(AUDIT), train, _load(SUPERSEDED))


def test_claim_limits_remain_non_isomorphic_and_non_codability() -> None:
    limits = _build()["claim_limits"]
    assert limits == {
        "codability_claim_permitted": False,
        "prompt_articulability_measured": False,
        "reference_reconstruction_measured": False,
        "isomorphism_measured": False,
        "negative_result_establishes_tacitness": False,
    }


def test_checked_in_freeze_equals_fresh_train_only_build() -> None:
    assert _load(ARTIFACT) == _build()
