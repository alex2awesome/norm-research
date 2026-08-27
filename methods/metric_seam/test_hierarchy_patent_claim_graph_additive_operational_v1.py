from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_patent_claim_graph_additive_operational_v1 import (
    build_summary,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "metric_seam_pilot" / "hierarchy_r123"
AUDIT = OUT / "patents_claim_graph_additive_construct_fidelity_v1.json"
FREEZE = OUT / "patents_claim_graph_additive_train_freeze_v1.json"
TRAIN = OUT / "patents_claim_graph_additive_compiler_train_v2.json"
HELDOUT = OUT / "patents_claim_graph_additive_heldout_pre_reference_v1.json"
ARTIFACT = OUT / "patents_claim_graph_additive_operational_summary_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build() -> dict:
    return build_summary(
        _load(AUDIT), _load(FREEZE), _load(TRAIN), _load(HELDOUT)
    )


def test_operational_funnel_stops_at_finite_witness_incidence() -> None:
    result = _build()
    assert result["summary"] == {
        "n_static_partial_cells": 8,
        "n_train_selected_cells": 8,
        "n_heldout_operational_cells": 7,
        "n_static_relation_mappings": 11,
        "n_heldout_operational_relation_mappings": 8,
        "heldout_operational_cells_by_level": {"R1": 1, "R2": 3, "R3": 3},
        "heldout_operational_cell_depth_counts": {"2": 3, "3": 4},
        "heldout_relation_status_mapping_counts": {
            "heldout_bounded_non_discovery": 3,
            "heldout_observed": 2,
            "heldout_observed_dense": 2,
            "heldout_observed_sparse": 4,
        },
        "train_items": 150,
        "heldout_items": 150,
        "train_items_at_character_cap": 119,
        "heldout_items_at_character_cap": 123,
        "whole_construct_cells": 0,
    }
    assert result["design"]["heldout_readout"] == (
        "finite relation witness incidence only"
    )


def test_formula_only_cell_is_bounded_non_discovery_not_tacitness() -> None:
    formula = _build()["relation_status"][
        "formula_variable_definition_alignment"
    ]
    assert formula["n_train_items_with_finite_certificates"] == 1
    assert formula["n_heldout_items_with_finite_certificates"] == 0
    assert formula["heldout_status"] == "heldout_bounded_non_discovery"
    r1_formula = next(
        row
        for row in _build()["cells"]
        if (row["level"], row["selection_rank"]) == ("R1", 26)
    )
    assert r1_formula["status"] == "heldout_bounded_non_discovery"
    assert not r1_formula["heldout_operational"]
    assert _build()["claim_limits"]["negative_result_establishes_tacitness"] is False


def test_sparse_numeric_relation_is_not_inflated() -> None:
    numeric = _build()["relation_status"]["numeric_constraint_definition_graph"]
    assert numeric["n_train_items_with_finite_certificates"] == 1
    assert numeric["n_heldout_items_with_finite_certificates"] == 1
    assert numeric["heldout_status"] == "heldout_observed_sparse"
    assert sum(
        row["relation_id"] == "numeric_constraint_definition_graph"
        and row["heldout_relation_operational"]
        for row in _build()["mappings"]
    ) == 3


def test_train_and_heldout_are_source_hash_bound_and_blind() -> None:
    result = _build()
    assert result["design"][
        "same_program_and_runner_source_hashes_train_to_heldout"
    ] is True
    assert result["design"]["compiler_train_selection_frozen_before_heldout"] is True
    assert result["design"]["outcome_or_reference_values_loaded"] is False
    assert result["design"]["model_or_api_calls_made"] is False
    assert result["design"]["accelerators_used"] is False

    heldout = deepcopy(_load(HELDOUT))
    heldout["sources"]["program"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="program provenance"):
        build_summary(_load(AUDIT), _load(FREEZE), _load(TRAIN), heldout)


def test_claim_limits_do_not_report_codability_or_isomorphism() -> None:
    limits = _build()["claim_limits"]
    assert limits == {
        "finite_witness_operational_is_not_reconstruction": True,
        "codability_claim_permitted": False,
        "prompt_articulability_measured": False,
        "reference_reconstruction_measured": False,
        "isomorphism_measured": False,
        "absence_from_full_patent_or_claim_set_established": False,
        "negative_result_establishes_tacitness": False,
    }


def test_checked_in_operational_summary_equals_fresh_build() -> None:
    assert _load(ARTIFACT) == _build()
