from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.adjudicate_science_claim_construct_fidelity import build_audit
from methods.metric_seam.hierarchy_science_claim_seed_mapper import (
    build_capability_inventory,
    build_seed_map,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
SCIENCE = ROOT / "methods/metric_seam/science_claims_v2"


def _seed_map():
    panel = json.loads((BASE / "panel_v3.json").read_text(encoding="utf-8"))
    capability = build_capability_inventory(
        SCIENCE / "core.py",
        SCIENCE / "core_relation_strict.py",
        repo_root=ROOT,
    )
    return build_seed_map(panel, capability)


def test_static_audit_exact_counts_and_levels():
    result = build_audit(_seed_map())
    assert result["status"] == "static-relation-local-adjudication-complete-pre-execution"
    assert result["summary"]["verdict_counts"] == {
        "no_candidate": 81,
        "partial_relation_local": 6,
        "relation_mismatch": 3,
    }
    assert result["summary"]["by_level"] == {
        "R1": {
            "n_cells": 30,
            "n_retrieved": 2,
            "n_partial_relation_local": 2,
            "n_relation_mismatch": 0,
        },
        "R2": {
            "n_cells": 30,
            "n_retrieved": 2,
            "n_partial_relation_local": 2,
            "n_relation_mismatch": 0,
        },
        "R3": {
            "n_cells": 30,
            "n_retrieved": 5,
            "n_partial_relation_local": 2,
            "n_relation_mismatch": 3,
        },
    }
    assert result["summary"]["n_partial_relation_local"] == 6
    assert result["summary"]["n_relation_mismatch"] == 3
    assert result["summary"]["n_exact_whole_construct"] == 0
    assert result["summary"]["n_execution_witnesses"] == 0
    assert result["summary"]["n_external_scientific_truth_claims"] == 0
    assert result["summary"]["n_automatic_discoveries"] == 0


def test_every_pass_audits_all_five_relation_dimensions_and_depth():
    result = build_audit(_seed_map())
    passed = [row for row in result["rows"] if row["verdict"] == "partial_relation_local"]
    assert len(passed) == 6
    for row in passed:
        assert row["object_assessment"]
        assert row["relation_assessment"]
        assert row["polarity_assessment"]
        assert row["applicability_assessment"]
        assert row["aggregation_assessment"]
        assert row["eligible_relation_local_depths"] == [3]
        assert row["maximum_matching_relation_depth"] == 3
        assert row["surrounding_relation_chain_depth"] == 3
        assert row["static_pure_code_capability"] is True
        assert row["eligible_for_later_relation_local_execution"] is True
        assert row["exact_whole_construct_fidelity"] is False
        assert row["execution_witness_established"] is False
        assert row["external_scientific_truth_established"] is False
        assert row["automatic_discovery"] is False
        assert len(row["matched_subrelations"]) == 1
        assert row["matched_subrelations"][0]["effective_code_depth"] == 3


def test_adjacent_but_wrong_objects_receive_no_depth_credit():
    result = build_audit(_seed_map())
    mismatches = {
        row["metric_name"]: row
        for row in result["rows"]
        if row["verdict"] == "relation_mismatch"
    }
    assert set(mismatches) == {
        "Citation practice quality, coverage, and ethics",
        "Causal inference and generalization claims rigor",
        "Discussion and conclusions — interpretation, balance, and implications",
    }
    for row in mismatches.values():
        assert row["matched_subrelations"] == []
        assert row["eligible_relation_local_depths"] == []
        assert row["maximum_matching_relation_depth"] is None
        assert row["eligible_for_later_relation_local_execution"] is False


def test_audit_is_static_and_keeps_certificate_scope_narrow():
    result = build_audit(_seed_map())
    assert result["execution_performed"] is False
    assert result["articles_or_items_loaded"] is False
    assert result["reference_values_loaded"] is False
    assert result["outcome_labels_loaded"] is False
    assert result["historical_certificates_or_program_outputs_loaded"] is False
    assert result["prompt_or_reconstruction_outputs_loaded"] is False
    limits = " ".join(result["interpretation_limits"])
    assert "document-internal consistency is not external scientific truth" in limits
    assert "not whole-criterion codability" in limits
    assert "bounded non-discovery" in limits


def test_frozen_adjudication_must_cover_exact_retrieved_set():
    seed_map = _seed_map()
    poisoned = copy.deepcopy(seed_map)
    row = next(row for row in poisoned["rows"] if row["selected_seed"] is None)
    template = next(row for row in poisoned["rows"] if row["selected_seed"] is not None)
    row["selected_seed"] = copy.deepcopy(template["selected_seed"])
    with pytest.raises(ValueError, match="lacks a frozen adjudication"):
        build_audit(poisoned)


def test_checked_in_audit_artifact_is_exact_builder_output():
    expected = build_audit(_seed_map())
    observed = json.loads(
        (BASE / "peer_review_science_claim_construct_fidelity_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert observed == expected
