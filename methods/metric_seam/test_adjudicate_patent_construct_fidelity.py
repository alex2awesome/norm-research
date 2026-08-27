from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.adjudicate_patent_construct_fidelity import build_audit


ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/patents_seed_map_v1.json"


def _seed_map():
    return json.loads(SEEDS.read_text(encoding="utf-8"))


def test_real_audit_is_90_cells_and_relation_depth_is_local():
    result = build_audit(_seed_map())
    assert result["n_cells"] == 90
    assert result["summary"]["verdict_counts"] == {
        "no_candidate": 84,
        "partial_relation_local": 6,
    }
    assert result["summary"]["n_exact_whole_construct"] == 0
    assert result["summary"]["n_pure_code_witnesses"] == 0
    assert result["summary"]["n_autonomous_unsupervised_discoveries"] == 0
    assert result["summary"]["maximum_matching_relation_depth_counts"] == {
        "1": 1,
        "3": 5,
    }
    utility = next(
        row for row in result["rows"] if row["metric_name"] == "Utility/industrial applicability"
    )
    assert utility["surrounding_program_depth"] == 3
    assert utility["eligible_relation_local_depths"] == [1]
    assert utility["maximum_matching_relation_depth"] == 1


def test_level_counts_follow_frozen_six_candidate_inventory():
    result = build_audit(_seed_map())
    assert result["summary"]["by_level"] == {
        "R1": {"n_cells": 30, "n_retrieved": 2, "n_partial_relation_local": 2},
        "R2": {"n_cells": 30, "n_retrieved": 1, "n_partial_relation_local": 1},
        "R3": {"n_cells": 30, "n_retrieved": 3, "n_partial_relation_local": 3},
    }


def test_extra_retrieved_candidate_fails_closed():
    source = _seed_map()
    poisoned = copy.deepcopy(source)
    row = next(item for item in poisoned["rows"] if item["selected_seed"] is None)
    row["selected_seed"] = copy.deepcopy(
        next(item for item in source["rows"] if item["selected_seed"])["selected_seed"]
    )
    row["decision"] = "candidate_seed_pending_independent_construct_fidelity_audit"
    with pytest.raises(ValueError, match="lacks a frozen adjudication"):
        build_audit(poisoned)


def test_candidate_family_change_fails_closed():
    poisoned = copy.deepcopy(_seed_map())
    row = next(
        item for item in poisoned["rows"] if item["metric_name"] == "Novelty requirement (statutory)"
    )
    row["selected_seed"]["aspect_id"] = "a60"
    with pytest.raises(ValueError, match="adjudication/seed mismatch"):
        build_audit(poisoned)


def test_unknown_outcome_fields_do_not_affect_static_adjudication():
    source = _seed_map()
    baseline = build_audit(source)
    poisoned = copy.deepcopy(source)
    for index, row in enumerate(poisoned["rows"]):
        row["judgement"] = index % 2
        row["rho"] = 0.999
        row["heldout_reconstruction"] = {"isomorphic": True}
    assert build_audit(poisoned) == baseline

