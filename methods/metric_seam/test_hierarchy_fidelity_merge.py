from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_fidelity_merge import FidelityAuditError, merge_audits


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def _inputs():
    seed = _load("code_review_seed_map_v2.json")
    audits = [
        _load(f"code_review_construct_fidelity_{level}_v1.json")
        for level in ("R1", "R2", "R3")
    ]
    adjudication = _load("code_review_construct_fidelity_adjudication_v1.json")
    return seed, audits, adjudication


def test_real_independent_level_audits_merge_to_canonical_90_rows():
    seed, audits, adjudication = _inputs()
    merged = merge_audits(seed, audits, adjudication=adjudication)
    assert len(merged["rows"]) == 90
    assert merged["summary"]["relation_local_static_fidelity_count"] == 56
    assert merged["summary"]["whole_construct_exact_count"] == 0
    assert merged["n_adjudicated_changes"] == 12
    assert all(
        row["candidate"] is not None
        for row in merged["rows"] if row["eligible_for_relation_local_execution"]
    )


def test_candidate_identity_cannot_drift_after_static_retrieval():
    seed, audits, _adjudication = _inputs()
    poisoned = copy.deepcopy(audits)
    row = next(row for row in poisoned[1]["rows"] if row["candidate"] is not None)
    row["candidate"]["aspect_id"] = "a999"
    with pytest.raises(FidelityAuditError, match="does not match seed"):
        merge_audits(seed, poisoned)


def test_partial_never_becomes_whole_construct_or_exact():
    seed, audits, _adjudication = _inputs()
    poisoned = copy.deepcopy(audits)
    row = next(row for row in poisoned[2]["rows"] if row["verdict"] == "partial")
    row["scope"] = "whole_construct"
    with pytest.raises(FidelityAuditError, match="requires scope subrelation_only"):
        merge_audits(seed, poisoned)


def test_abstention_cannot_be_reframed_as_tacitness_or_a_program():
    seed, audits, _adjudication = _inputs()
    poisoned = copy.deepcopy(audits)
    row = next(
        row for row in poisoned[0]["rows"]
        if row["verdict"] == "no_candidate_bounded_non_discovery"
    )
    row["verdict"] = "exact"
    row["scope"] = "whole_construct"
    row["eligible_for_relation_local_execution"] = True
    row["audited_depth"] = 1
    with pytest.raises(FidelityAuditError, match="abstained seed"):
        merge_audits(seed, poisoned)


def test_cross_adjudication_is_before_value_guarded():
    seed, audits, adjudication = _inputs()
    poisoned = copy.deepcopy(adjudication)
    poisoned["changes"][0]["before"]["audited_depth"] = 4
    with pytest.raises(FidelityAuditError, match="before-value drift"):
        merge_audits(seed, audits, adjudication=poisoned)
