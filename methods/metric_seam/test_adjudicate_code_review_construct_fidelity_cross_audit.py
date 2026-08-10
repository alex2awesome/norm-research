from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.adjudicate_code_review_construct_fidelity_cross_audit import (
    CrossAuditError,
    build_cross_audit,
    validate_cross_audit,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/"
    "code_review_construct_fidelity_v2.json"
)
ARTIFACT = (
    ROOT
    / "outputs/metric_seam_pilot/hierarchy_r123/"
    "code_review_construct_fidelity_independent_cross_audit_v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_complete_cross_audit_corrects_count_and_matched_depth() -> None:
    source = _load(SOURCE)
    audit = build_cross_audit(
        source,
        source_name="outputs/metric_seam_pilot/hierarchy_r123/code_review_construct_fidelity_v2.json",
    )
    assert audit["coverage"] == {
        "n_panel_cells": 90,
        "n_retrieved_reviewed": 68,
        "n_previously_accepted_reviewed": 56,
        "n_previously_mismatch_reviewed": 12,
        "n_unique_program_sources_reviewed": 33,
        "complete": True,
    }
    assert audit["before_summary"]["relation_local_static_fidelity_count"] == 56
    assert audit["after_summary"]["relation_local_static_fidelity_count"] == 50
    assert audit["before_summary"]["audited_depth_counts_eligible"] == {
        "1": 28,
        "2": 28,
    }
    assert audit["after_summary"]["audited_depth_counts_eligible"] == {
        "1": 25,
        "2": 25,
    }
    assert audit["after_summary"]["n_unique_eligible_programs"] == 26
    assert audit["n_guarded_changes"] == 7
    assert sum(review["matched_relation_depth"] is not None for review in audit["reviews"]) == 50


def test_all_program_sources_have_only_diff_text_code_channel() -> None:
    audit = build_cross_audit(_load(SOURCE))
    assert len(audit["program_source_audits"]) == 33
    for program in audit["program_source_audits"]:
        assert program["score_parameters"] == ["diff_text"]
        assert program["llm_or_judgment_field_accesses"] == []
        assert program["model_or_network_imports"] == []
    assert all(value is False for value in audit["sealed_inputs"].values())


def test_changes_are_guarded_and_only_expected_fields_change() -> None:
    audit = build_cross_audit(_load(SOURCE))
    expected_verdict_changes = {
        "TB::code-review::general::R1::merged_tree::171::33b7ed9b7e4e601644ef",
        "TB::code-review::general::R1::merged_tree::31::f239fc227b096b1638ef",
        "TB::code-review::general::R2::grandparent::43::12b78f2174bf884b965b",
        "TB::code-review::general::R2::grandparent::44::504809954bb08b978c5b",
        "TB::code-review::general::R2::merged_group::131::43ed2014b9a1669be3ca",
        "TB::code-review::general::R3::grandparent::3::681c2abce3bef33e3781",
    }
    verdict_changes = {
        change["cell_id"]
        for change in audit["changes"]
        if "verdict" in change["changed_fields"]
    }
    assert verdict_changes == expected_verdict_changes
    depth_change = next(
        change
        for change in audit["changes"]
        if change["cell_id"]
        == "TB::code-review::general::R2::merged_group::51::f8198da2d53d4f4219b2"
    )
    assert depth_change["before"]["audited_depth"] == 2
    assert depth_change["after"]["audited_depth"] == 1
    assert depth_change["changed_fields"] == ["audited_depth"]


def test_canonical_artifact_is_exact_guarded_rebuild() -> None:
    validate_cross_audit(_load(ARTIFACT), _load(SOURCE))


def test_artifact_tamper_is_rejected() -> None:
    artifact = _load(ARTIFACT)
    artifact["reviews"][0]["after"]["audited_depth"] = 4
    with pytest.raises(CrossAuditError, match="differs from guarded rebuild"):
        validate_cross_audit(artifact, _load(SOURCE))


def test_candidate_population_drift_is_rejected() -> None:
    source = copy.deepcopy(_load(SOURCE))
    row = next(row for row in source["rows"] if row.get("candidate"))
    row["candidate"]["aspect_id"] = "a999"
    with pytest.raises(CrossAuditError, match="retrieved population drift"):
        build_cross_audit(source)


def test_sealed_source_flag_violation_is_rejected() -> None:
    source = copy.deepcopy(_load(SOURCE))
    source["outcome_labels_loaded"] = True
    with pytest.raises(CrossAuditError, match="sealed flag outcome_labels_loaded"):
        build_cross_audit(source)
