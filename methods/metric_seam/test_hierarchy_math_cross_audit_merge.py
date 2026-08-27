from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_math_cross_audit_merge import merge_cross_audits


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
SOURCE_PATHS = [
    BASE / "math_stackexchange_construct_fidelity_R1_R2_v1.json",
    BASE / "math_stackexchange_construct_fidelity_R3_v1.json",
]
OVERLAY_PATHS = [
    BASE / "math_stackexchange_construct_fidelity_R1_R2_cross_adjudication_v1.json",
    BASE / "math_stackexchange_construct_fidelity_R3_cross_adjudication_v1.json",
]


def _inputs():
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in SOURCE_PATHS]
    overlays = [json.loads(path.read_text(encoding="utf-8")) for path in OVERLAY_PATHS]
    return sources, overlays


def test_real_cross_audits_merge_to_complete_47_candidate_review():
    result = merge_cross_audits(*_inputs())
    assert result["review_coverage"] == {
        "source_rows": 90,
        "retrieved_candidates_reviewed": 47,
        "changed_rows": 21,
        "unchanged_retrieved_rows": 26,
        "all_retrieved_candidates_reviewed": True,
    }
    assert result["before_counts"] == {
        "retrieved_candidates": 47,
        "retrieved_verdicts": {"mismatch": 13, "partial": 34},
        "retrieved_depths": {"1": 20, "2": 27},
        "eligible_depths": {"1": 10, "2": 24},
        "eligible_for_relation_local_execution": 34,
    }
    assert result["after_counts_if_overlay_applied"] == {
        "retrieved_candidates": 47,
        "retrieved_verdicts": {"mismatch": 14, "partial": 33},
        "retrieved_depths": {"1": 20, "2": 27},
        "eligible_depths": {"1": 10, "2": 23},
        "eligible_for_relation_local_execution": 33,
    }
    assert result["candidate_execution_performed"] is False
    assert result["model_or_api_calls_performed"] is False
    assert result["accelerators_used"] is False


def test_path_records_bind_both_source_pairs():
    sources, overlays = _inputs()
    result = merge_cross_audits(
        sources,
        overlays,
        source_audit_paths=SOURCE_PATHS,
        overlay_paths=OVERLAY_PATHS,
    )
    assert [row["levels"] for row in result["source_records"]] == [["R1", "R2"], ["R3"]]
    assert all(len(row["source_audit"]["sha256"]) == 64 for row in result["source_records"])
    assert all(len(row["source_overlay"]["sha256"]) == 64 for row in result["source_records"])


def test_level_overlap_or_incomplete_coverage_fails_closed():
    sources, overlays = copy.deepcopy(_inputs())
    overlays[1]["levels"] = ["R2"]
    with pytest.raises(ValueError, match="disjoint"):
        merge_cross_audits(sources, overlays)


def test_each_source_validator_remains_authoritative():
    sources, overlays = copy.deepcopy(_inputs())
    overlays[0]["changes"][0]["candidate_guard"]["program_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        merge_cross_audits(sources, overlays)

    sources, overlays = copy.deepcopy(_inputs())
    overlays[1]["forbidden_inputs_used"] = True
    with pytest.raises(ValueError):
        merge_cross_audits(sources, overlays)

