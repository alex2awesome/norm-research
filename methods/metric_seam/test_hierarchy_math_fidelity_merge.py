from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_math_fidelity_merge import (
    MathFidelityError,
    merge_math_audits,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def _inputs():
    return (
        _load("panel_v3.json"),
        _load("math_stackexchange_seed_map_v1.json"),
        [
            _load("math_stackexchange_construct_fidelity_R1_R2_v1.json"),
            _load("math_stackexchange_construct_fidelity_R3_v1.json"),
        ],
    )


def _overlay() -> dict:
    return _load(
        "math_stackexchange_construct_fidelity_cross_adjudication_merged_v1.json"
    )


def test_real_math_audits_merge_provisionally_without_execution():
    result = merge_math_audits(*_inputs())
    assert result["status"] == "provisional_static_merge_pending_cross_audit"
    assert result["cross_audit"] == {
        "status": "pending_independent_cross_audit",
        "n_guarded_changes": 0,
        "provisional_until_complete": True,
    }
    assert result["execution_performed"] is False
    assert result["items_loaded"] is False
    assert result["reference_values_loaded"] is False
    assert result["outcome_labels_loaded"] is False
    assert result["program_outputs_loaded"] is False
    summary = result["summary"]
    assert summary["n_cells"] == 90
    assert summary["n_retrieved_candidates"] == 47
    assert summary["eligible_for_relation_local_execution"] == 34
    assert summary["whole_construct_exact_count"] == 0
    assert summary["eligible_audited_depths"] == {"1": 10, "2": 24}
    assert {
        level: summary["by_level"][level]["eligible_for_relation_local_execution"]
        for level in ("R1", "R2", "R3")
    } == {"R1": 13, "R2": 6, "R3": 15}


def test_level_audit_top_row_and_candidate_shapes_fail_closed():
    panel, seed, audits = copy.deepcopy(_inputs())
    audits[0]["unexpected"] = True
    with pytest.raises(MathFidelityError, match="top-level shape"):
        merge_math_audits(panel, seed, audits)

    panel, seed, audits = copy.deepcopy(_inputs())
    audits[0]["rows"][0]["unexpected"] = True
    with pytest.raises(MathFidelityError, match="row shape"):
        merge_math_audits(panel, seed, audits)

    panel, seed, audits = copy.deepcopy(_inputs())
    row = next(row for row in audits[0]["rows"] if row["candidate"] is not None)
    row["candidate"]["unexpected"] = True
    with pytest.raises(MathFidelityError, match="candidate shape"):
        merge_math_audits(panel, seed, audits)


def test_panel_binding_disjoint_levels_and_exact_cell_closure_fail_closed():
    panel, seed, audits = copy.deepcopy(_inputs())
    seed["panel_content_sha256"] = "wrong"
    with pytest.raises(MathFidelityError, match="another panel"):
        merge_math_audits(panel, seed, audits)

    panel, seed, audits = copy.deepcopy(_inputs())
    for audit in audits:
        audit["source_candidate_map"] = "another_seed_map.json"
    with pytest.raises(MathFidelityError, match="another seed map"):
        merge_math_audits(panel, seed, audits)

    panel, seed, audits = copy.deepcopy(_inputs())
    audits[1]["levels"] = ["R2"]
    for row in audits[1]["rows"]:
        row["level"] = "R2"
    with pytest.raises(MathFidelityError, match="overlap"):
        merge_math_audits(panel, seed, audits)

    panel, seed, audits = copy.deepcopy(_inputs())
    audits[1]["rows"][0]["cell_id"] = audits[1]["rows"][1]["cell_id"]
    with pytest.raises(MathFidelityError, match="identities do not close"):
        merge_math_audits(panel, seed, audits)


def test_program_and_ops_source_digests_fail_closed():
    panel, seed, audits = copy.deepcopy(_inputs())
    row = next(row for row in audits[0]["rows"] if row["candidate"] is not None)
    row["candidate"]["program_sha256"] = "0" * 64
    with pytest.raises(MathFidelityError, match="candidate source digest"):
        merge_math_audits(panel, seed, audits)

    panel, seed, audits = copy.deepcopy(_inputs())
    for audit in audits:
        audit["ops_math_sha256"] = "0" * 64
    with pytest.raises(MathFidelityError, match="ops_math source digest"):
        merge_math_audits(panel, seed, audits)


def test_execution_forbidden_inputs_and_source_counts_fail_closed():
    panel, seed, audits = copy.deepcopy(_inputs())
    audits[0]["execution_performed"] = True
    with pytest.raises(MathFidelityError, match="execution state"):
        merge_math_audits(panel, seed, audits)

    panel, seed, audits = copy.deepcopy(_inputs())
    for audit in audits:
        audit["forbidden_inputs"] = ["outcomes were loaded"]
    with pytest.raises(MathFidelityError, match="forbidden-input contract"):
        merge_math_audits(panel, seed, audits)

    panel, seed, audits = copy.deepcopy(_inputs())
    audits[0]["counts"]["overall"]["n_retrieved_candidates"] += 1
    with pytest.raises(MathFidelityError, match="counts drifted"):
        merge_math_audits(panel, seed, audits)


def test_complete_guarded_overlay_reproduces_cross_audited_static_counts():
    panel, seed, audits = _inputs()
    result = merge_math_audits(panel, seed, audits, overlay=_overlay())
    assert result["status"] == "static_construct_fidelity_complete_pre_execution"
    assert result["cross_audit"] == {
        "status": "complete", "n_guarded_changes": 21,
        "provisional_until_complete": False,
    }
    assert result["summary"]["eligible_for_relation_local_execution"] == 33
    assert result["summary"]["verdicts"] == {
        "mismatch": 14,
        "no_candidate_bounded_non_discovery": 43,
        "partial": 33,
    }
    assert result["summary"]["eligible_audited_depths"] == {"1": 10, "2": 23}
    assert {
        level: result["summary"]["by_level"][level]
        ["eligible_for_relation_local_execution"]
        for level in ("R1", "R2", "R3")
    } == {"R1": 12, "R2": 6, "R3": 15}


def test_guarded_overlay_fails_closed_on_stale_before_or_candidate_identity():
    panel, seed, audits = _inputs()
    overlay = _overlay()
    overlay["changes"][0]["before"][
        next(iter(overlay["changes"][0]["before"]))
    ] = "stale test value"
    with pytest.raises(MathFidelityError, match="before-value drift"):
        merge_math_audits(panel, seed, audits, overlay=overlay)

    overlay = _overlay()
    overlay["changes"][0]["candidate_guard"]["aspect_id"] = "another-program"
    with pytest.raises(MathFidelityError, match="candidate guard drifted"):
        merge_math_audits(panel, seed, audits, overlay=overlay)


def test_overlay_cannot_be_unsealed_or_leave_incoherent_verdict_state():
    panel, seed, audits = _inputs()
    overlay = _overlay()
    verdict_change = next(
        change for change in overlay["changes"] if "verdict" in change["after"]
    )
    verdict_change["after"]["verdict"] = "partial"
    with pytest.raises(MathFidelityError, match="incoherent verdict"):
        merge_math_audits(panel, seed, audits, overlay=overlay)

    overlay = _overlay()
    overlay["forbidden_inputs_used"] = True
    with pytest.raises(MathFidelityError, match="not a sealed static audit"):
        merge_math_audits(panel, seed, audits, overlay=overlay)
