from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_code_review_corrected_funnel import (
    CorrectedFunnelError,
    build_corrected_funnel,
    validate_corrected_funnel,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
PATHS = {
    "panel": BASE / "panel_v3.json",
    "fidelity": BASE / "code_review_construct_fidelity_v2.json",
    "cross_audit": BASE / "code_review_construct_fidelity_independent_cross_audit_v1.json",
    "train_gate": BASE / "code_review_train_gate_v1.json",
    "heldout": BASE / "code_review_heldout_readiness_v1.json",
    "prevalence": BASE / "code_review_witness_prevalence_v3.json",
    "artifact": BASE / "code_review_corrected_funnel_v1.json",
}


def _load(key: str) -> dict:
    return json.loads(PATHS[key].read_text(encoding="utf-8"))


def _build(**overrides: dict) -> dict:
    values = {
        key: _load(key)
        for key in ("panel", "fidelity", "cross_audit", "train_gate", "heldout", "prevalence")
    }
    values.update(overrides)
    return build_corrected_funnel(**values)


def test_corrected_balanced_funnel_and_level_counts() -> None:
    artifact = _build()
    stages = artifact["corrected_readout"]["stages"]
    assert stages["retrieved_candidate"]["balanced_panel"]["n_positive"] == 68
    assert stages["relation_local_static_fidelity"]["balanced_panel"] == {
        "n_positive": 50,
        "denominator": 90,
        "rate": 0.555556,
    }
    assert stages["train_operational_relation_witness"]["balanced_panel"] == {
        "n_positive": 27,
        "denominator": 90,
        "rate": 0.3,
    }
    assert stages["heldout_confirmatory_reconstruction_evaluable"]["balanced_panel"] == {
        "n_positive": 18,
        "denominator": 90,
        "rate": 0.2,
    }
    by_level = artifact["corrected_readout"]["by_level"]
    expected = {
        "R1": (14, 9, 7),
        "R2": (15, 6, 4),
        "R3": (21, 12, 7),
    }
    for level, counts in expected.items():
        observed = tuple(
            by_level[level][stage]["balanced_panel"]["n_positive"]
            for stage in (
                "relation_local_static_fidelity",
                "train_operational_relation_witness",
                "heldout_confirmatory_reconstruction_evaluable",
            )
        )
        assert observed == counts


def test_corrected_depth_counts_include_relation_local_depth_change() -> None:
    by_depth = _build()["corrected_readout"]["by_depth"]
    assert {
        depth: row["n_positive"]
        for depth, row in by_depth["relation_local_static_fidelity"].items()
    } == {"1": 25, "2": 25}
    assert {
        depth: row["n_positive"]
        for depth, row in by_depth["train_operational_relation_witness"].items()
    } == {"1": 19, "2": 8}
    assert {
        depth: row["n_positive"]
        for depth, row in by_depth[
            "heldout_confirmatory_reconstruction_evaluable"
        ].items()
    } == {"1": 12, "2": 6}


def test_conditional_inventory_expansion_is_recomputed() -> None:
    comparisons = _build()["before_after"]
    assert comparisons["relation_local_static_fidelity"] == {
        "balanced_n_before": 56,
        "balanced_n_after": 50,
        "balanced_rate_before": 0.622222,
        "balanced_rate_after": 0.555556,
        "balanced_rate_change": -0.066666,
        "expanded_rate_before": 0.478369,
        "expanded_rate_after": 0.418085,
        "expanded_rate_change": -0.060284,
    }
    assert comparisons["train_operational_relation_witness"]["expanded_rate_after"] == 0.231028
    assert comparisons[
        "heldout_confirmatory_reconstruction_evaluable"
    ]["expanded_rate_after"] == 0.160461


def test_exact_historical_operational_and_heldout_removals() -> None:
    artifact = _build()
    expected = {
        (
            "TB::code-review::general::R1::merged_tree::171::33b7ed9b7e4e601644ef",
            "a8",
        ),
        (
            "TB::code-review::general::R2::merged_group::131::43ed2014b9a1669be3ca",
            "a131",
        ),
        (
            "TB::code-review::general::R3::grandparent::3::681c2abce3bef33e3781",
            "a131",
        ),
    }
    for stage in ("train_operational", "heldout_confirmatory"):
        observed = {
            (row["cell_id"], row["candidate_aspect_id"])
            for row in artifact["removed_mappings"][stage]
        }
        assert observed == expected
    assert artifact["program_counts"] == {
        "static_unique_eligible_before": 30,
        "static_unique_eligible_after": 26,
        "train_selected_before": 16,
        "train_selected_after_static_filter": 14,
        "heldout_confirmatory_before": 12,
        "heldout_confirmatory_after_static_filter": 10,
    }


def test_documentation_ia_depth_change_stays_train_only() -> None:
    correction = _build()["depth_corrections"][0]
    assert correction["candidate_aspect_id"] == "a52"
    assert correction["before_depth"] == 2
    assert correction["after_matched_relation_depth"] == 1
    assert correction["train_operational_after"] is True
    assert correction["heldout_confirmatory_after"] is False


def test_canonical_artifact_is_exact_guarded_rebuild() -> None:
    validate_corrected_funnel(
        _load("artifact"),
        _load("panel"),
        _load("fidelity"),
        _load("cross_audit"),
        _load("train_gate"),
        _load("heldout"),
        _load("prevalence"),
    )


def test_cross_audit_tamper_is_rejected() -> None:
    cross_audit = copy.deepcopy(_load("cross_audit"))
    cross_audit["reviews"][0]["after"]["audited_depth"] = 4
    with pytest.raises(CorrectedFunnelError, match="failed validation"):
        _build(cross_audit=cross_audit)


def test_train_reference_flag_violation_is_rejected() -> None:
    train_gate = copy.deepcopy(_load("train_gate"))
    train_gate["reference_values_used"] = True
    with pytest.raises(CorrectedFunnelError, match="train gate violates sealed flag"):
        _build(train_gate=train_gate)
