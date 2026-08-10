from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_math_lclamp_operational_summary import (
    EXPANSION_KEY,
    MathLClampSummaryError,
    build_math_lclamp_operational_summary,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
NAMES = (
    "panel_v3.json",
    "math_stackexchange_construct_fidelity_merged_v1.json",
    "math_stackexchange_lclamp_compiler_train_v1.json",
    "math_stackexchange_lclamp_train_profile_gate_v1.json",
    "math_stackexchange_lclamp_heldout_pre_reference_v1.json",
)


def _inputs() -> list[dict]:
    return [json.loads((BASE / name).read_text(encoding="utf-8")) for name in NAMES]


@pytest.fixture(scope="module")
def canonical() -> tuple[list[dict], dict]:
    inputs = _inputs()
    return inputs, build_math_lclamp_operational_summary(*inputs)


def test_canonical_funnel_closes_at_all_four_boundaries(canonical):
    _inputs, result = canonical
    assert result["status"] == "complete_static_train_and_pre_reference_heldout_funnel"
    validation = result["validation"]
    assert validation["construct_fidelity"] == {
        "n_cells": 90,
        "n_static_relation_mappings": 33,
        "n_unique_programs": 16,
        "cross_audit": {
            "status": "complete",
            "n_guarded_changes": 21,
            "provisional_until_complete": False,
        },
    }
    assert validation["compiler_train"]["n_profile_runs"] == 240
    assert validation["compiler_train"]["three_state_totals"] == {
        "measured": 36_000,
        "abstained": 0,
        "failed": 0,
    }
    assert validation["train_only_profile_gate"]["n_selected_programs"] == 16
    assert validation["train_only_profile_gate"]["n_selected_relation_mappings"] == 33
    assert validation["heldout_pre_reference"]["n_profile_runs"] == 16
    assert validation["heldout_pre_reference"]["three_state_totals"] == {
        "measured": 2_400,
        "abstained": 0,
        "failed": 0,
    }
    assert set(validation["stage_relation_mapping_counts"].values()) == {33}


def test_banked_canonical_artifact_is_exact_rebuild(canonical):
    inputs, _result = canonical
    source_names = (
        "panel",
        "construct_fidelity",
        "compiler_train_execution",
        "train_profile_gate",
        "heldout_pre_reference_execution",
    )
    sources = {
        label: {
            "path": str(Path("outputs/metric_seam_pilot/hierarchy_r123") / filename),
            "sha256": hashlib.sha256((BASE / filename).read_bytes()).hexdigest(),
        }
        for label, filename in zip(source_names, NAMES)
    }
    expected = build_math_lclamp_operational_summary(*inputs, sources=sources)
    observed = json.loads(
        (BASE / "math_stackexchange_lclamp_operational_prevalence_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert observed == expected


def test_balanced_and_conditional_expansion_are_stage_explicit(canonical):
    _inputs, result = canonical
    pooled = result["pooled_eligible_action_nodes"]
    for estimates in (pooled["balanced_panel"], pooled[EXPANSION_KEY]):
        assert len(estimates) == 3
        assert len({row["rate"] for row in estimates.values()}) == 1
    assert pooled["balanced_panel"]["static_relation_local_witness"] == {
        "n_sampled_nodes": 90,
        "expanded_population_nodes": 90.0,
        "expanded_positive_nodes": 33.0,
        "rate": 0.366667,
    }
    assert pooled[EXPANSION_KEY]["heldout_measurable_constant_l_slice"] == {
        "n_sampled_nodes": 90,
        "expanded_population_nodes": 1185.0,
        "expanded_positive_nodes": 428.1,
        "rate": 0.361266,
    }
    assert pooled["stage_retention"] == {
        "train_given_static": {"numerator": 33, "denominator": 33, "fraction": 1.0},
        "heldout_given_train_operational": {
            "numerator": 33,
            "denominator": 33,
            "fraction": 1.0,
        },
    }


def test_level_and_audited_depth_decompositions_use_declared_denominators(canonical):
    _inputs, result = canonical
    assert {
        level: result["by_level"][level]["balanced_panel"]
        ["heldout_measurable_constant_l_slice"]["expanded_positive_nodes"]
        for level in ("R1", "R2", "R3")
    } == {"R1": 12.0, "R2": 6.0, "R3": 15.0}
    assert {
        depth: result["by_audited_depth"][depth]["balanced_panel"]
        ["heldout_measurable_constant_l_slice"]["expanded_positive_nodes"]
        for depth in ("0", "1", "2", "3", "4")
    } == {"0": 0.0, "1": 10.0, "2": 23.0, "3": 0.0, "4": 0.0}
    depth_one = result["by_audited_depth"]["1"]
    assert depth_one["balanced_panel"]["static_relation_local_witness"]["rate"] == 0.111111
    assert depth_one[EXPANSION_KEY]["static_relation_local_witness"]["rate"] == 0.088917
    assert "full enclosing panel" in depth_one["interpretation"]


def test_unsupervised_sentinel_sensitivity_is_target_free_and_not_a_gate(canonical):
    _inputs, result = canonical
    diagnostic = result["unsupervised_sentinel_sensitivity"]
    assert diagnostic["used_for_train_gate_selection"] is False
    assert diagnostic["used_for_heldout_decisions"] is False
    assert diagnostic["reference_values_used"] is False
    assert diagnostic["outcome_labels_used"] is False
    assert diagnostic["score_direction_or_target_used"] is False
    assert diagnostic["constant_profiles"] == "abstained before pairing"
    assert diagnostic["pooled_pair_weighted"] == {
        "n_programs": 16,
        "n_nondegenerate_profiles": 230,
        "n_profile_pairs": 2285,
        "n_spearman_pairs": 2285,
        "n_abstained_pairs": 0,
        "n_identical_vector_pairs": 959,
        "identical_vector_pair_rate": 0.419694,
        "spearman_median": 1.0,
        "spearman_min": 0.705609,
        "spearman_max": 1.0,
    }
    assert "not reconstruction" in diagnostic["interpretation"]


def test_channel_and_claim_limits_do_not_upgrade_conditional_slices(canonical):
    _inputs, result = canonical
    assert result["scientific_object"]["executable_object"] == (
        "constant-L conditional slices g_c(x)=f(x,c)"
    )
    channel = result["channel_contract"]
    assert channel["program_execution_outputs_read"] is True
    assert all(value is False for key, value in channel.items() if key != "program_execution_outputs_read")
    claims = " ".join(result["claim_limits"]).lower()
    for phrase in (
        "not a pure-code rewrite",
        "prompt articulability",
        "reconstruction",
        "isomorphism",
        "codability",
        "no references",
    ):
        assert phrase in claims
    assert result["uncertainty_intervals_emitted"] is False


def test_train_gate_and_execution_tampering_fail_closed(canonical):
    inputs, _result = canonical
    panel, audit, train, gate, heldout = inputs

    poisoned_gate = copy.deepcopy(gate)
    poisoned_gate["thresholds"]["min_coverage"] = 0.0
    with pytest.raises(MathLClampSummaryError, match="frozen policy"):
        build_math_lclamp_operational_summary(
            panel, audit, train, poisoned_gate, heldout
        )

    poisoned_train = copy.deepcopy(train)
    poisoned_train["summary"]["three_state_totals"]["measured"] -= 1
    with pytest.raises(MathLClampSummaryError, match="aggregate summary drifted"):
        build_math_lclamp_operational_summary(
            panel, audit, poisoned_train, gate, heldout
        )


def test_heldout_profile_or_forbidden_channel_tampering_fails_closed(canonical):
    inputs, _result = canonical
    panel, audit, train, gate, heldout = inputs

    poisoned = copy.deepcopy(heldout)
    poisoned["outcome_fields_passed_to_worker"] = True
    with pytest.raises(MathLClampSummaryError, match="forbidden boundary"):
        build_math_lclamp_operational_summary(panel, audit, train, gate, poisoned)

    poisoned = copy.deepcopy(heldout)
    poisoned["programs"][0]["profiles"][0]["rows"][0]["score"] = 0.123456
    with pytest.raises(ValueError, match="profile summary/status"):
        build_math_lclamp_operational_summary(panel, audit, train, gate, poisoned)


def test_panel_weight_drift_fails_closed(canonical):
    inputs, _result = canonical
    panel, audit, train, gate, heldout = inputs
    poisoned = copy.deepcopy(panel)
    cell = next(cell for cell in poisoned["cells"] if cell["task"] == "math-stackexchange")
    cell["design_weight"] += 1
    with pytest.raises(MathLClampSummaryError, match="design weight drifted"):
        build_math_lclamp_operational_summary(poisoned, audit, train, gate, heldout)
