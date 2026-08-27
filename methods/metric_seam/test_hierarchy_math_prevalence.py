from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_math_fidelity_merge import merge_math_audits
from methods.metric_seam.hierarchy_math_prevalence import (
    EXPANSION_KEY,
    MathPrevalenceError,
    build_math_prevalence,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def _inputs():
    panel = _load("panel_v3.json")
    fidelity = merge_math_audits(
        panel,
        _load("math_stackexchange_seed_map_v1.json"),
        [
            _load("math_stackexchange_construct_fidelity_R1_R2_v1.json"),
            _load("math_stackexchange_construct_fidelity_R3_v1.json"),
        ],
    )
    return panel, fidelity


def _cross_audited_inputs():
    panel = _load("panel_v3.json")
    fidelity = merge_math_audits(
        panel,
        _load("math_stackexchange_seed_map_v1.json"),
        [
            _load("math_stackexchange_construct_fidelity_R1_R2_v1.json"),
            _load("math_stackexchange_construct_fidelity_R3_v1.json"),
        ],
        overlay=_load(
            "math_stackexchange_construct_fidelity_cross_adjudication_merged_v1.json"
        ),
    )
    return panel, fidelity


def _rebind_panel(panel: dict, fidelity: dict) -> None:
    core = {key: value for key, value in panel.items() if key != "panel_content_sha256"}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    panel["panel_content_sha256"] = digest
    fidelity["panel_content_sha256"] = digest


def test_real_math_static_rates_are_provisional_and_descriptive_only():
    result = build_math_prevalence(*_inputs())
    assert result["status"] == "provisional_static_rates_pending_cross_audit"
    assert result["cross_audit"]["status"] == "pending_independent_cross_audit"
    pooled = result["pooled_eligible_action_nodes"]
    assert pooled["balanced_panel"]["retrieved_candidate"]["rate"] == 0.522222
    assert pooled["balanced_panel"]["relation_local_static_fidelity"]["rate"] == 0.377778
    assert pooled[EXPANSION_KEY]["retrieved_candidate"]["rate"] == 0.573333
    assert pooled[EXPANSION_KEY]["relation_local_static_fidelity"]["rate"] == 0.383713
    assert pooled[EXPANSION_KEY]["whole_construct_exact"]["rate"] == 0.0
    assert result["uncertainty_intervals_emitted"] is False
    assert result["execution_or_outcome_stages_emitted"] is False


def test_cross_audited_math_static_rates_use_corrected_relation_local_witnesses():
    result = build_math_prevalence(*_cross_audited_inputs())
    assert result["status"] == "static_descriptive_rates_cross_audited"
    assert result["cross_audit"] == {
        "status": "complete",
        "n_guarded_changes": 21,
        "provisional_until_complete": False,
    }
    pooled = result["pooled_eligible_action_nodes"]
    assert pooled["balanced_panel"]["relation_local_static_fidelity"] == {
        "n_sampled_nodes": 90,
        "expanded_population_nodes": 90.0,
        "expanded_positive_nodes": 33.0,
        "rate": 0.366667,
    }
    assert pooled[EXPANSION_KEY]["relation_local_static_fidelity"] == {
        "n_sampled_nodes": 90,
        "expanded_population_nodes": 1185.0,
        "expanded_positive_nodes": 428.1,
        "rate": 0.361266,
    }
    assert pooled[EXPANSION_KEY]["whole_construct_exact"]["rate"] == 0.0
    assert result["eligible_static_witness_by_audited_depth"]["1"][
        EXPANSION_KEY
    ]["rate"] == 0.088917
    assert result["eligible_static_witness_by_audited_depth"]["2"][
        EXPANSION_KEY
    ]["rate"] == 0.272349
    assert result["uncertainty_intervals_emitted"] is False
    assert result["execution_or_outcome_stages_emitted"] is False


def test_math_sampling_frame_is_the_eligible_action_node_inventory():
    result = build_math_prevalence(*_inputs())
    assert result["sampling_frame"] == {
        "n_complete_action_node_records": 1185,
        "n_eligible_action_node_records": 1185,
        "n_excluded_by_frozen_eligibility_rule": 0,
        "complete_by_level": {"R1": 909, "R2": 233, "R3": 43},
        "eligible_by_level": {"R1": 909, "R2": 233, "R3": 43},
        "n_sampling_strata": 18,
        "selected_per_stratum": [3, 4, 5, 6],
        "eligibility_rule": "nonempty name, at least 8 description words, and at least 1 child",
    }
    assert "conditional" in result["estimands"][EXPANSION_KEY]
    assert "not estimated" in result["estimands"]["sampling_uncertainty"]


def test_math_static_counts_decompose_by_level_and_audited_depth():
    result = build_math_prevalence(*_inputs())
    assert {
        level: result["by_level"][level]["balanced_panel"]
        ["relation_local_static_fidelity"]["expanded_positive_nodes"]
        for level in ("R1", "R2", "R3")
    } == {"R1": 13.0, "R2": 6.0, "R3": 15.0}
    depths = result["eligible_static_witness_by_audited_depth"]
    assert depths["1"]["balanced_panel"]["expanded_positive_nodes"] == 10.0
    assert depths["2"]["balanced_panel"]["expanded_positive_nodes"] == 24.0
    assert depths["1"][EXPANSION_KEY]["rate"] == 0.095499
    assert depths["2"][EXPANSION_KEY]["rate"] == 0.288214


def test_math_prevalence_contains_no_operational_or_heldout_stage():
    result = build_math_prevalence(*_inputs())
    text = json.dumps(result).lower()
    for forbidden in ("train_operational", "heldout_confirmatory", "reconstruction_evaluable"):
        assert forbidden not in text
    assert "confidence_interval" not in text
    assert any("not metric codability" in claim for claim in result["claim_limits"])


def test_math_fidelity_binding_and_forbidden_boundary_fail_closed():
    panel, fidelity = copy.deepcopy(_inputs())
    fidelity["panel_content_sha256"] = "wrong"
    with pytest.raises(MathPrevalenceError, match="another panel"):
        build_math_prevalence(panel, fidelity)

    panel, fidelity = copy.deepcopy(_inputs())
    fidelity["program_outputs_loaded"] = True
    with pytest.raises(MathPrevalenceError, match="forbidden boundary"):
        build_math_prevalence(panel, fidelity)


def test_math_design_weight_and_inventory_totals_fail_closed():
    panel, fidelity = copy.deepcopy(_inputs())
    cell = next(cell for cell in panel["cells"] if cell["task"] == "math-stackexchange")
    cell["design_weight"] += 1
    _rebind_panel(panel, fidelity)
    with pytest.raises(MathPrevalenceError, match="design weight drifted"):
        build_math_prevalence(panel, fidelity)

    panel, fidelity = copy.deepcopy(_inputs())
    inventory = next(
        row for row in panel["inventory"]
        if row["task"] == "math-stackexchange" and row["level"] == "R1"
    )
    inventory["n_eligible_nodes"] += 1
    _rebind_panel(panel, fidelity)
    with pytest.raises(MathPrevalenceError, match="eligible inventory"):
        build_math_prevalence(panel, fidelity)
