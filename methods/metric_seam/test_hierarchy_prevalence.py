from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_prevalence import (
    EXPANSION_KEY,
    PrevalenceError,
    build_prevalence,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def _inputs():
    return (
        _load("panel_v3.json"),
        _load("code_review_construct_fidelity_v2.json"),
        _load("code_review_train_gate_v1.json"),
        _load("code_review_heldout_readiness_v1.json"),
    )


def _rebind_panel(panel: dict, audit: dict) -> None:
    core = {key: value for key, value in panel.items() if key != "panel_content_sha256"}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    panel["panel_content_sha256"] = digest
    audit["panel_content_sha256"] = digest


def test_real_prevalence_separates_balanced_and_conditional_expansion_estimands():
    result = build_prevalence(*_inputs(), n_resamples=100)
    pooled = result["pooled_eligible_action_nodes"]
    balanced = pooled["balanced_panel"]
    expanded = pooled[EXPANSION_KEY]
    assert result["schema"] == "metric-seam.hierarchy-witness-prevalence.v2"
    assert balanced["relation_local_static_fidelity"]["rate"] == 0.622222
    assert expanded["relation_local_static_fidelity"]["rate"] == 0.478369
    assert expanded["train_operational_relation_witness"]["rate"] == 0.246099
    assert expanded["heldout_confirmatory_reconstruction_evaluable"]["rate"] == 0.175532
    assert expanded["relation_local_static_fidelity"]["estimated_population_nodes"] == 1128
    assert result["whole_construct_exact"] == {"n": 0, "denominator": 90, "rate": 0.0}
    assert result["supersedes"]["point_estimates_changed"] is False


def test_sampling_frame_distinguishes_complete_from_eligible_inventory():
    result = build_prevalence(*_inputs(), n_resamples=100)
    frame = result["sampling_frame"]
    assert frame == {
        "n_complete_action_node_records": 1132,
        "n_eligible_action_node_records": 1128,
        "n_excluded_by_frozen_eligibility_rule": 4,
        "complete_by_level": {"R1": 860, "R2": 217, "R3": 55},
        "eligible_by_level": {"R1": 856, "R2": 217, "R3": 55},
        "n_sampling_strata": 18,
        "selected_per_stratum": [5],
        "eligibility_rule": "nonempty name, at least 8 description words, and at least 1 child",
    }
    assert "conditional" in result["estimands"][EXPANSION_KEY]
    assert "not estimated" in result["estimands"]["sampling_uncertainty"]


def test_funnel_is_nested_at_every_level():
    result = build_prevalence(*_inputs(), n_resamples=100)
    for scope in result["by_level"].values():
        rates = scope["balanced_panel"]
        assert (
            rates["heldout_confirmatory_reconstruction_evaluable"]["rate"]
            <= rates["train_operational_relation_witness"]["rate"]
            <= rates["relation_local_static_fidelity"]["rate"]
            <= rates["retrieved_candidate"]["rate"]
        )


def test_panel_binding_and_unknown_cells_fail_closed():
    inputs = list(_inputs())
    bad_audit = copy.deepcopy(inputs[1])
    bad_audit["panel_content_sha256"] = "wrong"
    with pytest.raises(PrevalenceError, match="another panel"):
        build_prevalence(inputs[0], bad_audit, inputs[2], inputs[3], n_resamples=100)

    bad_readiness = copy.deepcopy(inputs[3])
    bad_readiness["confirmatory_programs"][0]["cell_ids"].append("invented")
    with pytest.raises(PrevalenceError, match="unknown cell"):
        build_prevalence(inputs[0], inputs[1], inputs[2], bad_readiness, n_resamples=100)


def test_static_fidelity_without_a_retrieved_candidate_fails_closed():
    panel, audit, gate, readiness = copy.deepcopy(_inputs())
    operational = {
        cell_id for program in gate["selected_programs"] for cell_id in program["cell_ids"]
    }
    row = next(
        row for row in audit["rows"]
        if row["eligible_for_relation_local_execution"] and row["cell_id"] not in operational
    )
    row["candidate"] = None
    with pytest.raises(PrevalenceError, match="not nested"):
        build_prevalence(panel, audit, gate, readiness, n_resamples=100)


def test_program_ownership_and_candidate_identity_fail_closed():
    panel, audit, gate, readiness = copy.deepcopy(_inputs())
    gate["selected_programs"].append(copy.deepcopy(gate["selected_programs"][0]))
    with pytest.raises(PrevalenceError, match="repeats one candidate program"):
        build_prevalence(panel, audit, gate, readiness, n_resamples=100)

    panel, audit, gate, readiness = copy.deepcopy(_inputs())
    readiness["confirmatory_programs"][0]["source_path"] = "wrong.py"
    with pytest.raises(PrevalenceError, match="does not match audit candidate"):
        build_prevalence(panel, audit, gate, readiness, n_resamples=100)


def test_summary_and_artifact_binding_drift_fail_closed():
    panel, audit, gate, readiness = copy.deepcopy(_inputs())
    gate["summary"]["n_selected_relation_mappings"] += 1
    with pytest.raises(PrevalenceError, match="summary drifted"):
        build_prevalence(panel, audit, gate, readiness, n_resamples=100)

    panel, audit, gate, readiness = copy.deepcopy(_inputs())
    readiness["compiler_train_gate_source"] = "different_gate.json"
    with pytest.raises(PrevalenceError, match="another train gate"):
        build_prevalence(panel, audit, gate, readiness, n_resamples=100)


def test_design_weight_and_stratum_counts_fail_closed():
    panel, audit, gate, readiness = copy.deepcopy(_inputs())
    cell = next(cell for cell in panel["cells"] if cell["task"] == "code-review")
    cell["design_weight"] += 1
    _rebind_panel(panel, audit)
    with pytest.raises(PrevalenceError, match="design weight drifted"):
        build_prevalence(panel, audit, gate, readiness, n_resamples=100)

    panel, audit, gate, readiness = copy.deepcopy(_inputs())
    cell = next(cell for cell in panel["cells"] if cell["task"] == "code-review")
    cell["stratum_selected_n"] += 1
    _rebind_panel(panel, audit)
    with pytest.raises(PrevalenceError, match="inconsistent population/sample counts"):
        build_prevalence(panel, audit, gate, readiness, n_resamples=100)


def test_one_way_perturbations_are_deterministic_and_narrowly_labeled():
    first = build_prevalence(*_inputs(), n_resamples=100)
    second = build_prevalence(*_inputs(), n_resamples=100)
    a = first["by_level"]["R1"][EXPANSION_KEY]
    b = second["by_level"]["R1"][EXPANSION_KEY]
    assert a == b
    diagnostic = a["relation_local_static_fidelity"][
        "dependency_one_way_observed_block_perturbation"
    ]
    assert diagnostic["method"] == "uniform pairs perturbation of observed one-way blocks"
    assert "does not preserve the stratified sampling design" in diagnostic["interpretation"]
    assert "confidence interval" in diagnostic["interpretation"]


def test_cross_level_program_and_joint_component_diagnostics_are_structure_only():
    result = build_prevalence(*_inputs(), n_resamples=100)
    diagnostics = result["sensitivities"]["dependence_component_diagnostics"]
    assert diagnostics["status"] == "component_structure_only_no_interval"
    assert diagnostics["not_an_interval"] is True
    assert diagnostics["cross_level_raw_support"]["n_components"] == 35
    assert diagnostics["cross_level_raw_support"]["largest_component"] == 33
    assert diagnostics["cross_level_raw_support"]["n_cross_level_components"] == 12
    assert diagnostics["shared_candidate_program"]["n_components"] == 55
    assert diagnostics["shared_candidate_program"]["largest_component"] == 5
    assert diagnostics["joint_dependency_raw_program_union"]["n_components"] == 25
    assert diagnostics["joint_dependency_raw_program_union"]["largest_component"] == 49
    assert diagnostics["program_reuse_by_outcome"][
        "heldout_confirmatory_reconstruction_evaluable"
    ] == {"n_positive_mappings": 21, "n_unique_candidate_programs": 12}
    assert len(diagnostics["cell_assignments"]) == 90


def test_terminal_frontier_and_required_sensitivities_are_explicitly_outstanding():
    result = build_prevalence(*_inputs(), n_resamples=100)
    frontier = result["sensitivities"]["tightest_first_terminal_frontier"]
    assert frontier["status"] == "not_yet_measured"
    assert len(result["outstanding_sensitivities"]) == 3
    assert any("joint or multiway" in item for item in result["outstanding_sensitivities"])
