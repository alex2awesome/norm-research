from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_science_claim_prevalence import (
    EXPANSION_KEY,
    ScienceClaimPrevalenceError,
    build_science_claim_prevalence,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def _inputs():
    return (
        _load("panel_v3.json"),
        _load("peer_review_science_claim_construct_fidelity_v1.json"),
    )


def _rebind_panel(panel: dict, fidelity: dict) -> None:
    core = {key: value for key, value in panel.items() if key != "panel_content_sha256"}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    panel["panel_content_sha256"] = digest
    fidelity["source_panel_content_sha256"] = digest


def test_science_static_rates_are_narrow_descriptive_relation_rates():
    result = build_science_claim_prevalence(*_inputs())
    assert result["status"] == "static_descriptive_rates_complete_pre_execution"
    pooled = result["pooled_eligible_action_nodes"]
    assert pooled["balanced_panel"]["retrieved_candidate"] == {
        "n_sampled_nodes": 90,
        "expanded_population_nodes": 90.0,
        "expanded_positive_nodes": 9.0,
        "rate": 0.1,
    }
    assert pooled["balanced_panel"]["relation_local_static_fidelity"] == {
        "n_sampled_nodes": 90,
        "expanded_population_nodes": 90.0,
        "expanded_positive_nodes": 6.0,
        "rate": 0.066667,
    }
    assert pooled[EXPANSION_KEY]["retrieved_candidate"] == {
        "n_sampled_nodes": 90,
        "expanded_population_nodes": 675.0,
        "expanded_positive_nodes": 40.4,
        "rate": 0.059852,
    }
    assert pooled[EXPANSION_KEY]["relation_local_static_fidelity"] == {
        "n_sampled_nodes": 90,
        "expanded_population_nodes": 675.0,
        "expanded_positive_nodes": 37.4,
        "rate": 0.055407,
    }
    assert pooled[EXPANSION_KEY]["whole_construct_exact"]["rate"] == 0.0


def test_science_sampling_frame_is_the_eligible_action_node_inventory():
    result = build_science_claim_prevalence(*_inputs())
    assert result["sampling_frame"] == {
        "n_complete_action_node_records": 676,
        "n_eligible_action_node_records": 675,
        "n_excluded_by_frozen_eligibility_rule": 1,
        "complete_by_level": {"R1": 509, "R2": 136, "R3": 31},
        "eligible_by_level": {"R1": 508, "R2": 136, "R3": 31},
        "n_sampling_strata": 18,
        "selected_per_stratum": [3, 5, 6],
        "eligibility_rule": (
            "nonempty name, at least 8 description words, and at least 1 child"
        ),
    }
    assert "conditional" in result["estimands"][EXPANSION_KEY]
    assert "not estimated" in result["estimands"]["sampling_uncertainty"]


def test_all_and_only_static_relation_matches_are_depth_three():
    result = build_science_claim_prevalence(*_inputs())
    pooled = result["pooled_eligible_action_nodes"]
    for frame in ("balanced_panel", EXPANSION_KEY):
        assert (
            pooled[frame]["depth3_relation_local_static_fidelity"]
            == pooled[frame]["relation_local_static_fidelity"]
        )
    assert result["matched_relation_depth"] == {
        "depth": 3,
        "depth_meaning": "document-local retrieval relation chain",
        "all_and_only_relation_local_static_matches_at_this_depth": True,
    }


def test_balanced_levels_are_equal_but_no_trend_is_claimed():
    result = build_science_claim_prevalence(*_inputs())
    assert {
        level: result["by_level"][level]["balanced_panel"]
        ["relation_local_static_fidelity"]["expanded_positive_nodes"]
        for level in ("R1", "R2", "R3")
    } == {"R1": 2.0, "R2": 2.0, "R3": 2.0}
    assert {
        level: result["by_level"][level][EXPANSION_KEY]
        ["relation_local_static_fidelity"]["rate"]
        for level in ("R1", "R2", "R3")
    } == {"R1": 0.051969, "R2": 0.066176, "R3": 0.064516}
    assert any("do not establish" in claim for claim in result["claim_limits"])


def test_science_prevalence_emits_no_execution_prompt_or_reconstruction_stage():
    result = build_science_claim_prevalence(*_inputs())
    assert result["uncertainty_intervals_emitted"] is False
    assert result["execution_or_outcome_stages_emitted"] is False
    assert result["prompt_or_model_stages_emitted"] is False
    assert result["reconstruction_or_isomorphism_stages_emitted"] is False
    text = json.dumps(result).lower()
    for forbidden in (
        "train_operational",
        "heldout_confirmatory",
        "reconstruction_evaluable",
        "confidence_interval",
    ):
        assert forbidden not in text
    assert any("not peer-review metric codability" in claim for claim in result["claim_limits"])
    assert any("not external scientific truth" in claim for claim in result["claim_limits"])


def test_science_channel_provenance_is_manual_document_local_pure_code():
    result = build_science_claim_prevalence(*_inputs())
    assert result["channel_provenance"] == {
        "historical_pipeline": "manually designed full-article pure-code verifier",
        "evidence_scope": "distinct body sentences within the same presented article",
        "retrieval_scope": "document-local BM25; no corpus or external retrieval",
        "certificate_scope": (
            "numeric/comparative document-internal consistency, not external scientific truth"
        ),
        "automatic_discovery": False,
    }


def test_fidelity_binding_and_forbidden_boundary_fail_closed():
    panel, fidelity = copy.deepcopy(_inputs())
    fidelity["source_panel_content_sha256"] = "wrong"
    with pytest.raises(ScienceClaimPrevalenceError, match="another panel"):
        build_science_claim_prevalence(panel, fidelity)

    panel, fidelity = copy.deepcopy(_inputs())
    fidelity["historical_certificates_or_program_outputs_loaded"] = True
    with pytest.raises(ScienceClaimPrevalenceError, match="forbidden boundary"):
        build_science_claim_prevalence(panel, fidelity)


def test_design_weight_and_inventory_totals_fail_closed():
    panel, fidelity = copy.deepcopy(_inputs())
    cell = next(cell for cell in panel["cells"] if cell["task"] == "peer-review")
    cell["design_weight"] += 1
    _rebind_panel(panel, fidelity)
    with pytest.raises(ScienceClaimPrevalenceError, match="design weight drifted"):
        build_science_claim_prevalence(panel, fidelity)

    panel, fidelity = copy.deepcopy(_inputs())
    inventory = next(
        row
        for row in panel["inventory"]
        if row["task"] == "peer-review" and row["level"] == "R1"
    )
    inventory["n_eligible_nodes"] += 1
    _rebind_panel(panel, fidelity)
    with pytest.raises(ScienceClaimPrevalenceError, match="eligible inventory"):
        build_science_claim_prevalence(panel, fidelity)


def test_depth_and_summary_drift_fail_closed():
    panel, fidelity = copy.deepcopy(_inputs())
    row = next(row for row in fidelity["rows"] if row["verdict"] == "partial_relation_local")
    row["eligible_relation_local_depths"] = [2]
    with pytest.raises(ScienceClaimPrevalenceError, match="relation/depth contract"):
        build_science_claim_prevalence(panel, fidelity)

    panel, fidelity = copy.deepcopy(_inputs())
    fidelity["summary"]["n_partial_relation_local"] += 1
    with pytest.raises(ScienceClaimPrevalenceError, match="summary drifted"):
        build_science_claim_prevalence(panel, fidelity)


def test_checked_in_prevalence_artifact_is_exact_builder_output():
    panel, fidelity = _inputs()
    expected = build_science_claim_prevalence(
        panel,
        fidelity,
        sources={
            "panel": "outputs/metric_seam_pilot/hierarchy_r123/panel_v3.json",
            "construct_fidelity": (
                "outputs/metric_seam_pilot/hierarchy_r123/"
                "peer_review_science_claim_construct_fidelity_v1.json"
            ),
        },
    )
    observed = _load("peer_review_science_claim_static_prevalence_v1.json")
    assert observed == expected

