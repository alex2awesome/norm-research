from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from methods.metric_seam.adjudicate_patent_construct_fidelity import build_audit
from methods.metric_seam.hierarchy_patent_prevalence import (
    EXPANSION_KEY,
    PatentPrevalenceError,
    build_patent_prevalence,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def _inputs():
    panel = _load("panel_v3.json")
    fidelity = build_audit(_load("patents_seed_map_v1.json"))
    return panel, fidelity


def _rebind_panel(panel: dict, fidelity: dict) -> None:
    core = {key: value for key, value in panel.items() if key != "panel_content_sha256"}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    panel["panel_content_sha256"] = digest
    fidelity["source_panel_content_sha256"] = digest


def test_real_patent_rates_are_narrow_static_witness_rates():
    result = build_patent_prevalence(*_inputs())
    pooled = result["pooled_eligible_action_nodes"]
    assert pooled["balanced_panel"]["retrieved_candidate"]["rate"] == 0.066667
    assert pooled["balanced_panel"]["relation_local_static_fidelity"]["rate"] == 0.066667
    assert pooled[EXPANSION_KEY]["retrieved_candidate"]["rate"] == 0.051754
    assert pooled[EXPANSION_KEY]["relation_local_static_fidelity"]["rate"] == 0.051754
    assert pooled[EXPANSION_KEY]["depth3_evidence_relation"]["rate"] == 0.050877
    assert pooled[EXPANSION_KEY]["pure_code_witness"]["rate"] == 0.0
    assert pooled[EXPANSION_KEY]["whole_construct_exact"]["rate"] == 0.0
    assert result["uncertainty_intervals_emitted"] is False
    assert result["execution_or_outcome_stages_emitted"] is False


def test_patent_sampling_frame_is_eligible_action_node_inventory():
    result = build_patent_prevalence(*_inputs())
    assert result["sampling_frame"] == {
        "n_complete_action_node_records": 1369,
        "n_eligible_action_node_records": 1368,
        "n_excluded_by_frozen_eligibility_rule": 1,
        "complete_by_level": {"R1": 1071, "R2": 256, "R3": 42},
        "eligible_by_level": {"R1": 1070, "R2": 256, "R3": 42},
        "n_sampling_strata": 18,
        "selected_per_stratum": [4, 5, 6],
        "eligibility_rule": "nonempty name, at least 8 description words, and at least 1 child",
    }
    assert "conditional" in result["estimands"][EXPANSION_KEY]
    assert "not estimated" in result["estimands"]["sampling_uncertainty"]


def test_relation_depth_is_not_misreported_as_pure_code():
    result = build_patent_prevalence(*_inputs())
    assert result["channel_provenance"] == {
        "historical_programs": "manual hybrids",
        "prior_art_candidates": "examiner/oracle conditioned",
        "disclosure_relations": "precomputed reading-model verdicts",
        "autonomous_retrieval": False,
        "pure_code": False,
    }
    assert any("not pure code" in claim for claim in result["claim_limits"])
    assert result["outcome_definitions"]["depth3_evidence_relation"].endswith("not pure code")


def test_patent_prevalence_contains_no_operational_or_reconstruction_stage():
    result = build_patent_prevalence(*_inputs())
    text = json.dumps(result).lower()
    for forbidden in (
        "train_operational",
        "heldout_confirmatory",
        "reconstruction_evaluable",
        "confidence_interval",
    ):
        assert forbidden not in text
    assert any("not patent-metric codability" in claim for claim in result["claim_limits"])


def test_fidelity_binding_and_forbidden_boundary_fail_closed():
    panel, fidelity = copy.deepcopy(_inputs())
    fidelity["source_panel_content_sha256"] = "wrong"
    with pytest.raises(PatentPrevalenceError, match="another panel"):
        build_patent_prevalence(panel, fidelity)

    panel, fidelity = copy.deepcopy(_inputs())
    fidelity["outcome_labels_loaded"] = True
    with pytest.raises(PatentPrevalenceError, match="forbidden boundary"):
        build_patent_prevalence(panel, fidelity)


def test_design_weight_and_inventory_totals_fail_closed():
    panel, fidelity = copy.deepcopy(_inputs())
    cell = next(cell for cell in panel["cells"] if cell["task"] == "patents")
    cell["design_weight"] += 1
    _rebind_panel(panel, fidelity)
    with pytest.raises(PatentPrevalenceError, match="design weight drifted"):
        build_patent_prevalence(panel, fidelity)

    panel, fidelity = copy.deepcopy(_inputs())
    inventory = next(
        row
        for row in panel["inventory"]
        if row["task"] == "patents" and row["level"] == "R1"
    )
    inventory["n_eligible_nodes"] += 1
    _rebind_panel(panel, fidelity)
    with pytest.raises(PatentPrevalenceError, match="eligible inventory"):
        build_patent_prevalence(panel, fidelity)
