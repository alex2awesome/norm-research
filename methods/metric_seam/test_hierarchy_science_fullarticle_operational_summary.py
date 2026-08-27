from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_science_fullarticle_operational_summary import (
    EXPANSION_KEY,
    STAGES,
    ScienceOperationalSummaryError,
    build_science_operational_summary,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
NAMES = (
    "panel_v3.json",
    "peer_review_science_claim_seed_map_v1.json",
    "peer_review_science_claim_construct_fidelity_v1.json",
    "peer_review_science_fullarticle_compiler_train_v1.json",
    "peer_review_science_fullarticle_train_gate_v1.json",
    "peer_review_science_fullarticle_heldout_pre_reference_v1.json",
)
OUT = BASE / "peer_review_science_fullarticle_operational_prevalence_v1.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs():
    return [_load(BASE / name) for name in NAMES]


@pytest.fixture(scope="module")
def result():
    return build_science_operational_summary(*_inputs())


def test_science_additive_funnel_retains_all_six_relation_mappings(result):
    assert result["status"] == (
        "additive_representation_static_train_heldout_funnel_complete"
    )
    assert result["validation"]["stage_relation_mapping_counts"] == {
        STAGES[0]: 6,
        STAGES[1]: 6,
        STAGES[2]: 6,
    }
    assert result["pooled_eligible_action_nodes"]["stage_retention"] == {
        "train_given_static": {
            "numerator": 6,
            "denominator": 6,
            "fraction": 1.0,
        },
        "heldout_given_train_operational": {
            "numerator": 6,
            "denominator": 6,
            "fraction": 1.0,
        },
    }


def test_mapping_prevalence_and_item_measurability_use_separate_denominators(result):
    pooled = result["pooled_eligible_action_nodes"]
    for stage in STAGES:
        assert pooled["balanced_panel"][stage] == {
            "n_sampled_nodes": 90,
            "expanded_population_nodes": 90.0,
            "expanded_positive_nodes": 6.0,
            "rate": 0.066667,
        }
        assert pooled[EXPANSION_KEY][stage] == {
            "n_sampled_nodes": 90,
            "expanded_population_nodes": 675.0,
            "expanded_positive_nodes": 37.4,
            "rate": 0.055407,
        }
    assert result["item_execution"]["compiler_train"][
        "three_state_totals_unique_items"
    ] == {"measured": 118, "abstained": 32, "failed": 0}
    assert result["item_execution"]["heldout_pre_reference"][
        "three_state_totals_unique_items"
    ] == {"measured": 108, "abstained": 42, "failed": 0}
    assert result["item_execution"]["compiler_train"]["measured_coverage"] == 0.786667
    assert result["item_execution"]["heldout_pre_reference"]["measured_coverage"] == 0.72
    assert "not a codability rate" in result["estimands"]["item_measurability"]


def test_representation_and_claim_limits_keep_the_additive_scope_narrow(result):
    representation = result["representation"]
    assert representation["canonical_hierarchy_items"] is False
    assert representation["same_bytes_for_future_prompt_and_current_code"] is True
    assert representation["direct_comparison_to_canonical_abstract_only_execution"] is False
    assert representation["complete_pdf_claimed"] is False
    assert representation["upstream_corpus_historically_outcome_stratified"] is True
    assert representation[
        "outcome_values_used_by_current_split_gate_or_execution"
    ] is False
    assert result["scientific_object"]["external_scientific_truth"] is False
    assert result["scientific_object"]["whole_peer_review_construct"] is False
    claims = " ".join(result["claim_limits"]).lower()
    for phrase in (
        "not whole-metric codability",
        "not codability",
        "not directly comparable",
        "not external scientific truth",
        "prompt articulability",
        "reconstruction",
        "isomorphism",
        "no reference judgement",
        "upstream evidence corpus was historically outcome-stratified",
    ):
        assert phrase in claims
    channel = result["channel_contract"]
    assert channel["program_execution_outputs_read"] is True
    assert all(
        value is False
        for key, value in channel.items()
        if key != "program_execution_outputs_read"
    )


def test_banked_operational_artifact_is_exact_rebuild():
    sources = {
        "panel": str(Path("outputs/metric_seam_pilot/hierarchy_r123") / NAMES[0]),
        "seed_map": str(Path("outputs/metric_seam_pilot/hierarchy_r123") / NAMES[1]),
        "construct_fidelity": str(
            Path("outputs/metric_seam_pilot/hierarchy_r123") / NAMES[2]
        ),
        "compiler_train_execution": str(
            Path("outputs/metric_seam_pilot/hierarchy_r123") / NAMES[3]
        ),
        "train_gate": str(Path("outputs/metric_seam_pilot/hierarchy_r123") / NAMES[4]),
        "heldout_pre_reference_execution": str(
            Path("outputs/metric_seam_pilot/hierarchy_r123") / NAMES[5]
        ),
    }
    expected = build_science_operational_summary(*_inputs(), sources=sources)
    assert _load(OUT) == expected


def test_train_gate_and_execution_tampering_fail_closed():
    panel, seed, fidelity, train, gate, heldout = _inputs()

    poisoned = copy.deepcopy(train)
    poisoned["summary"]["three_state_totals_unique_items"]["measured"] -= 1
    with pytest.raises(ScienceOperationalSummaryError, match="summary drifted"):
        build_science_operational_summary(
            panel, seed, fidelity, poisoned, gate, heldout
        )

    poisoned_gate = copy.deepcopy(gate)
    poisoned_gate["criteria"]["minimum_measured_items"]["threshold"] = 0
    with pytest.raises(ScienceOperationalSummaryError, match="deterministic train-only"):
        build_science_operational_summary(
            panel, seed, fidelity, train, poisoned_gate, heldout
        )

    poisoned_heldout = copy.deepcopy(heldout)
    poisoned_heldout["execution_policy"]["outcome_values_loaded"] = True
    with pytest.raises(ScienceOperationalSummaryError, match="forbidden boundary"):
        build_science_operational_summary(
            panel, seed, fidelity, train, gate, poisoned_heldout
        )


def test_panel_weight_drift_fails_closed():
    panel, seed, fidelity, train, gate, heldout = _inputs()
    poisoned = copy.deepcopy(panel)
    cell = next(cell for cell in poisoned["cells"] if cell["task"] == "peer-review")
    cell["design_weight"] += 1
    with pytest.raises(ScienceOperationalSummaryError, match="design weight drifted"):
        build_science_operational_summary(
            poisoned, seed, fidelity, train, gate, heldout
        )
