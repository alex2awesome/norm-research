from __future__ import annotations

import numpy as np
import pytest

from methods.metric_implementer.experiments.v14_value_bound import (
    aggregate_state_tables,
    classify_status,
    dkw_expected_best_gain_bound,
    fidelity_legibility_diagnostic,
    novelty_collapse_curves,
    process_gain_certificate,
    record_rank_gain_bound,
    reject_adaptive_zero_hit_cp,
    split_sample_cp_gain_bound,
    validate_state_tables,
)


def test_exact_state_aggregation_cap_raw_ties_and_dead_status():
    rng = np.random.default_rng(14)
    panels = [list(range(0, 8)), list(range(8, 16))]
    signatures = rng.integers(0, 2, size=(20, 16))
    raw = np.tile(np.linspace(-0.2, 0.3, 256), (2, 1))
    clipped = np.maximum(raw, 0.0)
    validate_state_tables(raw, clipped)
    result = aggregate_state_tables(
        raw_lift=raw, clipped_value=clipped, prompt_signatures=signatures,
        panels=panels, prompt_ids=[f"p{i}" for i in range(20)],
        decoder_families=["qwen", "llama"],
    )
    assert result["free_recombination_cap"] >= result["achieved_value"]
    assert np.any(result["prompt_raw_lift"] < 0.0)
    assert result["legibility_argmax"]["canonical_representative"] in result["prompt_ids"]
    dead = classify_status(
        achieved=0.0, cap=0.0, raw_panel_caps=[-0.1, -0.2],
        blind_value=0.4, annotated_canonical_value=0.3,
    )
    assert dead["status"] == "DEAD_INSTRUMENT"


def test_record_rank_split_cp_and_dkw_are_valid_and_adaptive_zero_hit_is_rejected():
    record = record_rank_gain_bound(n=400, m=100, achieved=0.0188, cap=0.0357)
    assert record["improvement_probability_upper"] == pytest.approx(0.2)
    assert record["gain_upper"] == pytest.approx(0.00338)
    cp = split_sample_cp_gain_bound(
        discovery_achieved=0.2, audit_values=[0.1, 0.3, 0.15, 0.25],
        cap=0.5, future_horizon=100,
    )
    assert cp["n_audit_improvements"] == 2
    assert cp["audit_achieved_lower_bound"] == pytest.approx(0.3)
    dkw = dkw_expected_best_gain_bound(
        observed_values=[0.1, 0.2, 0.3], achieved=0.3, cap=0.5,
        future_horizon=100,
    )
    assert 0.0 <= dkw["gain_upper"] <= 0.2
    combined = process_gain_certificate(
        observed_values=[0.1, 0.2], discovery_achieved=0.2,
        audit_values=[0.15, 0.25], cap=0.5, future_horizon=100,
    )
    assert combined["headline_gain_upper"] == min(
        row["gain_upper"] for row in combined["bounds"].values()
    )
    with pytest.raises(ValueError, match="invalid bound"):
        reject_adaptive_zero_hit_cp()


def test_fidelity_legibility_tie_gate_and_novelty_ladder():
    target = np.asarray([0, 0, 1, 1], dtype=int)
    signatures = np.asarray([
        target, 1 - target, [0, 0, 0, 0], target,
    ], dtype=float)
    diagnostic = fidelity_legibility_diagnostic(
        prompt_signatures_on_h=signatures, target_on_h=target,
        legibility_values=[0.1, 0.2, 0.3, 0.1],
        prompt_ids=["a", "b", "c", "d"],
    )
    assert diagnostic["fidelity_argmax"]["size"] >= 2
    curves = novelty_collapse_curves(
        full_signatures=signatures,
        joint_codes=np.asarray([[0, 1], [0, 1], [1, 1], [0, 1]]),
        values=[0.1, 0.2, 0.3, 0.1], frozen_incumbent=0.25,
    )
    assert len(curves) == 4
    assert all(row["cumulative_behavior_novelty_rate"] >= row[
        "cumulative_code_novelty_rate"
    ] for row in curves)

