import pytest

from scripts.tools.silver_match_v3.freeze_cross_encoder_queue import (
    _audit_dev_gate_feasibility,
)


def _gate() -> dict[str, float | int]:
    return {
        "minimum_retained_predictions": 30,
        "minimum_exact_match_precision": 0.9,
        "minimum_exact_match_precision_wilson_95_lower": 0.8,
    }


def test_dev_gate_feasibility_rejects_math_select60_oracle_upper_bound():
    dev = {f"n-{index}" for index in range(60)}
    matches = {f"n-{index}" for index in range(23)}
    with pytest.raises(ValueError, match="mathematically infeasible"):
        _audit_dev_gate_feasibility(
            dev_uids=dev,
            possible_match_uids=matches,
            gate=_gate(),
        )


def test_dev_gate_feasibility_accepts_code_select60_oracle_upper_bound():
    dev = {f"n-{index}" for index in range(60)}
    matches = {f"n-{index}" for index in range(46)}
    audit = _audit_dev_gate_feasibility(
        dev_uids=dev,
        possible_match_uids=matches,
        gate=_gate(),
    )
    assert audit["feasible_under_oracle"] is True
    assert audit["possible_gold_match_upper_bound"] == 46
    assert audit["first_feasible_oracle_case"]["predicted_match_count"] >= 30


def test_dev_gate_feasibility_rejects_panel_smaller_than_support_floor():
    dev = {f"n-{index}" for index in range(20)}
    with pytest.raises(ValueError, match="mathematically infeasible"):
        _audit_dev_gate_feasibility(
            dev_uids=dev,
            possible_match_uids=dev,
            gate=_gate(),
        )
