from __future__ import annotations

from methods.metric_seam.pilot.patent_ws3_family_retrospective import evaluate


def test_full_patent_family_is_retained_and_outcomes_are_not_read() -> None:
    result = evaluate(permutation_samples=199, bootstrap_samples=200)
    assert [row["criterion_id"] for row in result["criteria"]] == [
        "a26",
        "a34",
        "a60",
        "a35",
    ]
    assert result["summary"]["registered_criteria"] == 4
    assert result["summary"]["bh_family_size"] == 4
    assert result["input_policy"]["items_fields_read"] == ["datapoint_id", "ctext"]
    assert result["input_policy"]["items_judgement_read"] is False
    assert result["input_policy"]["model_calls"] is False
    assert result["input_policy"]["gpu_used"] is False


def test_observed_correlations_reproduce_historical_ws3_report() -> None:
    result = evaluate(permutation_samples=199, bootstrap_samples=200)
    by_id = {row["criterion_id"]: row for row in result["criteria"]}
    expected = {
        "a26": (0.455, 0.244, 0.211),
        "a34": (0.745, 0.084, 0.661),
        "a60": (0.096, -0.027, 0.123),
        "a35": (0.451, -0.159, 0.609),
    }
    for aspect, values in expected.items():
        row = by_id[aspect]
        assert round(row["rho_full_evidence_operation"], 3) == values[0]
        assert round(row["rho_null_operation"], 3) == values[1]
        assert round(row["delta_spearman"], 3) == values[2]
        assert row["heldout_n"] == 100
        assert row["reference_common_n"] == 100
    assert by_id["a60"]["reference_reliability_floor_met"] is False
    assert all(
        by_id[aspect]["reference_reliability_floor_met"]
        for aspect in ("a26", "a34", "a35")
    )
