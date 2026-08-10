from methods.metric_seam.analyze_technical_target_discrimination import summarize_vector


def test_resolution_diagnostic_passes_spread_vector():
    row = summarize_vector(
        task="math", vector_id="x", values=[0, 1, 2, 0, 1, 2] * 20, depth=3
    )
    assert row["passes_resolution_diagnostic"]
    assert row["n_unique_values"] == 3
    assert row["mode_fraction"] == 1 / 3


def test_resolution_diagnostic_separates_coverage_and_mode_failures():
    sparse = summarize_vector(
        task="science", vector_id="sparse", values=[None] * 20 + [0, 1, 2], depth=3
    )
    assert not sparse["passes_resolution_diagnostic"]
    assert "coverage_below_0.90" in sparse["failed_checks"]
    tied = summarize_vector(
        task="patents", vector_id="tied", values=[1] * 95 + [0, 2, 3, 4, 5], depth=2
    )
    assert not tied["passes_resolution_diagnostic"]
    assert "mode_fraction_above_0.85" in tied["failed_checks"]


def test_none_is_not_a_measured_target_value():
    row = summarize_vector(
        task="math", vector_id="missing", values=[0, 1, 2, None], depth=None
    )
    assert row["n_measured"] == 3
    assert row["coverage"] == 0.75
