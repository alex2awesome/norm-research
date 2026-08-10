from __future__ import annotations

import math

import numpy as np

from methods.metric_seam.analyze_code_review_target_resolution import (
    analyze,
    bootstrap_auc,
    main,
    mann_whitney_auc,
    tie_expanded_terciles,
)
import pytest


def test_mann_whitney_auc_counts_prediction_ties_as_half() -> None:
    assert mann_whitney_auc([0.0, 1.0], [1.0, 2.0]) == 0.875
    assert mann_whitney_auc([], [1.0]) is None


def test_tie_expanded_terciles_do_not_split_target_boundary_ties() -> None:
    targets = {
        "a0": 0.0,
        "a1": 0.0,
        "a2": 1.0,
        "a3": 1.0,
        "a4": 1.0,
        "a5": 2.0,
        "a6": 2.0,
        "a7": 3.0,
        "a8": 3.0,
    }
    partition = tie_expanded_terciles(targets)

    assert partition.bottom_nominal_n == 3
    assert partition.top_nominal_n == 3
    assert partition.bottom == ("a0", "a1", "a2", "a3", "a4")
    assert partition.top == ("a5", "a6", "a7", "a8")
    assert partition.bottom_boundary == 1.0
    assert partition.top_boundary == 2.0
    assert partition.has_spread


def test_bootstrap_auc_is_seeded_and_stratified() -> None:
    first = bootstrap_auc([0.0, 1.0], [1.0, 2.0], draws=25, seed=7)
    repeat = bootstrap_auc([0.0, 1.0], [1.0, 2.0], draws=25, seed=7)
    different = bootstrap_auc([0.0, 1.0], [1.0, 2.0], draws=25, seed=8)

    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, different)
    assert np.all((0.0 <= first) & (first <= 1.0))


def test_frozen_artifacts_reproduce_named_auc_but_not_claimed_median() -> None:
    readout = analyze(bootstrap_draws=20, bootstrap_seed=11)
    by_aspect_level = {
        (row["aspect_id"], row["level"]): row for row in readout["per_cell"]
    }

    assert len(readout["per_cell"]) == 18
    assert readout["aggregate"]["target_spread_mappings"] == 8
    assert readout["aggregate"]["target_no_spread_mappings"] == 10
    assert math.isclose(
        by_aspect_level[("a0", "R3")]["tercile_auc"],
        0.7197802197802198,
    )
    assert math.isclose(
        by_aspect_level[("a37", "R3")]["tercile_auc"],
        0.7107988165680473,
    )
    assert math.isclose(
        by_aspect_level[("a92", "R3")]["tercile_auc"],
        0.7096153846153846,
    )
    assert math.isclose(
        readout["aggregate"]["median_tercile_auc"], 0.5733498527616174
    )
    assert not readout["headline_reproduction"]["median_reproduces_to_3dp"]

    parser = readout["parser_provenance"]
    assert parser["observed_transport_evidence"] == {
        "valid_markdown_fenced_rows": 4424,
        "valid_literal_tab_rows": 206,
    }
    recovered = readout["source_adjudication"]["recovered_implementation_summary"]
    assert not any(
        result["any_mapping_reproduces_to_3dp"]
        for result in recovered["headline_checks_under_identical_estimator"].values()
    )


def test_cli_refuses_existing_append_only_output_before_analysis(tmp_path) -> None:
    output = tmp_path / "readout.json"
    output.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--output", str(output)])
    assert output.read_text(encoding="utf-8") == "sentinel\n"
