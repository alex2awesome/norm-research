from __future__ import annotations

import math
import gzip
import json
from pathlib import Path

from methods.metric_seam.analyze_code_review_reconstruction import (
    MappingSeries,
    hierarchical_bootstrap,
    mapping_statistics,
    render_report,
    resolve_analysis_channel,
    wrong_relation_statistics,
)


def _series(
    *,
    cell: str = "cell-a",
    cluster: str = "cluster-a",
    code: list[float],
    pass1: list[float | None],
    pass2: list[float | None],
) -> MappingSeries:
    items = tuple(f"i{index:03d}" for index in range(len(code)))
    return MappingSeries(
        cell_id=cell,
        aspect_id="aspect-" + cell,
        vector_cluster_id=cluster,
        level="R1",
        metric_name="synthetic",
        item_keys=items,
        code_scores={item: value for item, value in zip(items, code)},
        pass1_scores={
            item: value
            for item, value in zip(items, pass1)
            if value is not None
        },
        pass2_scores={
            item: value
            for item, value in zip(items, pass2)
            if value is not None
        },
    )


def test_perfect_inverse_tied_constant_sparse_and_abstaining_vectors() -> None:
    code = [float(index) for index in range(40)]
    perfect = mapping_statistics(_series(code=code, pass1=code, pass2=code))
    assert perfect["raw_rho"] == 1.0
    assert perfect["support_interpretation"] == "confirmatory_estimate"
    assert perfect["common_support_n"] == 40

    inverse_values = list(reversed(code))
    inverse = mapping_statistics(
        _series(code=code, pass1=inverse_values, pass2=inverse_values)
    )
    assert inverse["raw_rho"] == -1.0

    tied_code = [float(index // 4) for index in range(40)]
    tied = mapping_statistics(
        _series(code=tied_code, pass1=tied_code, pass2=tied_code)
    )
    assert tied["raw_rho"] == 1.0
    assert tied["code_unique_score_count"] == 10
    assert tied["largest_code_tie_fraction"] == 0.1

    constant = mapping_statistics(
        _series(code=[1.0] * 40, pass1=code, pass2=code)
    )
    assert constant["raw_rho"] is None
    assert constant["code_unique_score_count"] == 1

    sparse_pass = code[:9] + [None] * 31
    sparse = mapping_statistics(
        _series(code=code, pass1=sparse_pass, pass2=sparse_pass)
    )
    assert sparse["common_support_n"] == 9
    assert sparse["support_interpretation"] == "no_correlation_estimate"
    assert sparse["raw_rho"] is None

    abstaining_pass2 = code[:23] + [None] * 17
    abstaining = mapping_statistics(
        _series(code=code, pass1=code, pass2=abstaining_pass2)
    )
    assert abstaining["common_support_n"] == 23
    assert abstaining["support_interpretation"] == "exploratory_estimate"
    assert abstaining["prompt_pass2_scored_coverage"] == 23 / 40


def test_two_pass_averaging_and_reliability() -> None:
    code = [float(index) for index in range(30)]
    pass1 = [value + (0.4 if index % 2 else -0.4) for index, value in enumerate(code)]
    pass2 = [2.0 * value - pass1[index] for index, value in enumerate(code)]
    stats = mapping_statistics(_series(code=code, pass1=pass1, pass2=pass2))

    assert stats["raw_rho"] == 1.0
    reliability = stats["pass_to_pass_reliability"]
    expected_sb = 2.0 * reliability / (1.0 + reliability)
    assert math.isclose(stats["two_pass_spearman_brown_reliability"], expected_sb)
    assert math.isclose(stats["attenuation_ceiling"], math.sqrt(expected_sb))
    assert math.isclose(
        stats["ceiling_normalized_rho"], 1.0 / math.sqrt(expected_sb)
    )

    anti = mapping_statistics(
        _series(code=code, pass1=code, pass2=list(reversed(code)))
    )
    assert anti["pass_to_pass_reliability"] == -1.0
    assert anti["two_pass_spearman_brown_reliability"] is None
    assert anti["attenuation_ceiling"] is None


def test_wrong_relation_uses_one_identical_support() -> None:
    code = [float(index) for index in range(20)]
    correct = _series(code=code, pass1=code, pass2=code)
    wrong_values = list(reversed(code))
    wrong_pass1: list[float | None] = list(wrong_values)
    wrong_pass2: list[float | None] = list(wrong_values)
    wrong_pass1[3] = None
    wrong = _series(
        cell="cell-wrong",
        cluster="cluster-wrong",
        code=code,
        pass1=wrong_pass1,
        pass2=wrong_pass2,
    )

    stats = wrong_relation_statistics(correct, wrong)

    assert stats["identical_common_support_n"] == 19
    assert "i003" not in stats["identical_common_support_item_keys"]
    assert stats["rho_correct"] == 1.0
    assert stats["rho_wrong"] == -1.0
    assert stats["rho_correct_minus_wrong"] == 2.0


def test_vector_cluster_bootstrap_is_reproducible() -> None:
    code = [float(index) for index in range(40)]
    a = _series(code=code, pass1=code, pass2=code)
    noisy = [value + (5.0 if index % 5 == 0 else 0.0) for index, value in enumerate(code)]
    b = _series(
        cell="cell-b",
        cluster="cluster-b",
        code=code,
        pass1=noisy,
        pass2=noisy,
    )
    series = {a.cell_id: a, b.cell_id: b}
    clusters = {"cluster-a": [a.cell_id], "cluster-b": [b.cell_id]}

    first = hierarchical_bootstrap(
        series_by_cell=series,
        cluster_to_cells=clusters,
        item_keys=a.item_keys,
        draws=100,
        seed=20260713,
    )
    second = hierarchical_bootstrap(
        series_by_cell=series,
        cluster_to_cells=clusters,
        item_keys=a.item_keys,
        draws=100,
        seed=20260713,
    )
    different = hierarchical_bootstrap(
        series_by_cell=series,
        cluster_to_cells=clusters,
        item_keys=a.item_keys,
        draws=100,
        seed=17,
    )

    assert first == second
    assert first != different
    assert len(first) == 100


def test_unusable_response_report_is_explicit() -> None:
    readout = {
        "execution_accounting": {"status_counts": {"contract_error": 4500}},
        "aggregate": {
            "median_raw_rho": None,
            "ci95": [None, None],
            "mappings_with_confirmatory_support": 0,
            "median_reliability_ceiling": None,
        },
        "wrong_relation_control": {
            "median_rho_correct_minus_wrong": None,
            "ci95": [None, None],
        },
        "per_mapping": [],
        "claim_limits": ["No reconstruction estimate was available."],
    }

    report = render_report(readout)

    assert report.startswith("**0 responses were valid; 0 of 18 mappings")
    assert "Median raw Spearman rho was undefined" in report
    assert "not an observed rho of zero" in report


def test_analysis_channel_resolution_preserves_v3_default_and_infers_ceiling(
    tmp_path: Path,
) -> None:
    def write(path: Path, channels: list[str]) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for channel in channels:
                handle.write(json.dumps({"audit_metadata": {"channel": channel}}) + "\n")

    v3 = tmp_path / "v3.jsonl.gz"
    write(v3, ["source_only_whole_construct", "implementation_disclosed"])
    assert resolve_analysis_channel(v3) == "implementation_disclosed"
    assert resolve_analysis_channel(v3, "source_only_whole_construct") == (
        "source_only_whole_construct"
    )

    ceiling = tmp_path / "ceiling.jsonl.gz"
    write(ceiling, ["full_executable_contract"])
    assert resolve_analysis_channel(ceiling) == "full_executable_contract"


def test_ceiling_report_uses_predeclared_branches() -> None:
    def readout(rho: float) -> dict:
        return {
            "scope": {"channel": "full_executable_contract"},
            "execution_accounting": {"status_counts": {"valid": 4500}},
            "aggregate": {
                "median_raw_rho": rho,
                "ci95": [rho - 0.1, rho + 0.1],
                "mappings_with_confirmatory_support": 18,
                "median_reliability_ceiling": 0.9,
            },
            "wrong_relation_control": {
                "median_rho_correct_minus_wrong": None,
                "ci95": [None, None],
            },
            "per_mapping": [],
            "claim_limits": [],
        }

    assert "high-ceiling branch" in render_report(readout(0.70))
    assert "low-ceiling branch" in render_report(readout(0.39))
    assert "intermediate-ceiling branch" in render_report(readout(0.55))
