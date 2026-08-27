from __future__ import annotations

import math

from methods.metric_seam.analyze_code_review_applicability_ladder import (
    build_comparison,
    summarize_status_counts,
)


def test_prompt_status_summary_keeps_valid_and_expected_denominators() -> None:
    summary = summarize_status_counts(
        {
            "valid": 90,
            "contract_error": 10,
            "not_applicable": 45,
            "applicable_abstain": 9,
            "scored": 36,
        },
        100,
    )

    assert summary["rates_over_expected_rows"]["not_applicable"] == 0.45
    assert summary["rates_over_valid_rows"]["not_applicable"] == 0.5
    assert summary["unscored_rate_over_expected_rows"] == 0.54
    assert summary["unscored_rate_over_valid_rows"] == 0.6


def test_comparison_does_not_collapse_code_abstention_into_applicability() -> None:
    code = {
        "rates": {"not_applicable": 0.2, "abstained": 0.3, "scored": 0.5},
        "unscored_rate": 0.5,
    }
    implementation = {
        "rates_over_valid_rows": {"not_applicable": 0.84},
        "unscored_rate_over_valid_rows": 0.85,
    }
    ceiling = {
        "rates_over_valid_rows": {"not_applicable": 0.25},
        "unscored_rate_over_valid_rows": 0.52,
    }

    comparison = build_comparison(code, implementation, ceiling)

    assert math.isclose(
        comparison["differences_from_code"]["ceiling_not_applicable_minus_code"],
        0.05,
    )
    assert math.isclose(
        comparison["differences_from_code"]["ceiling_unscored_minus_code"],
        0.02,
    )
    assert comparison[
        "ceiling_is_closer_to_code_not_applicability_than_implementation"
    ]
    assert "No numerical threshold" in comparison["interpretation_limit"]
