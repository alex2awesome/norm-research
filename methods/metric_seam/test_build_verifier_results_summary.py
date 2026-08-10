from __future__ import annotations

from methods.metric_seam.build_verifier_results_summary import _rate


def test_rate_uses_explicit_denominator_and_does_not_round_counts() -> None:
    assert _rate(1, 3) == {"numerator": 1, "denominator": 3, "percent": 100 / 3}
