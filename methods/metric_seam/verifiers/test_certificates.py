from __future__ import annotations

import math

import pytest

from methods.metric_seam.verifiers.certificates import (
    aggregate_witness_ablation,
    applies_agreement,
    cohen_kappa,
    evaluate_heldout_certificate,
    jointly_applicable_polarity_agreement,
    violated_ablation_pass_rate,
    witness_line_set_jaccard,
)
from methods.metric_seam.verifiers.schema import Span, Verdict


def _verdict(applies: bool, violated: bool = False) -> Verdict:
    return Verdict(
        applies,
        violated,
        (Span("src/check.py", 1, 1),) if applies else (),
    )


def test_cohen_kappa_known_value_and_degenerate_case() -> None:
    assert math.isclose(cohen_kappa([0, 0, 1, 1], [0, 1, 1, 1]), 0.5)
    assert cohen_kappa([True, True], [True, True]) is None
    with pytest.raises(ValueError, match="equal length"):
        cohen_kappa([0], [0, 1])
    with pytest.raises(ValueError, match="nonempty"):
        cohen_kappa([], [])


def test_applies_agreement_keeps_polarity_out_of_the_statistic() -> None:
    left = [_verdict(False), _verdict(True), _verdict(True, True), _verdict(False)]
    right = [_verdict(False), _verdict(False), _verdict(True), _verdict(False)]
    stats = applies_agreement(left, right)
    assert stats.n == 4
    assert stats.agreements == 3
    assert stats.observed_agreement == 0.75
    assert math.isclose(stats.kappa, 0.5)


def test_jointly_applicable_polarity_agreement_filters_nonjoint_items() -> None:
    left = [
        _verdict(True, True),
        _verdict(True),
        _verdict(False),
        _verdict(True, True),
    ]
    right = [
        _verdict(True, True),
        _verdict(True, True),
        _verdict(True),
        _verdict(False),
    ]
    stats = jointly_applicable_polarity_agreement(left, right)
    assert stats.n == 2
    assert stats.agreements == 1
    assert stats.observed_agreement == 0.5
    assert stats.kappa == 0.0

    no_joint = jointly_applicable_polarity_agreement(
        [_verdict(False)], [_verdict(True)]
    )
    assert no_joint.n == 0
    assert no_joint.observed_agreement is None
    assert no_joint.kappa is None


def test_witness_jaccard_uses_union_of_inclusive_line_sets() -> None:
    assert witness_line_set_jaccard(
        [Span("src/a.py", 1, 3)], [Span("src/a.py", 3, 5)]
    ) == 0.2
    assert witness_line_set_jaccard(
        [Span("src/a.py", 1, 2), Span("src/a.py", 2, 3)],
        [Span("src/a.py", 1, 3)],
    ) == 1.0
    assert witness_line_set_jaccard(
        [Span("src/a.py", 10, 10)], [Span("src/b.py", 10, 10)]
    ) == 0.0
    assert witness_line_set_jaccard([], []) == 1.0
    assert witness_line_set_jaccard([Span("src/a.py", 1, 1)], []) == 0.0


def test_witness_ablation_aggregates_named_outcome_transitions() -> None:
    original = [
        _verdict(True, True),
        _verdict(True, True),
        _verdict(True),
        _verdict(False),
    ]
    ablated = [
        _verdict(True),
        _verdict(False),
        _verdict(True),
        _verdict(False),
    ]
    stats = aggregate_witness_ablation(original, ablated)
    assert stats.n == 4
    assert stats.unchanged == 2
    assert stats.changed == 2
    assert stats.changed_fraction == 0.5
    assert stats.transitions == {
        "applies_not_violated->applies_not_violated": 1,
        "not_applicable->not_applicable": 1,
        "violated->applies_not_violated": 1,
        "violated->not_applicable": 1,
    }
    assert violated_ablation_pass_rate(original, ablated) == 1.0


def test_violated_ablation_pass_rate_ignores_nonviolated_originals() -> None:
    original = [
        _verdict(True, True),
        _verdict(True, True),
        _verdict(True, True),
        _verdict(True),
        _verdict(False),
    ]
    ablated = [
        _verdict(True),       # satisfied: pass
        _verdict(False),      # non-applicable: pass
        _verdict(True, True), # still violated: fail
        _verdict(True, True), # ignored
        _verdict(True, True), # ignored
    ]
    assert violated_ablation_pass_rate(original, ablated) == pytest.approx(2 / 3)
    assert violated_ablation_pass_rate([_verdict(True)], [_verdict(False)]) is None


def test_heldout_certificate_enforces_kappa_referent_and_both_ablations() -> None:
    left = [
        _verdict(True, True),
        _verdict(True),
        _verdict(True, True),
        _verdict(True),
    ]
    right = list(left)
    ablated = [_verdict(True), _verdict(True), _verdict(False), _verdict(True)]
    certificate = evaluate_heldout_certificate(
        left,
        right,
        left_ablated=ablated,
        right_ablated=ablated,
    )
    assert certificate.passed
    assert certificate.polarity.kappa == 1.0
    assert certificate.mean_witness_jaccard == 1.0
    assert certificate.left_violated_ablation_rate == 1.0

    no_violations = [_verdict(True), _verdict(True)]
    failed = evaluate_heldout_certificate(
        no_violations,
        no_violations,
        left_ablated=no_violations,
        right_ablated=no_violations,
        same_referent_adjudication_passed=True,
    )
    assert not failed.passed
    assert "polarity_kappa_below_minimum_or_undefined" in failed.failed_checks
    assert "left_witness_ablation_below_minimum_or_undefined" in failed.failed_checks


def test_same_referent_adjudication_only_overrides_line_jaccard() -> None:
    left = [
        Verdict(True, True, (Span("src/a.py", 1, 1),)),
        Verdict(True, False, (Span("src/a.py", 2, 2),)),
    ]
    right = [
        Verdict(True, True, (Span("src/b.py", 1, 1),)),
        Verdict(True, False, (Span("src/b.py", 2, 2),)),
    ]
    left_ablated = [_verdict(False), left[1]]
    right_ablated = [_verdict(False), right[1]]
    without_override = evaluate_heldout_certificate(
        left,
        right,
        left_ablated=left_ablated,
        right_ablated=right_ablated,
    )
    assert not without_override.passed
    assert "witness_referent_agreement_below_minimum" in without_override.failed_checks
    with_override = evaluate_heldout_certificate(
        left,
        right,
        left_ablated=left_ablated,
        right_ablated=right_ablated,
        same_referent_adjudication_passed=True,
    )
    assert with_override.passed
