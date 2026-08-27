from __future__ import annotations

from fractions import Fraction

import pytest

from methods.metric_seam.verifiers.schema import Span, Verdict
from methods.metric_seam.verifiers.train_gate import (
    DiscriminationThresholds,
    FailureReason,
    ProbeOutcome,
    TrainObservation,
    evaluate_train_discrimination,
)


def _verdict(applies: bool, violated: bool = False) -> Verdict:
    return Verdict(
        applies,
        violated,
        (Span("src/check.py", 1, 1),) if applies else (),
    )


def _observations(
    *,
    not_applicable: int,
    applies_not_violated: int,
    violated: int,
    failed: int = 0,
    patterns: int = 3,
    both_corpora: bool = True,
) -> list[TrainObservation]:
    natural_verdicts = (
        [_verdict(False)] * not_applicable
        + [_verdict(True)] * applies_not_violated
        + [_verdict(True, True)] * violated
        + [None] * failed
    )
    rows = []
    for index, verdict in enumerate(natural_verdicts):
        rows.append(
            TrainObservation(
                item_id=f"item-{index}",
                pattern_id=f"pattern-{index % patterns}",
                corpus_kind="natural",
                verdict=verdict,
            )
        )
    if both_corpora:
        for offset, verdict in enumerate(
            (_verdict(False), _verdict(True), _verdict(True, True))
        ):
            rows.append(
                TrainObservation(
                    item_id=f"plant-{offset}",
                    pattern_id=f"plant-pattern-{offset}",
                    corpus_kind="planted",
                    verdict=verdict,
                )
            )
    return rows


PASSING_PROBES = [
    ProbeOutcome.CORRECT,
    ProbeOutcome.CORRECT,
    ProbeOutcome.CORRECT,
    ProbeOutcome.INCORRECT,
]


def test_default_gate_passes_and_reports_exact_rational_rates() -> None:
    result = evaluate_train_discrimination(
        _observations(not_applicable=8, applies_not_violated=6, violated=6),
        PASSING_PROBES,
    )
    assert result.passed is True
    assert result.failure_reason is None
    assert result.applies_rate == Fraction(3, 5)
    assert result.violated_given_applies == Fraction(1, 2)
    assert result.completeness == 1
    assert result.probe_correct_rate == Fraction(3, 4)
    assert result.corpus_kinds == {"natural", "planted"}
    assert result.distinct_patterns == 3
    assert result.all_total_items == result.total_items + 3
    assert result.all_completed_items == result.completed_items + 3


@pytest.mark.parametrize(
    "counts,expected_rate",
    [
        ((80, 18, 2), Fraction(1, 5)),
        ((5, 45, 50), Fraction(95, 100)),
    ],
)
def test_applies_boundaries_are_inclusive(
    counts: tuple[int, int, int], expected_rate: Fraction
) -> None:
    result = evaluate_train_discrimination(
        _observations(
            not_applicable=counts[0],
            applies_not_violated=counts[1],
            violated=counts[2],
        ),
        PASSING_PROBES,
    )
    assert result.passed
    assert result.applies_rate == expected_rate


@pytest.mark.parametrize(
    "compliant,violated,expected_rate",
    [(45, 5, Fraction(1, 10)), (8, 72, Fraction(9, 10))],
)
def test_conditional_violation_boundaries_are_inclusive(
    compliant: int, violated: int, expected_rate: Fraction
) -> None:
    result = evaluate_train_discrimination(
        _observations(
            not_applicable=20,
            applies_not_violated=compliant,
            violated=violated,
        ),
        PASSING_PROBES,
    )
    assert result.passed
    assert result.violated_given_applies == expected_rate


def test_mode_boundary_is_inclusive_but_values_above_it_fail() -> None:
    at_boundary = evaluate_train_discrimination(
        _observations(not_applicable=5, applies_not_violated=10, violated=85),
        PASSING_PROBES,
    )
    assert at_boundary.passed
    assert at_boundary.mode_rate == Fraction(85, 100)

    above = evaluate_train_discrimination(
        _observations(not_applicable=5, applies_not_violated=9, violated=86),
        PASSING_PROBES,
        thresholds=DiscriminationThresholds(max_violated_given_applies=Fraction(1)),
    )
    assert above.failure_reason is FailureReason.INSUFFICIENT_DISCRIMINATION
    assert "mode_fraction_above_maximum" in above.failed_checks


def test_completeness_boundary_and_execution_failure_taxonomy() -> None:
    boundary = evaluate_train_discrimination(
        _observations(
            not_applicable=30, applies_not_violated=30, violated=30, failed=10
        ),
        PASSING_PROBES,
    )
    assert boundary.passed
    assert boundary.completeness == Fraction(9, 10)

    below = evaluate_train_discrimination(
        _observations(
            not_applicable=30, applies_not_violated=30, violated=29, failed=11
        ),
        PASSING_PROBES,
    )
    assert below.failure_reason is FailureReason.EXECUTION_FAILURE
    assert below.failed_checks == ("completeness_below_minimum",)


def test_probes_require_75_percent_correct_and_zero_inversions() -> None:
    rows = _observations(not_applicable=8, applies_not_violated=6, violated=6)
    assert evaluate_train_discrimination(rows, PASSING_PROBES).passed

    low = evaluate_train_discrimination(
        rows,
        [ProbeOutcome.CORRECT, ProbeOutcome.CORRECT, ProbeOutcome.INCORRECT],
    )
    assert low.failure_reason is FailureReason.PROBE_FAILURE
    assert "probe_accuracy_below_minimum" in low.failed_checks

    inverted = evaluate_train_discrimination(
        rows,
        [
            ProbeOutcome.CORRECT,
            ProbeOutcome.CORRECT,
            ProbeOutcome.CORRECT,
            ProbeOutcome.INVERTED,
        ],
    )
    assert inverted.probe_correct_rate == Fraction(3, 4)
    assert inverted.failure_reason is FailureReason.PROBE_FAILURE
    assert inverted.failed_checks == ("probe_inversion",)


def test_pattern_coverage_counts_verdict_states_not_caller_pattern_ids() -> None:
    # Unique caller-controlled labels cannot make a one-state verifier look diverse.
    too_few_rows = _observations(
        not_applicable=0,
        applies_not_violated=20,
        violated=0,
        patterns=20,
        both_corpora=False,
    )
    assert len({row.pattern_id for row in too_few_rows}) == 20
    too_few = evaluate_train_discrimination(too_few_rows, PASSING_PROBES)
    assert too_few.failure_reason is FailureReason.CORPUS_UNMEASURABLE
    assert too_few.distinct_patterns == 1
    assert "insufficient_pattern_corpus_coverage" in too_few.failed_checks

    # Conversely, repeated provenance labels do not hide three observed states.
    repeated_labels = evaluate_train_discrimination(
        _observations(
            not_applicable=8,
            applies_not_violated=6,
            violated=6,
            patterns=1,
        ),
        PASSING_PROBES,
    )
    assert repeated_labels.passed
    assert repeated_labels.distinct_patterns == 3


def test_pattern_coverage_requires_natural_and_planted_corpora() -> None:
    one_corpus = evaluate_train_discrimination(
        _observations(
            not_applicable=8,
            applies_not_violated=6,
            violated=6,
            both_corpora=False,
        ),
        PASSING_PROBES,
    )
    assert one_corpus.failure_reason is FailureReason.INSUFFICIENT_DISCRIMINATION


def test_plants_cannot_rescue_degenerate_natural_prevalence_or_mode() -> None:
    rows = [
        TrainObservation(
            item_id=f"natural-{index}",
            pattern_id="natural-flat",
            corpus_kind="natural",
            verdict=_verdict(True),
        )
        for index in range(20)
    ]
    rows.extend(
        [
            TrainObservation("plant-na", "plant-na", "planted", _verdict(False)),
            TrainObservation("plant-ok", "plant-ok", "planted", _verdict(True)),
            TrainObservation(
                "plant-bad", "plant-bad", "planted", _verdict(True, True)
            ),
        ]
    )
    result = evaluate_train_discrimination(rows, PASSING_PROBES)
    assert result.distinct_patterns == 3
    assert result.all_applies_count == 22
    assert result.applies_count == 20
    assert result.failure_reason is FailureReason.CORPUS_UNMEASURABLE
    assert "applies_rate_out_of_bounds" in result.failed_checks
    assert "violated_given_applies_out_of_bounds" in result.failed_checks
    assert "mode_fraction_above_maximum" in result.failed_checks


def test_unmeasurable_corpus_and_input_invariants() -> None:
    result = evaluate_train_discrimination([], PASSING_PROBES)
    assert result.failure_reason is FailureReason.CORPUS_UNMEASURABLE

    failed = _observations(
        not_applicable=0, applies_not_violated=0, violated=0, failed=4
    )
    assert (
        evaluate_train_discrimination(failed, PASSING_PROBES).failure_reason
        is FailureReason.CORPUS_UNMEASURABLE
    )

    duplicated = _observations(not_applicable=8, applies_not_violated=6, violated=6)
    duplicated[1] = TrainObservation(
        item_id=duplicated[0].item_id,
        pattern_id=duplicated[1].pattern_id,
        corpus_kind=duplicated[1].corpus_kind,
        verdict=duplicated[1].verdict,
    )
    with pytest.raises(ValueError, match="unique"):
        evaluate_train_discrimination(duplicated, PASSING_PROBES)
