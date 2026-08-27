"""Training-only discrimination gate for verifier-native verdicts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Iterable, Literal, Sequence

from .schema import Verdict


class FailureReason(str, Enum):
    """Complete, mutually exclusive top-level failure taxonomy."""

    CORPUS_UNMEASURABLE = "corpus_unmeasurable"
    PROBE_FAILURE = "probe_failure"
    INSUFFICIENT_DISCRIMINATION = "insufficient_discrimination"
    EXECUTION_FAILURE = "execution_failure"


class ProbeOutcome(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INVERTED = "inverted"


@dataclass(frozen=True)
class DiscriminationThresholds:
    min_applies: Fraction = Fraction(20, 100)
    max_applies: Fraction = Fraction(95, 100)
    min_violated_given_applies: Fraction = Fraction(10, 100)
    max_violated_given_applies: Fraction = Fraction(90, 100)
    min_patterns: int = 3
    max_mode: Fraction = Fraction(85, 100)
    min_completeness: Fraction = Fraction(90, 100)
    min_probe_correct: Fraction = Fraction(75, 100)

    def __post_init__(self) -> None:
        unit_interval = (
            self.min_applies,
            self.max_applies,
            self.min_violated_given_applies,
            self.max_violated_given_applies,
            self.max_mode,
            self.min_completeness,
            self.min_probe_correct,
        )
        if any(not Fraction(0) <= value <= Fraction(1) for value in unit_interval):
            raise ValueError("fractional discrimination thresholds must lie in [0, 1]")
        if self.min_applies > self.max_applies:
            raise ValueError("min_applies exceeds max_applies")
        if self.min_violated_given_applies > self.max_violated_given_applies:
            raise ValueError("conditional violation minimum exceeds maximum")
        if self.min_patterns < 1:
            raise ValueError("min_patterns must be positive")


@dataclass(frozen=True)
class TrainObservation:
    """One TRAIN item; ``verdict=None`` denotes a failed execution."""

    item_id: str
    pattern_id: str
    corpus_kind: Literal["natural", "planted"]
    verdict: Verdict | None

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must be nonempty")
        if not self.pattern_id:
            raise ValueError("pattern_id must be nonempty")
        if self.corpus_kind not in ("natural", "planted"):
            raise ValueError("corpus_kind must be natural or planted")
        if self.verdict is not None and not isinstance(self.verdict, Verdict):
            raise TypeError("verdict must be Verdict or None")


@dataclass(frozen=True)
class TrainDiscriminationResult:
    passed: bool
    failure_reason: FailureReason | None
    failed_checks: tuple[str, ...]
    # Unprefixed counts are the natural-corpus gate denominator.  Plants may
    # demonstrate a third behavioral state, but cannot rescue natural-corpus
    # prevalence, resolution, or execution completeness.
    total_items: int
    completed_items: int
    applies_count: int
    violated_count: int
    distinct_patterns: int
    corpus_kinds: frozenset[str]
    modal_count: int
    probe_total: int
    probe_correct: int
    probe_inversions: int
    all_total_items: int
    all_completed_items: int
    all_applies_count: int
    all_violated_count: int
    all_modal_count: int

    @property
    def completeness(self) -> Fraction:
        return Fraction(self.completed_items, self.total_items) if self.total_items else Fraction(0)

    @property
    def applies_rate(self) -> Fraction | None:
        return Fraction(self.applies_count, self.completed_items) if self.completed_items else None

    @property
    def violated_given_applies(self) -> Fraction | None:
        return Fraction(self.violated_count, self.applies_count) if self.applies_count else None

    @property
    def mode_rate(self) -> Fraction | None:
        return Fraction(self.modal_count, self.completed_items) if self.completed_items else None

    @property
    def probe_correct_rate(self) -> Fraction | None:
        return Fraction(self.probe_correct, self.probe_total) if self.probe_total else None


def _duplicates(values: Iterable[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def evaluate_train_discrimination(
    observations: Sequence[TrainObservation],
    probes: Sequence[ProbeOutcome],
    *,
    thresholds: DiscriminationThresholds = DiscriminationThresholds(),
) -> TrainDiscriminationResult:
    """Evaluate one verifier on TRAIN natural examples, plants, and probes.

    Threshold boundaries are inclusive.  Applicability, conditional violation,
    modal fraction, and completeness are computed on natural TRAIN observations
    only.  Planted observations contribute only to behavioral-state/corpus
    coverage; this prevents synthetic cases from manufacturing natural-corpus
    discrimination.  Failure precedence is corpus measurability, execution
    completeness, probe behavior, then distributional discrimination.
    """

    if _duplicates(observation.item_id for observation in observations):
        raise ValueError("TRAIN observation item_id values must be unique")
    if any(not isinstance(probe, ProbeOutcome) for probe in probes):
        raise TypeError("probes must contain ProbeOutcome values")

    natural = [
        observation for observation in observations
        if observation.corpus_kind == "natural"
    ]
    natural_completed = [
        observation for observation in natural if observation.verdict is not None
    ]
    natural_verdicts = [observation.verdict for observation in natural_completed]
    completed = [
        observation for observation in observations if observation.verdict is not None
    ]
    all_verdicts = [observation.verdict for observation in completed]
    applies = sum(verdict.applies for verdict in natural_verdicts)
    violated = sum(verdict.violated for verdict in natural_verdicts)
    all_applies = sum(verdict.applies for verdict in all_verdicts)
    all_violated = sum(verdict.violated for verdict in all_verdicts)
    # Pattern diversity is behavioral: the three possible verifier states.
    # ``pattern_id`` remains provenance metadata but never contributes to this
    # gate, preventing item-specific identifiers from manufacturing diversity.
    patterns = {verdict.state for verdict in all_verdicts}
    kinds = frozenset(observation.corpus_kind for observation in completed)
    modes = Counter(
        (verdict.applies, verdict.violated) for verdict in natural_verdicts
    )
    modal_count = modes.most_common(1)[0][1] if modes else 0
    all_modes = Counter(
        (verdict.applies, verdict.violated) for verdict in all_verdicts
    )
    all_modal_count = all_modes.most_common(1)[0][1] if all_modes else 0
    probe_counts = Counter(probes)
    probe_correct = probe_counts[ProbeOutcome.CORRECT]
    probe_inversions = probe_counts[ProbeOutcome.INVERTED]

    common = dict(
        total_items=len(natural),
        completed_items=len(natural_completed),
        applies_count=applies,
        violated_count=violated,
        distinct_patterns=len(patterns),
        corpus_kinds=kinds,
        modal_count=modal_count,
        probe_total=len(probes),
        probe_correct=probe_correct,
        probe_inversions=probe_inversions,
        all_total_items=len(observations),
        all_completed_items=len(completed),
        all_applies_count=all_applies,
        all_violated_count=all_violated,
        all_modal_count=all_modal_count,
    )
    if not natural or not natural_completed:
        return TrainDiscriminationResult(
            False,
            FailureReason.CORPUS_UNMEASURABLE,
            ("no_completed_natural_train_observations",),
            **common,
        )

    completeness = Fraction(len(natural_completed), len(natural))
    if completeness < thresholds.min_completeness:
        return TrainDiscriminationResult(
            False,
            FailureReason.EXECUTION_FAILURE,
            ("completeness_below_minimum",),
            **common,
        )

    probe_failures: list[str] = []
    probe_rate = Fraction(probe_correct, len(probes)) if probes else Fraction(0)
    if probe_rate < thresholds.min_probe_correct:
        probe_failures.append("probe_accuracy_below_minimum")
    if probe_inversions:
        probe_failures.append("probe_inversion")
    if probe_failures:
        return TrainDiscriminationResult(
            False,
            FailureReason.PROBE_FAILURE,
            tuple(probe_failures),
            **common,
        )

    corpus_failures: list[str] = []
    discrimination_failures: list[str] = []
    applies_rate = Fraction(applies, len(natural_completed))
    if not thresholds.min_applies <= applies_rate <= thresholds.max_applies:
        corpus_failures.append("applies_rate_out_of_bounds")
    conditional = Fraction(violated, applies) if applies else None
    if conditional is None or not (
        thresholds.min_violated_given_applies
        <= conditional
        <= thresholds.max_violated_given_applies
    ):
        corpus_failures.append("violated_given_applies_out_of_bounds")
    if len(patterns) < thresholds.min_patterns or kinds != {"natural", "planted"}:
        discrimination_failures.append("insufficient_pattern_corpus_coverage")
    mode_rate = Fraction(modal_count, len(natural_completed))
    if mode_rate > thresholds.max_mode:
        discrimination_failures.append("mode_fraction_above_maximum")
    all_distribution_failures = tuple(corpus_failures + discrimination_failures)
    if corpus_failures:
        return TrainDiscriminationResult(
            False,
            FailureReason.CORPUS_UNMEASURABLE,
            all_distribution_failures,
            **common,
        )
    if discrimination_failures:
        return TrainDiscriminationResult(
            False,
            FailureReason.INSUFFICIENT_DISCRIMINATION,
            all_distribution_failures,
            **common,
        )

    return TrainDiscriminationResult(True, None, (), **common)
