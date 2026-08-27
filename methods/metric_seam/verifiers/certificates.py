"""Statistics for agreement and witness-ablation certificates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence

from .schema import Span, Verdict


def _paired(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError("paired certificate sequences must have equal length")
    if not left:
        raise ValueError("paired certificate sequences must be nonempty")


def cohen_kappa(left: Sequence[Hashable], right: Sequence[Hashable]) -> float | None:
    """Return unweighted Cohen's kappa, or ``None`` when chance agreement is 1."""

    _paired(left, right)
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(left_counts[label] * right_counts[label] for label in labels) / (n * n)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


@dataclass(frozen=True)
class AgreementStatistics:
    n: int
    agreements: int
    observed_agreement: float | None
    kappa: float | None


def applies_agreement(
    left: Sequence[Verdict], right: Sequence[Verdict]
) -> AgreementStatistics:
    """Agreement on applicability, separated from violation polarity."""

    _paired(left, right)
    left_applies = [verdict.applies for verdict in left]
    right_applies = [verdict.applies for verdict in right]
    agreements = sum(a == b for a, b in zip(left_applies, right_applies))
    return AgreementStatistics(
        n=len(left),
        agreements=agreements,
        observed_agreement=agreements / len(left),
        kappa=cohen_kappa(left_applies, right_applies),
    )


def jointly_applicable_polarity_agreement(
    left: Sequence[Verdict], right: Sequence[Verdict]
) -> AgreementStatistics:
    """Violation-polarity agreement restricted to jointly applicable items."""

    _paired(left, right)
    pairs = [
        (left_verdict.violated, right_verdict.violated)
        for left_verdict, right_verdict in zip(left, right)
        if left_verdict.applies and right_verdict.applies
    ]
    if not pairs:
        return AgreementStatistics(n=0, agreements=0, observed_agreement=None, kappa=None)
    left_polarity = [pair[0] for pair in pairs]
    right_polarity = [pair[1] for pair in pairs]
    agreements = sum(a == b for a, b in pairs)
    return AgreementStatistics(
        n=len(pairs),
        agreements=agreements,
        observed_agreement=agreements / len(pairs),
        kappa=cohen_kappa(left_polarity, right_polarity),
    )


def _line_set(spans: Sequence[Span]) -> frozenset[tuple[str, int]]:
    lines: set[tuple[str, int]] = set()
    for span in spans:
        if not isinstance(span, Span):
            raise TypeError("witness collections must contain Span values")
        lines.update(span.lines())
    return frozenset(lines)


def witness_line_set_jaccard(left: Sequence[Span], right: Sequence[Span]) -> float:
    """Jaccard similarity over inclusive ``(path, line)`` witness identities."""

    left_lines = _line_set(left)
    right_lines = _line_set(right)
    union = left_lines | right_lines
    if not union:
        return 1.0
    return len(left_lines & right_lines) / len(union)


def _outcome_label(verdict: Verdict) -> str:
    if not verdict.applies:
        return "not_applicable"
    return "violated" if verdict.violated else "applies_not_violated"


@dataclass(frozen=True)
class WitnessAblationStatistics:
    n: int
    unchanged: int
    changed: int
    transitions: dict[str, int]

    @property
    def changed_fraction(self) -> float:
        return self.changed / self.n


def aggregate_witness_ablation(
    original: Sequence[Verdict], ablated: Sequence[Verdict]
) -> WitnessAblationStatistics:
    """Aggregate outcome transitions after deleting the certified witness lines."""

    _paired(original, ablated)
    transitions = Counter(
        f"{_outcome_label(before)}->{_outcome_label(after)}"
        for before, after in zip(original, ablated)
    )
    unchanged = sum(
        count
        for transition, count in transitions.items()
        if transition.split("->", 1)[0] == transition.split("->", 1)[1]
    )
    return WitnessAblationStatistics(
        n=len(original),
        unchanged=unchanged,
        changed=len(original) - unchanged,
        transitions=dict(sorted(transitions.items())),
    )


def violated_ablation_pass_rate(
    original: Sequence[Verdict], ablated: Sequence[Verdict]
) -> float | None:
    """Return the violated-only witness-ablation pass rate.

    Only originally violated outcomes enter the denominator.  An ablation passes
    when its outcome flips to either satisfied or not-applicable.  ``None`` means
    there was no originally violated outcome to test.
    """

    _paired(original, ablated)
    eligible = [
        after
        for before, after in zip(original, ablated)
        if before.applies and before.violated
    ]
    if not eligible:
        return None
    return sum(not after.violated for after in eligible) / len(eligible)


@dataclass(frozen=True)
class HeldoutCertificate:
    """Decision-complete certificate over two frozen verifier implementations."""

    passed: bool
    failed_checks: tuple[str, ...]
    applicability: AgreementStatistics
    polarity: AgreementStatistics
    jointly_applicable_witness_n: int
    mean_witness_jaccard: float | None
    same_referent_override: bool
    left_violated_ablation_rate: float | None
    right_violated_ablation_rate: float | None


def evaluate_heldout_certificate(
    left: Sequence[Verdict],
    right: Sequence[Verdict],
    *,
    left_ablated: Sequence[Verdict],
    right_ablated: Sequence[Verdict],
    same_referent_adjudication_passed: bool = False,
    min_kappa: float = 0.80,
    min_witness_jaccard: float = 0.50,
    min_ablation_rate: float = 0.90,
) -> HeldoutCertificate:
    """Apply the frozen heldout certificate thresholds without external labels.

    Polarity kappa and witness overlap are restricted to jointly applicable
    items.  Ablation is implementation-local and restricted to that
    implementation's originally violated items.  A semantic same-referent
    adjudication may replace the line-overlap threshold, but never kappa or
    either ablation threshold.
    """

    _paired(left, right)
    _paired(left, left_ablated)
    _paired(right, right_ablated)
    for name, value in (
        ("min_kappa", min_kappa),
        ("min_witness_jaccard", min_witness_jaccard),
        ("min_ablation_rate", min_ablation_rate),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if type(same_referent_adjudication_passed) is not bool:
        raise TypeError("same_referent_adjudication_passed must be boolean")

    applicability = applies_agreement(left, right)
    polarity = jointly_applicable_polarity_agreement(left, right)
    joint_pairs = [
        (left_verdict, right_verdict)
        for left_verdict, right_verdict in zip(left, right)
        if left_verdict.applies and right_verdict.applies
    ]
    witness_scores = [
        witness_line_set_jaccard(a.witnesses, b.witnesses) for a, b in joint_pairs
    ]
    mean_jaccard = (
        sum(witness_scores) / len(witness_scores) if witness_scores else None
    )
    left_ablation = violated_ablation_pass_rate(left, left_ablated)
    right_ablation = violated_ablation_pass_rate(right, right_ablated)

    failures: list[str] = []
    if polarity.kappa is None or polarity.kappa < min_kappa:
        failures.append("polarity_kappa_below_minimum_or_undefined")
    if not same_referent_adjudication_passed and (
        mean_jaccard is None or mean_jaccard < min_witness_jaccard
    ):
        failures.append("witness_referent_agreement_below_minimum")
    if left_ablation is None or left_ablation < min_ablation_rate:
        failures.append("left_witness_ablation_below_minimum_or_undefined")
    if right_ablation is None or right_ablation < min_ablation_rate:
        failures.append("right_witness_ablation_below_minimum_or_undefined")
    return HeldoutCertificate(
        passed=not failures,
        failed_checks=tuple(failures),
        applicability=applicability,
        polarity=polarity,
        jointly_applicable_witness_n=len(joint_pairs),
        mean_witness_jaccard=mean_jaccard,
        same_referent_override=same_referent_adjudication_passed,
        left_violated_ablation_rate=left_ablation,
        right_violated_ablation_rate=right_ablation,
    )
