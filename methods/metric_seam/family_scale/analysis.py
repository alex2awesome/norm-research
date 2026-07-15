"""Family-scale analysis without denominator or certificate collapse.

The public objects in this module are deliberately task-agnostic.  Callers
must supply one family/domain/corpus identity explicitly; summaries never pool
domains or corpora merely because they share a family label.

Limitations
-----------
* C-index measures ordering concordance, not construct fidelity.
* Reliability normalization uses the signed-concordance analogue of the
  classical square-root attenuation ceiling.  It is undefined when the two
  passes have non-positive signed concordance and may exceed ``[0, 1]``; such
  values are diagnostics rather than probabilities.
* Clustered intervals condition on the observed families of clusters.  They do
  not create population representativeness.
* G1/G2 summaries are inputs to a certificate.  They do not establish whole-
  metric codability or tacitness.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
from typing import Hashable, Iterable, Mapping, Sequence


GroupKey = tuple[str, str, str]
STATES = frozenset({"not_applicable", "satisfied", "violated"})
G2_POLARITIES = frozenset({"positive_true_violation", "negative_proxy_trap"})


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _finite(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class FunnelCounts:
    """Ordered family funnel with every requested denominator retained."""

    family: str
    domain: str
    corpus: str
    proposed: int
    base_rate_killed: int
    authored: int
    gate_killed: int
    operational: int

    def __post_init__(self) -> None:
        for field in ("family", "domain", "corpus"):
            _nonempty(getattr(self, field), field)
        for field in (
            "proposed",
            "base_rate_killed",
            "authored",
            "gate_killed",
            "operational",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.base_rate_killed > self.proposed:
            raise ValueError("base_rate_killed exceeds proposed")
        if self.authored > self.after_base_rate:
            raise ValueError("authored exceeds candidates surviving the base-rate kill")
        if self.gate_killed > self.authored:
            raise ValueError("gate_killed exceeds authored")
        if self.operational > self.after_gate:
            raise ValueError("operational exceeds authored candidates surviving the gate")

    @property
    def key(self) -> GroupKey:
        return (self.family, self.domain, self.corpus)

    @property
    def after_base_rate(self) -> int:
        return self.proposed - self.base_rate_killed

    @property
    def unauthored(self) -> int:
        return self.after_base_rate - self.authored

    @property
    def after_gate(self) -> int:
        return self.authored - self.gate_killed

    @property
    def nonoperational_after_gate(self) -> int:
        return self.after_gate - self.operational

    @property
    def operational_per_proposed(self) -> float | None:
        return self.operational / self.proposed if self.proposed else None

    @property
    def operational_per_authored(self) -> float | None:
        return self.operational / self.authored if self.authored else None


@dataclass(frozen=True)
class ResolutionStatistics:
    n: int
    distinct: int
    mode_count: int
    mode_fraction: float | None
    tied_pair_count: int
    pair_count: int
    tie_fraction: float | None
    entropy_bits: float | None
    normalized_entropy: float | None


def resolution_statistics(values: Sequence[Hashable]) -> ResolutionStatistics:
    """Return exact-value resolution diagnostics.

    ``tie_fraction`` is the fraction of unordered observation pairs carrying
    identical values, rather than the less informative ``1 - distinct/n``.
    """

    counts = Counter(values)
    n = len(values)
    if not n:
        return ResolutionStatistics(0, 0, 0, None, 0, 0, None, None, None)
    mode = max(counts.values())
    pairs = n * (n - 1) // 2
    tied = sum(count * (count - 1) // 2 for count in counts.values())
    entropy = -sum(
        (count / n) * math.log2(count / n) for count in counts.values()
    )
    normalized = entropy / math.log2(n) if n > 1 else 0.0
    return ResolutionStatistics(
        n=n,
        distinct=len(counts),
        mode_count=mode,
        mode_fraction=mode / n,
        tied_pair_count=tied,
        pair_count=pairs,
        tie_fraction=tied / pairs if pairs else None,
        entropy_bits=entropy,
        normalized_entropy=normalized,
    )


@dataclass(frozen=True)
class CIndexResult:
    n: int
    comparable_pairs: int
    concordant_pairs: int
    discordant_pairs: int
    tied_prediction_pairs: int
    value: float | None


def c_index(targets: Sequence[float], predictions: Sequence[float]) -> CIndexResult:
    """Compute Harrell-style concordance with prediction ties worth one half.

    Pairs tied on the target are not comparable.  The implementation is
    ``O(n log n)`` and retains integer pair accounting.
    """

    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")
    rows = [
        (_finite(target, "target"), _finite(prediction, "prediction"))
        for target, prediction in zip(targets, predictions)
    ]
    n = len(rows)
    if n < 2:
        return CIndexResult(n, 0, 0, 0, 0, None)
    prediction_values = sorted({prediction for _, prediction in rows})
    ranks = {value: index + 1 for index, value in enumerate(prediction_values)}
    tree = [0] * (len(prediction_values) + 1)

    def add(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    concordant = discordant = tied = seen = 0
    ordered = sorted(rows)
    cursor = 0
    while cursor < n:
        end = cursor + 1
        while end < n and ordered[end][0] == ordered[cursor][0]:
            end += 1
        for _, prediction in ordered[cursor:end]:
            rank = ranks[prediction]
            lower = prefix(rank - 1)
            equal = prefix(rank) - lower
            greater = seen - lower - equal
            concordant += lower
            tied += equal
            discordant += greater
        for _, prediction in ordered[cursor:end]:
            add(ranks[prediction])
            seen += 1
        cursor = end
    comparable = concordant + discordant + tied
    value = (
        (concordant + 0.5 * tied) / comparable if comparable else None
    )
    return CIndexResult(
        n=n,
        comparable_pairs=comparable,
        concordant_pairs=concordant,
        discordant_pairs=discordant,
        tied_prediction_pairs=tied,
        value=value,
    )


@dataclass(frozen=True)
class ReliabilityCeilingResult:
    primary_c_index: float | None
    pass_c_index: float | None
    signed_pass_reliability: float | None
    signed_reliability_ceiling: float | None
    normalized_signed_concordance: float | None
    normalized_c_index: float | None


def reliability_ceiling_normalization(
    targets: Sequence[float],
    predictions: Sequence[float],
    pass_one: Sequence[float],
    pass_two: Sequence[float],
) -> ReliabilityCeilingResult:
    """Normalize primary concordance by a two-pass reliability ceiling.

    C-index is mapped to signed concordance ``2C-1``.  The ceiling is the
    square root of positive pass-to-pass signed concordance, mirroring the
    classical attenuation ceiling without pretending C-index is Pearson rho.
    """

    lengths = {len(targets), len(predictions), len(pass_one), len(pass_two)}
    if len(lengths) != 1:
        raise ValueError("all reliability inputs must have equal length")
    primary = c_index(targets, predictions).value
    reliability = c_index(pass_one, pass_two).value
    signed_reliability = 2 * reliability - 1 if reliability is not None else None
    if primary is None or signed_reliability is None or signed_reliability <= 0:
        return ReliabilityCeilingResult(
            primary, reliability, signed_reliability, None, None, None
        )
    ceiling = math.sqrt(signed_reliability)
    normalized_signed = (2 * primary - 1) / ceiling
    return ReliabilityCeilingResult(
        primary_c_index=primary,
        pass_c_index=reliability,
        signed_pass_reliability=signed_reliability,
        signed_reliability_ceiling=ceiling,
        normalized_signed_concordance=normalized_signed,
        normalized_c_index=0.5 + 0.5 * normalized_signed,
    )


@dataclass(frozen=True)
class ConcordanceObservation:
    target: float
    prediction: float
    metric_id: str | None = None
    document_id: str | None = None
    call_id: str | None = None

    def __post_init__(self) -> None:
        _finite(self.target, "target")
        _finite(self.prediction, "prediction")
        for field in ("metric_id", "document_id", "call_id"):
            value = getattr(self, field)
            if value is not None:
                _nonempty(value, field)


@dataclass(frozen=True)
class ClusteredBootstrapResult:
    point: float | None
    lower: float | None
    upper: float | None
    draws_requested: int
    draws_valid: int
    seed: int
    cluster_levels: tuple[str, ...]


def _available_cluster_levels(
    rows: Sequence[ConcordanceObservation], requested: Sequence[str]
) -> tuple[str, ...]:
    levels: list[str] = []
    allowed = {"metric_id", "document_id", "call_id"}
    if any(level not in allowed for level in requested):
        raise ValueError("cluster levels must be metric_id/document_id/call_id")
    for level in requested:
        present = [getattr(row, level) is not None for row in rows]
        if any(present) and not all(present):
            raise ValueError(f"cluster level {level} is only partially populated")
        if present and all(present):
            levels.append(level)
    return tuple(levels)


def _resample_nested(
    rows: Sequence[ConcordanceObservation],
    levels: Sequence[str],
    rng: random.Random,
) -> list[ConcordanceObservation]:
    if not levels:
        return list(rows)
    level = levels[0]
    groups: dict[str, list[ConcordanceObservation]] = defaultdict(list)
    for row in rows:
        value = getattr(row, level)
        assert value is not None
        groups[value].append(row)
    keys = sorted(groups)
    sampled: list[ConcordanceObservation] = []
    for _ in keys:
        key = rng.choice(keys)
        sampled.extend(_resample_nested(groups[key], levels[1:], rng))
    return sampled


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def clustered_bootstrap_c_index(
    observations: Sequence[ConcordanceObservation],
    *,
    draws: int = 2000,
    seed: int = 20260714,
    cluster_order: Sequence[str] = ("metric_id", "document_id", "call_id"),
) -> ClusteredBootstrapResult:
    """Hierarchically resample every available cluster level."""

    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not observations:
        raise ValueError("bootstrap requires observations")
    levels = _available_cluster_levels(observations, cluster_order)
    point = c_index(
        [row.target for row in observations],
        [row.prediction for row in observations],
    ).value
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sampled = (
            _resample_nested(observations, levels, rng)
            if levels
            else [rng.choice(observations) for _ in observations]
        )
        estimate = c_index(
            [row.target for row in sampled],
            [row.prediction for row in sampled],
        ).value
        if estimate is not None:
            estimates.append(estimate)
    return ClusteredBootstrapResult(
        point=point,
        lower=_quantile(estimates, 0.025) if estimates else None,
        upper=_quantile(estimates, 0.975) if estimates else None,
        draws_requested=draws,
        draws_valid=len(estimates),
        seed=seed,
        cluster_levels=levels if levels else ("observation",),
    )


@dataclass(frozen=True)
class BatchCalibrationObservation:
    observation_id: str
    batch_id: str
    unbatched: float
    batched: float

    def __post_init__(self) -> None:
        _nonempty(self.observation_id, "observation_id")
        _nonempty(self.batch_id, "batch_id")
        _finite(self.unbatched, "unbatched")
        _finite(self.batched, "batched")


@dataclass(frozen=True)
class BatchCalibrationResult:
    n: int
    batch_count: int
    mean_difference: float
    mean_absolute_difference: float
    root_mean_squared_difference: float
    pearson: float | None
    concordance: CIndexResult
    batch_mean_differences: Mapping[str, float]
    max_absolute_batch_mean_difference: float


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0 or right_ss == 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def batching_calibration(
    observations: Sequence[BatchCalibrationObservation],
) -> BatchCalibrationResult:
    """Compare matched batched and unbatched calls without selecting a winner."""

    if not observations:
        raise ValueError("batch calibration requires observations")
    ids = [row.observation_id for row in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("batch calibration observation IDs must be unique")
    differences = [row.batched - row.unbatched for row in observations]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, difference in zip(observations, differences):
        grouped[row.batch_id].append(difference)
    batch_means = {
        batch: sum(values) / len(values) for batch, values in sorted(grouped.items())
    }
    unbatched = [row.unbatched for row in observations]
    batched = [row.batched for row in observations]
    return BatchCalibrationResult(
        n=len(observations),
        batch_count=len(grouped),
        mean_difference=sum(differences) / len(differences),
        mean_absolute_difference=sum(abs(value) for value in differences)
        / len(differences),
        root_mean_squared_difference=math.sqrt(
            sum(value * value for value in differences) / len(differences)
        ),
        pearson=_pearson(unbatched, batched),
        concordance=c_index(unbatched, batched),
        batch_mean_differences=batch_means,
        max_absolute_batch_mean_difference=max(abs(value) for value in batch_means.values()),
    )


@dataclass(frozen=True)
class G1Observation:
    family: str
    domain: str
    corpus: str
    metric_id: str
    unit_id: str
    item_id: str
    left_state: str
    right_state: str
    left_witnesses: frozenset[str]
    right_witnesses: frozenset[str]

    def __post_init__(self) -> None:
        for field in ("family", "domain", "corpus", "metric_id", "unit_id", "item_id"):
            _nonempty(getattr(self, field), field)
        if self.left_state not in STATES or self.right_state not in STATES:
            raise ValueError("G1 states must use the frozen three-state vocabulary")
        for field in ("left_witnesses", "right_witnesses"):
            values = getattr(self, field)
            if not isinstance(values, frozenset) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{field} must be a frozenset of witness identities")
        if self.left_state != "not_applicable" and not self.left_witnesses:
            raise ValueError("an applicable left G1 outcome needs a witness")
        if self.right_state != "not_applicable" and not self.right_witnesses:
            raise ValueError("an applicable right G1 outcome needs a witness")
        if self.left_state == "not_applicable" and self.left_witnesses:
            raise ValueError("a non-applicable left G1 outcome cannot have witnesses")
        if self.right_state == "not_applicable" and self.right_witnesses:
            raise ValueError("a non-applicable right G1 outcome cannot have witnesses")

    @property
    def key(self) -> GroupKey:
        return (self.family, self.domain, self.corpus)


def _kappa(left: Sequence[Hashable], right: Sequence[Hashable]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[label] * right_counts[label]
        for label in set(left_counts) | set(right_counts)
    ) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class G1Summary:
    family: str
    domain: str
    corpus: str
    n: int
    state_agreement: float
    state_kappa: float | None
    applicability_agreement: float
    applicability_kappa: float | None
    jointly_applicable_n: int
    polarity_agreement: float | None
    polarity_kappa: float | None
    mean_witness_jaccard: float | None
    g1_ready: bool
    failed_checks: tuple[str, ...]

    @property
    def key(self) -> GroupKey:
        return (self.family, self.domain, self.corpus)


def summarize_g1(
    observations: Sequence[G1Observation],
    *,
    min_kappa: float = 0.80,
    min_witness_jaccard: float = 0.50,
) -> dict[GroupKey, G1Summary]:
    """Summarize dual implementation agreement within each domain/corpus."""

    for name, value in (("min_kappa", min_kappa), ("min_witness_jaccard", min_witness_jaccard)):
        value = _finite(value, name)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    groups: dict[GroupKey, list[G1Observation]] = defaultdict(list)
    seen: set[tuple[GroupKey, str, str, str]] = set()
    for row in observations:
        identity = (row.key, row.metric_id, row.unit_id, row.item_id)
        if identity in seen:
            raise ValueError("duplicate G1 metric/unit/item identity")
        seen.add(identity)
        groups[row.key].append(row)
    result: dict[GroupKey, G1Summary] = {}
    for key, rows in sorted(groups.items()):
        left = [row.left_state for row in rows]
        right = [row.right_state for row in rows]
        left_applies = [value != "not_applicable" for value in left]
        right_applies = [value != "not_applicable" for value in right]
        joint = [
            row for row in rows
            if row.left_state != "not_applicable" and row.right_state != "not_applicable"
        ]
        left_polarity = [row.left_state == "violated" for row in joint]
        right_polarity = [row.right_state == "violated" for row in joint]
        witness = [
            _jaccard(row.left_witnesses, row.right_witnesses) for row in joint
        ]
        state_kappa = _kappa(left, right)
        polarity_kappa = _kappa(left_polarity, right_polarity)
        mean_witness = sum(witness) / len(witness) if witness else None
        failures: list[str] = []
        if state_kappa is None or state_kappa < min_kappa:
            failures.append("state_kappa_below_minimum_or_undefined")
        if polarity_kappa is None or polarity_kappa < min_kappa:
            failures.append("polarity_kappa_below_minimum_or_undefined")
        if mean_witness is None or mean_witness < min_witness_jaccard:
            failures.append("witness_overlap_below_minimum_or_undefined")
        result[key] = G1Summary(
            family=key[0],
            domain=key[1],
            corpus=key[2],
            n=len(rows),
            state_agreement=sum(a == b for a, b in zip(left, right)) / len(rows),
            state_kappa=state_kappa,
            applicability_agreement=sum(a == b for a, b in zip(left_applies, right_applies)) / len(rows),
            applicability_kappa=_kappa(left_applies, right_applies),
            jointly_applicable_n=len(joint),
            polarity_agreement=(
                sum(a == b for a, b in zip(left_polarity, right_polarity)) / len(joint)
                if joint else None
            ),
            polarity_kappa=polarity_kappa,
            mean_witness_jaccard=mean_witness,
            g1_ready=not failures,
            failed_checks=tuple(failures),
        )
    return result


@dataclass(frozen=True)
class G2ControlObservation:
    family: str
    domain: str
    corpus: str
    implementation_id: str
    control_id: str
    polarity: str
    fired: bool

    def __post_init__(self) -> None:
        for field in (
            "family",
            "domain",
            "corpus",
            "implementation_id",
            "control_id",
        ):
            _nonempty(getattr(self, field), field)
        if self.polarity not in G2_POLARITIES:
            raise ValueError("invalid G2 control polarity")
        if type(self.fired) is not bool:
            raise TypeError("G2 fired must be boolean")

    @property
    def key(self) -> GroupKey:
        return (self.family, self.domain, self.corpus)

    @property
    def passed(self) -> bool:
        return self.fired if self.polarity == "positive_true_violation" else not self.fired


@dataclass(frozen=True)
class G2ImplementationSummary:
    implementation_id: str
    positive_n: int
    positive_passed: int
    proxy_trap_n: int
    proxy_trap_passed: int
    passed: bool


@dataclass(frozen=True)
class G2Summary:
    family: str
    domain: str
    corpus: str
    implementation_count: int
    implementations: Mapping[str, G2ImplementationSummary]
    positive_n: int
    positive_passed: int
    proxy_trap_n: int
    proxy_trap_passed: int
    g2_ready: bool
    failed_checks: tuple[str, ...]

    @property
    def key(self) -> GroupKey:
        return (self.family, self.domain, self.corpus)


def summarize_g2(
    observations: Sequence[G2ControlObservation],
    *,
    min_implementations: int = 2,
) -> dict[GroupKey, G2Summary]:
    """Require positive controls and proxy traps for every implementation."""

    if (
        isinstance(min_implementations, bool)
        or not isinstance(min_implementations, int)
        or min_implementations < 1
    ):
        raise ValueError("min_implementations must be positive")
    groups: dict[GroupKey, list[G2ControlObservation]] = defaultdict(list)
    seen: set[tuple[GroupKey, str, str]] = set()
    for row in observations:
        identity = (row.key, row.implementation_id, row.control_id)
        if identity in seen:
            raise ValueError("duplicate G2 implementation/control identity")
        seen.add(identity)
        groups[row.key].append(row)
    output: dict[GroupKey, G2Summary] = {}
    for key, rows in sorted(groups.items()):
        by_implementation: dict[str, list[G2ControlObservation]] = defaultdict(list)
        for row in rows:
            by_implementation[row.implementation_id].append(row)
        implementation_summaries: dict[str, G2ImplementationSummary] = {}
        for implementation, values in sorted(by_implementation.items()):
            positive = [row for row in values if row.polarity == "positive_true_violation"]
            traps = [row for row in values if row.polarity == "negative_proxy_trap"]
            positive_passed = sum(row.passed for row in positive)
            traps_passed = sum(row.passed for row in traps)
            implementation_summaries[implementation] = G2ImplementationSummary(
                implementation_id=implementation,
                positive_n=len(positive),
                positive_passed=positive_passed,
                proxy_trap_n=len(traps),
                proxy_trap_passed=traps_passed,
                passed=(
                    bool(positive)
                    and bool(traps)
                    and positive_passed == len(positive)
                    and traps_passed == len(traps)
                ),
            )
        failures: list[str] = []
        if len(implementation_summaries) < min_implementations:
            failures.append("insufficient_independent_implementations")
        if any(not value.positive_n for value in implementation_summaries.values()):
            failures.append("implementation_missing_positive_controls")
        if any(not value.proxy_trap_n for value in implementation_summaries.values()):
            failures.append("implementation_missing_proxy_traps")
        if any(
            value.positive_passed != value.positive_n
            for value in implementation_summaries.values()
        ):
            failures.append("positive_control_failure")
        if any(
            value.proxy_trap_passed != value.proxy_trap_n
            for value in implementation_summaries.values()
        ):
            failures.append("proxy_trap_fired")
        output[key] = G2Summary(
            family=key[0],
            domain=key[1],
            corpus=key[2],
            implementation_count=len(implementation_summaries),
            implementations=implementation_summaries,
            positive_n=sum(value.positive_n for value in implementation_summaries.values()),
            positive_passed=sum(value.positive_passed for value in implementation_summaries.values()),
            proxy_trap_n=sum(value.proxy_trap_n for value in implementation_summaries.values()),
            proxy_trap_passed=sum(value.proxy_trap_passed for value in implementation_summaries.values()),
            g2_ready=not failures,
            failed_checks=tuple(failures),
        )
    return output


@dataclass(frozen=True)
class FamilyCertificateInput:
    funnel: FunnelCounts
    g1: G1Summary | None
    g2: G2Summary | None
    ready_for_family_certificate: bool
    blockers: tuple[str, ...]


def assemble_family_certificate_inputs(
    funnels: Sequence[FunnelCounts],
    g1: Mapping[GroupKey, G1Summary],
    g2: Mapping[GroupKey, G2Summary],
) -> dict[GroupKey, FamilyCertificateInput]:
    """Join denominator, G1, and G2 inputs without changing the population."""

    output: dict[GroupKey, FamilyCertificateInput] = {}
    for funnel in funnels:
        if funnel.key in output:
            raise ValueError("duplicate family/domain/corpus funnel")
        g1_value = g1.get(funnel.key)
        g2_value = g2.get(funnel.key)
        blockers: list[str] = []
        if funnel.operational == 0:
            blockers.append("no_operational_units")
        if g1_value is None:
            blockers.append("missing_g1")
        elif not g1_value.g1_ready:
            blockers.append("g1_not_ready")
        if g2_value is None:
            blockers.append("missing_g2")
        elif not g2_value.g2_ready:
            blockers.append("g2_not_ready")
        output[funnel.key] = FamilyCertificateInput(
            funnel=funnel,
            g1=g1_value,
            g2=g2_value,
            ready_for_family_certificate=not blockers,
            blockers=tuple(blockers),
        )
    extra = (set(g1) | set(g2)) - set(output)
    if extra:
        raise ValueError(f"G1/G2 groups lack a proposed-funnel denominator: {sorted(extra)}")
    return output
