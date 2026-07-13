"""Typed outcome vocabulary for the metric-seam reconstruction-v2 lane.

The historical metric-seam artifacts use several overlapping meanings of
``articulable``, ``verifiable`` and ``ground truth``.  This module is the canonical
v2 vocabulary.  It deliberately keeps two questions separate:

* articulability: can a prompt/LLM program implement the articulated relation?
* verifiability: can executable code issue a replayable, scoped certificate?

Reconstruction/isomorphism is agreement with the frozen reference, not a synonym for
either axis.
Executable code may disagree with the reference and still earn the stronger, narrowly
scoped ``CONSTRUCTIVE_EXTENSION`` verdict, but only when a code-native certificate
adjudicates the disagreement.  Correlation alone can never establish that verdict.

Success is constructive and channel-specific: a prompt program witnesses articulability,
while a replayable executable certificate witnesses verifiability.  Failure is only bounded
non-discovery within the frozen program class, capabilities, representation, and budget; it
is not evidence that the relation is tacit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class Status(str, Enum):
    """Three-valued status used so missing evidence is never silently a failure."""

    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"


class DiscoveryMode(str, Enum):
    """How the decomposition/program entered the experiment."""

    AGENTIC = "agentic"
    MANUAL = "manual"
    MOCK = "mock"
    ORACLE = "oracle"
    REPLAY = "replay"


class PipelineStatus(str, Enum):
    """Experimental role, kept orthogonal to how an artifact was originally made."""

    SELECTED = "selected"
    CANDIDATE = "candidate"
    NOT_SELECTED = "not_selected"


class SelectionMode(str, Enum):
    """How the current experiment placed the artifact in its pipeline role."""

    BLIND_AGENTIC = "blind_agentic"
    RETROSPECTIVE_SEED = "retrospective_seed"
    PREDECLARED = "predeclared"


class Outcome(str, Enum):
    """Joint prompt/code outcome; channel scores remain separately reportable."""

    DUAL_RECONSTRUCTION = "dual_reconstruction"
    DUAL_IMPLEMENTATION = "dual_implementation"
    ARTICULABLE_ONLY = "articulable_only"
    VERIFIABLE_ONLY = "verifiable_only"
    HYBRID_COMPLEMENT = "hybrid_complement"
    CONSTRUCTIVE_EXTENSION = "constructive_extension"
    REFERENCE_DIVERGENCE = "reference_divergence"
    PROXY_MISMATCH = "proxy_mismatch"
    UNRESOLVED = "unresolved"


class RelationMatchVerdict(str, Enum):
    """How an implemented operation relates to one construct sub-relation.

    This vocabulary complements the outcome axes above; it does not replace them.
    In particular, a whole criterion can contain a code-native presence relation, a
    prompt-native function relation, and a hybrid position relation at the same time.
    """

    CODE_NATIVE = "code_native"
    PROMPT_NATIVE = "prompt_native"
    HYBRID_COMPLEMENT = "hybrid_complement"
    PROMPT_TAGGED_CODE_RESOLVABLE = "prompt_tagged_code_resolvable"
    CAPABILITY_MISMATCH = "capability_mismatch"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class AxisEvidence:
    """One channel's evidence, without imposing a universal numeric scale."""

    status: Status
    score: float | None = None
    metric: str | None = None
    artifact: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.score is not None and not isinstance(self.score, (int, float)):
            raise TypeError("axis score must be numeric or None")


@dataclass(frozen=True)
class ReconstructionEvidence:
    """Evidence needed to classify one criterion/sub-relation.

    ``reference_isomorphism`` asks whether the candidate reconstructs the frozen LLM
    judgement. ``construct_fidelity`` asks whether its executed path satisfies the frozen
    relation contract.  ``verified_reference_disagreement`` is intentionally demanding:
    it may be true only when a code-native certificate (e.g. execution, symbolic proof,
    typed graph invariant) directly adjudicates cases on which code and the LLM reference
    disagree.
    """

    criterion_id: str
    relation_id: str
    discovery_mode: DiscoveryMode
    articulability: AxisEvidence
    verifiability: AxisEvidence
    hybrid: AxisEvidence
    reference_isomorphism: AxisEvidence
    construct_fidelity: AxisEvidence
    reference_target: str = "frozen_llm_judgement"
    pipeline_status: PipelineStatus = PipelineStatus.CANDIDATE
    selection_mode: SelectionMode = SelectionMode.PREDECLARED
    verified_reference_disagreement: bool = False
    verifier_certificate: str | None = None
    provenance_note: str | None = None

    def __post_init__(self) -> None:
        if self.verified_reference_disagreement:
            if self.verifiability.status is not Status.PASS:
                raise ValueError(
                    "verified reference disagreement requires verifiability PASS"
                )
            if self.construct_fidelity.status is not Status.PASS:
                raise ValueError(
                    "verified reference disagreement requires construct-fidelity PASS"
                )
            if not self.verifier_certificate:
                raise ValueError(
                    "verified reference disagreement requires a replayable certificate"
                )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubrelationEvidence:
    """One independently auditable relation inside a criterion decomposition.

    ``construct_relation`` names the requested relation (for example presence,
    position, attribution, entailment, or execution). ``program_relation`` names what
    the implementation actually computes. Keeping both strings makes a relation-match
    claim falsifiable instead of treating the operation name as its own validation.
    """

    evidence: ReconstructionEvidence
    construct_relation: str
    program_relation: str
    relation_match: RelationMatchVerdict
    weight: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.construct_relation.strip():
            raise ValueError("construct_relation must be non-empty")
        if not self.program_relation.strip():
            raise ValueError("program_relation must be non-empty")
        if self.weight is not None and not 0.0 <= self.weight <= 1.0:
            raise ValueError("sub-relation weight must lie in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CriterionDecomposition:
    """Per-sub-relation evidence without an implicit whole-criterion collapse.

    ``aggregation_rule`` must describe a frozen rule if a later analysis wants a parent
    score. This class intentionally exposes no ``classify_parent`` method: sub-relation
    successes may be heterogeneous and are not evidence for an unregistered aggregation.
    """

    criterion_id: str
    subrelations: tuple[SubrelationEvidence, ...]
    aggregation_rule: str | None = None
    provenance_note: str | None = None

    def __post_init__(self) -> None:
        if not self.criterion_id.strip():
            raise ValueError("criterion_id must be non-empty")
        if not self.subrelations:
            raise ValueError("criterion decomposition needs at least one sub-relation")
        relation_ids = []
        for row in self.subrelations:
            if row.evidence.criterion_id != self.criterion_id:
                raise ValueError("every sub-relation must share the parent criterion_id")
            relation_ids.append(row.evidence.relation_id)
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("sub-relation relation_id values must be unique")
        if self.aggregation_rule is not None and not self.aggregation_rule.strip():
            raise ValueError("aggregation_rule must be non-empty when supplied")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_decomposition(
    criterion_id: str,
    subrelations: Sequence[SubrelationEvidence],
    *,
    aggregation_rule: str | None = None,
    provenance_note: str | None = None,
) -> CriterionDecomposition:
    """Construct a validated immutable decomposition from any finite sequence."""

    return CriterionDecomposition(
        criterion_id=criterion_id,
        subrelations=tuple(subrelations),
        aggregation_rule=aggregation_rule,
        provenance_note=provenance_note,
    )


def decomposition_readout(decomposition: CriterionDecomposition) -> dict[str, Any]:
    """Emit channel outcomes and claims for each sub-relation, never a guessed parent."""

    return {
        "criterion_id": decomposition.criterion_id,
        "aggregation_rule": decomposition.aggregation_rule,
        "parent_outcome": None,
        "parent_outcome_reason": (
            "not_inferred_from_subrelations; apply only a separately frozen aggregation rule"
        ),
        "subrelations": [
            {
                **row.as_dict(),
                "outcome": classify(row.evidence).value,
                "claim_permissions": claim_permissions(row.evidence),
            }
            for row in decomposition.subrelations
        ],
    }


def classify(evidence: ReconstructionEvidence) -> Outcome:
    """Classify without converting missing evidence into success or failure.

    Ordering matters: a construct-invalid proxy cannot be rescued by reference agreement,
    and a certified verifier-dominant disagreement is not mislabeled as poor isomorphism.
    """

    a = evidence.articulability.status
    v = evidence.verifiability.status
    h = evidence.hybrid.status
    iso = evidence.reference_isomorphism.status
    fidelity = evidence.construct_fidelity.status

    if fidelity is Status.FAIL:
        return Outcome.PROXY_MISMATCH

    if evidence.verified_reference_disagreement:
        return Outcome.CONSTRUCTIVE_EXTENSION

    if fidelity is Status.UNAVAILABLE:
        return Outcome.UNRESOLVED

    if iso is Status.FAIL and (a is Status.PASS or v is Status.PASS or h is Status.PASS):
        return Outcome.REFERENCE_DIVERGENCE

    # Channel implementation can be established even when no frozen-reference comparison
    # exists.  Keeping it visible is the point of separating articulability/verifiability
    # from isomorphism; ``UNAVAILABLE`` must not erase a positive channel witness.
    if iso is Status.UNAVAILABLE:
        if a is Status.PASS and v is Status.PASS:
            return Outcome.DUAL_IMPLEMENTATION
        if a is Status.PASS:
            return Outcome.ARTICULABLE_ONLY
        if v is Status.PASS:
            return Outcome.VERIFIABLE_ONLY

    if iso is Status.PASS:
        if a is Status.PASS and v is Status.PASS:
            return Outcome.DUAL_RECONSTRUCTION
        if a is Status.PASS and v is not Status.PASS:
            return Outcome.ARTICULABLE_ONLY
        if v is Status.PASS and a is not Status.PASS:
            return Outcome.VERIFIABLE_ONLY
        if h is Status.PASS:
            return Outcome.HYBRID_COMPLEMENT

    return Outcome.UNRESOLVED


def claim_permissions(evidence: ReconstructionEvidence) -> dict[str, Any]:
    """Return the strongest channel-specific claims licensed by one evidence record.

    Negative results never license a tacitness claim.  This makes the constructive
    asymmetry machine-checkable instead of leaving it as a prose caveat in reports.
    """

    outcome = classify(evidence)
    fidelity_ok = evidence.construct_fidelity.status is Status.PASS
    return {
        "outcome": outcome.value,
        "may_claim_selected_pipeline": (
            evidence.pipeline_status is PipelineStatus.SELECTED
        ),
        "may_claim_automatic_decomposition": (
            evidence.discovery_mode is DiscoveryMode.AGENTIC
            and evidence.selection_mode is SelectionMode.BLIND_AGENTIC
        ),
        "may_claim_prompt_articulability": (
            fidelity_ok and evidence.articulability.status is Status.PASS
        ),
        "may_claim_code_verifiability": (
            fidelity_ok and evidence.verifiability.status is Status.PASS
        ),
        "may_claim_isomorphic_reconstruction": (
            fidelity_ok and evidence.reference_isomorphism.status is Status.PASS
        ),
        "may_claim_constructive_extension": (
            outcome is Outcome.CONSTRUCTIVE_EXTENSION
        ),
        "may_claim_tacitness": False,
        "failure_interpretation": (
            "bounded_non_discovery_within_frozen_program_class_capabilities_"
            "representation_and_budget"
        ),
    }


def validate_record(record: Mapping[str, Any]) -> ReconstructionEvidence:
    """Parse a JSON-like record and apply the same invariants as direct construction."""

    def axis(name: str) -> AxisEvidence:
        raw = record[name]
        return AxisEvidence(
            status=Status(raw["status"]),
            score=raw.get("score"),
            metric=raw.get("metric"),
            artifact=raw.get("artifact"),
            note=raw.get("note"),
        )

    return ReconstructionEvidence(
        criterion_id=str(record["criterion_id"]),
        relation_id=str(record["relation_id"]),
        discovery_mode=DiscoveryMode(record["discovery_mode"]),
        articulability=axis("articulability"),
        verifiability=axis("verifiability"),
        hybrid=axis("hybrid"),
        reference_isomorphism=axis("reference_isomorphism"),
        construct_fidelity=axis("construct_fidelity"),
        reference_target=str(record.get("reference_target", "frozen_llm_judgement")),
        pipeline_status=PipelineStatus(record.get("pipeline_status", "candidate")),
        selection_mode=SelectionMode(record.get("selection_mode", "predeclared")),
        verified_reference_disagreement=bool(
            record.get("verified_reference_disagreement", False)
        ),
        verifier_certificate=record.get("verifier_certificate"),
        provenance_note=record.get("provenance_note"),
    )
