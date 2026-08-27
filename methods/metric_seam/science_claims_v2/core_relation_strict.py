"""Additive relation-fidelity correction for the science claim verifier.

The earlier v2.2 program corrected truncated quantities and a small-count entity
collision, but its matching predicate could still reuse one evidence quantity for
multiple obligations and treated most units and large counts as entity-free.  This
module keeps the selected retrospective decomposition while tightening only the
executable relation layer:

* numeric obligations are matched one-to-one;
* values and normalized units must agree exactly up to floating-point noise;
* local metric/entity heads must agree for counts, ratios, and percentages;
* change direction is checked when the claim articulates one;
* codec identifiers such as ``H.264`` are not measurements; and
* questions and hypothetical comparisons cannot certify support.

Certificates remain document-local parser witnesses, not external scientific truth.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from . import core as v2
from . import core_corrected as v22


Sentence = v2.Sentence
Quantity = v2.Quantity
Comparison = v2.Comparison
Claim = v2.Claim
Edge = v2.Edge


_V22_EXTRACT_QUANTITIES = v22.extract_quantities

_GENERIC_LOCAL = v2._STOP | {  # type: ignore[attr-defined]
    "about", "across", "achieve", "achieved", "achieves", "achieving",
    "approximately", "around", "average", "averaged", "compared", "comparing",
    "demonstrate", "demonstrated", "demonstrates", "demonstrating", "different",
    "experiment", "experiments", "following", "given", "including", "introduce",
    "introduced", "introduces", "leading", "method", "nearly", "optimal", "overall", "report",
    "reported", "reports", "result", "results", "save", "saved", "saves", "show",
    "showed", "showing", "shown", "shows", "significant", "significantly", "total",
    "using", "various", "with", "without",
}

# Left/left agreement is otherwise easily satisfied by a shared method name.  Permit it
# only for terms that name the measured quantity or counted entity.
_MEASURE_HEADS = {
    "accuracy", "adapter", "algorithm", "auc", "auroc", "benchmark", "byte",
    "caption", "class", "core", "cost", "dataset", "document", "elo", "error",
    "example", "fact", "fid", "flop", "frame", "game", "hour", "image", "iteration",
    "language", "latency", "layer", "loss", "memory", "metric", "minute", "model",
    "mse", "node", "parameter", "participant", "point", "precision", "question",
    "rate", "recall", "record", "resource", "robustness", "round", "r2", "sample",
    "scenario", "score", "second", "speed", "speedup", "task", "time", "token",
}

_PASSIVE_COMPARATORS: tuple[tuple[re.Pattern[str], int], ...] = (
    (
        re.compile(
            r"\b(?:is|are|was|were|be|been|being)\s+"
            r"(?:outperformed|surpassed|exceeded|beaten)\s+by\b",
            re.I,
        ),
        -1,
    ),
)
_ACTIVE_COMPARATORS: tuple[tuple[re.Pattern[str], int], ...] = (
    (
        re.compile(
            r"\b(?:outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|"
            r"exceed(?:s|ed|ing)?|beat(?:s|en|ing)?)\b",
            re.I,
        ),
        1,
    ),
    (re.compile(r"\bimprov(?:e|es|ed|ing)\s+(?:on|over|upon)\b", re.I), 1),
    (
        re.compile(
            r"\b(?:better|higher|faster|stronger|more accurate|more efficient)\s+than\b",
            re.I,
        ),
        1,
    ),
    (re.compile(r"\b(?:superior)\s+to\b", re.I), 1),
    (
        re.compile(r"\b(?:worse|lower|slower|weaker|inferior)\s+than\b", re.I),
        -1,
    ),
    (re.compile(r"\bunderperform(?:s|ed|ing)?\b", re.I), -1),
)

_POSITIVE_CHANGE = re.compile(
    r"\b(?:boost|boosted|boosts|gain|gained|gains|higher|improv(?:e|ed|ement|ements|es|ing)|"
    r"increas(?:e|ed|es|ing)|rise|rises|rose)\b",
    re.I,
)
_NEGATIVE_CHANGE = re.compile(
    r"\b(?:decreas(?:e|ed|es|ing)|drop|dropped|drops|lower|reduc(?:e|ed|es|ing|tion|tions)|"
    r"sav(?:e|ed|es|ing)|decline|declined|declines)\b",
    re.I,
)
_NONASSERTIVE = re.compile(
    r"\b(?:can|could|may|might|would|whether|if|potentially|hypothes(?:is|ize|ized)|"
    r"we\s+ask|question)\b",
    re.I,
)
_MONTH_BEFORE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s*$",
    re.I,
)


def _normal_term(term: str) -> tuple[str, ...]:
    parts = re.findall(r"[a-z0-9]+", term.lower())
    out: list[str] = []
    for part in parts:
        if len(part) > 4 and part.endswith("ies"):
            part = part[:-3] + "y"
        elif len(part) > 4 and part.endswith("ses"):
            part = part[:-2]
        elif len(part) > 3 and part.endswith("s") and not part.endswith("ss"):
            part = part[:-1]
        if part and part not in _GENERIC_LOCAL and not part.isdigit():
            out.append(part)
    return tuple(out)


def _terms(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for token in v2.tokens(text, content_only=True):
        out.extend(_normal_term(token))
    return tuple(dict.fromkeys(out))


def _local_sides(text: str, quantity: Quantity) -> tuple[set[str], set[str]]:
    before = text[max(0, quantity.start - 96):quantity.start]
    after = text[quantity.end:min(len(text), quantity.end + 96)]
    before = re.split(r"[.;:!?]", before)[-1]
    after = re.split(r"[.;:!?]", after, maxsplit=1)[0]
    left = set(_terms(before)[-6:])
    # The grammatical head is normally immediately after the quantity.  A wider
    # window leaks shared method names and conclusions (``1000 nodes ... optimal``
    # vs ``1000 rounds ... optimal``) into what should be an entity check.
    right = set(_terms(after)[:3])
    return left, right


def quantity_context_terms(text: str, quantity: Quantity) -> tuple[str, ...]:
    """Expose the conservative local relation heads used by strict matching."""

    left, right = _local_sides(text, quantity)
    return tuple(sorted(left | right))


def extract_quantities(text: str) -> tuple[Quantity, ...]:
    """Retain v2.2 normalization while excluding numeric identifiers."""

    source = text or ""
    quantities: list[Quantity] = []
    for quantity in _V22_EXTRACT_QUANTITIES(source):
        start = quantity.start
        end = quantity.end
        before = source[max(0, start - 48):start]
        after = source[end:min(len(source), end + 20)]
        # ``H.264``/``H.265``/``AVC.264`` are identifiers.  Decimal measurements
        # still begin at their leading digit, so this does not suppress ``0.264``.
        if start >= 2 and source[start - 1] == "." and source[start - 2].isalpha():
            continue
        # Function parameters such as ``Linex(1/2)`` are not result quantities.
        last_open = source.rfind("(", max(0, start - 48), start)
        last_close = source.rfind(")", max(0, start - 48), start)
        if last_open > last_close:
            # A function call has no prose-space boundary before ``(``.  This
            # distinction preserves ordinary counts such as ``tasks (12 datasets)``.
            if last_open > 0 and (
                source[last_open - 1].isalnum() or source[last_open - 1] in "_-"
            ):
                continue
            # Likewise, the leading constant in ``(1+O(alpha))`` is part of a
            # symbolic expression, not an empirical numeric obligation.
            if re.match(r"\s*[+*/^]", after) or re.search(r"[/^]\s*$", before):
                continue
        # TeX and Unicode norm/order identifiers (``\ell_{2}``, ``ℓ2``) label a
        # perturbation family; their numeral is not a measured outcome.
        if re.search(r"(?:\\ell|ℓ)\s*(?:_\{?)?\s*$", before, re.I):
            continue
        # Calendar days repeated in prose do not verify the neighboring efficacy
        # claim.  Years are already excluded by v2.2.
        if _MONTH_BEFORE.search(before) and re.match(
            r"\s*,?\s*(?:19|20)\d{2}\b", after
        ):
            continue
        quantities.append(quantity)
    return tuple(quantities)


def _value_unit_equal(left: Quantity, right: Quantity) -> bool:
    if left.unit != right.unit:
        return False
    scale = max(abs(left.value), abs(right.value), 1.0)
    return math.isclose(left.value, right.value, rel_tol=1e-9, abs_tol=1e-12 * scale)


def _context_equal(
    claim_text: str,
    claim_quantity: Quantity,
    evidence_text: str,
    evidence_quantity: Quantity,
) -> bool:
    claim_left, claim_right = _local_sides(claim_text, claim_quantity)
    evidence_left, evidence_right = _local_sides(evidence_text, evidence_quantity)
    # Prefer heads following the quantity, but permit a metric written before the
    # value on one side (``accuracy of 43.8%`` vs ``43.8% accuracy``).
    if claim_right & evidence_right:
        return True
    if claim_right & evidence_left:
        return True
    if claim_left & evidence_right:
        return True
    return bool((claim_left & evidence_left) & _MEASURE_HEADS)


def quantity_relation_equal(
    claim_text: str,
    claim_quantity: Quantity,
    evidence_text: str,
    evidence_quantity: Quantity,
) -> bool:
    return _value_unit_equal(claim_quantity, evidence_quantity) and _context_equal(
        claim_text, claim_quantity, evidence_text, evidence_quantity
    )


def _quantity_matching(
    claim_text: str,
    claim_quantities: tuple[Quantity, ...],
    evidence_text: str,
    evidence_quantities: tuple[Quantity, ...],
) -> int:
    """Maximum-cardinality one-to-one matching of numeric obligations."""

    neighbors = tuple(
        tuple(
            index
            for index, evidence_quantity in enumerate(evidence_quantities)
            if quantity_relation_equal(
                claim_text,
                claim_quantity,
                evidence_text,
                evidence_quantity,
            )
        )
        for claim_quantity in claim_quantities
    )

    @lru_cache(maxsize=None)
    def solve(claim_index: int, used: tuple[int, ...]) -> int:
        if claim_index >= len(neighbors):
            return 0
        best = solve(claim_index + 1, used)
        occupied = set(used)
        for evidence_index in neighbors[claim_index]:
            if evidence_index in occupied:
                continue
            best = max(
                best,
                1 + solve(
                    claim_index + 1,
                    tuple(sorted((*used, evidence_index))),
                ),
            )
        return best

    return solve(0, ())


def _comparison_terms(text: str) -> tuple[str, ...]:
    return v2._entity_terms(text)  # type: ignore[attr-defined]


def extract_comparison(text: str) -> Comparison | None:
    """Parse directed roles, including stronger/weaker and passive forms."""

    source = str(text or "")
    for pattern, base_polarity in (*_PASSIVE_COMPARATORS, *_ACTIVE_COMPARATORS):
        match = pattern.search(source)
        if not match:
            continue
        left = source[max(0, match.start() - 120):match.start()]
        right = source[match.end():min(len(source), match.end() + 120)]
        left = re.split(r"[.;:]|\b(?:that|whether)\b", left, flags=re.I)[-1]
        right = re.split(r"[.;:]", right)[0]
        right = re.split(
            r"\b(?:by|on|with|using|for|at|while|under|across)\b",
            right,
            maxsplit=1,
            flags=re.I,
        )[0]
        polarity = base_polarity
        if re.search(
            r"\b(?:not|never|fails?\s+to|doesn['’]t|didn['’]t)\b",
            left[-45:],
            re.I,
        ):
            polarity *= -1
        return Comparison(
            cue=match.group(0),
            polarity=polarity,
            left_terms=_comparison_terms(left),
            right_terms=tuple(_comparison_terms(right)[:8]),
        )
    return None


def _comparison_is_assertive(text: str, comparison: Comparison | None) -> bool:
    if comparison is None:
        return False
    if "?" in text:
        return False
    cue_match = re.search(re.escape(comparison.cue), text, re.I)
    if not cue_match:
        return False
    local = text[max(0, cue_match.start() - 55):cue_match.start()]
    return _NONASSERTIVE.search(local) is None


def _change_direction(text: str, quantities: tuple[Quantity, ...]) -> int | None:
    if not quantities:
        return None
    start = max(0, min(quantity.start for quantity in quantities) - 90)
    end = min(len(text), max(quantity.end for quantity in quantities) + 90)
    local = text[start:end]
    positive = bool(_POSITIVE_CHANGE.search(local))
    negative = bool(_NEGATIVE_CHANGE.search(local))
    if positive == negative:
        return None
    return 1 if positive else -1


def evaluate_edge(claim: Claim, evidence: Sentence, bm25: float) -> Edge | None:
    """Evaluate one relation edge with strict numeric and assertion semantics."""

    claim_tokens = v2.tokens(claim.sentence.text, content_only=True)
    evidence_tokens = v2.tokens(evidence.text, content_only=True)
    coverage = len(set(claim_tokens) & set(evidence_tokens)) / max(
        1, len(set(claim_tokens))
    )
    if coverage < 0.08:
        return None

    evidence_quantities = extract_quantities(evidence.text)
    quantity_matches = _quantity_matching(
        claim.sentence.text,
        claim.quantities,
        evidence.text,
        evidence_quantities,
    )
    relation_state = v2._comparison_state(  # type: ignore[attr-defined]
        claim.comparison, extract_comparison(evidence.text)
    )
    claim_direction = _change_direction(claim.sentence.text, claim.quantities)
    evidence_direction = _change_direction(evidence.text, evidence_quantities)
    numeric_direction_ok = (
        claim_direction is None
        or (
            evidence_direction is not None
            and claim_direction == evidence_direction
        )
    )

    decision, witness_kind = "insufficient", "none"
    reason = "retrieved_but_relation_not_certified"
    if claim.relation == "comparative":
        evidence_comparison = extract_comparison(evidence.text)
        if not _comparison_is_assertive(evidence.text, evidence_comparison):
            reason = "nonassertive_comparison_evidence"
        elif relation_state in {"reversed_roles", "direction_mismatch"} and coverage >= 0.18:
            decision, witness_kind, reason = (
                "contradicted", "relation_certificate", relation_state
            )
        elif relation_state not in {"aligned", "aligned_reversed"}:
            reason = relation_state
        elif claim.quantities and quantity_matches < len(claim.quantities):
            reason = "quantity_unit_entity_or_uniqueness_failed"
        elif coverage >= 0.16:
            decision, witness_kind, reason = (
                "supported", "relation_certificate", "assertive_aligned_comparison"
            )
    elif claim.relation == "numeric":
        if quantity_matches < len(claim.quantities) or not quantity_matches:
            reason = "quantity_unit_entity_or_uniqueness_failed"
        elif not numeric_direction_ok:
            reason = (
                "quantity_direction_missing"
                if evidence_direction is None
                else "quantity_direction_mismatch"
            )
        elif coverage >= 0.13:
            decision, witness_kind, reason = (
                "supported",
                "relation_certificate",
                "bijective_quantity_unit_entity_direction_match",
            )
    elif claim.relation == "theoretical":
        if v2._THEORY_RE.search(evidence.text) and coverage >= 0.18:  # type: ignore[attr-defined]
            decision, witness_kind, reason = (
                "evidence_link", "evidence_link", "theory_marker_and_terms_match"
            )
        else:
            reason = "missing_theory_witness"
    elif claim.relation == "empirical":
        if (
            v2._EVIDENCE_RE.search(evidence.text)  # type: ignore[attr-defined]
            and v2._ASSERTION_RE.search(evidence.text)  # type: ignore[attr-defined]
            and coverage >= 0.20
        ):
            decision, witness_kind, reason = (
                "evidence_link", "evidence_link", "empirical_artifact_and_terms_match"
            )
        else:
            reason = "missing_empirical_assertion_witness"
    elif v2._EVIDENCE_RE.search(evidence.text) and coverage >= 0.25:  # type: ignore[attr-defined]
        decision, witness_kind, reason = (
            "evidence_link", "evidence_link", "qualitative_evidence_and_terms_match"
        )

    relation_bonus = {
        "aligned": 1.0,
        "aligned_reversed": 1.0,
        "not_required": 0.45,
        "reversed_roles": 0.35,
        "direction_mismatch": 0.35,
        "missing": 0.0,
        "baseline_mismatch": 0.0,
    }.get(relation_state, 0.0)
    weight = (
        coverage
        + 0.12 * math.log1p(bm25)
        + 0.20 * quantity_matches
        + relation_bonus
    )
    return Edge(
        claim.index,
        evidence.index,
        weight,
        coverage,
        bm25,
        quantity_matches,
        len(claim.quantities),
        relation_state,
        decision,
        witness_kind,
        reason,
    )


@contextmanager
def _strict_bindings() -> Iterator[None]:
    old_extract_quantities = v2.extract_quantities
    old_extract_comparison = v2.extract_comparison
    old_evaluate_edge = v2._evaluate_edge  # type: ignore[attr-defined]
    v2.extract_quantities = extract_quantities
    v2.extract_comparison = extract_comparison
    v2._evaluate_edge = evaluate_edge  # type: ignore[attr-defined]
    try:
        yield
    finally:
        v2.extract_quantities = old_extract_quantities
        v2.extract_comparison = old_extract_comparison
        v2._evaluate_edge = old_evaluate_edge  # type: ignore[attr-defined]


def verify_document(paper_id: str, abstract: str, body: str) -> dict:
    """Run the additive strict relation layer over continuous article text."""

    with _strict_bindings():
        return v2.verify_document(paper_id, abstract, body)


def metamorphic_self_check() -> dict[str, bool]:
    checks = {
        "positive_numeric_relation_supports": verify_document(
            "positive-numeric",
            "We show a 28% improvement in robustness on the benchmark.",
            "Table 2 shows a 28% improvement in robustness on the benchmark.",
        )["status"] == "supported",
        "percentage_metric_swap_rejected": verify_document(
            "metric-swap",
            "We show a 28% improvement in robustness compared with routing methods.",
            "The method saves 28% of computational resources compared with routing methods.",
        )["status"] != "supported",
        "large_count_entity_swap_rejected": verify_document(
            "entity-swap",
            "Experiments with 1000 nodes demonstrate nearly optimal solutions.",
            "After 1000 rounds of iteration, the method is nearly optimal.",
        )["status"] != "supported",
        "quantity_reuse_rejected": verify_document(
            "quantity-reuse",
            "Results show accuracy gains of 90.9% and 91.3% on two settings.",
            "Table 2 reports one accuracy gain of 91.3% on the settings.",
        )["status"] != "supported",
        "codec_identifier_excluded": extract_quantities("AVC/H.264 codec") == (),
        "interrogative_comparison_rejected": verify_document(
            "question",
            "We show that OTDF outperforms prior strong baselines.",
            "Can OTDF beat prior strong baselines across varied shifts?",
        )["status"] != "supported",
        "assertive_comparison_supports": verify_document(
            "comparison",
            "We show that our method outperforms BERT.",
            "Table 2 shows that our method outperforms BERT.",
        )["status"] == "supported",
        "role_swap_contradicts": verify_document(
            "role-swap",
            "We show that our method outperforms BERT.",
            "Table 2 shows that BERT outperforms our method.",
        )["status"] == "contradicted",
        "unit_swap_rejected": verify_document(
            "unit-swap",
            "We report a latency of 5 seconds on the benchmark.",
            "Table 2 reports a latency of 5 milliseconds on the benchmark.",
        )["status"] != "supported",
        "stronger_weaker_roles_execute": verify_document(
            "stronger-weaker",
            "Our agent is 400 Elo stronger than behavioral cloning.",
            "Behavioral cloning is 400 Elo weaker than our agent.",
        )["status"] == "supported",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"strict science relation invariant(s) failed: {failed}")
    return checks
