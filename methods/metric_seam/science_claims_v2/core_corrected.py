"""Audited correction to the v2 science claim verifier.

The original :mod:`core` remains frozen so the 2026-07-12 audit can reproduce the
reported 171 certificates exactly.  This module changes only quantity extraction and
quantity-to-entity binding, then reuses the frozen v2 retrieval and matching machinery.

This is still a conservative document-local verifier.  A certificate witnesses a
declared relation in two passages; it does not establish external scientific truth.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from typing import Any, Iterator

from . import core as v2


# Keep the v2 public structures and text algorithms as the single implementation of the
# unchanged parts of the pipeline.
Sentence = v2.Sentence
Quantity = v2.Quantity
Comparison = v2.Comparison
Claim = v2.Claim
Edge = v2.Edge
segment_sentences = v2.segment_sentences
tokens = v2.tokens
quantity_equal = v2.quantity_equal
extract_comparison = v2.extract_comparison


_UNIT_PATTERN = (
    r"percentage\s+points?|percent|%|points?|times|fold|x|"
    r"nanometers?|nanometres?|nm|"
    r"milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|min|"
    r"hours?|hrs?|hr|h|kilobytes?|megabytes?|gigabytes?|bytes?|kb|mb|gb"
)

# The final boundary includes digits.  That detail prevents a failed suffix parse from
# backtracking into a shorter prefix (the audited ``100k -> 10`` failure).
_QUANTITY_RE_CORRECTED = re.compile(
    rf"(?<![A-Za-z0-9])(?P<sign>[+\-−]?)\s*"
    rf"(?P<value>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)\+?"
    rf"(?:\s*(?P<unit>(?i:{_UNIT_PATTERN}))|(?P<compact_mag>[kKMB])|"
    rf"\s+(?P<word_mag>(?i:thousand|million|billion)))?"
    rf"(?!(?:[A-Za-z0-9]|\.\d))",
)

_UNIT_ALIASES_CORRECTED = dict(v2._UNIT_ALIASES)
_UNIT_ALIASES_CORRECTED.update({
    "nm": "meter",
    "nanometer": "meter",
    "nanometers": "meter",
    "nanometre": "meter",
    "nanometres": "meter",
})
_UNIT_SCALE_CORRECTED = dict(v2._UNIT_SCALE)
_UNIT_SCALE_CORRECTED.update({
    "nm": 1e-9,
    "nanometer": 1e-9,
    "nanometers": 1e-9,
    "nanometre": 1e-9,
    "nanometres": 1e-9,
})
_MAGNITUDE_SCALE = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
}

_INDEX_PREFIX_RE = re.compile(
    r"\b(?:table|tab\.?|figure|fig\.?|equation|eq\.?|stage|step|phase|"
    r"section|sec\.?|appendix|chapter|part|item|case|setting|algorithm)\s*"
    r"(?:[:#-]\s*)?$",
    re.I,
)
_KNOWN_VERSION_NAMES = {"habitat", "lean"}


def _has_version_identifier_prefix(before: str) -> bool:
    """Detect bare numbers that are part of a named system/dataset version."""

    # Slash-composed identifiers such as CIFAR-10/100 and ViT-B/16.
    if re.search(r"[A-Za-z][A-Za-z0-9_-]*\d+\s*/\s*$", before):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", before)
    if not words:
        return False
    last = words[-1]
    internal_upper = any(char.isupper() for char in last[1:])
    acronym = len(last) >= 2 and last.isupper()
    named_version = last.lower() in _KNOWN_VERSION_NAMES
    # DALL·E leaves the final single-letter token E immediately before the version.
    dotted_acronym = len(last) == 1 and last.isupper() and bool(
        re.search(r"[A-Z]{2,}\s*[·.]\s*[A-Z]\s*$", before)
    )
    return internal_upper or acronym or named_version or dotted_acronym


def extract_quantities(text: str) -> tuple[Quantity, ...]:
    """Extract complete quantities without accepting numeric token prefixes.

    Compact count magnitudes (``100k``, ``6.7B``), word magnitudes, and nanometers
    are normalized.  Display indices, enumeration indices, years, model versions,
    and math superscripts are excluded.
    """

    source = text or ""
    out: list[Quantity] = []
    for match in _QUANTITY_RE_CORRECTED.finditer(source):
        start, end = match.start(), match.end()
        before = source[max(0, start - 28):start]
        after = source[end:min(len(source), end + 12)]

        # Model names such as GPT-4 and dimensions such as 3D are identifiers.
        if start > 1 and source[start - 1] in "-_" and source[start - 2].isalnum():
            continue
        # Do not restart inside a decimal whose proper beginning was blocked by an
        # identifier, e.g. Human3.6M must not yield a fabricated ``6M`` quantity.
        if start > 1 and source[start - 1] == "." and source[start - 2].isdigit():
            continue
        # TeX/plain-text superscripts such as P$^3$EFT are identifiers, not counts.
        if re.search(r"\^\{?\s*$", before) or (
            re.search(r"\$\s*$", after) and re.search(r"\$\^\s*$", before)
        ):
            continue
        # Section/table/stage/etc. labels are document or process indices.
        if _INDEX_PREFIX_RE.search(before):
            continue

        raw_value = match.group("value").replace(",", "")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if match.group("sign") in {"-", "−"}:
            value = -value

        unit_raw = (match.group("unit") or "").lower().strip()
        magnitude_raw = (match.group("compact_mag") or match.group("word_mag") or "").lower()
        if not unit_raw and not magnitude_raw and _has_version_identifier_prefix(before):
            continue
        value *= _MAGNITUDE_SCALE.get(magnitude_raw, 1.0)

        is_small_integer = value.is_integer() and 0 <= abs(value) <= 50
        if not unit_raw and not magnitude_raw and is_small_integer:
            parenthesized_index = (
                bool(re.search(r"[\(\[]\s*$", before))
                and bool(re.match(r"\s*[\)\]]", after))
            )
            bare_list_index = bool(re.match(r"\s*[\)\.]\s+[A-Z]", after))
            if parenthesized_index or bare_list_index:
                continue
        if not unit_raw and not magnitude_raw and value.is_integer() and 1900 <= value <= 2100:
            continue

        unit = _UNIT_ALIASES_CORRECTED.get(unit_raw, "unitless")
        value *= _UNIT_SCALE_CORRECTED.get(unit_raw, 1.0)
        out.append(Quantity(match.group(0), value, unit, start, end))
    return tuple(out)


_ANCHOR_STOP = v2._STOP | {
    "about", "approximately", "around", "different", "equipped", "following",
    "given", "including", "nearly", "public", "representative", "selected", "several",
    "spanning", "such", "total", "various", "via", "where", "within",
}


def _anchor_form(term: str) -> str:
    """A deliberately small inflection normalizer for local quantity heads."""

    term = term.lower()
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("ses"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def quantity_anchor_terms(text: str, quantity: Quantity) -> tuple[str, ...]:
    """Return conservative local entity terms bound to a bare count.

    The first three informative terms after a quantity are preferred.  Only when no
    such term exists do we use the two closest preceding terms.  This prevents broad
    sentence overlap (for example shared model names) from binding ``28 adapters`` to
    an unrelated ``28 tasks``.
    """

    source = text or ""
    after = source[quantity.end:min(len(source), quantity.end + 72)]
    after = re.split(r"[.;:!?]|\b(?:but|whereas|while|although)\b", after, maxsplit=1, flags=re.I)[0]
    right = [
        _anchor_form(t) for t in tokens(after, content_only=True)
        if t not in _ANCHOR_STOP and not re.fullmatch(r"\d+(?:\.\d+)?", t)
    ][:3]
    if right:
        return tuple(dict.fromkeys(right))

    before = source[max(0, quantity.start - 72):quantity.start]
    before = re.split(r"[.;:!?]", before)[-1]
    left = [
        _anchor_form(t) for t in tokens(before, content_only=True)
        if t not in _ANCHOR_STOP and not re.fullmatch(r"\d+(?:\.\d+)?", t)
    ][-2:]
    return tuple(dict.fromkeys(left))


def _requires_entity_binding(quantity: Quantity) -> bool:
    return (
        quantity.unit == "unitless"
        and quantity.value.is_integer()
        and 0 <= abs(quantity.value) <= 100
    )


def quantity_relation_equal(
    claim_text: str,
    claim_quantity: Quantity,
    evidence_text: str,
    evidence_quantity: Quantity,
) -> bool:
    """Match value/unit plus a local entity head for ambiguous bare integers."""

    if not quantity_equal(claim_quantity, evidence_quantity):
        return False
    if not _requires_entity_binding(claim_quantity):
        return True
    claim_anchors = set(quantity_anchor_terms(claim_text, claim_quantity))
    evidence_anchors = set(quantity_anchor_terms(evidence_text, evidence_quantity))
    return bool(claim_anchors and evidence_anchors and claim_anchors & evidence_anchors)


def _evaluate_edge(claim: Claim, evidence: Sentence, bm25: float) -> Edge | None:
    """Frozen v2 edge rule with entity-bound quantity matching."""

    ctokens = tokens(claim.sentence.text, content_only=True)
    etokens = tokens(evidence.text, content_only=True)
    coverage = len(set(ctokens) & set(etokens)) / max(1, len(set(ctokens)))
    if coverage < 0.08:
        return None
    evidence_quantities = extract_quantities(evidence.text)
    matches = sum(
        1
        for claim_quantity in claim.quantities
        if any(
            quantity_relation_equal(
                claim.sentence.text, claim_quantity, evidence.text, evidence_quantity
            )
            for evidence_quantity in evidence_quantities
        )
    )
    raw_value_matches = sum(
        1
        for claim_quantity in claim.quantities
        if any(quantity_equal(claim_quantity, evidence_quantity)
               for evidence_quantity in evidence_quantities)
    )
    comp_state = v2._comparison_state(claim.comparison, extract_comparison(evidence.text))
    decision = "insufficient"
    witness_kind = "none"
    reason = "retrieved_but_relation_not_certified"

    if claim.relation == "comparative":
        if comp_state in {"reversed_roles", "direction_mismatch"} and coverage >= 0.18:
            decision, witness_kind, reason = "contradicted", "relation_certificate", comp_state
        elif comp_state not in {"aligned", "aligned_reversed"}:
            reason = comp_state
        elif claim.quantities and matches < len(claim.quantities):
            reason = (
                "quantity_entity_binding_failed"
                if raw_value_matches == len(claim.quantities)
                else "claim_quantity_not_reproduced"
            )
        elif coverage >= 0.16:
            decision, witness_kind, reason = "supported", "relation_certificate", "aligned_comparison"
    elif claim.relation == "numeric":
        if matches == len(claim.quantities) and matches > 0 and coverage >= 0.13:
            decision, witness_kind, reason = (
                "supported", "relation_certificate", "normalized_quantity_entity_and_terms_match"
            )
        else:
            reason = (
                "quantity_entity_binding_failed"
                if raw_value_matches == len(claim.quantities) and raw_value_matches > matches
                else "claim_quantity_not_reproduced"
            )
    elif claim.relation == "theoretical":
        if v2._THEORY_RE.search(evidence.text) and coverage >= 0.18:
            decision, witness_kind, reason = "evidence_link", "evidence_link", "theory_marker_and_terms_match"
        else:
            reason = "missing_theory_witness"
    elif claim.relation == "empirical":
        if v2._EVIDENCE_RE.search(evidence.text) and v2._ASSERTION_RE.search(evidence.text) and coverage >= 0.20:
            decision, witness_kind, reason = "evidence_link", "evidence_link", "empirical_artifact_and_terms_match"
        else:
            reason = "missing_empirical_assertion_witness"
    elif v2._EVIDENCE_RE.search(evidence.text) and coverage >= 0.25:
        decision, witness_kind, reason = "evidence_link", "evidence_link", "qualitative_evidence_and_terms_match"

    relation_bonus = {
        "aligned": 1.0, "aligned_reversed": 1.0, "not_required": 0.45,
        "reversed_roles": 0.35, "direction_mismatch": 0.35, "missing": 0.0,
        "baseline_mismatch": 0.0,
    }.get(comp_state, 0.0)
    weight = coverage + 0.12 * math.log1p(bm25) + 0.20 * matches + relation_bonus
    return Edge(
        claim.index, evidence.index, weight, coverage, bm25, matches,
        len(claim.quantities), comp_state, decision, witness_kind, reason,
    )


@contextmanager
def _corrected_v2_bindings() -> Iterator[None]:
    """Install corrected pure functions only for one single-threaded verification call."""

    old_extract = v2.extract_quantities
    old_evaluate = v2._evaluate_edge
    v2.extract_quantities = extract_quantities
    v2._evaluate_edge = _evaluate_edge
    try:
        yield
    finally:
        v2.extract_quantities = old_extract
        v2._evaluate_edge = old_evaluate


def verify_document(paper_id: str, abstract: str, body: str) -> dict[str, Any]:
    """Run frozen v2 retrieval/matching with the audited quantity corrections."""

    with _corrected_v2_bindings():
        return v2.verify_document(paper_id, abstract, body)


def metamorphic_self_check() -> dict[str, bool]:
    """Executable invariants covering each counterexample in the external audit."""

    suffix_expectations = {
        "100k examples": (100_000.0, "unitless"),
        "33B parameters": (33_000_000_000.0, "unitless"),
        "6.7B parameters": (6_700_000_000.0, "unitless"),
        "1.5B parameters": (1_500_000_000.0, "unitless"),
        "30nm process": (30e-9, "meter"),
    }
    suffix_checks = {}
    for text, expected in suffix_expectations.items():
        quantities = extract_quantities(text)
        suffix_checks[f"complete_token_{text.split()[0]}"] = (
            len(quantities) == 1
            and math.isclose(quantities[0].value, expected[0], rel_tol=1e-12, abs_tol=1e-18)
            and quantities[0].unit == expected[1]
        )

    checks = {
        **suffix_checks,
        "stage_step_phase_are_indices": not any(
            extract_quantities(text)
            for text in ("Stage 1 focuses on retrieval.", "Step 2 trains it.", "Phase 3 evaluates it.")
        ),
        "math_superscript_is_not_quantity": extract_quantities("We propose P$^3$EFT.") == (),
        "adapter_count_does_not_bind_to_task_count": verify_document(
            "counterexample",
            "We evaluate 28 existing LoRA adapters and show robust performance.",
            "Table 2 reports robust performance across 28 tasks.",
        )["status"] != "supported",
        "adapter_count_binds_to_adapter_count": verify_document(
            "positive-control",
            "We evaluate 28 existing LoRA adapters and show robust performance.",
            "Table 2 reports robust performance for all 28 LoRA adapters.",
        )["status"] == "supported",
        "p3eft_does_not_bind_to_three_measures": verify_document(
            "counterexample",
            "Using this analysis, we propose P$^3$EFT and demonstrate lower overhead.",
            "We report the 3 following measures for an unrelated baseline evaluation.",
        )["status"] != "supported",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"corrected science verifier invariant(s) failed: {failed}")
    return checks
