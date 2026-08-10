"""a3 hybrid: RFC evidentiary support (SSR 96-8p).

Criterion asks whether the RFC is supported by and consistent with the
medical evidence -- a strong remand basis is an RFC unsupported by any
medical opinion (the ALJ "playing doctor" on raw data), or one that
conflicts with an opinion the ALJ itself credited. A weak basis is an RFC
that tracks a credited opinion and is record-supported. Higher score =
stronger remand basis (unsupported/conflicting RFC).

Design: code owns a structural PREDICATE -- it counts RFC functional
terms (lift/carry/stand/walk/sit/postural/concentrate) and counts nearby
citation markers ("(Tr.", "opinion of Dr.", "Dr. X opined") to compute an
"unsupported ratio": lots of functional limitations, few grounding
citations. Regex cannot determine WHICH opinion (if any) an ALJ actually
relied on for a given limitation, or spot a limitation that contradicts an
opinion the ALJ elsewhere called persuasive -- both are reading-
comprehension tasks, so they are routed to two LLM fields. Code owns the
predicate: it treats a missing/NONE opinion-basis field as evidence of an
unsupported RFC and a non-NONE conflict field as direct evidence of
inconsistency, blending both with the structural ratio.
"""
import re

LLM_FIELDS = {
    "rfc_opinion_basis": (
        "Name the medical opinion or source the ALJ cited to support the "
        "key RFC limitation, else NONE."
    ),
    "opinion_conflict": (
        "Quote an RFC limitation that conflicts with a medical opinion "
        "the ALJ found persuasive, else NONE."
    ),
}

_RFC_TERM = re.compile(r"\blift\b|\bcarry\b|\bstand\b|\bwalk\b|\bsit\b|\bpostural\b|\bconcentrat", re.I)
_CITE_MARKER = re.compile(r"\(Tr\.|\(Exhibit|opinion of Dr\.|Dr\.\s?\w+\s?opined|found persuasive|found (?:un)?supportable", re.I)

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""

        rfc_terms = len(_RFC_TERM.findall(t))
        cite_markers = len(_CITE_MARKER.findall(t))
        if rfc_terms:
            unsupported_ratio = max(0.0, 1.0 - min(cite_markers / rfc_terms, 1.0))
        else:
            unsupported_ratio = 0.5  # no RFC functional language found at all: ambiguous

        ex = extracted if isinstance(extracted, dict) else {}
        basis_missing = 1.0 if _is_none(ex.get("rfc_opinion_basis")) else 0.0
        conflict_present = 0.0 if _is_none(ex.get("opinion_conflict")) else 1.0

        final = 0.35 * basis_missing + 0.35 * conflict_present + 0.3 * unsupported_ratio
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
