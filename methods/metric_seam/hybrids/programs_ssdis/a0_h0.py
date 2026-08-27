"""a0 hybrid: Logical bridge articulation (Chenery reviewability).

Criterion asks whether the ALJ built an accurate, traceable logical bridge
from the record evidence to the disputed findings -- conclusory findings,
unexplained leaps, and undiscussed contrary evidence are strong remand
bases; a decision that cites the record at each step and weighs conflicting
evidence is weak. Higher score = stronger remand basis (bridge failure).

Design: code owns two structural, regex-legible PREDICATE signals that
approximate "grounded in the record" without needing to understand the
reasoning itself -- (1) record-citation density (Tr./Exhibit/AR markers per
100 words: sparse citation is consistent with unsupported findings) and (2)
canned-conclusion phrase hits ("substantial evidence supports", "after
careful consideration of the entire record", etc. used as a summary label
rather than an explanation). Neither regex signal can tell whether a
SPECIFIC finding actually lacks explanation or whether contrary evidence
was actually dropped -- that requires reading comprehension over the
narrative, so it is routed to two LLM fields for thick-input grounding.
Code still owns the predicate: it maps the LLM's short answers to
presence/absence flags and blends them with the structural code score
rather than trusting an LLM verdict directly.
"""
import re

LLM_FIELDS = {
    "unexplained_leap": (
        "Quote the ALJ's conclusory finding that lacks any record citation "
        "or explanation, else NONE."
    ),
    "contrary_evidence_ignored": (
        "Name one piece of contrary medical evidence the ALJ never "
        "discussed or explained rejecting, else NONE."
    ),
}

_CITE_MARKER = re.compile(r"\(Tr\.|\(R\.|\bExhibit\s?\d|\bAR\s?\d{2,}|\bpage\s?\d+\s?of\s?the\s?record", re.I)
_CANNED_CONCLUSION = re.compile(
    r"substantial evidence supports|after (?:careful|due) consideration of the (?:entire )?record"
    r"|as discussed above|for the reasons (?:stated|discussed) (?:above|herein)",
    re.I,
)

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

        words = max(len(t.split()), 1)
        cite_markers = len(_CITE_MARKER.findall(t))
        cite_density = cite_markers / (words / 100.0)
        # sparse citation -> higher structural bridge-failure signal
        density_term = max(0.0, 1.0 - min(cite_density / 3.0, 1.0))

        canned_hits = len(_CANNED_CONCLUSION.findall(t))
        canned_term = min(canned_hits / 3.0, 1.0)

        ex = extracted if isinstance(extracted, dict) else {}
        leap_flag = 0.0 if _is_none(ex.get("unexplained_leap")) else 1.0
        ignored_flag = 0.0 if _is_none(ex.get("contrary_evidence_ignored")) else 1.0

        final = 0.25 * density_term + 0.15 * canned_term + 0.3 * leap_flag + 0.3 * ignored_flag
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
