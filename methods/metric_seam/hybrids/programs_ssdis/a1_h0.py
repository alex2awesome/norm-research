"""a1 hybrid: Legal error reviewed de novo.

Criterion asks whether the ALJ applied the wrong legal standard or ignored
a binding regulation/SSR -- a strong remand basis names a discrete,
articulable rule violated (wrong medical-opinion framework, skipped
listing analysis, misallocated step-5 burden, ignored SSR procedure); a
weak basis is a routine evidentiary disagreement dressed up as "legal
error". Harmless misstatements that did not affect the outcome discount
the score. Higher score = stronger remand basis (real, outcome-relevant
legal error).

Design: code owns a structural PREDICATE -- regex-detectable citations to
specific regulations/SSRs/step numbers -- as a weak supporting signal that
the argument is anchored to real doctrine rather than vibes. What regex
cannot do is confirm that a CONCRETE rule is actually named as violated
(vs. a vague complaint) or that the claimant/facts flag the error as
outcome-determinative -- both require reading comprehension, so they are
routed to two LLM fields. Code owns the predicate: it checks whether the
named rule is itself concrete (contains a citation-like token) before
crediting it, and applies a harmless-error penalty only when the LLM field
actually returns text.
"""
import re

LLM_FIELDS = {
    "rule_violated": (
        "Name the specific regulation, SSR, or legal standard the ALJ "
        "allegedly misapplied, else NONE."
    ),
    "harmless_note": (
        "Quote any statement suggesting the alleged legal error did not "
        "change the outcome, else NONE."
    ),
}

_REG_CITE = re.compile(
    r"20\s?C\.?F\.?R\.?\s?§?\s?404\.\d+|SSR\s?\d{2}-\d+p|\bstep\s(?:two|2|three|3|four|4|five|5)\b",
    re.I,
)
_CONCRETE_RULE = re.compile(r"\d|SSR|C\.?F\.?R\.?|listing", re.I)

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

        reg_hits = len(_REG_CITE.findall(t))
        cite_term = min(reg_hits / 3.0, 1.0)

        ex = extracted if isinstance(extracted, dict) else {}
        rule = "" if _is_none(ex.get("rule_violated")) else str(ex.get("rule_violated")).strip()
        rule_present = 1.0 if rule else 0.0
        rule_concrete = 1.0 if rule and _CONCRETE_RULE.search(rule) else 0.5

        harmless = ex.get("harmless_note")
        harmless_penalty = 0.25 if not _is_none(harmless) else 0.0

        base = 0.5 * rule_present * rule_concrete + 0.3 * cite_term
        final = base - harmless_penalty
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
