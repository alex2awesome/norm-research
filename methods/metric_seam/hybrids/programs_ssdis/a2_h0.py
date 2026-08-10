"""a2 hybrid: Step 2 severity/duration mishandling.

Criterion asks whether a documented, medically determinable impairment was
wrongly found non-severe (or duration/combination mishandled) at the de
minimis Step 2 screen. The error is usually harmless if the ALJ found at
least one other severe impairment and the omitted impairment's limitations
still reappear later in the RFC discussion; it is a strong remand basis
only if those limitations never reappear anywhere. Higher score = stronger
remand basis (real, uncured Step 2 omission).

Design: code owns a cheap structural PREDICATE -- regex hits for
"non-severe"/"not severe"/"de minimis" as a weak fallback signal when no
concrete impairment is named. What regex cannot do is name a SPECIFIC
documented impairment the ALJ treated as non-severe, and -- critically --
determine whether that same impairment's functional limitations show up
anywhere else in the RFC narrative (the harmless-error question). Both
require reading comprehension across the whole document, so they are
routed to two LLM fields. Code owns the predicate: it only escalates the
score when a concrete impairment is named AND the reappearance field
explicitly signals absence; it discounts when the field signals presence
(harmless), consistent with the criterion's own harmless-error carve-out.
"""
import re

LLM_FIELDS = {
    "nonsevere_impairment": (
        "Name one medically documented impairment the ALJ found "
        "non-severe or omitted entirely at step two, else NONE."
    ),
    "reappears_in_rfc": (
        "State briefly whether that impairment's limitations appear "
        "anywhere in the RFC discussion, else NONE."
    ),
}

_NONSEVERE_KW = re.compile(r"non-severe|not severe|de minimis", re.I)
_REAPPEAR_NO = re.compile(r"\bno\b|not (?:mentioned|discussed|addressed|included)|absent|never|omitted", re.I)
_REAPPEAR_YES = re.compile(r"\byes\b|appears|included|considered|addressed|reflected", re.I)

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

        ex = extracted if isinstance(extracted, dict) else {}
        imp = ex.get("nonsevere_impairment")

        if _is_none(imp):
            # weak code-only fallback: no concrete impairment named
            nonsevere_hits = len(_NONSEVERE_KW.findall(t))
            return max(0.0, min(1.0, 0.15 * min(nonsevere_hits / 2.0, 1.0)))

        reappear = ex.get("reappears_in_rfc")
        base = 0.5  # a concrete non-severe/omitted impairment was named
        if not _is_none(reappear):
            r = str(reappear).strip()
            if _REAPPEAR_NO.search(r):
                base += 0.4
            elif _REAPPEAR_YES.search(r):
                base -= 0.4

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
