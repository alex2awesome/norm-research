"""a10 hybrid: code classifies error-target dispositive weight + applies a harmless-grounds cap; two LLM fields carry the "which finding did the error hit" and "did the ALJ give an independent alternative ground" judgments regex can't reach."""

# Criterion: whether an identified ALJ error is outcome-determinative (higher
# score) or harmless (lower score) -- courts affirm despite error where an
# independent adequate ground exists or the same result would obtain
# regardless. Strong remand basis: error infects RFC/credibility/step-5 with
# no curative alternative reasoning. Weak (harmless): collateral issue, or
# the ALJ backstopped with independent valid grounds.
#
# Design: code buckets the LLM's short error-target phrase into a
# dispositive-weight tier (RFC/credibility/step-5 = high; step-2/other =
# lower; NONE = ambiguous) -- regex cannot itself tell WHICH finding a given
# narrative's error actually targets, that's the field's job. A non-NONE
# alt-grounds field caps the score low, operationalizing "harmless if an
# independent adequate ground exists" directly in code. A code-only
# structural backstop (dispositive-stage keyword found near an
# error-signaling verb) keeps the function informative even when both
# fields come back NONE/empty.
import re

LLM_FIELDS = {
    "error_target": (
        "In <=10 words, which finding does the identified ALJ error affect: "
        "RFC, credibility, step-5/job-numbers, step-2/severity, or OTHER; else NONE."
    ),
    "alt_grounds": (
        "Quote/name any independent alternative valid ground the ALJ gave "
        "besides the challenged finding, else NONE."
    ),
}

_DISPOSITIVE_HIGH = re.compile(
    r"\brfc\b|residual functional capacity|credibility|symptom|step[- ]?5|job numbers?|vocational", re.I)
_DISPOSITIVE_MID = re.compile(r"step[- ]?2|severity|severe", re.I)
_ERROR_VERB = re.compile(r"\bfail(?:ed|ure)?\b|\bignor(?:ed|es)\b|\bomit(?:ted)?\b|\bwithout considering\b|\berror\b", re.I)

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


def _structural_backstop(t):
    # dispositive-stage keyword found near an error-signaling verb: fallback
    # signal for when the LLM fields are empty
    hits_high = 0
    for m in _DISPOSITIVE_HIGH.finditer(t):
        window = t[max(0, m.start() - 150): m.end() + 150]
        if _ERROR_VERB.search(window):
            hits_high += 1
    hits_mid = 0
    for m in _DISPOSITIVE_MID.finditer(t):
        window = t[max(0, m.start() - 150): m.end() + 150]
        if _ERROR_VERB.search(window):
            hits_mid += 1
    return min(1.0, 0.4 * hits_high + 0.2 * hits_mid)


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""

        ex = extracted if isinstance(extracted, dict) else {}
        target = str(ex.get("error_target") or "").strip().lower()
        alt = ex.get("alt_grounds")

        if _is_none(target):
            base = 0.4  # no clear error target named: ambiguous, mildly weak
        elif _DISPOSITIVE_HIGH.search(target):
            base = 0.85
        elif _DISPOSITIVE_MID.search(target):
            base = 0.55
        else:
            base = 0.45  # OTHER: collateral-leaning

        backstop = _structural_backstop(t)
        pre_cap = 0.75 * base + 0.25 * backstop

        if not _is_none(alt):
            # ALJ gave an independent alternative ground: harmless-error cap
            final = min(pre_cap, 0.2)
        else:
            final = pre_cap

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
