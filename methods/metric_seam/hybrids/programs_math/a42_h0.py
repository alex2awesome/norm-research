"""a42 - Use of visuals and concrete examples (Math StackExchange).

Reading of the training residuals: the v2_holistic baseline (train rho 0.338)
keys off lexical markers ("example", "consider", "for instance") wherever they
occur -- including inside the quoted QUESTION text (d01778, d02961) -- and it
rewards answers that merely crunch the specific numbers already given in the
question (d01366's apples/mangoes/bananas, d03220's dice problem, d03806's
digit-count problem all get nonzero baseline score but judge_score_0_1 == 0.0).

What the judge actually rewards, reading the nonzero-score docs (d01054,
d00566, d04424, d04412, d04848, d01855, d03818, d00671, d04081, d02552): the
ANSWER introduces a concrete numeric instance, a specific named object, or a
small worked toy case to GROUND a general argument -- distinct from just
answering the literal question asked. The single clearest example (d02552,
judge 0.6, highest in the sample) walks through four small enumerated toy
cases ("Eg: 2 indistinct balls in 2 indistinct boxes..."). That is a
thick-input distinction ("is this example illustrating a general idea, or is
it just the arithmetic the question demanded?") that a regex over surface
keywords cannot make reliably -- hence the two LLM_FIELDS below. The
predicate itself (turning that judgment + LaTeX-aware structure into a score)
stays in code.
"""
import re

LLM_FIELDS = {
    "grounds_with_example": (
        "Does the answer use a concrete numeric example, specific named object, "
        "or small worked toy case to illustrate a general idea -- NOT merely "
        "computing the specific numbers already given in the question? "
        "Answer YES or NO."
    ),
    "example_richness": (
        "How many separate concrete illustrative examples or toy cases does "
        "the answer give to ground its reasoning: NONE, ONE, or MULTIPLE?"
    ),
}

_MARKER_RE = re.compile(
    r"\b(e\.?g\.?|for example|for instance|such as|consider the (?:case|example))\b",
    re.IGNORECASE,
)
_CASE_RE = re.compile(r"\bcase\s*\d", re.IGNORECASE)


def _tier_from_llm(extracted):
    """0 = no grounding example, 1 = one, 2 = multiple."""
    g = (extracted.get("grounds_with_example") or "").strip().upper()
    r = (extracted.get("example_richness") or "").strip().upper()
    has = g.startswith("Y")
    if not has:
        return 0
    if "MULTI" in r or "MANY" in r or "SEVERAL" in r:
        return 2
    return 1


def _richness_proxy(text, ops):
    """Cheap, bounded, code-only tie-breaker within a tier: counts marker
    phrases, enumerated 'Case N' splits, and the fraction of parsed math
    spans that carry literal digits (a rough proxy for numeric grounding
    density, independent of the LLM's own tally)."""
    try:
        markers = len(_MARKER_RE.findall(text))
    except Exception:
        markers = 0
    try:
        cases = len(_CASE_RE.findall(text))
    except Exception:
        cases = 0
    density = 0.0
    try:
        spans = ops.extract_math_spans(text)
        total = len(spans)
        if total:
            numeric = sum(1 for _kind, content in spans if any(c.isdigit() for c in content))
            density = numeric / total
    except Exception:
        pass
    raw = 0.5 * min(markers, 4) + 0.5 * min(cases, 4) + 2.0 * density
    return raw


def _malformed_penalty(text, ops):
    """Small penalty for badly broken LaTeX -- an example that isn't
    'well-integrated' (contract's phrase) shouldn't get full credit."""
    try:
        dh = ops.delimiter_health(text) or {}
    except Exception:
        return 0.0
    issues = 0
    for v in dh.values():
        try:
            if int(v) > 0:
                issues += 1
        except Exception:
            continue
    if issues >= 4:
        return 0.08
    if issues >= 2:
        return 0.04
    return 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if len(t) < 30:
            return 0.0

        extracted = extracted or {}
        tier = _tier_from_llm(extracted)
        tier_base = {0: 0.03, 1: 0.32, 2: 0.55}[tier]

        proxy = _richness_proxy(t, ops)
        nudge = min(0.10, 0.02 * proxy)

        penalty = _malformed_penalty(t, ops)

        s = tier_base + nudge - penalty
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
