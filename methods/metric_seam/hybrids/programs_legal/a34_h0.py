"""a34 hybrid: discrete act vs. hostile-environment claim grounding.

Criterion: the claim is based on a discrete act (termination / demotion /
failure-to-promote) OR a continuing-violation hostile environment.

Reading the judge scores against the baseline_code_score column shows the
criterion is NOT rewarding one pole over the other -- it rewards whichever
pole is CONCRETELY established (a specific act with detail, or a specific
recurring-incident pattern with detail), and punishes vague/generic/
procedural-only/off-topic narratives that establish neither. The v2_holistic
baseline gets this backwards: it SUBTRACTS a hostile-keyword hit
(`-0.35*hostile_signal`) from a discrete-keyword hit, so a clean, detailed,
fully-grounded hostile-environment narrative (school harassment: judge=1.0,
baseline=0.24; EEOC pattern-or-practice abuse: judge=1.0, baseline=0.18;
supervisor groping over years: judge=1.0, baseline=0.37) gets crushed. It also
over-trusts bare keyword hits regardless of whether the surrounding text
actually states a concrete fact (a prisoner's-rights case with no employment
adverse action at all still scored 0.82 on the baseline; judge=0.0).

Per the pack's failure note: legal-TERM presence is a weak proxy for the
CONSTRUCT being factually present. So the construct check (is a specific,
concrete discrete act or hostile-incident pattern actually stated, as opposed
to a vague/hedged assertion) is delegated to an LLM field that must name the
specific act/pattern or answer NONE. Code then only: (a) scores how concrete/
specific that short answer is (date-anchored, quoted, non-generic vs. hedged/
empty), (b) adds small, GATED corroboration from quantities/temporal
structure that code can see directly and reliably (date density via
ops.extract_dates, recurring-language regex) -- gated on the LLM having
already confirmed the construct, so date-heavy PROCEDURAL filings (which are
common in this corpus and previously fooled the date-keyword signal) can't
manufacture score on their own, and (c) keeps the old keyword list as a small
ADDITIVE (never subtractive) backstop. A vagueness-disclaimer regex
('not explicitly stated', 'not detailed in the provided text', etc. -- a
recurring template artifact in this scraped/summarized corpus) is a global
dampener, since a document that admits it lacks facts cannot be concretely
grounded in either pole.
"""
import re

LLM_FIELDS = {
    "discrete_act": (
        "In <=20 words, name the ONE concrete adverse employment action "
        "(termination, demotion, failure to promote/hire, disqualification) "
        "with its date if the text gives one; answer NONE if no such "
        "concrete action is stated."
    ),
    "hostile_pattern": (
        "In <=20 words, name the specific recurring or pervasive hostile-"
        "conduct pattern (who did what, how often/over what period) that "
        "grounds a hostile-work-environment claim; answer NONE if no such "
        "specific pattern is stated."
    ),
}

# code-side backstop: original baseline keyword lists, kept as a small
# ADDITIVE corroboration only (never subtracted from each other).
_DISCRETE_KWS = [
    "terminat", "fired", "discharg", "demot", "failure to promote",
    "failed to promote", "denied promotion", "laid off", "not promoted",
    "refused to hire", "denied the position", "non-selection", "not selected",
    "disqualified",
]
_HOSTILE_KWS = [
    "hostile work environment", "hostile environment", "harassment",
    "harassed", "pervasive", "severe and pervasive", "repeatedly",
    "on multiple occasions",
]
_RECURRING_RE = re.compile(
    r"\b(multiple times|numerous occasions|pattern of|repeated(ly)?|"
    r"on \d+ occasions|day after day|for years|over the course of)\b"
)
_VAGUE_RE = re.compile(
    r"not explicitly stated|not detailed in|is not clear from|"
    r"cannot be determined|no specific facts|not specified in the (?:text|record)"
)
_NEG_MARKERS = (
    "none", "n/a", "na", "not stated", "not specified", "not clear",
    "not established", "unclear", "unknown", "not applicable", "no clear",
    "not explicit", "not detailed", "cannot determine", "ambiguous",
    "no concrete",
)
_GENERIC_WORDS = {
    "various", "some", "several", "things", "actions", "ways", "matters",
    "issues", "certain", "something",
}
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTH_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
)
_QUOTE_RE = re.compile(r'["‘’“”][^"‘’“”]{2,}["‘’“”]')


def _sat(x, k):
    import math
    return 1.0 - math.exp(-x / max(1e-6, k))


def _specificity(ans):
    """How concrete/specific is the LLM's short answer (0=empty/hedged/none,
    ~1=date- or quote-anchored, non-generic)."""
    if not isinstance(ans, str):
        return 0.0
    a = ans.strip()
    if not a:
        return 0.0
    al = a.lower()
    if al in ("none", "n/a", "na", "unclear", "unknown"):
        return 0.0
    if any(m in al for m in _NEG_MARKERS):
        return 0.12
    hits = 0
    if _YEAR_RE.search(a):
        hits += 1
    if _MONTH_RE.search(al):
        hits += 1
    if _QUOTE_RE.search(a):
        hits += 1
    words = a.split()
    generic_hits = sum(1 for w in words if w.strip(".,;:").lower() in _GENERIC_WORDS)
    length_bonus = 0.1 if len(words) >= 4 else 0.0
    base = 0.35 + 0.18 * min(hits, 2) + length_bonus - 0.12 * min(generic_hits, 2)
    return max(0.05, min(1.0, base))


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        if not isinstance(t, str):
            t = str(text) if text is not None else ""
        tl = t.lower()

        ext = extracted if isinstance(extracted, dict) else {}
        discrete_ans = ext.get("discrete_act", "")
        hostile_ans = ext.get("hostile_pattern", "")

        discrete_score = _specificity(discrete_ans)
        hostile_score = _specificity(hostile_ans)

        core = max(discrete_score, hostile_score)
        both_bonus = 0.08 * min(discrete_score, hostile_score)

        # code-side quantity signal (dates), gated on LLM-confirmed discrete act
        try:
            dates = ops.extract_dates(t)
            n_dates = len(dates) if dates else 0
        except Exception:
            n_dates = 0
        date_boost = _sat(n_dates, 6.0) * 0.12 if discrete_score >= 0.3 else 0.0

        # code-side temporal-structure signal, gated on LLM-confirmed hostile pattern
        recurring_hits = len(_RECURRING_RE.findall(tl))
        recurring_boost = _sat(recurring_hits, 2.0) * 0.10 if hostile_score >= 0.3 else 0.0

        # small additive (never subtractive) keyword backstop
        kw_discrete = sum(tl.count(k) for k in _DISCRETE_KWS)
        kw_hostile = sum(tl.count(k) for k in _HOSTILE_KWS)
        kw_boost = 0.0
        if discrete_score >= 0.3:
            kw_boost += _sat(kw_discrete, 3.0) * 0.06
        if hostile_score >= 0.3:
            kw_boost += _sat(kw_hostile, 3.0) * 0.06

        # global vagueness-disclaimer dampener
        vague_hits = len(_VAGUE_RE.findall(tl))
        vague_penalty = min(0.3, 0.12 * vague_hits)

        s = 0.08 + 0.82 * (core + both_bonus) + date_boost + recurring_boost + kw_boost - vague_penalty
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
