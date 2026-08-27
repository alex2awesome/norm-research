"""alternatives_analysis hybrid: Proposes distinct policy alternatives with trade-offs.

Construct: ~1.0 = comment proposes >=1 distinct, named alternative to the rule as proposed
AND discusses the trade-offs between the alternative and the proposal; ~0.5 = an alternative
is gestured at but not enumerated/compared, or trade-offs discussed with no concrete
alternative; ~0.0 = pure support/opposition with no alternative offered.

INPUT = comment text. Code sees: alternative-marker syntax ("instead of", "in lieu of",
"phase in", conditional constructions) and enumeration structure (numbered/lettered lists,
"first/second/third"), which are the syntactic signatures of laying out real alternatives.
Code CANNOT judge whether the alternative is substantively viable (needs domain expertise) —
out of scope for h0.
"""
import re

LLM_FIELDS = {
    "alternatives_proposed": (
        "Comma-separated distinct policy alternatives the comment proposes instead of the "
        "rule as proposed (e.g. 'phase-in over 3 years', 'voluntary compliance program', "
        "'apply only to facilities over 500 employees'). Answer NONE if none proposed."
    ),
    "tradeoffs_discussed": (
        "In <=20 words, any discussion of trade-offs or costs-vs-benefits between the "
        "proposed rule and an alternative. Answer NONE if none discussed."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_ALT_MARKER_RE = re.compile(
    r'\b(instead of|in lieu of|as an alternative|alternatively|rather than|in place of|'
    r'phase[- ]?in|phase[- ]?out|could instead|option [abc123]\b|as an option|'
    r'a better approach|we propose that instead|one alternative)\b', re.I)
_CONDITIONAL_RE = re.compile(
    r'\b(if\s+[^.]{3,60}\bthen\b|unless\b|provided that\b|in the event that\b|'
    r'should the agency instead\b)\b', re.I)
_ENUM_RE = re.compile(
    r'(?m)^\s*(?:\(?[a-hA-H1-9]\)|[1-9]\.\s|first,|second,|third,|option \d)', re.I)


def _split_names(val):
    if not isinstance(val, str) or val.strip().lower().strip(". ") in _NONE:
        return []
    return [p.strip() for p in re.split(r"[,;]", val) if p.strip() and p.strip().lower() not in _NONE]


def _code_score(t):
    n_markers = len(_ALT_MARKER_RE.findall(t))
    has_conditional = bool(_CONDITIONAL_RE.search(t))
    n_enum = len(_ENUM_RE.findall(t))

    marker_part = min(0.40, 0.12 * n_markers)
    conditional_part = 0.20 if has_conditional else 0.0
    enum_part = min(0.40, 0.15 * n_enum)
    return max(0.0, min(1.0, marker_part + conditional_part + enum_part))


def _llm_score(extracted):
    alts = _split_names(extracted.get("alternatives_proposed"))
    n = len(alts)
    alt_part = {0: 0.1, 1: 0.5}.get(n, 0.75 if n <= 3 else 0.9)
    tradeoffs = extracted.get("tradeoffs_discussed")
    has_tradeoffs = isinstance(tradeoffs, str) and tradeoffs.strip().lower() not in _NONE
    tr_part = 0.15 if has_tradeoffs else 0.0
    return max(0.0, min(1.0, alt_part * 0.85 + tr_part))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5
