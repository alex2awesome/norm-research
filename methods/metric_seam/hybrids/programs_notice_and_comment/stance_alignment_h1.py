"""stance_alignment hybrid: How aligned the comment is with the rule AS PROPOSED.

Construct: ~1.0 = comment supports the rule as proposed or asks only for refinements within
the existing framework ("support ... but", "generally agree, however"); ~0.5 = mixed/neutral,
or opposes specific provisions while leaving the core rule intact; ~0.0 = wholesale
opposition or an explicit demand to withdraw/rescind/scrap the rule entirely.

INPUT = comment text. Code sees: graded support/oppose lexicons with negation handling,
"support ... but/however/except" hedged-support constructions, explicit withdraw/rescind
demands (treated as maximal opposition), and positional markers referencing the rule AS
PROPOSED ("as proposed", "the proposed rule correctly", "we support the proposed"). Code
CANNOT verify that the STATED position matches the commenter's true underlying preference
(needs discourse-level inference) — out of scope for h1.
"""
import re

LLM_FIELDS = {
    "overall_position": (
        "The comment's overall position toward the rule, in exactly one word from this set: "
        "support, support_with_changes, neutral, oppose_in_part, oppose_fully. Pick the single "
        "best-fitting label."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_SUPPORT_RE = re.compile(
    r'\b(support[s]?|favor[s]?|welcome[s]?|applaud[s]?|commend[s]?|agree[s]? with|'
    r'in favor of|pleased (?:to see|with)|glad to see|endorse[s]?|approve[s]? of)\b', re.I)
_OPPOSE_RE = re.compile(
    r'\b(oppose[s]?|object[s]? to|disagree[s]? with|against\s+(?:this|the)\s+rule|'
    r'concerned? (?:that|about)|troubled by|disappointed (?:in|with|by)|reject[s]?)\b', re.I)
_NEGATION_RE = re.compile(r'\b(not|never|hardly|cannot|can\'?t|don\'?t|does not|doesn\'?t)\b', re.I)
_HEDGE_SUPPORT_RE = re.compile(
    r'\bsupport[s]?\b[^.;\n]{0,60}\b(but|however|except|although|though|provided that|'
    r'so long as|as long as)\b', re.I)
_WITHDRAW_RE = re.compile(
    r'\b(withdraw|rescind|scrap|abandon|kill|shelve)\b[^.;\n]{0,40}\b(this|the)\s+(?:proposed\s+)?rule\b|'
    r'\b(withdraw|rescind)\s+(?:this|the)\s+(?:proposed\s+)?(?:rule|regulation|proposal)\b', re.I)
_AS_PROPOSED_RE = re.compile(
    r'\bas\s+proposed\b|\bthe\s+proposed\s+rule\s+correctly\b|\bsupport[s]?\s+the\s+proposed\b|'
    r'\bfinaliz(?:e|ing)\s+(?:the\s+rule\s+)?as\s+proposed\b', re.I)


def _polarity_hits(t, re_, window=6):
    """Count hits of re_ that are NOT preceded by a negation within `window` words."""
    words = t.split()
    hits = 0
    for m in re_.finditer(t):
        pre = t[max(0, m.start() - 40):m.start()]
        pre_words = pre.split()[-window:]
        if _NEGATION_RE.search(" ".join(pre_words)):
            continue
        hits += 1
    return hits


def _code_score(t):
    n_support = _polarity_hits(t, _SUPPORT_RE)
    n_oppose = _polarity_hits(t, _OPPOSE_RE)
    has_hedge_support = bool(_HEDGE_SUPPORT_RE.search(t))
    has_withdraw = bool(_WITHDRAW_RE.search(t))
    has_as_proposed = bool(_AS_PROPOSED_RE.search(t))

    net = n_support - n_oppose
    # base: -0.5 .. +0.5 mapped from net polarity, centered at 0.5 (neutral)
    base = 0.5 + max(-0.5, min(0.5, 0.10 * net))
    if has_hedge_support:
        base += 0.15
    if has_as_proposed:
        base += 0.10
    if has_withdraw:
        base = min(base, 0.05)  # withdrawal demand overrides -> near-zero regardless of counts
    return max(0.0, min(1.0, base))


_POSITION_SCORE = {
    "support": 0.95,
    "support_with_changes": 0.75,
    "neutral": 0.5,
    "oppose_in_part": 0.25,
    "oppose_fully": 0.05,
}


def _llm_score(extracted):
    pos = (extracted.get("overall_position") or "").strip().lower()
    pos = re.sub(r'[^a-z_]', '', pos)
    if pos in _NONE or not pos:
        return 0.5
    return _POSITION_SCORE.get(pos, 0.5)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        extracted = extracted or {}
        return max(0.0, min(1.0, 0.65 * _code_score(t) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5
