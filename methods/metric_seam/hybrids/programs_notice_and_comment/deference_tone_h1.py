"""deference_tone hybrid: Professional/deferential register vs hostile/demanding (CODE ONLY).

Construct: ~1.0 = comment uses politeness/deference markers ("respectfully", "we appreciate
the opportunity", "thank you for considering"), hedged phrasing, and low imperative/insult
density; ~0.5 = neutral, businesslike tone with neither strong politeness nor hostility
markers; ~0.0 = hostile/demanding tone: direct accusatory second-person address ("you
people", "your agency"), heavy imperatives, ALL-CAPS shouting, exclamation stacking, or
outright insults.

INPUT = comment text only. This is a pure-code aspect (no LLM extraction needed — tone
register is directly observable from lexical/orthographic markers, not content requiring
semantic judgment). Uses ops.sent_stats only incidentally (not required for this construct).
"""
import re

LLM_FIELDS = {}

_POLITE_RE = re.compile(
    r'\b(respectfully|we appreciate|thank you for (?:the opportunity|considering|your time)|'
    r'we are grateful|please consider|we ask that you kindly|we would (?:like to )?request|'
    r'we urge you to consider|appreciate the opportunity to comment)\b', re.I)
_HEDGE_RE = re.compile(
    r'\b(we believe|it (?:seems|appears)|in our view|we would suggest|perhaps|might '
    r'(?:consider|want)|we respectfully (?:submit|disagree)|to the extent that)\b', re.I)
_ACCUSATORY_2P_RE = re.compile(
    r'\byou people\b|\byour agency\b|\byou (?:clearly|obviously) '
    r'(?:don\'?t|do not|fail(?:ed)?)\b|\byou (?:should be ashamed|owe us)\b', re.I)
_IMPERATIVE_RE = re.compile(
    r'(?m)^\s*(?:stop|do not|don\'?t|must|shall|fix|reject|kill|scrap)\b', re.I)
_INSULT_RE = re.compile(
    r'\b(incompetent|corrupt|idiotic|stupid|ridiculous|absurd|shameful|disgraceful|'
    r'outrageous|pathetic|liars?|lying)\b', re.I)


def _allcaps_ratio(t):
    words = re.findall(r"[A-Za-z']{3,}", t)
    if not words:
        return 0.0
    caps = sum(1 for w in words if w.isupper())
    return caps / len(words)


def _exclaim_density(t):
    n_excl = t.count("!")
    n_sent = max(1, len(re.split(r'(?<=[.!?])\s+', t)))
    return n_excl / n_sent


def _code_score(t):
    n_polite = len(_POLITE_RE.findall(t))
    n_hedge = len(_HEDGE_RE.findall(t))
    n_accus = len(_ACCUSATORY_2P_RE.findall(t))
    n_imper = len(_IMPERATIVE_RE.findall(t))
    n_insult = len(_INSULT_RE.findall(t))
    caps_ratio = _allcaps_ratio(t)
    excl_density = _exclaim_density(t)

    positive = min(0.35, 0.15 * n_polite) + min(0.15, 0.06 * n_hedge)
    negative = (
        min(0.25, 0.15 * n_accus)
        + min(0.15, 0.04 * n_imper)
        + min(0.25, 0.12 * n_insult)
        + min(0.15, 1.5 * caps_ratio)
        + min(0.15, 0.20 * excl_density)
    )
    return max(0.0, min(1.0, 0.5 + positive - negative))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        return _code_score(t)
    except Exception:
        return 0.5
