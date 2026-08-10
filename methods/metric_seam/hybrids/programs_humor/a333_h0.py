"""a333: Rule of three / triples (hybrid v0).

Criterion: use of triads to establish a minimal rhythmic pattern that primes
a surprising or escalated third beat (and can imply plurality visually).

Design:
  - Keyword-only signals (e.g. bare "three", "and", "finally") are WEAK
    proxies: many texts mention "three" as a plain fact/quantity, or use
    "first"/"second" without ever completing a triad, and the v0 baseline
    over-rewards these coincidental hits.
  - The real signal is STRUCTURAL: three parallel beats (dialogue turns,
    repeated actions, list items) where the third (or an inserted 4th
    element that displaces the expected third) delivers an escalation,
    subversion, or surprise. That semantic judgment (is beat 3 different/
    escalated vs. beats 1-2?) is a THICK-INPUT construct code cannot
    reliably reach on its own, so it is delegated to the LLM fields below.
  - Code keeps the PREDICATE: it parses/validates the LLM's short answers,
    and independently corroborates with cheap structural regex evidence
    (ordinal-word sequencing "first...second...third", the bare numeral/
    "triple" family, and repeated-content-word counts in the 2-4 range,
    which catches non-ordinal triads like a phrase repeated three times
    with escalating punctuation, e.g. "Ouch! Ouch!! OUCH!!!").
  - Corroboration is weighted lightly when the LLM already found a triad
    (nudge only) and more heavily when the LLM found nothing (partial
    rescue for extractor misses), but can never alone reach the top of
    the range - genuine top scores require LLM-confirmed escalation.
"""
import re
from collections import Counter

LLM_FIELDS = {
    "triad_items": "List the 3 parallel beats/items (comma or semicolon separated) if this text uses a rule-of-three structure, else NONE.",
    "third_twist": "In up to 8 words, say how the third/final beat escalates or subverts the first two beats, else NONE.",
}

_NONE_TOKENS = {"", "none", "n/a", "na", "no", "nothing", "not applicable"}

_NEGATIVE_TWIST_MARKERS = (
    "same as", "no differ", "identical", "no escalat", "no surpris",
    "nothing differ", "no twist", "no change", "not differ",
)

_STOPWORDS = {
    "that", "this", "with", "from", "have", "those", "these", "would",
    "could", "should", "there", "their", "when", "what", "your", "were",
    "been", "into", "just", "like", "some", "them", "then", "also",
    "because", "after", "before", "during", "which", "about", "where",
    "being", "doing", "having", "only", "very", "such", "most", "more",
    "than", "over", "under", "again", "further", "once", "here", "other",
    "same", "each", "both", "against", "between", "through", "above",
    "below", "down", "near", "upon", "within", "without", "will", "shall",
    "went", "walk", "walks", "walked", "walking",
}

_ORDINAL_WORDS = ("first", "second", "third")
_NUMERAL_RE = re.compile(r"\b(three|trio|triple|triad)\b")
_CONTENT_WORD_RE = re.compile(r"\b[a-zA-Z]{4,}\b")


def _clean_field(value) -> str:
    try:
        return (value or "").strip()
    except Exception:
        return ""


def _is_none_token(s: str) -> bool:
    return s.strip().lower() in _NONE_TOKENS


def _max_repeated_content_word_count(t: str) -> int:
    counts = Counter()
    for m in _CONTENT_WORD_RE.finditer(t):
        w = m.group(0).lower()
        if w in _STOPWORDS:
            continue
        counts[w] += 1
    if not counts:
        return 0
    return max(counts.values())


def _ordinal_sequence_score(t: str) -> float:
    positions = {}
    for w in _ORDINAL_WORDS:
        m = re.search(r"\b" + w + r"\b", t)
        positions[w] = m.start() if m else None
    if all(positions[w] is not None for w in _ORDINAL_WORDS):
        if positions["first"] < positions["second"] < positions["third"]:
            return 0.20
        return 0.12
    if positions["third"] is not None and (
        positions["first"] is not None or positions["second"] is not None
    ):
        return 0.08
    return 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        if not raw.strip():
            return 0.5

        try:
            normed = ops.normalize(raw)
        except Exception:
            normed = raw
        t = (normed or raw).lower()
        if not t.strip():
            return 0.5

        extracted = extracted or {}
        triad_field = _clean_field(extracted.get("triad_items", ""))
        twist_field = _clean_field(extracted.get("third_twist", ""))
        triad_present = not _is_none_token(triad_field)

        # --- LLM-grounded component: did the extractor find a real triad? ---
        if triad_present:
            items = [x.strip() for x in re.split(r"[;,]", triad_field) if x.strip()]
            n = len(items)
            if n == 3:
                a = 0.65
            elif n in (2, 4):
                a = 0.45
            else:
                a = 0.35

            if twist_field and not _is_none_token(twist_field):
                twist_l = twist_field.lower()
                if any(marker in twist_l for marker in _NEGATIVE_TWIST_MARKERS):
                    twist_bonus = 0.0
                else:
                    twist_bonus = 0.25
            else:
                twist_bonus = 0.0
            a = min(1.0, a + twist_bonus)
        else:
            a = 0.15

        # --- code-side structural corroboration (cheap, deterministic) ---
        ordinal_score = _ordinal_sequence_score(t)
        numeral_score = 0.05 if _NUMERAL_RE.search(t) else 0.0
        rep_count = _max_repeated_content_word_count(t)
        if rep_count == 3:
            repeat_score = 0.15
        elif rep_count == 4:
            repeat_score = 0.10
        elif rep_count == 2:
            repeat_score = 0.03
        else:
            repeat_score = 0.0
        b = min(0.35, ordinal_score + numeral_score + repeat_score)

        # Best-effort sentence-count gate: a single run-on "sentence" is a
        # weaker candidate for a multi-beat structure. Never fatal.
        n_sent = None
        try:
            stats = ops.sent_stats(raw)
            if isinstance(stats, dict):
                for k in ("n_sent", "num_sentences", "n_sentences", "sentence_count"):
                    if k in stats:
                        n_sent = stats[k]
                        break
            elif stats:
                n_sent = stats[0]
        except Exception:
            n_sent = None
        if isinstance(n_sent, (int, float)) and n_sent <= 1:
            b *= 0.5

        # If the LLM already confirmed a triad, corroboration only nudges;
        # if the LLM found nothing, allow structural evidence to partially
        # rescue (extractor miss) but never reach the top of the range.
        weight = 0.25 if triad_present else 0.6
        final = a + weight * b
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
