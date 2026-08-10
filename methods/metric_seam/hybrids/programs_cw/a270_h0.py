"""Hybrid metric channel for a270: Compression and focus in flash/short forms.

Criterion: tight scope, high signal, single moment/image/insight, line-by-line
compression. Train residual analysis: the judge ranking is dominated by a
monotone-decreasing function of length within this corpus (flash pieces of
~150-450 words with one conceit score 0.5-0.85; multi-scene sprawl scores
0.0-0.35). Within the short band, sloppy line-level craft (typo-dense prose),
verse/poem form, and trailing author-note cruft pull scores down. The keyword
baseline fails because it treats 100-1500 words as uniformly ideal.

Predicate stays in code; two LLM fields ground what regex cannot see:
scene span (focus) and line-level error frequency (compression of the line).
"""

import re

LLM_FIELDS = {
    "scene_span": (
        "Does the story stay in one continuous scene, a few linked scenes, "
        "or many scenes with time jumps? Answer one word: ONE, FEW, or MANY."
    ),
    "prose_errors": (
        "How frequent are spelling, grammar, or punctuation errors in the "
        "story text? Answer one word: NONE, SOME, or MANY."
    ),
}

# Piecewise-linear length curve: word_count -> [0,1].
_LEN_PTS = [
    (0, 0.25),
    (40, 0.80),
    (120, 1.00),
    (260, 1.00),
    (500, 0.72),
    (900, 0.45),
    (1600, 0.22),
    (3000, 0.08),
]

# Missing-apostrophe contractions that are unambiguous errors.
_ERR_TOKENS = re.compile(
    r"\b(dont|cant|wont|didnt|doesnt|isnt|wasnt|werent|couldnt|wouldnt|"
    r"shouldnt|hadnt|havent|im|ive|youre|theyre|thats|theres|whats|wheres)\b"
)
# Standalone lowercase "i" as a pronoun.
_LOWER_I = re.compile(r"(?<![\w'])i(?=[\s,.!?;:])")

_META_PATTERNS = [
    r"/?r/\w+",
    r"\bedit\s*:",
    r"\bfirst\s+(?:wp\s+)?post\b",
    r"\bthanks?\s+(?:you\s+)?for\s+(?:reading|the\s+gold)\b",
    r"\bfeedback\s+is\s+welcome\b",
    r"\bany\s+feedback\b",
    r"\bpart\s+\d+.{0,30}comments\b",
    r"\bcheck\s+out\b",
    r"\bmy\s+sub(?:reddit)?\b",
    r"\bupdate\s*:",
]


def _len_score(w):
    if w <= _LEN_PTS[0][0]:
        return _LEN_PTS[0][1]
    for (x0, y0), (x1, y1) in zip(_LEN_PTS, _LEN_PTS[1:]):
        if w <= x1:
            return y0 + (y1 - y0) * (w - x0) / float(x1 - x0)
    return 0.06


def _field_token(value, tokens):
    """Find which enum token a short LLM answer contains; None if unclear."""
    if not value:
        return None
    up = str(value).upper()
    hits = [t for t in tokens if re.search(r"\b" + t + r"\b", up)]
    return hits[0] if len(hits) == 1 else None


def _regex_error_rate(text, n_words):
    hits = len(_ERR_TOKENS.findall(text.lower()))
    hits += len(_LOWER_I.findall(text))
    return 1000.0 * hits / max(1, n_words)


def _verse_like(text, lines):
    head = text[:300].lower()
    if "[poem]" in head or "[pi]" in head:
        return True
    if len(lines) < 12:
        return False
    short = 0
    for ln in lines:
        wds = len(re.findall(r"[\w']+", ln))
        if (
            0 < wds <= 7
            and ":" not in ln
            and '"' not in ln
            and not re.search(r"[.!?][\"']?\s*$", ln)
        ):
            short += 1
    return short / float(len(lines)) >= 0.6


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        try:
            norm = ops.normalize(text)
            if not isinstance(norm, str) or not norm.strip():
                norm = text
        except Exception:
            norm = text

        words = re.findall(r"[\w']+", norm)
        n_words = len(words)
        lines = [ln.strip() for ln in norm.splitlines() if ln.strip()]

        base = _len_score(n_words)

        # Focus: LLM-grounded scene span (regex cannot see narrative scope).
        span = _field_token((extracted or {}).get("scene_span"), ("ONE", "FEW", "MANY"))
        span_mult = {"ONE": 1.0, "FEW": 0.82, "MANY": 0.55}.get(span, 0.90)

        # Line-level craft: LLM-grounded error frequency, regex fallback.
        errs = _field_token(
            (extracted or {}).get("prose_errors"), ("NONE", "SOME", "MANY")
        )
        if errs is not None:
            err_mult = {"NONE": 1.0, "SOME": 0.85, "MANY": 0.50}[errs]
        else:
            rate = _regex_error_rate(norm, n_words)
            err_mult = 0.50 if rate > 8.0 else (0.80 if rate > 3.0 else 1.0)

        # Trailing meta-cruft (plugs, edit notes, feedback begging) breaks
        # closure; mild penalty (a strong story survives a plug).
        zone = (norm[:200] + "\n" + norm[-600:]).lower()
        meta_hits = sum(1 for p in _META_PATTERNS if re.search(p, zone))
        meta_mult = 1.0 if meta_hits == 0 else (0.87 if meta_hits == 1 else 0.76)

        # Verse/ballad form is not compressed flash prose in this corpus.
        verse_mult = 0.50 if _verse_like(norm, lines) else 1.0

        raw = base * span_mult * err_mult * meta_mult * verse_mult
        out = 0.05 + 0.90 * raw
        return float(max(0.0, min(1.0, out)))
    except Exception:
        return 0.5
