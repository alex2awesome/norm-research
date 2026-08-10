"""p907_v2_holistic — Criterion: Comprehensive detail (composite of weak signals).

Blends many weak, mutually independent cues of informational richness:
prose volume, sentence count and shape, lexical diversity, numeric
density, proper-noun density (specific actors/places/products),
discourse connectives (background/context linkage), quotation presence,
and paragraph depth — with a penalty for navigation-chrome noise.
No single cue dominates; a terse stub fails nearly all of them, a
full release with background, figures, quotes and context passes most.

Contract: score(text: str) -> float in [0.0, 1.0]; deterministic;
imports limited to re/math/statistics/collections; returns 0.5 on
unexpected internal error.
"""

import re
import math
from collections import Counter


# --- mojibake normalization (UTF-8 read as cp1252 artifacts) --------------
# Built via chr() because several sequences end in INVISIBLE characters
# (U+009D, NBSP) that do not survive copy/paste. Longer sequences MUST
# precede the bare 2-char stub _A.
_A = chr(0xE2) + chr(0x20AC)  # "a-circumflex + euro" mojibake lead-in
_MOJIBAKE = (
    (_A + chr(0x0153), '"'),    # mojibake left curly double quote
    (_A + chr(0x009D), '"'),    # mojibake right curly double (invisible tail)
    (_A + chr(0x2122), "'"),    # mojibake apostrophe
    (_A + chr(0x02DC), "'"),    # mojibake left single quote
    (_A + chr(0x201D), " - "),  # mojibake em dash
    (_A + chr(0x201C), " - "),  # mojibake en dash
    (_A + chr(0x00A6), "..."),  # mojibake ellipsis
    (_A, '"'),                  # bare stub (lost final byte)
    (chr(0xC2) + chr(0xA0), " "),   # mojibake A-circ + NBSP -> space
    (chr(0xC2), ""),                # stray mojibake A-circ
    (chr(0x201C), '"'), (chr(0x201D), '"'),  # real curly doubles
    (chr(0x2018), "'"), (chr(0x2019), "'"),  # real curly singles
    (chr(0xA0), " "),                        # NBSP -> space
)


def _clean(text):
    for bad, good in _MOJIBAKE:
        if bad in text:
            text = text.replace(bad, good)
    return text


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENT_RE = re.compile(r"[^.!?]*[.!?]")
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_QUOTE_RE = re.compile(r'"[^"]{20,600}"')
_CONNECTIVE_RE = re.compile(
    r"\b(?:according to|in addition|additionally|furthermore|moreover|"
    r"as part of|as a result|meanwhile|previously|since|because|"
    r"for example|such as|including|while|although|however|therefore)\b",
    re.IGNORECASE,
)
_PROPER_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
_CHROME_RE = re.compile(
    r"(?i)\b(?:home|menu|subscribe|sign in|log ?in|share this|tweet|"
    r"privacy policy|terms of use|cookies?|read more|related articles|"
    r"click here|newsletter|follow us)\b"
)
_CONTACT_RE = re.compile(
    r"(?i)(?:media|press|investor)?\s*contacts?:|for (?:more|further) information|"
    r"[\w.\-]+@[\w\-]+\.[A-Za-z]{2,}"
)
_END_PUNCT_RE = re.compile(r"[.!?][\"')\]]?\s*$")


def _sat(x, k):
    """Smooth saturation: 0 at 0, ~0.63 at k, -> 1.0 asymptotically."""
    if x <= 0:
        return 0.0
    return 1.0 - math.exp(-float(x) / float(k))


def _band(x, lo, hi):
    """1.0 inside [lo, hi], linear falloff outside, floor 0."""
    if x <= 0:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return max(0.0, x / lo)
    return max(0.0, 1.0 - (x - hi) / (2.0 * hi))


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _clean(text).replace("[...]", " ")

        words = _WORD_RE.findall(t)
        n_words = len(words)
        if n_words < 5:
            return 0.0

        sents = [s for s in _SENT_RE.findall(t) if len(_WORD_RE.findall(s)) >= 4]
        n_sents = len(sents)
        sent_lens = [len(_WORD_RE.findall(s)) for s in sents]
        mean_sl = (sum(sent_lens) / float(n_sents)) if n_sents else 0.0

        if re.search(r"\n\s*\n", t):
            paras = [p for p in re.split(r"\n\s*\n+", t) if p.strip()]
        else:
            paras = [p for p in t.split("\n") if p.strip()]
        n_meaty_paras = sum(1 for p in paras if len(_WORD_RE.findall(p)) >= 30)

        # Lexical diversity on a fixed prefix (length-independent).
        prefix = [w.lower() for w in words[:300]]
        ttr = len(Counter(prefix)) / float(len(prefix)) if prefix else 0.0

        n_nums = len(_NUM_RE.findall(t))
        num_per100 = 100.0 * n_nums / float(n_words)
        n_conn = len(_CONNECTIVE_RE.findall(t))
        n_quotes = len(_QUOTE_RE.findall(t))
        n_proper = len(_PROPER_RE.findall(t))
        proper_per100 = 100.0 * n_proper / float(n_words)

        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        chrome_hits = sum(1 for ln in lines if _CHROME_RE.search(ln))
        short_unpunct = sum(
            1 for ln in lines
            if len(_WORD_RE.findall(ln)) <= 4 and not _END_PUNCT_RE.search(ln)
        )
        chrome_frac = (
            (chrome_hits + short_unpunct) / float(len(lines)) if lines else 0.0
        )

        # --- weak signals, each in [0,1] ------------------------------------
        z_len = _sat(n_words, 300.0)               # prose volume
        z_sents = _sat(n_sents, 12.0)              # number of full sentences
        z_shape = _band(mean_sl, 12.0, 32.0)       # journalistic sentence heft
        z_ttr = _band(ttr, 0.45, 0.80)             # varied vocabulary
        z_nums = _band(num_per100, 1.0, 10.0)      # quantitative specificity
        z_conn = _sat(n_conn, 4.0)                 # background/context linkage
        z_quote = _sat(n_quotes, 1.5)              # voices quoted
        z_proper = _band(proper_per100, 3.0, 18.0)  # named specifics
        z_paras = _sat(n_meaty_paras, 3.0)         # paragraph depth

        s = (
            0.22 * z_len
            + 0.13 * z_sents
            + 0.08 * z_shape
            + 0.07 * z_ttr
            + 0.12 * z_nums
            + 0.10 * z_conn
            + 0.10 * z_quote
            + 0.08 * z_proper
            + 0.10 * z_paras
        )
        # Chrome-heavy scrapes: most "content" is navigation, not detail.
        if chrome_frac > 0.4:
            s *= 1.0 - 0.6 * min(1.0, (chrome_frac - 0.4) / 0.6)
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
