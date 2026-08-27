"""p906 v0 -- Persuasive cadence, surface/lexical heuristic.

Criterion: "The prose rhythm builds momentum across paragraphs:
sentence-length variation, paragraph pacing, and transitions sustain the
reader's attention from the headline through the closing boilerplate."

Approach (purely lexical / regex-count based):
  * density of discourse-transition markers (sentence-initial and
    mid-sentence connectives) per prose sentence,
  * density of "rhythm punctuation" (em-dash, semicolon, colon,
    rhetorical question marks) per 100 words,
  * variety of distinct transition types used (a cadenced writer varies
    connectives instead of repeating one).
Each raw density is mapped through a trapezoidal "sweet-spot" band
(too few markers = flat, choppy prose; too many = stilted) and the
banded signals are combined with fixed weights.

Contract: score(text) -> float in [0, 1]; deterministic; never raises.
"""

import re
import math
import statistics
from collections import Counter

# ---------------------------------------------------------------- cleanup

_MOJIBAKE = [
    ("\xc3\xa2\xc2\x80\xc2\x9c", '"'),  # rare double-encoded forms
    ("\xe2\x80\x9c", '"'),
    ("â€œ", '"'),
    ("â€\x9d", '"'),
    ("â€™", "'"),
    ("â€˜", "'"),
    ("â€”", " -- "),
    ("â€“", " - "),
    ("â€¦", "..."),
    ("â€¢", " "),
    ("Â\xa0", " "),
    ("Â ", " "),
    ("Â", ""),
    ("\xa0", " "),
]

_ENTITIES = [
    ("&amp;", "&"), ("&nbsp;", " "), ("&quot;", '"'), ("&#39;", "'"),
    ("&apos;", "'"), ("&lt;", "<"), ("&gt;", ">"), ("&rsquo;", "'"),
    ("&lsquo;", "'"), ("&ldquo;", '"'), ("&rdquo;", '"'),
    ("&mdash;", " -- "), ("&ndash;", " - "), ("&hellip;", "..."),
]


def _clean(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    for bad, good in _ENTITIES:
        text = text.replace(bad, good)
    # elision marker from truncation -> hard boundary, not prose
    text = text.replace("[...]", "\n")
    return text


# ------------------------------------------------------- sentence carving

_ABBREV = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Inc|Corp|Ltd|Co|St|No|vs|etc|approx|Jr|Sr|"
    r"U\.S|U\.K|e\.g|i\.e|Fig|Rev)\.$"
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def _prose_sentences(text):
    """Extract sentences that look like running prose (skip nav chrome)."""
    sentences = []
    for rawline in text.split("\n"):
        line = rawline.strip()
        if len(line) < 40:
            continue
        words = line.split()
        # nav-chrome heuristic: no sentence punctuation and few words
        if not re.search(r"[.!?]", line) and len(words) < 12:
            continue
        parts = _SENT_SPLIT.split(line)
        buf = ""
        for p in parts:
            buf = (buf + " " + p).strip() if buf else p
            if _ABBREV.search(buf):
                continue
            if len(buf.split()) >= 5 and re.search(r"[a-z]", buf):
                sentences.append(buf)
            buf = ""
        if buf and len(buf.split()) >= 5 and re.search(r"[a-z]", buf):
            sentences.append(buf)
    return sentences


# ------------------------------------------------------ keyword lexicons

_INITIAL_TRANSITIONS = re.compile(
    r"^(however|moreover|furthermore|additionally|in addition|meanwhile|"
    r"as a result|therefore|thus|consequently|in fact|indeed|notably|"
    r"importantly|significantly|first|second|third|finally|ultimately|"
    r"beyond that|building on|at the same time|in turn|for example|"
    r"for instance|similarly|likewise|still|yet|already|now|today|"
    r"looking ahead|going forward|together|what is more|in short|"
    r"in other words|to that end|with that|since then|earlier|next|"
    r"crucially|better yet|more broadly)\b[ ,]",
    re.IGNORECASE,
)

_MID_CONNECTIVES = re.compile(
    r"\b(but|because|so that|which means|not only|as well as|while|"
    r"whereas|although|even as|thanks to|driven by|paving the way|"
    r"building on|coupled with|in order to|resulting in|leading to|"
    r"underscoring|highlighting|marking|signaling)\b",
    re.IGNORECASE,
)

_RHYTHM_PUNCT = re.compile(r"(--|—|–|;|:)")


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        text = _clean(text)
        sentences = _prose_sentences(text)
        n = len(sentences)
        if n < 3:
            return 0.1  # no sustained prose -> cadence cannot exist

        total_words = sum(len(s.split()) for s in sentences)
        if total_words < 40:
            return 0.15

        # --- signal 1: sentence-initial transition density -------------
        init_hits = sum(1 for s in sentences if _INITIAL_TRANSITIONS.match(s))
        init_density = init_hits / n
        # sweet spot ~ 8%..35% of sentences opening with a connective
        s1 = _band(init_density, 0.03, 0.08, 0.35, 0.60)

        # --- signal 2: mid-sentence connective density ------------------
        mid_hits = sum(len(_MID_CONNECTIVES.findall(s)) for s in sentences)
        mid_density = mid_hits / n
        # sweet spot ~ 0.3..1.2 connectives per sentence
        s2 = _band(mid_density, 0.10, 0.30, 1.20, 2.20)

        # --- signal 3: rhythm punctuation per 100 words -----------------
        punct_hits = sum(len(_RHYTHM_PUNCT.findall(s)) for s in sentences)
        punct_per100 = 100.0 * punct_hits / total_words
        # sweet spot ~ 0.4..2.5 per 100 words
        s3 = _band(punct_per100, 0.10, 0.40, 2.50, 5.00)

        # --- signal 4: variety of transition vocabulary -----------------
        kinds = Counter()
        for s in sentences:
            m = _INITIAL_TRANSITIONS.match(s)
            if m:
                kinds[m.group(1).lower()] += 1
        distinct = len(kinds)
        s4 = min(1.0, distinct / 4.0)  # >=4 distinct connectives = full marks
        if kinds and max(kinds.values()) / max(1, sum(kinds.values())) > 0.7 \
                and sum(kinds.values()) >= 4:
            s4 *= 0.5  # one connective hammered repeatedly

        raw = 0.35 * s1 + 0.25 * s2 + 0.20 * s3 + 0.20 * s4
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5


def _band(x, lo0, lo, hi, hi0):
    """Trapezoid: 0 below lo0, ramp to 1 at lo, flat to hi, ramp to 0 at hi0."""
    if x <= lo0 or x >= hi0:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return (x - lo0) / (lo - lo0)
    return (hi0 - x) / (hi0 - hi)
