"""p906 v2 -- Persuasive cadence, holistic composite.

Criterion: "The prose rhythm builds momentum across paragraphs:
sentence-length variation, paragraph pacing, and transitions sustain the
reader's attention from the headline through the closing boilerplate."

Approach: combine seven weak signals, each mapped to [0,1], into a
weighted composite:
  1. sentence-length variation (coefficient of variation in a sweet band),
  2. rhythmic alternation (mean absolute jump between successive
     sentence lengths -- the short/long "punch" pattern),
  3. length-bucket mix entropy (short / medium / long sentences all
     present, not a monotone drone),
  4. transition-marker density (sentence-initial + bridging connectives),
  5. paragraph pacing (variation of prose-paragraph sizes),
  6. prose share of the page (nav-chrome-dominated scrapes cannot carry
     a reader anywhere),
  7. rhythm punctuation (dashes/semicolons/colons in moderation).
Signals 1-4 carry most weight; the composite is clipped to [0,1].

Contract: score(text) -> float in [0, 1]; deterministic; never raises.
"""

import re
import math
import statistics
from collections import Counter

# ---------------------------------------------------------------- cleanup

_REPL = [
    ("â€œ", '"'), ("â€\x9d", '"'), ("â€™", "'"), ("â€˜", "'"),
    ("â€”", " -- "), ("â€“", " - "), ("â€¦", "..."), ("â€¢", " "),
    ("Â\xa0", " "), ("Â ", " "), ("Â", ""), ("\xa0", " "),
    ("&amp;", "&"), ("&nbsp;", " "), ("&quot;", '"'), ("&#39;", "'"),
    ("&apos;", "'"), ("&lt;", "<"), ("&gt;", ">"), ("&rsquo;", "'"),
    ("&lsquo;", "'"), ("&ldquo;", '"'), ("&rdquo;", '"'),
    ("&mdash;", " -- "), ("&ndash;", " - "), ("&hellip;", "..."),
]


def _clean(text):
    for bad, good in _REPL:
        text = text.replace(bad, good)
    return text.replace("[...]", "\n\n")


# --------------------------------------------------------------- carving

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z\"'(])")

_TRANSITION = re.compile(
    r"\b(however|moreover|furthermore|additionally|in addition|meanwhile|"
    r"as a result|therefore|thus|consequently|in fact|indeed|notably|"
    r"importantly|significantly|finally|ultimately|building on|"
    r"at the same time|in turn|for example|for instance|similarly|"
    r"likewise|looking ahead|going forward|to that end|since then|"
    r"not only|as well as|coupled with|resulting in|leading to|"
    r"underscoring|marking|paving the way)\b",
    re.IGNORECASE,
)

_RHYTHM_PUNCT = re.compile(r"(--|—|–|;|:)")


def _paragraphs(text):
    """Paragraph-like blocks of adjacent lines."""
    paras, cur = [], []
    for rawline in text.split("\n"):
        line = rawline.strip()
        if not line:
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        cur.append(line)
        if len(line) < 30 and not re.search(r"[.!?,;:]$", line):
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return paras


def _prose_paras(paras):
    out = []
    for p in paras:
        w = p.split()
        if len(w) < 15 or not re.search(r"[.!?]", p):
            continue
        lower = sum(1 for t in w if t[:1].islower())
        if lower / len(w) < 0.35:
            continue
        out.append(p)
    return out


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        text = _clean(text)
        paras = _paragraphs(text)
        prose = _prose_paras(paras)
        total_words = sum(len(p.split()) for p in paras)
        prose_words = sum(len(p.split()) for p in prose)
        if not prose or prose_words < 60:
            return 0.1

        sentences = []
        for p in prose:
            for s in _SENT_SPLIT.split(p):
                if len(s.split()) >= 3:
                    sentences.append(s)
        n = len(sentences)
        if n < 4:
            return 0.15

        slen = [len(s.split()) for s in sentences]
        mean_len = statistics.mean(slen)

        # 1. sentence-length variation -----------------------------------
        cv = statistics.pstdev(slen) / mean_len if mean_len else 0.0
        s_cv = _band(cv, 0.10, 0.30, 0.75, 1.20)

        # 2. rhythmic alternation (successive jumps) ---------------------
        diffs = [abs(a - b) for a, b in zip(slen, slen[1:])]
        alt = (statistics.mean(diffs) / mean_len) if diffs and mean_len else 0.0
        s_alt = _band(alt, 0.08, 0.25, 0.85, 1.40)

        # 3. bucket-mix entropy (short <=9 / medium / long >=23 words) ----
        buckets = Counter(
            "s" if L <= 9 else ("l" if L >= 23 else "m") for L in slen
        )
        probs = [c / n for c in buckets.values()]
        ent = -sum(p * math.log(p, 3) for p in probs if p > 0)  # 0..1
        s_mix = min(1.0, ent / 0.85)

        # 4. transition density ------------------------------------------
        hits = sum(len(_TRANSITION.findall(s)) for s in sentences)
        dens = hits / n
        s_tr = _band(dens, 0.02, 0.12, 0.90, 1.80)

        # 5. paragraph pacing --------------------------------------------
        plens = [len(p.split()) for p in prose]
        if len(plens) >= 3:
            pmean = statistics.mean(plens)
            pcv = statistics.pstdev(plens) / pmean if pmean else 0.0
            s_pace = _band(pcv, 0.05, 0.20, 0.85, 1.50)
        else:
            s_pace = 0.3

        # 6. prose share of page -----------------------------------------
        s_share = min(1.0, (prose_words / max(1, total_words)) / 0.55)

        # 7. rhythm punctuation per 100 prose words ----------------------
        punct = sum(len(_RHYTHM_PUNCT.findall(s)) for s in sentences)
        per100 = 100.0 * punct / max(1, prose_words)
        s_punct = _band(per100, 0.05, 0.40, 2.50, 5.00)

        raw = (0.20 * s_cv + 0.16 * s_alt + 0.14 * s_mix + 0.16 * s_tr
               + 0.14 * s_pace + 0.12 * s_share + 0.08 * s_punct)
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
