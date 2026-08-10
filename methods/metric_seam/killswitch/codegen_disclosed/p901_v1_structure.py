"""p901_v1_structure -- Quantitative support (criterion p901), structural arm.

Positional / structural reasoning: strips navigation chrome (short,
punctuation-free menu lines) and the trailing contact/boilerplate block,
reconstructs the prose body, then asks WHERE concrete figures appear:
what fraction of body sentences carry a figure (early sentences weighted
up, tail sentences weighted down), whether the lede leads with a hard
number, and whether figures are spread across the document rather than
clumped in one spot.
"""

import re
import math

# --- mojibake / entity normalization (longest first; residual last) ---------
_MOJIBAKE = [
    ("â€œ", '"'),   # curly left double quote
    ("â€", '"'),   # curly right double quote
    ("â€™", "'"),   # curly apostrophe
    ("â€˜", "'"),   # curly left single quote
    ("â€“", "-"),   # en dash
    ("â€”", "-"),   # em dash
    ("â€¦", "..."), # ellipsis
    ("â‚¬", "€"),  # euro sign
    ("â€", '"'),         # residual quote fragment
    ("Â ", " "),         # mojibake non-breaking space
    ("Â ", " "),
    ("Â", ""),
    (" ", " "),
    ("&amp;", "&"),
    ("&nbsp;", " "),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("&rsquo;", "'"),
    ("&lsquo;", "'"),
    ("&rdquo;", '"'),
    ("&ldquo;", '"'),
    ("&gt;", " "),
    ("&lt;", " "),
]

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:\+?1[\s\-.])?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# markers that open a trailing contact / boilerplate section
_TAIL_MARKER_RE = re.compile(
    r"^(?:media\s+)?contacts?\b|^about\s+\S+|^for\s+(?:more|further)\s+"
    r"information\b|^source[:\s]|^###|^investor\s+relations\b|"
    r"^press\s+(?:contact|office)\b|forward-looking\s+statements|"
    r"safe\s+harbor", re.IGNORECASE)

# any concrete figure (digit-anchored)
_FIGURE_RE = re.compile(r"\d")
# "hard" figures for the lede bonus: money / percent / magnitude
_HARD_FIGURE_RE = re.compile(
    r"[$£€¥]\s?\d"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent|million|billion|trillion|thousand)\b",
    re.IGNORECASE)

_WORD_RE = re.compile(r"\b\w+\b")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize(text):
    for bad, good in _MOJIBAKE:
        if bad in text:
            text = text.replace(bad, good)
    text = text.replace("[...]", "\n")
    text = _URL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return text


def _is_prose_line(line):
    """Nav chrome is short and unpunctuated; prose is long or sentence-like."""
    n = len(line)
    if n >= 60:
        return True
    if n >= 25 and re.search(r"[.!?](?:\s|$)", line):
        return True
    return False


def _body_sentences(text):
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    # cut the trailing contact/boilerplate block (search last 30% of lines)
    tail_start = int(len(lines) * 0.70)
    cut = len(lines)
    for i in range(tail_start, len(lines)):
        if _TAIL_MARKER_RE.search(lines[i]):
            cut = i
            break
    lines = lines[:cut]

    # keep prose lines only, join into blocks
    prose = [ln for ln in lines if _is_prose_line(ln)]
    if not prose:
        return []
    blob = " ".join(prose)

    sents = _SENT_SPLIT_RE.split(blob)
    return [s for s in sents if len(_WORD_RE.findall(s)) >= 6]


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _normalize(text)
        sents = _body_sentences(t)

        if len(sents) < 3:
            # No recoverable prose body (nav page / stub): crude fallback,
            # capped low -- claims cannot be well-supported without prose.
            lines = [ln for ln in (x.strip() for x in t.split("\n")) if ln]
            if not lines:
                return 0.0
            frac = sum(1 for ln in lines if _FIGURE_RE.search(ln)) / len(lines)
            return max(0.0, min(0.25, 0.25 * frac))

        n = len(sents)
        q1 = max(1, int(math.ceil(n / 4.0)))   # first quarter: lede zone
        q4 = int(n * 0.8)                       # last fifth: contact-adjacent

        wsum = 0.0
        whit = 0.0
        for i, s in enumerate(sents):
            w = 1.4 if i < q1 else (0.5 if i >= q4 else 1.0)
            wsum += w
            if _FIGURE_RE.search(s):
                whit += w
        weighted_frac = whit / wsum if wsum > 0 else 0.0

        # lede bonus: does the opening make a numeric claim?
        lede = " ".join(sents[:3])
        if _HARD_FIGURE_RE.search(lede):
            lede_score = 1.0
        elif _FIGURE_RE.search(lede):
            lede_score = 0.5
        else:
            lede_score = 0.0

        # spread: figures distributed across thirds of the body, not clumped
        third = max(1, n // 3)
        hits_by_third = [0, 0, 0]
        for i, s in enumerate(sents):
            if _FIGURE_RE.search(s):
                hits_by_third[min(2, i // third)] = 1
        spread = sum(hits_by_third) / 3.0

        sat = 1.0 - math.exp(-weighted_frac / 0.22)
        raw = 0.70 * sat + 0.15 * lede_score + 0.15 * spread
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
