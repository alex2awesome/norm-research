"""p907_v1_structure — Criterion: Comprehensive detail (structural/positional).

Scores completeness of a scraped press release from its STRUCTURE:
how many substantive body paragraphs it has, whether it exhibits the
canonical release skeleton (headline, dateline lede, quote paragraph,
"About X" boilerplate section, trailing contact block), and how much
body sits BEFORE the trailing contact/boilerplate zone. A terse stub
has one or two thin paragraphs and little or no skeleton; a full
release stacks several meaty paragraphs plus the standard furniture.

Contract: score(text: str) -> float in [0.0, 1.0]; deterministic;
imports limited to re/math/statistics/collections; returns 0.5 on
unexpected internal error.
"""

import re
import math
import statistics


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

# Dateline lede: "SAN FRANCISCO, June 12, 2024 -" / "LONDON -- ..." styles.
_DATELINE_RE = re.compile(
    r"^\s*[A-Z][A-Za-z.\- ]{1,40},?\s*(?:[A-Z]{2},?\s*)?"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?\s+\d{1,2},?\s+(?:19|20)\d{2}"
)
_DATELINE_DASH_RE = re.compile(
    r"^\s*[A-Z][A-Z.\- ]{2,40}(?:,\s*[A-Za-z. ]+)?\s*"
    r"[-" + chr(0x2013) + chr(0x2014) + r"]{1,2}\s+\w"
)

# Trailing furniture markers (positional: expected near the END).
_CONTACT_RE = re.compile(
    r"(?im:^\s*(?:(?:media|press|investor)\s+)?contacts?\s*:?\s*$)"
    r"|(?i:\bfor (?:more|further|additional) information\b)"
    r"|[\w.\-]+@[\w\-]+\.[A-Za-z]{2,}"
    r"|(?:\+?\d[\d ()\-.]{7,}\d)"
    r"|(?m:^\s*#{3,}\s*$)"
)
_ABOUT_RE = re.compile(r"(?im)^\s*about\s+[A-Z0-9][\w&.,' \-]{1,60}:?\s*$")

_QUOTE_PARA_RE = re.compile(r'"[^"]{25,600}"')
_ATTRIB_RE = re.compile(
    r"\b(?:said|says|stated|noted|commented|added|explained)\b", re.IGNORECASE
)

# Nav-chrome-ish line: very short, no sentence punctuation.
_END_PUNCT_RE = re.compile(r"[.!?][\"')\]]?\s*$")


def _sat(x, k):
    """Smooth saturation: 0 at 0, ~0.63 at k, -> 1.0 asymptotically."""
    if x <= 0:
        return 0.0
    return 1.0 - math.exp(-float(x) / float(k))


def _paragraphs(t):
    if re.search(r"\n\s*\n", t):
        parts = re.split(r"\n\s*\n+", t)
    else:
        parts = t.split("\n")
    return [p.strip() for p in parts if p.strip()]


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        # Elision marker acts as a paragraph boundary, not content.
        t = _clean(text).replace("[...]", "\n\n")
        paras = _paragraphs(t)
        if not paras:
            return 0.0

        n_words_total = sum(len(_WORD_RE.findall(p)) for p in paras)
        if n_words_total < 5:
            return 0.0

        # --- locate trailing contact/boilerplate zone (last ~25%) ---------
        cut = len(paras)
        tail_start = max(1, int(math.floor(len(paras) * 0.75)))
        for i in range(len(paras) - 1, tail_start - 1, -1):
            if _CONTACT_RE.search(paras[i]) or _ABOUT_RE.search(paras[i]):
                cut = i
        body = paras[:cut]
        tail = paras[cut:]

        # --- substantive body paragraphs -----------------------------------
        para_words = [len(_WORD_RE.findall(p)) for p in body]
        substantive = [w for w in para_words if w >= 25]
        n_sub = len(substantive)
        body_words = sum(para_words)

        # chrome fraction: short unpunctuated lines (menus, link lists)
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        chrome = sum(
            1 for ln in lines
            if len(_WORD_RE.findall(ln)) <= 4 and not _END_PUNCT_RE.search(ln)
        )
        chrome_frac = chrome / float(len(lines)) if lines else 0.0

        # --- skeleton components (each a completeness cue) ------------------
        head = "\n".join(paras[:3])
        has_dateline = bool(
            _DATELINE_RE.search(head) or _DATELINE_DASH_RE.search(head)
        )
        has_quote = any(
            _QUOTE_PARA_RE.search(p) and _ATTRIB_RE.search(p) for p in body
        )
        has_about = any(_ABOUT_RE.search(p) for p in paras)
        has_contact_tail = any(_CONTACT_RE.search(p) for p in tail) or (
            cut < len(paras)
        )

        # --- component scores -----------------------------------------------
        s_paras = _sat(n_sub, 3.5)                   # depth: meaty grafs
        s_body = _sat(body_words, 320.0)             # extent before tail
        mean_pw = statistics.mean(para_words) if para_words else 0.0
        s_meat = max(0.0, min(1.0, mean_pw / 55.0))  # avg graf heft
        s_skel = (
            0.30 * has_dateline
            + 0.35 * has_quote
            + 0.20 * has_about
            + 0.15 * has_contact_tail
        )

        s = 0.35 * s_paras + 0.30 * s_body + 0.15 * s_meat + 0.20 * s_skel
        # Nav chrome dilutes structural evidence of a full document.
        if chrome_frac > 0.5:
            s *= 1.0 - 0.5 * (chrome_frac - 0.5) * 2.0
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
