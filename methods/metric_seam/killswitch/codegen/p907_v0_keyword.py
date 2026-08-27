"""p907_v0_keyword — Criterion: Comprehensive detail (surface/lexical heuristic).

Scores how much substantial informative detail a scraped press release
contains, using only surface lexical signals: raw volume of prose,
numeric/quantitative specificity, detail-bearing keywords, and
attribution/quotation cues. Higher = fuller, more complete release.

Contract: score(text: str) -> float in [0.0, 1.0]; deterministic;
imports limited to re/math/statistics/collections; returns 0.5 on
unexpected internal error.
"""

import re
import math


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


# Detail-bearing vocabulary: background, specifics, supporting facts, context.
_DETAIL_TERMS = (
    r"according to", r"in addition", r"additionally", r"furthermore",
    r"moreover", r"as part of", r"as a result", r"for example", r"such as",
    r"including", r"more than", r"approximately", r"up to", r"per cent",
    r"percent", r"million", r"billion", r"founded", r"headquartered",
    r"based in", r"established", r"previously", r"history", r"background",
    r"research", r"study", r"survey", r"report(?:ed|s)?", r"data",
    r"results", r"findings", r"expanded?", r"growth", r"customers",
    r"partnership", r"agreement", r"initiative", r"program(?:me)?",
    r"technology", r"available", r"details", r"features", r"designed to",
    r"aims to", r"will (?:be|provide|offer|enable|allow)",
)
_DETAIL_RE = re.compile(r"\b(?:" + "|".join(_DETAIL_TERMS) + r")\b", re.IGNORECASE)

# Attribution / quotation verbs typical of fleshed-out releases.
_ATTRIB_RE = re.compile(
    r"\b(?:said|says|stated|noted|commented|added|explained|announced|"
    r"remarked|emphasized|emphasised|according to)\b",
    re.IGNORECASE,
)

_QUOTE_RE = re.compile(r'"([^"]{20,600})"')
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_MONEY_PCT_RE = re.compile(
    r"(?:[$" + chr(0xA3) + chr(0x20AC) + r"]\s?\d"
    r"|\d+(?:\.\d+)?\s?(?:%|percent|per cent))",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Navigation-chrome noise lines that should not count as "detail".
_CHROME_RE = re.compile(
    r"(?im)^\s*(?:home|menu|search|subscribe|sign in|log ?in|share|tweet|"
    r"print|contact us|about us|privacy policy|terms of use|cookies?|"
    r"read more|related (?:articles|posts|stories)|newsletter)\s*$"
)


def _sat(x, k):
    """Smooth saturation: 0 at 0, ~0.63 at k, -> 1.0 asymptotically."""
    if x <= 0:
        return 0.0
    return 1.0 - math.exp(-float(x) / float(k))


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _clean(text)
        # Drop pure navigation-chrome lines so menus don't count as content.
        t = _CHROME_RE.sub(" ", t)
        # Elision marker is scrape metadata, not content.
        t = t.replace("[...]", " ")

        words = _WORD_RE.findall(t)
        n_words = len(words)
        if n_words < 5:
            return 0.0

        n_detail = len(_DETAIL_RE.findall(t))
        n_attrib = len(_ATTRIB_RE.findall(t))
        n_quotes = len(_QUOTE_RE.findall(t))
        n_nums = len(_NUM_RE.findall(t))
        n_money = len(_MONEY_PCT_RE.findall(t))
        n_years = len(_YEAR_RE.findall(t))

        # Volume of prose: a full release typically runs 300-800+ words.
        s_len = _sat(n_words, 350.0)
        # Quantitative specificity (figures, money, percentages, dates).
        s_num = _sat(n_nums + 2.0 * n_money + n_years, 8.0)
        # Detail-bearing vocabulary (background/context/supporting facts).
        s_kw = _sat(n_detail, 7.0)
        # Voices and attribution (quotes flesh out a release).
        s_quote = _sat(n_attrib + 2.0 * n_quotes, 4.0)

        s = 0.40 * s_len + 0.20 * s_num + 0.25 * s_kw + 0.15 * s_quote
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
