"""p901_v0_keyword -- Quantitative support (criterion p901), surface/lexical arm.

Pure keyword/regex counting: finds typed numeric evidence (currency amounts,
percentages, magnitude figures like "5 million", unit measurements, dates,
years) plus remaining bare numbers, weights them by specificity, normalizes
by document length, and squashes the weighted density per 100 words through
a saturating exponential. No structural or positional reasoning.
"""

import re
import math

# --- mojibake / entity normalization ----------------------------------------
# Longest sequences first; the bare "â€" residual must come after
# every longer "â€?" sequence it prefixes.
_MOJIBAKE = [
    ("â€œ", '"'),    # curly left double quote
    ("â€", '"'),    # curly right double quote
    ("â€™", "'"),    # curly apostrophe
    ("â€˜", "'"),    # curly left single quote
    ("â€“", "-"),    # en dash
    ("â€”", "-"),    # em dash
    ("â€¦", "..."),  # ellipsis
    ("â‚¬", "€"),  # euro sign
    ("â€", '"'),          # residual quote fragment (must be last)
    ("Â ", " "),          # mojibake non-breaking space
    ("Â ", " "),
    ("Â", ""),
    (" ", " "),                # plain non-breaking space
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

# --- typed numeric patterns, matched in priority order ----------------------
# (name, compiled regex, weight)
_PATTERNS = [
    ("currency", re.compile(
        r"[$£€¥]\s?\d[\d,]*(?:\.\d+)?"
        r"(?:\s?(?:million|billion|trillion|thousand|[MBK])\b)?"
        r"|\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion|thousand)?\s?"
        r"(?:dollars|euros|pounds|cents|USD|EUR|GBP|CAD|AUD|JPY)\b",
        re.IGNORECASE), 2.2),
    ("percent", re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent(?:age\s+points?)?)",
        re.IGNORECASE), 2.2),
    ("magnitude", re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion|thousand|mn|bn)\b",
        re.IGNORECASE), 2.0),
    ("measure", re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:kg|km|mg|ml|mm|cm|kWh?|MWh?|GWh?|GW|MW|"
        r"mph|sq\.?\s?ft|square\s+(?:feet|meters|metres|miles|kilometers)|"
        r"acres?|miles?|meters?|metres?|feet|tons?|tonnes?|barrels?|"
        r"gallons?|liters?|litres?|hectares?|ounces?|degrees?|"
        r"hours?|minutes?|days?|weeks?|months?|years?|employees|patients|"
        r"customers|members|students|countries|states|cities|stores|"
        r"locations|units|sites|jobs|people)\b",
        re.IGNORECASE), 1.5),
    ("date", re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|"
        r"Nov|Dec)\.?\s+\d{1,2}(?:,?\s+\d{4})?\b"
        r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.IGNORECASE), 0.8),
    ("year", re.compile(r"\b(?:19|20)\d{2}\b"), 0.6),
]

_BARE_NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_BARE_WEIGHT = 1.0

_WORD_RE = re.compile(r"\b\w+\b")


def _normalize(text):
    for bad, good in _MOJIBAKE:
        if bad in text:
            text = text.replace(bad, good)
    text = text.replace("[...]", " ")
    text = _URL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    return text


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _normalize(text)
        words = len(_WORD_RE.findall(t))
        if words < 10:
            return 0.0

        claimed = []

        def _free(s, e):
            for cs, ce in claimed:
                if s < ce and e > cs:
                    return False
            return True

        weighted = 0.0
        for _name, pat, w in _PATTERNS:
            for m in pat.finditer(t):
                if _free(m.start(), m.end()):
                    claimed.append((m.start(), m.end()))
                    weighted += w
        for m in _BARE_NUM_RE.finditer(t):
            if _free(m.start(), m.end()):
                claimed.append((m.start(), m.end()))
                weighted += _BARE_WEIGHT

        density = 100.0 * weighted / float(words)  # weighted hits per 100 words
        raw = 1.0 - math.exp(-density / 2.0)
        raw *= min(1.0, words / 40.0)              # damp very short docs
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
