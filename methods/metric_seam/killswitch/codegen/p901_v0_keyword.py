# -*- coding: utf-8 -*-
"""p901_v0_keyword -- surface/lexical metric for 'Quantitative support'.

Approach: weighted regex counts of numeric evidence (currency amounts,
percentages, magnitude numbers like '4.2 million', number+unit
measurements, numeric dates, comma-grouped and bare numbers) are turned
into a per-100-words density and squashed through a saturating
exponential.  Mojibake is normalized first, and URLs / e-mails / phone
numbers / copyright years / '[...]' elision markers are stripped so that
navigation-chrome and contact-boilerplate digits do not count as
quantitative evidence.  Higher-value patterns are matched first and their
spans blanked out so '$4.2 million' is counted exactly once.

Source is pure ASCII: all non-ASCII characters are built with chr() so
the module is immune to encoding mangling.
"""

import re
import math

# Mojibake building blocks (UTF-8 bytes mis-decoded as cp1252).
_AC = chr(0x00E2)            # a-circumflex
_EU = chr(0x20AC)            # euro sign
_MOJI_PREFIX = _AC + _EU     # common two-char prefix of the 3-char forms

# Longest sequences first; the bare two-char prefix and the lone
# A-circumflex come last so they cannot pre-empt the longer forms.
_MOJIBAKE = [
    (_MOJI_PREFIX + chr(0x0153), '"'),   # left curly double quote
    (_MOJI_PREFIX + chr(0x009D), '"'),   # right curly double quote (invisible U+009D)
    (_MOJI_PREFIX + chr(0x2122), "'"),   # apostrophe / right single quote
    (_MOJI_PREFIX + chr(0x02DC), "'"),   # left single quote
    (_MOJI_PREFIX + chr(0x201C), "-"),   # en dash
    (_MOJI_PREFIX + chr(0x201D), "-"),   # em dash
    (_MOJI_PREFIX + chr(0x00A6), "..."), # ellipsis
    (_AC + chr(0x201A) + chr(0x00AC), _EU),  # euro mojibake -> euro sign
    (_MOJI_PREFIX, '"'),                 # leftover bare form
    (chr(0x00C2), ""),                   # A-circumflex before nbsp; recovers pound
]

_STRIP_RES = [
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),          # URLs
    re.compile(r"\S+@\S+"),                                       # e-mails
    re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b"),                   # phones 555-123-4567
    re.compile(r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}"),                 # phones (555) 123 4567
    re.compile("(?:" + chr(0x00A9) + r"|\(c\)|copyright)\s*\d{4}",
                re.IGNORECASE),                                   # copyright years
    re.compile(r"\b\d+\s+[A-Z][A-Za-z]+\s+(?:Street|St\.?|Avenue|Ave\.?|"
               r"Way|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|"
               r"Parkway|Pkwy\.?|Plaza|Square)\b"),               # street addresses
    re.compile(r"\b(?:Suite|Ste\.?|Floor|Fl\.?|Room|Rm\.?|Unit)\s+\d+\b",
               re.IGNORECASE),                                    # suite/floor numbers
    re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"),               # state + ZIP
    re.compile(r"\[\.\.\.\]"),                                    # elision marker
]

_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
           r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
           r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")

_UNITS = (r"(?:units?|employees?|customers?|users?|members?|subscribers?|"
          r"patients?|students?|stores?|locations?|branches?|offices?|"
          r"countries|states|cities|markets?|acres?|hectares?|"
          r"square\s+(?:feet|meters?|miles?)|sq\.?\s?ft\.?|miles?|"
          r"kilometers?|km|meters?|feet|ft|tons?|tonnes?|pounds?|lbs?|kg|"
          r"kilograms?|grams?|liters?|gallons?|barrels?|megawatts?|"
          r"gigawatts?|kilowatts?|MW|GW|kWh|MWh|jobs?|shares?|people|"
          r"homes?|households?|vehicles?|devices?|downloads?|"
          r"transactions?|hours?|minutes?|days?|weeks?|months?|years?|"
          r"quarters?)")

# Currency symbols: $, euro, pound, yen.
_CURRENCY_CLASS = "[$" + chr(0x20AC) + chr(0x00A3) + chr(0x00A5) + "]"

# (compiled pattern, weight) -- applied in order, matched spans blanked out.
_PATTERNS = [
    # Currency amounts (optionally with a scale word)
    (re.compile("(?:" + _CURRENCY_CLASS + r"|\bUS\$|\bUSD\s|\bEUR\s|\bGBP\s)"
                r"\s?\d[\d,]*(?:\.\d+)?"
                r"(?:\s?(?:million|billion|trillion|thousand|mn|bn|[MBK])\b)?",
                re.IGNORECASE), 2.0),
    (re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion|thousand)"
                r"\s+dollars\b", re.IGNORECASE), 2.0),
    # Percentages
    (re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|"
                r"percentage\s+points?\b|pct\b|basis\s+points\b|bps\b)",
                re.IGNORECASE), 2.0),
    # Magnitude numbers ('3.5 billion')
    (re.compile(r"\b\d[\d,]*(?:\.\d+)?\s(?:million|billion|trillion|"
                r"thousand|hundred)\b", re.IGNORECASE), 2.0),
    # Number + unit measurements
    (re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?" + _UNITS + r"\b",
                re.IGNORECASE), 1.5),
    # Numeric dates
    (re.compile(r"\b" + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?"
                r"(?:,?\s+\d{4})?\b", re.IGNORECASE), 1.0),
    (re.compile(r"\b\d{1,2}\s+" + _MONTHS + r"\b\.?(?:,?\s+\d{4})?",
                re.IGNORECASE), 1.0),
    (re.compile(r"\b(?:Q[1-4]|FY|H[12])\s?'?\d{2,4}\b"), 1.0),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), 1.0),
    # Comma-grouped numbers ('1,200')
    (re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"), 1.5),
    # Bare years
    (re.compile(r"\b(?:19|20)\d{2}\b"), 0.4),
    # Any remaining bare number
    (re.compile(r"\b\d+(?:\.\d+)?\b"), 0.7),
]

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        cleaned = text
        for bad, good in _MOJIBAKE:
            cleaned = cleaned.replace(bad, good)
        for rx in _STRIP_RES:
            cleaned = rx.sub(" ", cleaned)

        n_words = len(_WORD_RE.findall(cleaned))
        if n_words == 0:
            return 0.0
        effective_words = max(n_words, 40)

        total_weight = 0.0
        working = cleaned
        for rx, weight in _PATTERNS:
            count = len(rx.findall(working))
            if count:
                total_weight += weight * count
                working = rx.sub(lambda m: " " * len(m.group(0)), working)

        density = 100.0 * total_weight / float(effective_words)
        value = 1.0 - math.exp(-density / 4.5)
        return max(0.0, min(1.0, value))
    except Exception:
        return 0.5
