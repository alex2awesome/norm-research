# -*- coding: utf-8 -*-
"""p901_v2_holistic -- composite metric for 'Quantitative support'.

Approach: blends four weak signals after mojibake normalization and
URL/e-mail/phone/copyright stripping:
  (1) strong-figure density per 100 words (saturating), weight 0.30;
  (2) figure-TYPE diversity -- how many of {currency, percent, magnitude,
      comma-grouped count, number+unit, numeric date} appear, weight 0.25;
  (3) fraction of sentences containing a figure, weight 0.25;
  (4) specificity contrast -- strong figures vs vague quantifiers
      ("significant", "numerous", "industry-leading", ...), a damped
      ratio so hype vocabulary alone cannot sink a genuinely quantified
      release, weight 0.20.

Source is pure ASCII: all non-ASCII characters are built with chr() so
the module is immune to encoding mangling.
"""

import re
import math

_AC = chr(0x00E2)            # a-circumflex
_EU = chr(0x20AC)            # euro sign
_MOJI_PREFIX = _AC + _EU

_MOJIBAKE = [
    (_MOJI_PREFIX + chr(0x0153), '"'),   # left curly double quote
    (_MOJI_PREFIX + chr(0x009D), '"'),   # right curly double quote (invisible U+009D)
    (_MOJI_PREFIX + chr(0x2122), "'"),   # apostrophe
    (_MOJI_PREFIX + chr(0x02DC), "'"),   # left single quote
    (_MOJI_PREFIX + chr(0x201C), "-"),   # en dash
    (_MOJI_PREFIX + chr(0x201D), "-"),   # em dash
    (_MOJI_PREFIX + chr(0x00A6), "..."), # ellipsis
    (_AC + chr(0x201A) + chr(0x00AC), _EU),  # euro mojibake -> euro sign
    (_MOJI_PREFIX, '"'),                 # leftover bare form
    (chr(0x00C2), ""),                   # A-circumflex before nbsp
]

_STRIP_RES = [
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    re.compile(r"\S+@\S+"),
    re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b"),
    re.compile(r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}"),
    re.compile("(?:" + chr(0x00A9) + r"|\(c\)|copyright)\s*\d{4}",
               re.IGNORECASE),
    re.compile(r"\b\d+\s+[A-Z][A-Za-z]+\s+(?:Street|St\.?|Avenue|Ave\.?|"
               r"Way|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|"
               r"Parkway|Pkwy\.?|Plaza|Square)\b"),
    re.compile(r"\b(?:Suite|Ste\.?|Floor|Fl\.?|Room|Rm\.?|Unit)\s+\d+\b",
               re.IGNORECASE),
    re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"),
    re.compile(r"\[\.\.\.\]"),
]

_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
           r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
           r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")

_UNITS = (r"(?:units?|employees?|customers?|users?|members?|subscribers?|"
          r"patients?|stores?|locations?|offices?|countries|states|cities|"
          r"acres?|square\s+(?:feet|meters?)|miles?|kilometers?|km|meters?|"
          r"tons?|tonnes?|pounds?|kg|liters?|gallons?|barrels?|megawatts?|"
          r"gigawatts?|MW|GW|kWh|MWh|jobs?|shares?|people|homes?|"
          r"households?|vehicles?|devices?|downloads?|transactions?|"
          r"hours?|days?|weeks?|months?|years?|quarters?)")

_CURRENCY_CLASS = "[$" + chr(0x20AC) + chr(0x00A3) + chr(0x00A5) + "]"

# (type name, compiled regex) -- order fixed for determinism.
_TYPED_RES = [
    ("currency", re.compile("(?:" + _CURRENCY_CLASS +
                            r"|\bUS\$|\bUSD\s|\bEUR\s|\bGBP\s)"
                            r"\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)),
    ("percent", re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|"
                           r"percentage\s+points?\b|basis\s+points\b|bps\b)",
                           re.IGNORECASE)),
    ("magnitude", re.compile(r"\b\d[\d,]*(?:\.\d+)?\s(?:million|billion|"
                             r"trillion|thousand)\b", re.IGNORECASE)),
    ("comma_count", re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")),
    ("unit", re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?" + _UNITS + r"\b",
                        re.IGNORECASE)),
    ("date", re.compile(r"\b" + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?"
                        r"(?:,?\s+\d{4})?\b"
                        r"|\b(?:Q[1-4]|FY|H[12])\s?'?\d{2,4}\b",
                        re.IGNORECASE)),
]

_VAGUE_RE = re.compile(
    r"\b(?:many|numerous|several|countless|significant(?:ly)?|"
    r"substantial(?:ly)?|considerabl[ey]|huge|vast|tremendous|massive|"
    r"major|extensive|robust|dramatic(?:ally)?|rapid(?:ly)?|remarkable|"
    r"outstanding|unparalleled|unmatched|world-class|best-in-class|"
    r"industry-leading|market-leading|cutting-edge|state-of-the-art|"
    r"next-generation|innovative|revolutionary|game-changing|"
    r"a\s+(?:lot|great\s+deal)\s+of|wide\s+range\s+of)\b",
    re.IGNORECASE)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ANY_FIG_RE = re.compile(r"\d")


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

        # Signals 1 + 2: strong-figure density and type diversity,
        # counting each span once (typed patterns consume their matches).
        n_strong = 0
        types_present = 0
        working = cleaned
        for _name, rx in _TYPED_RES:
            found = rx.findall(working)
            if found:
                types_present += 1
                n_strong += len(found)
                working = rx.sub(lambda m: " " * len(m.group(0)), working)
        density = 100.0 * n_strong / float(effective_words)
        density_sig = 1.0 - math.exp(-density / 4.0)
        diversity_sig = min(1.0, types_present / 5.0)

        # Signal 3: fraction of real sentences carrying any figure.
        sentences = [s for s in _SENT_SPLIT_RE.split(cleaned)
                     if s and len(_WORD_RE.findall(s)) >= 4]
        if sentences:
            quantified = 0
            for s in sentences:
                if _ANY_FIG_RE.search(s):
                    quantified += 1
            sent_ratio = quantified / float(len(sentences))
        else:
            sent_ratio = 0.0

        # Signal 4: specificity contrast (damped so vague words alone
        # cannot invert the ordering for a well-quantified text).
        n_vague = len(_VAGUE_RE.findall(cleaned))
        if n_strong + n_vague > 0:
            contrast = n_strong / (n_strong + 0.5 * n_vague)
        else:
            contrast = 0.5

        value = (0.30 * density_sig + 0.25 * diversity_sig +
                 0.25 * sent_ratio + 0.20 * contrast)
        return max(0.0, min(1.0, value))
    except Exception:
        return 0.5
