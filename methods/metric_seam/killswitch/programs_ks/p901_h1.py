# -*- coding: utf-8 -*-
"""p901_h1 -- hybrid metric channel for 'Quantitative support' (round 1).

Pure-code channel (no LLM fields).  h1 keeps h0's typed-count /
per-word-density / coverage architecture and makes exactly two GENERAL
changes, each motivated by a class-level read of the round-1 train
residual (not by per-document patching):

1. TYPE-WEIGHT COMPRESSION.  The judge behaves largely type-BLIND about
   digits: contact/phone-digit-dense pages keep being judged 1.0 (third
   such doc in the residual, after two in round 0), while scattered
   high-value currency/percent tokens inside long prose are judged
   0.0-0.5.  h0's semantic-value spread (2.6 for currency/percent vs
   0.6 for phone) therefore misranks in BOTH directions.  h1 compresses
   the spread toward uniform digit counting: currency/percent 2.6->1.8,
   magnitude/measurement 2.2->1.7, grouped 2.0->1.6, decimal 1.5->1.3,
   phone 0.6->1.5.  Date/year/time/ordinal/bare weights are left
   untouched -- a well-fit anchor whose numeric content is almost all
   dates is judged mid-scale, so calendar credit is already calibrated.

2. COVERAGE AS MULTIPLIER, NOT FLOOR.  h0's additive blend
   (0.8*density + 0.2*coverage) hands every long document with digits
   sprinkled across most chunks a free ~0.2 floor; the entire
   judge<=0.1 residual class (qualitative releases whose only numbers
   are dates/years plus an isolated figure) sits in the 0.34-0.44 band
   that this floor creates.  Spread across the document should modulate
   the density signal, never substitute for it:
       val = sat_density * (0.75 + 0.25 * cov_sat)
   Low-density documents now stay low no matter how widely their few
   digits are scattered; the ordering among genuinely dense documents
   is preserved.

Everything else (normalization, stripping of URLs/e-mails/hex ids,
pattern set, blank-once matching, chunking, word floor, saturation
constant) is byte-identical to h0.
"""

import re
import math

LLM_FIELDS = {}

# --------------------------------------------------------------------------
# Fallback mojibake repair (used only if ops.normalize is unavailable).
_AC = chr(0x00E2)
_MOJI_PREFIX = _AC + chr(0x20AC)
_MOJIBAKE = [
    (_MOJI_PREFIX + chr(0x0153), '"'),
    (_MOJI_PREFIX + chr(0x009D), '"'),
    (_MOJI_PREFIX + chr(0x2122), "'"),
    (_MOJI_PREFIX + chr(0x02DC), "'"),
    (_MOJI_PREFIX + chr(0x201C), "-"),
    (_MOJI_PREFIX + chr(0x201D), "-"),
    (_MOJI_PREFIX + chr(0x00A6), "..."),
    (_MOJI_PREFIX, '"'),
    (chr(0x00C2), ""),
    (chr(0x00A0), " "),
]

# --------------------------------------------------------------------------
# True non-signal: digits inside URLs / e-mails / hex ids / elision markers
# are markup noise, not figures a reader would perceive.
_STRIP_RES = [
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{8,}\b", re.IGNORECASE),  # hex ids (needs a letter)
    re.compile(r"\[\.\.\.\]"),
]

_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
           r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
           r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")

_UNITS = (r"(?:units?|shares?|employees?|customers?|users?|members?|"
          r"subscribers?|patients?|students?|volunteers?|people|jobs?|"
          r"stores?|locations?|branches?|offices?|projects?|hours?|"
          r"minutes?|seconds?|countries|states|cities|markets?|"
          r"acres?|hectares?|"
          r"square\s+(?:feet|meters?|metres?|miles?)|sq\.?\s?ft\.?|"
          r"miles?|kilometers?|km\b|meters?|metres?|feet|ft\b|inches|"
          r"tons?|tonnes?|pounds?|lbs?|kg\b|mm\b|cm\b|mwh?\b|gwh?\b|kwh?\b|"
          r"barrels?|gallons?|liters?|litres?|basis\s+points?|"
          r"percentage\s+points?)")

_CURRENCY_SYM = "[$" + chr(0x20AC) + chr(0x00A3) + chr(0x00A5) + "]"
_MAGNITUDE = r"(?:million|billion|trillion|thousand|bn|mn|m\b|k\b|crore|lakh)"
_NUM = r"\d[\d,]*(?:\.\d+)?"

# (name, compiled regex, weight) -- matched in order, spans blanked out so
# each numeric span is counted exactly once at its highest-value reading.
# Weights compressed vs h0 (see module docstring, change 1).
_PATTERNS = [
    ("currency", re.compile(
        _CURRENCY_SYM + r"\s?" + _NUM + r"(?:\s?" + _MAGNITUDE + r")?|"
        r"\b" + _NUM + r"\s?" + _MAGNITUDE +
        r"?\s?(?:dollars?|euros?|pounds?|kroner|yen|cents?)\b",
        re.IGNORECASE), 1.8),
    ("percent", re.compile(
        r"\b" + _NUM + r"\s?(?:%|percent|per\s?cent)", re.IGNORECASE), 1.8),
    ("magnitude", re.compile(
        r"\b" + _NUM + r"\s?" + _MAGNITUDE + r"\b", re.IGNORECASE), 1.7),
    ("measurement", re.compile(
        r"\b" + _NUM + r"\s?" + _UNITS, re.IGNORECASE), 1.7),
    ("time", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"), 1.2),
    ("date", re.compile(
        r"\b" + _MONTHS + r"\.?\s+\d{1,2}(?:\s?,\s?\d{4})?\b|"
        r"\b\d{1,2}\s+" + _MONTHS + r"\.?(?:\s+\d{4})?\b|"
        r"\b" + _MONTHS + r"\.?\s+\d{4}\b|"
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
        re.IGNORECASE), 0.7),
    ("range", re.compile(
        r"\b\d[\d,]*\s?(?:-|" + chr(0x2013) + r"|to)\s?\d[\d,]*\b"), 1.0),
    ("phone", re.compile(
        r"\+\d[\d\s().-]{5,14}\d|\b\d{3}[-.]\d{3}[-.]\d{4}\b|"
        r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}"), 1.5),
    ("grouped", re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"), 1.6),
    ("decimal", re.compile(r"\b\d+\.\d+\b"), 1.3),
    ("ordinal", re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE), 0.8),
    ("year", re.compile(r"\b(?:19|20)\d{2}\b"), 0.4),
    ("bare", re.compile(r"\b\d{2,}\b"), 0.8),
    ("digit", re.compile(r"\b\d\b"), 0.3),
]

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_CHUNK_MIN_CHARS = 25
_DIGIT_RE = re.compile(r"\d")

# Squash calibration: density d (weighted count per 100 words)
#   d ~ 0.5 -> ~0.15 ; d ~ 1.5 -> ~0.39 ; d ~ 3 -> ~0.63 ; d >= 8 -> ~0.93+
_D0 = 3.0
_COV_SAT = 0.6
# Coverage modulates density multiplicatively (docstring, change 2):
# full spread scales density by 1.0, zero spread by 0.75.
_COV_BASE = 0.75
_COV_MOD = 0.25


def _fallback_normalize(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    return text


def _chunks(text):
    """Split into sentence/line-ish chunks for the coverage signal."""
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
    return [p for p in parts if len(p.strip()) >= _CHUNK_MIN_CHARS]


def score(text, extracted, ops):
    try:
        if not text or not text.strip():
            return 0.0
        try:
            t = ops.normalize(text)
            if not isinstance(t, str) or not t.strip():
                t = _fallback_normalize(text)
        except Exception:
            t = _fallback_normalize(text)

        for rx in _STRIP_RES:
            t = rx.sub(" ", t)

        n_words = len(_WORD_RE.findall(t))
        if n_words < 20:
            # Too little prose to judge density on; digits-only fragments
            # should not explode.
            n_words = 20

        # Coverage BEFORE blanking numeric spans.
        chunks = _chunks(t)
        if chunks:
            covered = sum(1 for c in chunks if _DIGIT_RE.search(c))
            cov = covered / float(len(chunks))
        else:
            cov = 1.0 if _DIGIT_RE.search(t) else 0.0

        # Typed weighted counts; blank each matched span so a token is
        # counted once at its highest-value reading.
        total_w = 0.0
        for _name, rx, w in _PATTERNS:
            matches = list(rx.finditer(t))
            if not matches:
                continue
            total_w += w * len(matches)
            out = []
            prev = 0
            for m in matches:
                out.append(t[prev:m.start()])
                out.append(" " * (m.end() - m.start()))
                prev = m.end()
            out.append(t[prev:])
            t = "".join(out)

        density = 100.0 * total_w / float(n_words)
        sat_density = 1.0 - math.exp(-density / _D0)
        cov_sat = min(1.0, cov / _COV_SAT)

        val = sat_density * (_COV_BASE + _COV_MOD * cov_sat)
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
