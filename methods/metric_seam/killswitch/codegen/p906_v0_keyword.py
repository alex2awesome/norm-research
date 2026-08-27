"""p906 v0 -- Persuasive cadence: surface/lexical heuristic.

Criterion: prose rhythm builds momentum across paragraphs; sentence-length
variation, paragraph pacing, and transitions sustain the reader's attention.

This variant uses ONLY surface lexical evidence, normalized per 100 words:
  * density and variety of transition/connective markers,
  * rhythm punctuation (commas, semicolons, colons, dashes),
  * sentence-terminal punctuation rate (a band: too few terminals = nav
    chrome / no prose; too many = choppy monotone fragments).

score(text) -> float in [0.0, 1.0]; deterministic; 0.5 on unexpected error.
"""

import re
import math

# Mojibake repairs (UTF-8 bytes mis-decoded as cp1252). Order matters:
# three-char sequences first, then the two-char right-double-quote remnant.
# Written as \u escapes because several code points are invisible.
_MOJIBAKE = (
    ("â€œ", '"'),    # left curly double quote
    ("â€", '"'),    # right curly double quote (invisible)
    ("â€™", "'"),    # right single quote / apostrophe
    ("â€˜", "'"),    # left single quote
    ("â€“", " - "),  # en dash
    ("â€”", " - "),  # em dash
    ("â€¦", " . "),  # ellipsis
    ("â€", '"'),          # right dbl quote whose 0x9D was stripped
    ("Â ", " "),          # A-circumflex + non-breaking space
    ("Â", ""),                 # stray A-circumflex
    (" ", " "),                # remaining non-breaking space
    ("", ""),                 # stray invisible control char
)

_TRANSITION_TERMS = (
    r"however", r"moreover", r"furthermore", r"additionally",
    r"consequently", r"therefore", r"meanwhile", r"ultimately",
    r"finally", r"notably", r"importantly", r"crucially", r"indeed",
    r"similarly", r"likewise", r"nonetheless", r"nevertheless",
    r"in addition", r"as a result", r"at the same time", r"in turn",
    r"what's more", r"for example", r"for instance", r"in fact",
    r"on top of that", r"building on", r"looking ahead", r"going forward",
    r"as part of", r"to that end", r"this means", r"that is why",
    r"most importantly", r"since then", r"first", r"second", r"third",
    r"next", r"then",
)

_TRANSITION_RE = re.compile(
    r"\b(?:" + "|".join(_TRANSITION_TERMS) + r")\b", re.IGNORECASE
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_RHYTHM_PUNCT_RE = re.compile(r"[,;:]|\s-\s|--")
_TERMINAL_RE = re.compile(r"[.!?]")
_ELISION_RE = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]")
_MULTIDOT_RE = re.compile(r"\.{2,}")


def _normalize(t):
    for bad, good in _MOJIBAKE:
        t = t.replace(bad, good)
    t = _ELISION_RE.sub(" ", t)
    t = _MULTIDOT_RE.sub(".", t)
    return t


def _terminal_band(e_per100):
    # Healthy narrative prose lands around 4-9 sentence terminals per 100
    # words. Near zero => not prose (menus/chrome). Very high => choppy
    # monotone fragments, the opposite of built momentum.
    if e_per100 <= 0.0:
        return 0.0
    if e_per100 < 4.0:
        return e_per100 / 4.0
    if e_per100 <= 9.0:
        return 1.0
    if e_per100 >= 18.0:
        return 0.1
    return 1.0 - 0.9 * (e_per100 - 9.0) / 9.0


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _normalize(text)
        words = _WORD_RE.findall(t)
        n_words = len(words)
        if n_words == 0:
            return 0.0
        per100 = 100.0 / float(n_words)

        matches = _TRANSITION_RE.findall(t)
        trans_per100 = len(matches) * per100
        s_trans = 1.0 - math.exp(-trans_per100 / 1.8)
        if trans_per100 > 9.0:  # keyword-stuffed junk, taper back
            s_trans *= max(0.4, 1.0 - (trans_per100 - 9.0) / 12.0)
        distinct = len(set(m.lower() for m in matches))
        s_variety = min(distinct, 6) / 6.0

        rhythm_per100 = len(_RHYTHM_PUNCT_RE.findall(t)) * per100
        s_punct = 1.0 - math.exp(-rhythm_per100 / 4.0)

        s_end = _terminal_band(len(_TERMINAL_RE.findall(t)) * per100)

        gate = min(1.0, n_words / 70.0)
        raw = (0.40 * s_trans + 0.15 * s_variety
               + 0.20 * s_punct + 0.25 * s_end)
        return max(0.0, min(1.0, gate * raw))
    except Exception:
        return 0.5
