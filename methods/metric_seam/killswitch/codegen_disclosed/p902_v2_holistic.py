"""p902_v2_holistic -- Temporal anchoring, composite of weak signals.

Criterion: the release anchors its claims to specific calendar dates
(event dates, availability dates, deadlines, fiscal periods).

Approach: blend seven weak signals, each in [0, 1], into a weighted sum:
  1. COUNT      (0.30) distinct explicit dates, saturating curve
  2. DIVERSITY  (0.10) how many distinct date-format families appear
  3. FISCAL     (0.10) fiscal-period anchoring present (Q3 2024, FY24 ...)
  4. CUES       (0.15) commitment vocabulary near a date ("deadline",
                 "available", "effective", "beginning", "expires",
                 "expected to close", ...) -- dates tied to claims,
                 not just decoration
  5. DATELINE   (0.15) a date in the opening region (classic press-release
                 "CITY, March 3, 2024 /Wire/ --" lead)
  6. DENSITY    (0.10) date mentions per 1000 words, capped
  7. SPREAD     (0.10) dates occur across multiple thirds of the document
A document with zero explicit dates is clamped to (near) zero regardless
of the other components.
"""

import re
import math

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
          r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
          r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_YEAR = r"(?:19|20)\d{2}"
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"

# (family, regex) -- family used for the diversity signal
_FAMILIES = [
    ("month_day_year",
     re.compile(r"\b" + _MONTH + r"\.?\s+" + _DAY +
                r"(?:st|nd|rd|th)?\s*,?\s+" + _YEAR + r"\b", re.IGNORECASE)),
    ("month_day_year",
     re.compile(r"\b" + _DAY + r"(?:st|nd|rd|th)?\s+" + _MONTH +
                r"\.?,?\s+" + _YEAR + r"\b", re.IGNORECASE)),
    ("month_year",
     re.compile(r"\b" + _MONTH + r"\.?,?\s+" + _YEAR + r"\b", re.IGNORECASE)),
    ("numeric",
     re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]" + _YEAR + r"\b")),
    ("numeric",
     re.compile(r"\b" + _YEAR + r"-(?:0?[1-9]|1[0-2])-" + _DAY + r"\b")),
    ("fiscal",
     re.compile(r"\bQ[1-4][\s\-]*(?:of\s+)?(?:FY\s*)?" + _YEAR + r"\b",
                re.IGNORECASE)),
    ("fiscal",
     re.compile(r"\b(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?"
                r"(?:fiscal\s+(?:year\s+)?)?" + _YEAR + r"\b", re.IGNORECASE)),
    ("fiscal",
     re.compile(r"\b(?:fiscal\s+(?:year\s+)?|FY\s*)(?:19|20)?\d{2}\b",
                re.IGNORECASE)),
    ("half_year",
     re.compile(r"\b(?:H[12]\s+|(?:first|second)\s+half\s+of\s+)" + _YEAR +
                r"\b", re.IGNORECASE)),
]

_CUES = re.compile(
    r"\b(?:deadline|due|expir\w*|effective|beginning|begins?|starting|"
    r"starts?|launch\w*|available|availability|commenc\w*|scheduled|"
    r"expected|anticipated|closes?|closing|until|through|ends?|ending|"
    r"held|hosted?|open\w*|ship\w*|releas\w*|announce\w*|report\w*)\b",
    re.IGNORECASE)

_WIRE = re.compile(
    r"(?:/?PR\s*Newswire/?|BUSINESS\s+WIRE|GLOBE\s+NEWSWIRE|ACCESSWIRE|"
    r"Marketwired|PRWeb|Newsfile)", re.IGNORECASE)

_COPYRIGHT = re.compile(r"(?:©|\(c\)|copyright|all rights reserved)",
                        re.IGNORECASE)


def _demojibake(t):
    for bad, good in (("â€œ", '"'), ("â€", '"'),
                      ("â€™", "'"), ("â€˜", "'"),
                      ("â€“", "-"), ("â€”", "-"),
                      ("â€¦", "..."), ("Â ", " "),
                      (" ", " "), ("Â", " "),
                      ("&amp;", "&"), ("&nbsp;", " ")):
        t = t.replace(bad, good)
    return t


def _collect(t):
    """Return list of (pos, dedup_key, family) with surface-form dedup,
    dropping matches that sit in an obvious copyright context."""
    seen = set()
    out = []
    for fam, rx in _FAMILIES:
        for m in rx.finditer(t):
            ctx = t[max(0, m.start() - 40):m.start()]
            if _COPYRIGHT.search(ctx):
                continue
            key = re.sub(r"[\s,./\-]+", " ", m.group(0).lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            out.append((m.start(), key, fam))
    return out


def score(text: str) -> float:
    try:
        t = _demojibake(str(text))
        n_chars = len(t)
        if n_chars < 40:
            return 0.0
        hits = _collect(t)
        n = len(hits)
        if n == 0:
            return 0.0

        # 1. count of distinct dates, saturating (~4 -> 0.63, ~10 -> 0.92)
        s_count = 1.0 - math.exp(-n / 4.0)

        # 2. format diversity: distinct families out of 5
        fams = set(f for _, _, f in hits)
        s_div = min(1.0, len(fams) / 3.0)

        # 3. fiscal anchoring present
        s_fiscal = 1.0 if ("fiscal" in fams or "half_year" in fams) else 0.0

        # 4. commitment cues within +/-70 chars of a date
        cued = 0
        for p, k, _ in hits:
            lo = max(0, p - 70)
            hi = min(n_chars, p + len(k) + 70)
            if _CUES.search(t[lo:hi]):
                cued += 1
        s_cues = cued / float(n)

        # 5. dateline: a date in the opening 12% (>=600 chars) of the doc
        open_len = max(600, int(n_chars * 0.12))
        s_dateline = 0.0
        if any(p < open_len for p, _, _ in hits):
            s_dateline = 1.0 if _WIRE.search(t[:open_len]) else 0.75

        # 6. density per 1000 words, capped at 5
        n_words = max(1, len(t.split()))
        s_density = min(1.0, (n * 1000.0 / n_words) / 5.0)

        # 7. spread across thirds of the document
        third = max(1, n_chars // 3)
        terciles = set(min(2, p // third) for p, _, _ in hits)
        s_spread = {1: 0.33, 2: 0.66, 3: 1.0}[len(terciles)]

        val = (0.30 * s_count + 0.10 * s_div + 0.10 * s_fiscal +
               0.15 * s_cues + 0.15 * s_dateline + 0.10 * s_density +
               0.10 * s_spread)
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
