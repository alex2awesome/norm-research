"""p902_v0_keyword -- Temporal anchoring, surface/lexical approach.

Criterion: the release anchors its claims to specific calendar dates
(event dates, availability dates, deadlines, fiscal periods).  More
distinct explicit dates -> stronger anchoring; no concrete dates -> low.

Approach (pure lexical): scan the raw text with a battery of regexes
covering the common surface forms of explicit dates --
  * "March 3, 2024" / "March 3rd, 2024" / "Mar. 3, 2024"
  * "3 March 2024"
  * "March 2024" (month + year, no day)
  * "03/04/2024", "03-04-2024", "3/4/24"
  * ISO "2024-03-04"
  * fiscal periods: "Q3 2024", "third quarter of 2024", "fiscal 2024",
    "FY2024", "first half of 2024", "H1 2024"
Matches are canonicalised and de-duplicated so repeating the same date
does not inflate the count; the distinct-date count is mapped through a
saturating curve to [0, 1].
"""

import re
import math

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
          r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
          r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_YEAR = r"(?:19|20)\d{2}"
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"

_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Month D, YYYY  (e.g. "March 3, 2024", "Mar. 3rd 2024")
_RE_MDY = re.compile(
    r"\b(" + _MONTH + r")\.?\s+(" + _DAY + r")(?:st|nd|rd|th)?\s*,?\s+(" + _YEAR + r")\b",
    re.IGNORECASE)
# D Month YYYY  (e.g. "3 March 2024")
_RE_DMY = re.compile(
    r"\b(" + _DAY + r")(?:st|nd|rd|th)?\s+(" + _MONTH + r")\.?,?\s+(" + _YEAR + r")\b",
    re.IGNORECASE)
# Month YYYY  (e.g. "March 2024") -- no day
_RE_MY = re.compile(
    r"\b(" + _MONTH + r")\.?,?\s+(" + _YEAR + r")\b", re.IGNORECASE)
# Numeric with 4-digit year: 03/04/2024, 03-04-2024, 3.4.2024
_RE_NUM4 = re.compile(
    r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](" + _YEAR + r")\b")
# Numeric with 2-digit year, slashes only: 3/4/24
_RE_NUM2 = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b")
# ISO: 2024-03-04
_RE_ISO = re.compile(
    r"\b(" + _YEAR + r")-(0?[1-9]|1[0-2])-(" + _DAY + r")\b")
# Fiscal quarter: Q3 2024, Q3 FY2024, Q3-2024
_RE_Q = re.compile(
    r"\bQ([1-4])[\s\-]*(?:of\s+)?(?:FY\s*)?(" + _YEAR + r")\b", re.IGNORECASE)
# Spelled quarter: "third quarter of 2024", "first quarter fiscal 2024"
_RE_QW = re.compile(
    r"\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(?:fiscal\s+(?:year\s+)?)?(" + _YEAR + r")\b",
    re.IGNORECASE)
# Fiscal year: "fiscal 2024", "fiscal year 2024", "FY2024", "FY 24"
_RE_FY = re.compile(
    r"\b(?:fiscal\s+(?:year\s+)?|FY\s*)((?:19|20)?\d{2})\b", re.IGNORECASE)
# Half-year: "H1 2024", "first half of 2024"
_RE_H = re.compile(
    r"\b(?:H([12])\s+|(first|second)\s+half\s+of\s+)(" + _YEAR + r")\b",
    re.IGNORECASE)

_QW_NUM = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _demojibake(t):
    for bad, good in (("â€œ", '"'), ("â€", '"'),
                      ("â€™", "'"), ("â€˜", "'"),
                      ("â€“", "-"), ("â€”", "-"),
                      ("â€¦", "..."), ("Â ", " "),
                      (" ", " "), ("Â", " "),
                      ("&amp;", "&"), ("&nbsp;", " ")):
        t = t.replace(bad, good)
    return t


def _mnum(name):
    return _MONTH_NUM.get(name[:3].lower(), 0)


def _distinct_dates(t):
    keys = set()
    for m in _RE_MDY.finditer(t):
        keys.add(("full", int(m.group(3)), _mnum(m.group(1)), int(m.group(2))))
    for m in _RE_DMY.finditer(t):
        keys.add(("full", int(m.group(3)), _mnum(m.group(2)), int(m.group(1))))
    for m in _RE_ISO.finditer(t):
        keys.add(("full", int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _RE_NUM4.finditer(t):
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 31 and 1 <= b <= 31 and (a <= 12 or b <= 12):
            keys.add(("num", int(m.group(3)), min(a, b), max(a, b)))
    for m in _RE_NUM2.finditer(t):
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 31 and 1 <= b <= 31 and (a <= 12 or b <= 12):
            keys.add(("num", 2000 + int(m.group(3)), min(a, b), max(a, b)))
    for m in _RE_MY.finditer(t):
        k = ("my", int(m.group(2)), _mnum(m.group(1)))
        # only add if no full date already covers this month/year
        if not any(x[0] in ("full",) and x[1] == k[1] and x[2] == k[2]
                   for x in keys):
            keys.add(k)
    for m in _RE_Q.finditer(t):
        keys.add(("q", int(m.group(2)), int(m.group(1))))
    for m in _RE_QW.finditer(t):
        keys.add(("q", int(m.group(2)), _QW_NUM[m.group(1).lower()]))
    for m in _RE_FY.finditer(t):
        y = m.group(1)
        keys.add(("fy", int(y) if len(y) == 4 else 2000 + int(y)))
    for m in _RE_H.finditer(t):
        h = m.group(1) or ("1" if (m.group(2) or "").lower() == "first" else "2")
        keys.add(("h", int(m.group(3)), int(h)))
    return keys


def score(text: str) -> float:
    try:
        t = _demojibake(str(text))
        keys = _distinct_dates(t)
        # full dates count 1.0, month/year and fiscal periods 0.6
        weight = 0.0
        for k in keys:
            weight += 1.0 if k[0] in ("full", "num") else 0.6
        # saturating map: 0 -> 0.0, ~3 -> 0.5, ~10 -> 0.85
        val = 1.0 - math.exp(-weight / 4.0)
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
