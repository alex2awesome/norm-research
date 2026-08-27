"""p902_v1_structure -- Temporal anchoring, structural/positional approach.

Criterion: the release anchors its claims to specific calendar dates.

Approach (structural): a genuine press release places dates in
characteristic POSITIONS.  We segment the document into
  (1) a nav-chrome header (runs of short, punctuation-free menu lines),
  (2) the body,
  (3) a trailing contact/boilerplate block (media contact, "About X",
      forward-looking statements, copyright footer),
and then score:
  * DATELINE  -- a date in the opening region of the body (the classic
    "CITY, March 3, 2024 /PRNewswire/ --" lead), with a bonus when a
    wire-service marker sits nearby;
  * BODY DATES -- distinct dates inside the body proper (dates that only
    live in the footer/boilerplate earn almost nothing);
  * SPREAD -- dates appearing in more than one tercile of the body,
    i.e. the whole document is anchored, not just one sentence.
"""

import re
import math

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
          r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
          r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_YEAR = r"(?:19|20)\d{2}"
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"

# any explicit-date surface form; each match yields (position, dedup-key)
_DATE_RES = [
    re.compile(r"\b" + _MONTH + r"\.?\s+" + _DAY +
               r"(?:st|nd|rd|th)?\s*,?\s+" + _YEAR + r"\b", re.IGNORECASE),
    re.compile(r"\b" + _DAY + r"(?:st|nd|rd|th)?\s+" + _MONTH +
               r"\.?,?\s+" + _YEAR + r"\b", re.IGNORECASE),
    re.compile(r"\b" + _MONTH + r"\.?,?\s+" + _YEAR + r"\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]" + _YEAR + r"\b"),
    re.compile(r"\b" + _YEAR + r"-(?:0?[1-9]|1[0-2])-" + _DAY + r"\b"),
    re.compile(r"\bQ[1-4][\s\-]*(?:of\s+)?(?:FY\s*)?" + _YEAR + r"\b",
               re.IGNORECASE),
    re.compile(r"\b(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?"
               r"(?:fiscal\s+(?:year\s+)?)?" + _YEAR + r"\b", re.IGNORECASE),
    re.compile(r"\b(?:fiscal\s+(?:year\s+)?|FY\s*)(?:19|20)?\d{2}\b",
               re.IGNORECASE),
]

_WIRE = re.compile(
    r"(?:/?PR\s*Newswire/?|BUSINESS\s+WIRE|GLOBE\s+NEWSWIRE|ACCESSWIRE|"
    r"Marketwired|PRWeb|Newsfile)", re.IGNORECASE)

_TAIL_MARKERS = [
    "media contact", "press contact", "investor relations", "contact:",
    "for more information", "forward-looking statements", "safe harbor",
    "about ", "all rights reserved", "copyright", "cookie", "privacy policy",
    "terms of use", "©",
]


def _demojibake(t):
    for bad, good in (("â€œ", '"'), ("â€", '"'),
                      ("â€™", "'"), ("â€˜", "'"),
                      ("â€“", "-"), ("â€”", "-"),
                      ("â€¦", "..."), ("Â ", " "),
                      (" ", " "), ("Â", " "),
                      ("&amp;", "&"), ("&nbsp;", " ")):
        t = t.replace(bad, good)
    return t


def _strip_header(t):
    """Skip leading nav-chrome: short menu-like lines with no sentences."""
    lines = t.split("\n")
    start = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if len(s) >= 120 or (". " in s and len(s) >= 60):
            start = i
            break
    else:
        start = 0
    return "\n".join(lines[start:])


def _split_tail(t):
    """Return (body, tail): cut at the earliest boilerplate marker found
    in the last 35% of the text."""
    n = len(t)
    if n < 400:
        return t, ""
    low = t.lower()
    floor = int(n * 0.65)
    cut = n
    for mk in _TAIL_MARKERS:
        i = low.find(mk, floor)
        if i != -1 and i < cut:
            cut = i
    return t[:cut], t[cut:]


def _date_hits(t):
    """List of (char_pos, matched_text_lower) for all date matches,
    de-duplicated on normalised surface form."""
    seen = set()
    hits = []
    for rx in _DATE_RES:
        for m in rx.finditer(t):
            key = re.sub(r"[\s,./\-]+", " ", m.group(0).lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            hits.append((m.start(), key))
    return hits


def score(text: str) -> float:
    try:
        t = _demojibake(str(text))
        t = _strip_header(t)
        body, _tail = _split_tail(t)
        if not body.strip():
            return 0.0

        hits = _date_hits(body)
        n = len(hits)

        # -- dateline: a date in the opening region of the body
        open_len = max(600, int(len(body) * 0.12))
        opening = body[:open_len]
        has_open_date = any(p < open_len for p, _ in hits)
        dateline = 0.0
        if has_open_date:
            dateline = 0.30
            if _WIRE.search(opening):
                dateline = 0.40
        elif _WIRE.search(opening):
            dateline = 0.10  # wire marker but no date up front

        # -- body dates: distinct dates in the body, saturating
        body_dates = 0.45 * (1.0 - math.exp(-n / 3.0))

        # -- spread: dates present in >= 2 of the 3 body terciles
        third = max(1, len(body) // 3)
        terciles = set(min(2, p // third) for p, _ in hits)
        spread = {0: 0.0, 1: 0.05, 2: 0.15, 3: 0.25}[len(terciles)]

        val = dateline + body_dates + spread
        if n == 0:
            val = min(val, 0.05)
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
