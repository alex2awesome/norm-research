"""p902_v1_structure -- Temporal anchoring, structural/positional approach.

Splits the document into lines and zones: navigation chrome (short,
punctuation-free, digit-free lines or pipe-separated menus) is nearly
ignored; contact/boilerplate zones (detected by explicit markers such as
"Media Contact:", "About X", "###", "SOURCE", or -- only in long docs --
the trailing 15% of lines) are heavily discounted.  Distinct dates found
in substantive body lines carry full weight.  A press-release dateline
(a full date in the first substantive lines, ideally "CITY, ... date -")
earns a bonus, and dates spread across the document earn a small spread
bonus.  Same-date mentions dedupe to one canonical form.
"""

import re
import math

# --- mojibake / unicode normalization -------------------------------------
# Built with chr() so the source stays pure ASCII (U+009D is invisible and
# vanishes if typed as a literal).  3-char sequences first; bare prefix LAST.
_MP = chr(0xE2) + chr(0x20AC)
_MOJIBAKE = [
    (_MP + chr(0x0153), '"'),   # mojibake left double quote
    (_MP + chr(0x009D), '"'),   # mojibake right double quote (invisible tail)
    (_MP + chr(0x2122), "'"),   # mojibake apostrophe / right single quote
    (_MP + chr(0x02DC), "'"),   # mojibake left single quote
    (_MP + chr(0x201C), "-"),   # mojibake en dash
    (_MP + chr(0x201D), "-"),   # mojibake em dash
    (_MP + chr(0x00A6), "..."), # mojibake ellipsis
    (chr(0x00C2) + chr(0x00A0), " "),   # mojibake non-breaking space
    (chr(0x00C2), ""),                  # stray A-circumflex
    (chr(0x00A0), " "),                 # real non-breaking space
    (chr(0x2013), "-"), (chr(0x2014), "-"),
    (chr(0x2018), "'"), (chr(0x2019), "'"),
    (chr(0x201C), '"'), (chr(0x201D), '"'),
    (_MP, '"'),                 # bare mojibake prefix -- keep LAST
]


def _normalize(t):
    for a, b in _MOJIBAKE:
        t = t.replace(a, b)
    return t


_MON = (r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?")
_MON3 = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
         "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_QWORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}

_COPY_RX = re.compile(
    r"(?:" + chr(0x00A9) + r"|\(c\)|copyright)\s*(?:19|20)\d{2}"
    r"(?:\s*-\s*(?:19|20)\d{2})?", re.I)


def _year(ys):
    y = int(ys)
    if y < 100:
        y = 2000 + y if y < 50 else 1900 + y
    return y


def _extract(t):
    """(canon, weight, start, end) mentions; spans masked between patterns."""
    work = _COPY_RX.sub(lambda m: " " * len(m.group(0)), t)
    out = []

    def consume(rx, flags, handler):
        nonlocal work
        for m in list(re.finditer(rx, work, flags)):
            res = handler(m)
            if res is None:
                continue
            canon, w = res
            out.append((canon, w, m.start(), m.end()))
            work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]

    def h_iso(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (("d", y, mo, d), 1.0) if 1 <= mo <= 12 and 1 <= d <= 31 else None

    def h_mdy(m):
        mo = _MON3.get(m.group(1)[:3].lower())
        d, y = int(m.group(2)), _year(m.group(3))
        return (("d", y, mo, d), 1.0) if mo and 1 <= d <= 31 else None

    def h_dmy(m):
        d = int(m.group(1))
        mo = _MON3.get(m.group(2)[:3].lower())
        y = _year(m.group(3))
        return (("d", y, mo, d), 1.0) if mo and 1 <= d <= 31 else None

    def h_num(m):
        a, sep, b, ys = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        if len(ys) == 3 or (len(ys) == 2 and sep != "/"):
            return None
        y = _year(ys)
        if not (1900 <= y <= 2099):
            return None
        if 1 <= a <= 12 and 1 <= b <= 31:
            mo, d = a, b
        elif 1 <= b <= 12 and 1 <= a <= 31:
            mo, d = b, a
        else:
            return None
        return ("d", y, mo, d), 1.0

    def h_q(m):
        return ("q", int(m.group(2)), int(m.group(1))), 0.8

    def h_q_rev(m):
        return ("q", int(m.group(1)), int(m.group(2))), 0.8

    def h_qword(m):
        return ("q", int(m.group(2)), _QWORD[m.group(1).lower()]), 0.8

    def h_fy(m):
        return ("fy", _year(m.group(1))), 0.7

    def h_my(m):
        mo = _MON3.get(m.group(1)[:3].lower())
        return (("m", _year(m.group(2)), mo), 0.6) if mo else None

    def h_yr(m):
        return ("y", int(m.group(1))), 0.25

    consume(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b", 0, h_iso)
    consume(r"\b(" + _MON + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s+"
            r"((?:19|20)\d{2})\b", re.I, h_mdy)
    consume(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MON +
            r")\.?\s*,?\s+((?:19|20)\d{2})\b", re.I, h_dmy)
    consume(r"\b(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})\b", 0, h_num)
    consume(r"\b[Qq]([1-4])\s*(?:of\s+)?(?:FY\s*)?((?:19|20)\d{2})\b", 0, h_q)
    consume(r"\b((?:19|20)\d{2})\s*[Qq]([1-4])\b", 0, h_q_rev)
    consume(r"\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter\s+"
            r"(?:of\s+)?((?:19|20)\d{2})\b", re.I, h_qword)
    consume(r"\b(?:fiscal\s+(?:year\s+)?|FY\s*)'?((?:19|20)\d{2}|\d{2})\b",
            re.I, h_fy)
    consume(r"\b(" + _MON + r")\.?,?\s+((?:19|20)\d{2})\b", 0, h_my)
    consume(r"\b((?:19|20)\d{2})\b", 0, h_yr)
    return out


def _is_nav(s):
    if s.count("|") >= 2 or s.count("\t") >= 2:
        return True
    if re.search(r"\d", s):          # lines with digits may hold dates
        return False
    words = s.split()
    if len(s) < 35 and len(words) <= 5 and not re.search(r"[.!?:]$", s):
        return True
    return False


_BOILER_RX = [
    re.compile(r"^(?:media\s+|press\s+|investor\s+)?contacts?\b[:.]?\s*", re.I),
    re.compile(r"^about\s+\S", re.I),
    re.compile(r"^#\s*#\s*#"),
    re.compile(r"^source[:\s]+\S", re.I),
    re.compile(r"^for\s+(?:more|further)\s+information\b", re.I),
]


def _boiler_start(lines):
    for j, s in enumerate(lines):
        if len(s) < 80:
            for rx in _BOILER_RX:
                if rx.match(s):
                    return j
    return None


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = _normalize(text[:200000])
        lines = [s.strip() for s in t.split("\n") if s.strip()]
        n = len(lines)
        if n == 0:
            return 0.0

        boiler = _boiler_start(lines)
        tail = int(n * 0.85) if n > 10 else n   # positional rule: long docs only

        zone = []
        for j, s in enumerate(lines):
            if _is_nav(s):
                zone.append(0.1)
            elif boiler is not None and j >= boiler:
                zone.append(0.3)
            elif j >= tail:
                zone.append(0.5)
            else:
                zone.append(1.0)

        best = {}                 # canon -> zone-adjusted weight (max)
        line_kinds = []           # per line: set of canon kinds found
        date_lines = []           # substantive lines holding any date
        for j, s in enumerate(lines):
            mentions = _extract(s)
            kinds = set()
            for canon, w, _a, _b in mentions:
                kinds.add(canon[0])
                zw = w * zone[j]
                if zw > best.get(canon, 0.0):
                    best[canon] = zw
            line_kinds.append(kinds)
            if kinds and zone[j] == 1.0:
                date_lines.append(j)

        ysum = sum(w for c, w in best.items() if c[0] == "y")
        wsum = sum(w for c, w in best.items() if c[0] != "y") + min(0.5, ysum)

        # dateline bonus: full date in the first 3 substantive lines
        subst = [j for j in range(n) if zone[j] >= 0.5]
        dateline = 0.0
        for j in subst[:3]:
            if "d" in line_kinds[j]:
                dateline = 0.15
                if re.match(r"[A-Z][A-Za-z0-9 .,'()-]{0,60},\s", lines[j]) \
                        and " - " in lines[j]:
                    dateline = 0.25   # "CITY, ... <date> -" dateline shape
                break

        # spread bonus: body dates in more than one third of the document
        thirds = {min(2, (3 * j) // n) for j in date_lines}
        spread = 0.05 * max(0, len(thirds) - 1)

        s_val = dateline + 0.65 * (1.0 - math.exp(-wsum / 2.6)) + spread
        return max(0.0, min(1.0, s_val))
    except Exception:
        return 0.5
