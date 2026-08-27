"""p902_v0_keyword -- Temporal anchoring, surface/lexical heuristic.

Counts DISTINCT explicit calendar commitments via regex over the raw text:
full dates in many surface formats (March 3, 2024 / 3 March 2024 /
03/04/2024 / 2024-03-04), quarters (Q3 2024, "third quarter of 2024"),
fiscal years (fiscal year 2025, FY25), month-year, and bare years at a
small capped weight.  Mojibake is normalized first; copyright years are
masked out; each consumed span is blanked before the next (less specific)
pattern runs, so one date string can never match twice; repeated mentions
of the same date deduplicate to one canonical form.  The deduplicated
weight sum is mapped to [0, 1] with a saturating exponential.
"""

import re
import math

# --- mojibake / unicode normalization -------------------------------------
# Sequences are BUILT WITH chr() so this source stays pure ASCII: several
# of these code points (e.g. U+009D) are invisible and silently vanish
# when typed as literals.  _MP is the 2-char mojibake prefix (a-circumflex
# U+00E2 + euro U+20AC); order matters -- 3-char sequences first, the bare
# prefix fallback (for right-dquote whose U+009D was stripped) LAST.
_MP = chr(0xE2) + chr(0x20AC)
_MOJIBAKE = [
    (_MP + chr(0x0153), '"'),   # mojibake left double quote  (a-circ euro oe)
    (_MP + chr(0x009D), '"'),   # mojibake right double quote (invisible tail)
    (_MP + chr(0x2122), "'"),   # mojibake apostrophe / right single quote
    (_MP + chr(0x02DC), "'"),   # mojibake left single quote
    (_MP + chr(0x201C), "-"),   # mojibake en dash
    (_MP + chr(0x201D), "-"),   # mojibake em dash
    (_MP + chr(0x00A6), "..."), # mojibake ellipsis
    (chr(0x00C2) + chr(0x00A0), " "),   # mojibake non-breaking space
    (chr(0x00C2), ""),                  # stray A-circumflex
    (chr(0x00A0), " "),                 # real non-breaking space
    (chr(0x2013), "-"), (chr(0x2014), "-"),   # real en/em dash
    (chr(0x2018), "'"), (chr(0x2019), "'"),   # real curly squotes
    (chr(0x201C), '"'), (chr(0x201D), '"'),   # real curly dquotes
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

# copyright-year masker (chr(0xA9) is the copyright sign; ASCII source)
_COPY_RX = re.compile(
    r"(?:" + chr(0x00A9) + r"|\(c\)|copyright)\s*(?:19|20)\d{2}"
    r"(?:\s*-\s*(?:19|20)\d{2})?", re.I)


def _year(ys):
    y = int(ys)
    if y < 100:
        y = 2000 + y if y < 50 else 1900 + y
    return y


def _extract(t):
    """Return list of (canon_tuple, weight, start, end) for date mentions.

    Patterns run most-specific first; every accepted span is blanked
    (same length) before the next pattern, preventing double counting.
    """
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
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return ("d", y, mo, d), 1.0
        return None

    def h_mdy(m):
        mo = _MON3.get(m.group(1)[:3].lower())
        d, y = int(m.group(2)), _year(m.group(3))
        if mo and 1 <= d <= 31:
            return ("d", y, mo, d), 1.0
        return None

    def h_dmy(m):
        d = int(m.group(1))
        mo = _MON3.get(m.group(2)[:3].lower())
        y = _year(m.group(3))
        if mo and 1 <= d <= 31:
            return ("d", y, mo, d), 1.0
        return None

    def h_num(m):
        a, sep, b, ys = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        if len(ys) == 3:
            return None
        if len(ys) == 2 and sep != "/":       # avoid version strings 1.2.24
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
        if mo:
            return ("m", _year(m.group(2)), mo), 0.6
        return None

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
    consume(r"\b(" + _MON + r")\.?,?\s+((?:19|20)\d{2})\b", 0, h_my)  # cap. only
    consume(r"\b((?:19|20)\d{2})\b", 0, h_yr)
    return out


def _weight_sum(mentions):
    best = {}
    for canon, w, _s, _e in mentions:
        if w > best.get(canon, 0.0):
            best[canon] = w
    ysum = sum(w for c, w in best.items() if c[0] == "y")
    return sum(w for c, w in best.items() if c[0] != "y") + min(0.5, ysum)


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = _normalize(text[:200000])
        w = _weight_sum(_extract(t))
        s = 1.0 - math.exp(-w / 2.6)
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
