"""p902_v2_holistic -- Temporal anchoring, composite of weak signals.

Blends six weak signals into one score:
  1. distinct-date mass (deduped, saturating)                 weight 0.45
  2. format diversity (full date / quarter / FY / month-year)        0.10
  3. commitment context: dates near verbs like "will",
     "available", "deadline", "expires", "begins"                    0.15
  4. early anchoring: an explicit date in the first ~250 chars       0.10
  5. date density per 100 words (capped)                             0.10
  6. absence of vague-only time language ("soon",
     "in the coming months", "at a later date", ...)                 0.10
Mojibake normalized, copyright years masked, spans consumed between
patterns so no date string is counted twice, same dates deduped.
Documents with zero explicit dates are floored near 0 (lower still if
vague time language is all they offer).
"""

import re
import math
from collections import Counter

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


_COMMIT_RX = re.compile(
    r"\b(?:will|launch(?:es|ed|ing)?|releas(?:e|es|ed|ing)|"
    r"availab(?:le|ility)|begin(?:s|ning)?|start(?:s|ing)?|effective|"
    r"deadline|due|expir(?:e|es|ed|ing|ation)|open(?:s|ed|ing)?|"
    r"clos(?:e|es|ed|ing)|schedul(?:e|ed|ing)|host(?:s|ed|ing)?|"
    r"report(?:s|ed|ing)?|complet(?:e|ed|ion)|deliver(?:s|y|ed|ing)?|"
    r"announc(?:e|es|ed|ing)|expect(?:s|ed)?|held|until|through|"
    r"take\s+place|kicks?\s+off|commenc(?:e|es|ing))\b", re.I)

_VAGUE_RX = [
    re.compile(r"\bsoon\b", re.I),
    re.compile(r"\bshortly\b", re.I),
    re.compile(r"\bin\s+the\s+(?:coming|next\s+few|near)\s+"
               r"(?:days|weeks|months|years|future)\b", re.I),
    re.compile(r"\bat\s+a\s+later\s+date\b", re.I),
    re.compile(r"\bin\s+due\s+course\b", re.I),
    re.compile(r"\beventually\b", re.I),
    re.compile(r"\bsometime\b", re.I),
    re.compile(r"\bin\s+the\s+future\b", re.I),
    re.compile(r"\bdown\s+the\s+road\b", re.I),
]


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.0
        t = _normalize(text[:200000])
        mentions = _extract(t)

        best = {}
        for canon, w, _a, _b in mentions:
            if w > best.get(canon, 0.0):
                best[canon] = w
        ysum = sum(w for c, w in best.items() if c[0] == "y")
        wsum = sum(w for c, w in best.items() if c[0] != "y") + min(0.5, ysum)

        vague = sum(len(rx.findall(t)) for rx in _VAGUE_RX)

        if wsum <= 0.0:   # no explicit dates at all -> floor near zero
            return max(0.0, min(1.0, 0.08 - 0.02 * min(4, vague)))

        # 1. distinct-date mass
        s_count = 1.0 - math.exp(-wsum / 2.6)

        # 2. format diversity
        kind_counts = Counter(c[0] for c in best if c[0] != "y")
        s_div = min(1.0, len(kind_counts) / 3.0)

        # 3. commitment context around date mentions
        strong = [(a, b) for c, _w, a, b in mentions if c[0] != "y"]
        committed = sum(
            1 for a, b in strong
            if _COMMIT_RX.search(t[max(0, a - 70):min(len(t), b + 70)]))
        s_ctx = committed / len(strong) if strong else 0.0

        # 4. early anchoring (dateline territory)
        s_top = 1.0 if any(a < 250 for c, _w, a, _b in mentions
                           if c[0] != "y") else 0.0

        # 5. density per 100 words
        nwords = len(re.findall(r"\w+", t))
        n_distinct = sum(1 for c in best if c[0] != "y")
        s_dens = min(1.0, (n_distinct / max(1.0, nwords / 100.0)) / 2.0)

        # 6. vague-language (inverse) signal
        s_vague = 1.0 - min(1.0, vague / (vague + n_distinct + 0.5))

        s_val = (0.45 * s_count + 0.10 * s_div + 0.15 * s_ctx +
                 0.10 * s_top + 0.10 * s_dens + 0.10 * s_vague)
        return max(0.0, min(1.0, s_val))
    except Exception:
        return 0.5
