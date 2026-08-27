"""p907 v1 (structure) — structural/positional heuristic for 'Comprehensive detail'.

Criterion: the release provides substantial informative detail — background,
specifics, supporting facts, context. Fuller releases score high; terse stubs
score low.

Approach: structural, not lexical. Segment the document into line-blocks and
classify each as PROSE (real running text) vs CHROME (nav menus, link lists,
headings, contact boilerplate). Score how developed the document's structure
is: how many substantive prose paragraphs it has, how long they run, whether
prose is sustained through the beginning / middle / end of the document (a
comprehensive release keeps developing; a stub stops after one paragraph),
what fraction of the page is prose rather than chrome, and whether the
canonical full-form press-release skeleton is present (dateline near the top,
attributed quote in the body, "About X" boilerplate and a contact block near
the end — completeness of form). Positional rules keep the end-of-document
contact/boilerplate block from masquerading as body development.
"""
import math
import re

# --- cleanup: mojibake, HTML entities, truncation marker -----------------
_E = "â€"                       # mojibake prefix 'â€'
_REPLACEMENTS = [
    (_E + "œ", '"'), (_E + "\x9d", '"'), (_E + "™", "'"),
    (_E + "˜", "'"), (_E + "“", "-"), (_E + "”", "-"),
    (_E + "¦", "..."), (_E, '"'),
    ("Â\xa0", " "), ("Â ", " "), ("Â", ""), ("\xa0", " "),
    ("&amp;", "&"), ("&gt;", ">"), ("&lt;", "<"),
    ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
    ("[...]", "\n"),                       # truncation marker: block break
]


def _clean(text):
    for a, b in _REPLACEMENTS:
        text = text.replace(a, b)
    return text


_WORD = re.compile(r"[A-Za-z']+")
_SENT_END = re.compile(r"[.!?][\"')\]]?(?:\s|$)")

# full-form skeleton detectors
_DATELINE = re.compile(
    r"\b[A-Z][A-Za-z.]+(?:\s[A-Z][A-Za-z.]+)?,\s*"          # City,
    r"(?:[A-Z][a-z]+\.?|[A-Z]{2})\s*[-–—,]"                  # State/Country —
    r"|\([A-Z][A-Za-z ]+\)\s*[-–—]"                          # (BUSINESS WIRE) —
    r"|\b[A-Z]{2,},?\s+[A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4}")   # CITY, Month D, YYYY
_QUOTE_ATTR = re.compile(
    r"\"[^\"]{20,}\"[^\"]{0,80}\b(?:said|says|added|noted|commented|stated)\b"
    r"|\b(?:said|says|added|noted|commented|stated)\b[^\"]{0,80}\"[^\"]{20,}\"",
    re.IGNORECASE | re.DOTALL)
_ABOUT_HDR = re.compile(r"^\s*about\s+\S+", re.IGNORECASE)
_CONTACT = re.compile(
    r"\b(?:media|press|investor)?\s*contacts?\b|\bfor more information\b"
    r"|[\w.+-]+@[\w-]+\.\w{2,}|\(\d{3}\)\s?\d{3}[- ]\d{4}|\d{3}[-.]\d{3}[-.]\d{4}",
    re.IGNORECASE)


def _blocks(t):
    """Split into non-empty line blocks with (start_pos_fraction, text)."""
    out, pos, total = [], 0, max(1, len(t))
    for line in t.split("\n"):
        s = line.strip()
        if s:
            out.append((pos / total, s))
        pos += len(line) + 1
    return out


def _is_prose(s):
    """Real running text: enough words AND sentence-like punctuation."""
    n = len(_WORD.findall(s))
    if n >= 30:
        return True
    return n >= 12 and bool(_SENT_END.search(s))


def _sat(x, k):
    return 1.0 - math.exp(-float(x) / float(k))


def score(text: str) -> float:
    try:
        t = _clean("" if text is None else str(text))
        blocks = _blocks(t)
        if not blocks:
            return 0.0

        prose = [(p, s) for p, s in blocks if _is_prose(s)]
        # body prose = prose outside the final 15% (end zone is where
        # contact/boilerplate lives; it shouldn't count as development)
        body_prose = [(p, s) for p, s in prose if p < 0.85]

        n_body = len(body_prose)
        body_words = [len(_WORD.findall(s)) for _, s in body_prose]
        total_words = sum(len(_WORD.findall(s)) for _, s in blocks)
        prose_words = sum(body_words)

        # 1) volume of substantive paragraphs
        s_nblocks = _sat(n_body, 6)
        # 2) paragraph development: mean words per body paragraph
        mean_len = (sum(body_words) / n_body) if n_body else 0.0
        s_devel = _sat(mean_len, 40)
        # 3) sustained development across doc thirds (position of prose)
        thirds = set()
        for p, _ in body_prose:
            thirds.add(min(2, int(p * 3)))
        s_sustain = len(thirds) / 3.0
        # 4) prose share of the page (vs nav chrome / link lists)
        s_share = prose_words / max(1, total_words)
        # 5) full-form skeleton completeness (positional checklist)
        head = t[: max(1, int(len(t) * 0.30))]
        tail = t[int(len(t) * 0.60):]
        checklist = 0.0
        if _DATELINE.search(head):
            checklist += 0.25                       # dateline near the top
        if any(_QUOTE_ATTR.search(s) for _, s in body_prose):
            checklist += 0.25                       # attributed quote in body
        if any(_ABOUT_HDR.match(s) for p, s in blocks if p > 0.5):
            checklist += 0.25                       # About-X boilerplate late
        if _CONTACT.search(tail):
            checklist += 0.25                       # contact block near end
        s_form = checklist

        s = (0.30 * s_nblocks + 0.20 * s_devel + 0.20 * s_sustain
             + 0.15 * s_share + 0.15 * s_form)
        return float(max(0.0, min(1.0, s)))
    except Exception:
        return 0.5
