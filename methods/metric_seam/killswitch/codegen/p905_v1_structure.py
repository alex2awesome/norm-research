"""p905 v1 -- Authentic authorship, structural/positional approach.

Criterion: the release reads as written by someone with genuine
familiarity with the company (concrete, specific, internally coherent),
not assembled from generic template marketing language.

Approach (WHERE things appear, not just whether):
  - split into paragraph blocks, drop navigation-chrome lines, and cut the
    trailing contact/About/boilerplate tail so it neither helps nor hurts;
  - reward: a specific lede (numbers/dates + named entities in the first
    body paragraph), a dateline header, an attributed quotation, and
    specifics SPREAD across many body paragraphs (coherent narrative);
  - penalize: template-marketing phrases inside the body proper, and a
    body containing no specifics at all.

Contract: score(text) -> float in [0, 1]; higher = more authentic.
Deterministic; imports limited to re/math/statistics/collections.
"""

import re

# ----------------------------------------------------------------------
# Mojibake / unicode normalization (must run before quote detection).
# All non-ASCII is written as \u escapes so the invisible U+009D is exact.
# ----------------------------------------------------------------------
_MOJIBAKE = (
    ("â€œ", '"'),   # misdecoded left double quote
    ("â€", '"'),   # misdecoded right double quote
    ("â€™", "'"),   # misdecoded apostrophe
    ("â€˜", "'"),   # misdecoded left single quote
    ("â€“", "-"),   # misdecoded en dash
    ("â€”", "-"),   # misdecoded em dash
    ("â€¦", "..."), # misdecoded ellipsis
    ("â€¢", "*"),   # misdecoded bullet
    ("Â ", " "),         # A-circ + nbsp -> space
)
_LEFTOVER_MOJI = re.compile("â€.")


def _normalize(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    text = _LEFTOVER_MOJI.sub("'", text)
    text = text.replace("Â", "")
    text = text.replace(" ", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("[...]", "\n\n")
    return text


# ----------------------------------------------------------------------
# Structure detection
# ----------------------------------------------------------------------
_TAIL_MARKER_RE = re.compile(
    r"^\s*(?:about\s+\S|media\s+contacts?|press\s+contacts?|contacts?\s*:"
    r"|for\s+(?:more|further|additional)\s+information|investor\s+relations"
    r"|source\s*:|###|forward-looking\s+statements|safe\s+harbor|cautionary"
    r"|all\s+rights\s+reserved|copyright|©|follow\s+us|visit\s+(?:us|our)"
    r"|media\s+inquiries|tel[:.]|phone\s*:|e-?mail\s*:)",
    re.IGNORECASE)

_CHROME_LINE_RE = re.compile(r"^[\w\s|/>»-]{1,29}$")  # short link-ish line

_QUOTE_ATTRIB_RE = re.compile(
    r"\"[^\"\n]{15,600}\"\s*[,.]?\s*"
    r"(?:said|says|stated|added|noted|commented|explained|remarked"
    r"|according\s+to)\b",
    re.IGNORECASE)
_ATTRIB_NAME_RE = re.compile(
    r"\b(?:said|says|stated|added|noted|commented|explained|remarked)\s+"
    r"[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)?")
_PLAIN_QUOTE_RE = re.compile(r"\"[^\"\n]{20,600}\"")

_DIGIT_RE = re.compile(r"\d")
_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?"
    r"|dec(?:ember)?)\.?\s+\d{1,2}\b|\b(?:19|20)\d{2}\b",
    re.IGNORECASE)
_ALLCAPS_CITY_RE = re.compile(r"\b[A-Z]{3,}(?:\s+[A-Z]{3,})?\b")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z][\w'-]*")
_CAPWORD_RE = re.compile(r"^[A-Z][a-z]+$")
_ACRO_RE = re.compile(r"^[A-Z]{2,5}$")

# Small, high-precision template-phrase bank for the body penalty
_BODY_BUZZ_RE = re.compile(
    r"industry[\s-]+leading|world[\s-]+class|cutting[\s-]+edge"
    r"|state[\s-]+of[\s-]+the[\s-]+art|best[\s-]+in[\s-]+class"
    r"|leading\s+(?:provider|supplier)|innovative\s+solutions?"
    r"|one[\s-]+stop|unparalleled|unmatched|second\s+to\s+none"
    r"|world[\s-]+renowned|proven\s+track\s+record|trusted\s+partner"
    r"|commitment\s+to\s+excellence|exceed(?:s|ing)?\s+expectations"
    r"|thought\s+leader|game[\s-]+changing|seamless|synerg|leverag"
    r"|wide\s+range\s+of|passionate\s+about|striv(?:e|es|ing)\s+to"
    r"|top[\s-]+notch|award[\s-]+winning|next[\s-]+generation",
    re.IGNORECASE)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _paragraphs(t):
    """Paragraph blocks with navigation chrome filtered out."""
    blocks = re.split(r"\n\s*\n+", t)
    if len(blocks) <= 1 and t.count("\n") > 4:
        blocks = t.split("\n")
    paras = []
    for b in blocks:
        lines = [ln.strip() for ln in b.split("\n") if ln.strip()]
        kept = [ln for ln in lines
                if len(ln) >= 30
                or (ln[-1:] in ".!?\"'" and not _CHROME_LINE_RE.match(ln))]
        joined = " ".join(kept).strip()
        if len(joined) >= 40 and len(_WORD_RE.findall(joined)) >= 6:
            paras.append(joined)
    return paras


def _split_body_tail(paras):
    """Cut trailing contact/About boilerplate; only look in the last half."""
    n = len(paras)
    if n == 0:
        return [], []
    start = max(1, n // 2)
    for i in range(start, n):
        if _TAIL_MARKER_RE.match(paras[i]):
            return paras[:i], paras[i:]
    return paras, []


def _mid_sentence_propnouns(text):
    """Capitalized tokens excluding sentence-initial position; capped acronyms."""
    count = 0
    acro = 0
    for sent in _SENT_SPLIT_RE.split(text):
        toks = _TOKEN_RE.findall(sent)
        for tok in toks[1:]:
            if _CAPWORD_RE.match(tok):
                count += 1
            elif _ACRO_RE.match(tok):
                acro += 1
    return count + min(acro, 5)


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.5
        t = _normalize(text)
        paras = _paragraphs(t)
        if not paras:
            return 0.5

        body, _tail = _split_body_tail(paras)
        if not body:
            body = paras
        body_text = "\n".join(body)
        n_words = len(_WORD_RE.findall(body_text))
        if n_words < 8:
            return 0.5

        val = 0.45

        # -- lede specificity: first body paragraph carries who/what/when
        lede = body[0]
        if _DIGIT_RE.search(lede) or _DATE_RE.search(lede):
            val += 0.08
        if _mid_sentence_propnouns(lede) >= 1:
            val += 0.07

        # -- dateline header near the top ("CITY, State - Month D, YYYY -")
        head = lede[:150]
        if _ALLCAPS_CITY_RE.search(head) and ("-" in head or _DATE_RE.search(head)):
            val += 0.06

        # -- attributed quotation anywhere in the body
        if _QUOTE_ATTRIB_RE.search(body_text) or _ATTRIB_NAME_RE.search(body_text):
            val += 0.14
        elif _PLAIN_QUOTE_RE.search(body_text):
            val += 0.05

        # -- specifics spread across body paragraphs (positional coverage)
        with_spec = sum(
            1 for p in body
            if _DIGIT_RE.search(p) or _mid_sentence_propnouns(p) >= 1)
        spread = with_spec / float(len(body))
        val += 0.15 * min(1.0, spread / 0.6)

        # -- template-marketing pressure inside the body proper
        buzz = len(_BODY_BUZZ_RE.findall(body_text))
        buzz_rate = 100.0 * buzz / n_words
        val -= 0.30 * min(1.0, buzz_rate / 4.0)

        # -- a body with no specifics at all cannot be authentic
        if with_spec == 0:
            val -= 0.12

        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.5
