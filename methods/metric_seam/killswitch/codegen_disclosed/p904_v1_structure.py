"""p904 v1 -- Voice diversity: structural/positional heuristic.

Criterion: direct quotations from multiple distinct named people, each
clearly attributed. No quotes / single spokesperson -> low; 3+ distinct
quoted voices -> high.

Approach (document geometry rather than raw keyword counts):
  1. Repair mojibake, normalize quote glyphs, split into lines.
  2. Trim leading nav chrome (short link-y lines with no sentence
     punctuation) and trailing boilerplate (About/Contact/SOURCE/
     forward-looking blocks near the end) to isolate the release BODY.
  3. Detect QUOTE PARAGRAPHS: body lines that open with a double quote
     (the standard press-release quote-paragraph convention) or that
     carry a close-quote+comma attribution seam ('...," said Jane Doe').
  4. Read the attribution SLOT at each seam (the tokens right after
     '," said' / right before ', said') and dedupe the named people.
  5. Score from distinct attributed people, with small structural
     bonuses for quote paragraphs spread across different thirds of the
     body (multiple voices usually appear in separate sections).
"""

import re

# --- normalization ------------------------------------------------------
_MOJIBAKE = [
    ("â€œ", '"'), ("â€\x9d", '"'), ("â€™", "'"), ("â€˜", "'"),
    ("â€”", "-"), ("â€“", "-"), ("â€¦", "..."),
    ("â€", '"'),          # bare remainder; must follow longer sequences
    ("Â\xa0", " "), ("Â", ""),
]
_ENTITIES = [
    ("&quot;", '"'), ("&amp;", "&"), ("&#39;", "'"), ("&apos;", "'"),
    ("&nbsp;", " "), ("&ldquo;", '"'), ("&rdquo;", '"'),
    ("&lsquo;", "'"), ("&rsquo;", "'"), ("&gt;", ">"), ("&lt;", "<"),
]
_CURLY = [
    ("“", '"'), ("”", '"'), ("„", '"'), ("«", '"'), ("»", '"'),
    ("‘", "'"), ("’", "'"), ("\xa0", " "),
]


def _normalize(text):
    for table in (_MOJIBAKE, _ENTITIES, _CURLY):
        for a, b in table:
            text = text.replace(a, b)
    return text


# --- body isolation -----------------------------------------------------
_SENT_PUNCT = re.compile(r"[.!?]")

_TAIL_MARKERS = re.compile(
    r"^\s*(About\s+[A-Z]|Media\s+Contact|Press\s+Contact|Investor\s+"
    r"(Relations|Contact)|Contacts?\s*:|For\s+(more|further)\s+information|"
    r"Forward[- ]Looking\s+Statements?|Safe\s+Harbor|###|SOURCE\s+|View\s+"
    r"original\s+content|Related\s+Links|Cision|Logo\s*-|Photo\s*-|"
    r"CONTACT\s*:|Follow\s+us\s+on)",
    re.IGNORECASE,
)


def _paragraphs(text):
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) < 3 and text:
        # single-blob document: cut into ~400-char pseudo-paragraphs at
        # sentence ends so positional logic still has units to work with
        blob = " ".join(lines) if lines else text.strip()
        out, buf = [], []
        size = 0
        for sent in re.split(r"(?<=[.!?])\s+", blob):
            buf.append(sent)
            size += len(sent)
            if size > 400:
                out.append(" ".join(buf))
                buf, size = [], 0
        if buf:
            out.append(" ".join(buf))
        return out
    return lines


def _body_range(paras):
    """Indices [start, end) of the prose body inside the paragraph list."""
    n = len(paras)
    start = 0
    for i, p in enumerate(paras):
        words = p.split()
        # first 'proseful' paragraph ends the nav chrome
        if (len(words) >= 8 and _SENT_PUNCT.search(p)) or p.count('"') >= 2:
            start = i
            break
    else:
        start = 0
    end = n
    # scan the last 45% of paragraphs for boilerplate section heads
    floor = start + max(1, int((n - start) * 0.55))
    for i in range(floor, n):
        if _TAIL_MARKERS.match(paras[i]):
            end = i
            break
    if end <= start:
        start, end = 0, n
    return start, end


# --- quote-paragraph & attribution-seam detection -----------------------
_OPENS_WITH_QUOTE = re.compile(r'^\s*"')

_VERBS = (
    r"(?:said|says|stated|states|added|adds|noted|notes|commented|"
    r"comments|explained|explains|remarked|continued|concluded|told)"
)

_NAME = (
    r"((?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Professor|Mayor|Gov\.|Sen\.)?\s*"
    r"[A-Z][a-zA-Z'\-]+(?:\s+[A-Z]\.)?(?:\s+(?:van|von|de|der|da|del|la))?"
    r"\s+[A-Z][a-zA-Z'\-]+)"
)

# '," said Jane Doe'  /  '." says Dr. Jane Doe'
_SEAM_VERB_NAME = re.compile(r'[,.]"\s*,?\s*' + _VERBS + r"\s+" + _NAME)
# '," Jane Doe said'
_SEAM_NAME_VERB = re.compile(r'[,.]"\s*,?\s*' + _NAME + r"\s*,?\s+" + _VERBS)
# 'Jane Doe said: "' / 'Jane Doe, CEO, said "'
_PRE_NAME_VERB = re.compile(_NAME + r"[^\"\n]{0,60}?" + _VERBS + r'\s*[:,]?\s*"')

_BAD_FIRST = {
    "the", "a", "an", "in", "on", "at", "and", "but", "for", "with",
    "from", "chief", "senior", "executive", "vice", "president", "new",
    "our", "this", "that", "he", "she", "it", "they", "we",
}
_BAD_LAST = {
    "company", "inc", "corp", "corporation", "group", "llc", "ltd",
    "officer", "president", "director", "university", "institute",
    "release", "statement", "report", "study", "survey", "today",
}


def _key(raw):
    raw = re.sub(r"^(Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Professor|Mayor|Gov\.|Sen\.)\s*",
                 "", raw.strip())
    toks = [t for t in re.split(r"\s+", raw) if t and not re.match(r"^[A-Z]\.$", t)
            and t.lower() not in ("van", "von", "de", "der", "da", "del", "la")]
    if len(toks) < 2:
        return None
    first, last = toks[0], toks[-1]
    if first.lower() in _BAD_FIRST or last.lower() in _BAD_LAST:
        return None
    return (first[0].lower(), last.lower())


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _normalize(text)
        paras = _paragraphs(t)
        if not paras:
            return 0.0
        b0, b1 = _body_range(paras)
        body = paras[b0:b1]
        nb = len(body)
        if nb == 0:
            return 0.0

        voices = {}
        quote_para_pos = []

        for i, p in enumerate(body):
            is_qp = bool(_OPENS_WITH_QUOTE.match(p)) and p.count('"') >= 2
            found = []
            for m in _SEAM_VERB_NAME.finditer(p):
                found.append(m.group(1))
            for m in _SEAM_NAME_VERB.finditer(p):
                found.append(m.group(1))
            for m in _PRE_NAME_VERB.finditer(p):
                found.append(m.group(1))
            if is_qp or found:
                quote_para_pos.append(i / max(1, nb - 1) if nb > 1 else 0.0)
            for raw in found:
                k = _key(raw)
                if k and k not in voices:
                    voices[k] = i

        n_voices = len(voices)
        n_qp = len(quote_para_pos)

        if n_qp == 0 and n_voices == 0:
            return 0.0
        if n_voices == 0:
            return 0.15  # quote-shaped paragraphs but no attributable name

        base = {1: 0.4, 2: 0.7}.get(n_voices, 1.0)
        if n_voices >= 3:
            return 1.0

        # structural bonuses: quotes spread across distinct thirds of the
        # body, and multiple separate quote paragraphs, hint at additional
        # voices the seam regexes may have missed. (0.04 each so a real
        # score can never equal the 0.5 error sentinel: 0.4+0.08=0.48)
        thirds = {min(2, int(pos * 3)) for pos in quote_para_pos}
        bonus = 0.0
        if len(thirds) >= 2:
            bonus += 0.04
        if n_qp >= 3:
            bonus += 0.04
        return min(1.0, base + bonus)
    except Exception:
        return 0.5
