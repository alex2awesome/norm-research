"""p904_v1_structure -- Voice diversity (structural/positional approach).

Criterion: the release features direct quotations from multiple distinct
named people, each clearly attributed.

Approach: normalize mojibake -> segment the document into paragraphs ->
trim the trailing contact/About/### boilerplate block -> identify
quote-bearing paragraphs (paired-quote spans inside the paragraph) -> read
the attribution from each quote paragraph's NON-quoted remainder (or the top
of the following paragraph) -> count distinct speakers, with a bonus for
quotes positionally spread across the document body.
"""

import re


def _seq(*cps):
    return "".join(chr(c) for c in cps)


# --------------------------------------------------------------- mojibake
# UTF-8 bytes of curly punctuation re-read as cp1252 (or latin-1). Built via
# chr() codepoints because several members are INVISIBLE control characters
# (e.g. the U+009D tail of a mojibake closing double quote). Order matters:
# 3-char sequences first (incl. mojibake dashes whose THIRD char is itself a
# genuine curly quote), then bare 2-char stubs, then NBSP mojibake, then
# genuine curly punctuation.
_MOJIBAKE = [
    (_seq(0xE2, 0x20AC, 0x0153), '"'),    # left double quote (oe tail)
    (_seq(0xE2, 0x20AC, 0x009D), '"'),    # right double quote (invisible tail)
    (_seq(0xE2, 0x20AC, 0x2122), "'"),    # apostrophe (TM tail)
    (_seq(0xE2, 0x20AC, 0x02DC), "'"),    # left single quote (tilde tail)
    (_seq(0xE2, 0x20AC, 0x201C), "-"),    # en dash (tail is U+201C itself)
    (_seq(0xE2, 0x20AC, 0x201D), "--"),   # em dash (tail is U+201D itself)
    (_seq(0xE2, 0x20AC, 0x00A6), "..."),  # ellipsis
    (_seq(0xE2, 0x80, 0x9C), '"'),        # latin-1 variants (C1 controls)
    (_seq(0xE2, 0x80, 0x9D), '"'),
    (_seq(0xE2, 0x80, 0x99), "'"),
    (_seq(0xE2, 0x80, 0x98), "'"),
    (_seq(0xE2, 0x80, 0x93), "-"),
    (_seq(0xE2, 0x80, 0x94), "--"),
    (_seq(0xE2, 0x80, 0xA6), "..."),
    (_seq(0xE2, 0x20AC), '"'),            # bare stubs (tail stripped)
    (_seq(0xE2, 0x80), '"'),
    (_seq(0xC2, 0xA0), " "),              # NBSP mojibake
    (_seq(0xC2), ""),
    (_seq(0x201C), '"'), (_seq(0x201D), '"'), (_seq(0x201E), '"'),
    (_seq(0x2018), "'"), (_seq(0x2019), "'"),
    (_seq(0x00AB), '"'), (_seq(0x00BB), '"'),
    (_seq(0x2014), "--"), (_seq(0x2013), "-"),
    (_seq(0x00A0), " "),
]


def _normalize(text):
    for bad, good in _MOJIBAKE:
        if bad in text:
            text = text.replace(bad, good)
    return text


# ------------------------------------------------------------ attribution
_VERBS = (r"(?:said|says|say|stated|states|added|adds|noted|notes|commented|"
          r"comments|remarked|remarks|explained|explains|continued|concluded|"
          r"emphasized|emphasised|told|wrote|according\s+to)")

_TITLE = (r"(?:Dr|Mr|Ms|Mrs|Prof|Professor|Sen|Senator|Rep|Gov|Governor|Gen|"
          r"Col|Capt|Sgt|Rev|Mayor|Judge|President|Secretary|Commissioner)\.?")

_NAME = (r"(?:" + _TITLE + r"\s+)?"
         r"[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-\.]+){1,3}")

_P_AFTER = re.compile(r"\b" + _VERBS + r"\s+(?:that\s+)?(" + _NAME + r")")
_P_BEFORE = re.compile(r"(" + _NAME + r")\s*,?[^.!?\n\"]{0,60}?\s" + _VERBS + r"\b")
# In attribution-only fragments (quote content already blanked) a lone
# capitalized token right before the verb is a surname: '..., Doe added.'
_P_LONE_BEFORE = re.compile(
    r"(?:^|[,\"'.!?;:]\s*)\s*([A-Z][a-zA-Z'\-]{2,})\s+" + _VERBS + r"\b",
    re.MULTILINE)
_P_LONE_AFTER = re.compile(
    r"\b" + _VERBS + r"\s+([A-Z][a-zA-Z'\-]{2,})\b(?!\s+[A-Z])")

_TITLE_TOKENS = {"dr", "mr", "ms", "mrs", "prof", "professor", "sen",
                 "senator", "rep", "gov", "governor", "gen", "col", "capt",
                 "sgt", "rev", "mayor", "judge", "president", "secretary",
                 "commissioner"}

_STOP = {"the", "a", "an", "this", "that", "these", "those", "it", "he",
         "she", "we", "they", "i", "you", "who", "which", "in", "on", "at",
         "as", "and", "but", "for", "from", "with", "by", "of", "to",
         "according", "meanwhile", "however", "today", "yesterday",
         "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
         "sunday", "january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december",
         "company", "group", "press", "release", "statement", "sources",
         "spokesperson", "spokesman", "spokeswoman", "officials", "one"}

_CORP = {"inc", "corp", "corporation", "llc", "ltd", "co", "plc", "company",
         "group", "technologies", "solutions", "systems", "holdings",
         "partners", "university", "institute", "department", "agency",
         "association", "foundation", "committee", "council",
         "administration", "bureau", "office", "ministry", "robotics",
         "labs", "laboratories", "software", "networks", "pharmaceuticals",
         "energy", "capital", "ventures", "media", "bank", "airlines",
         "motors", "international"}


def _key_from_name(name):
    toks = [t for t in (tok.strip(".,;:'\"") for tok in name.split()) if t]
    toks = [t for t in toks if t.lower().rstrip(".") not in _TITLE_TOKENS]
    if not toks:
        return None
    if toks[0].lower() in _STOP:
        return None
    last = toks[-1].lower().rstrip(".")
    if len(last) < 2 or last in _STOP or last in _CORP:
        return None
    return last


def _speaker_key(fragment):
    for pat in (_P_AFTER, _P_BEFORE, _P_LONE_BEFORE, _P_LONE_AFTER):
        for m in pat.finditer(fragment):
            k = _key_from_name(m.group(1))
            if k:
                return k
    return None


# --------------------------------------------------------------- structure
def _quote_spans(text):
    spans = []
    for m in re.finditer(r'"([^"]{15,600})"', text):
        inner = m.group(1)
        if " " in inner.strip() and re.search(r"[A-Za-z]", inner):
            spans.append((m.start(), m.end()))
    return spans


def _paragraphs(t):
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in t.split("\n") if p.strip()]
    return paras


_TAIL = re.compile(
    r"(?i)^\s*(about\s+(?:the\s+)?[A-Za-z]|media\s+contacts?|press\s+contacts?|"
    r"contacts?:|for\s+(?:more|further|additional)\s+information|source:|"
    r"###|#\s?#\s?#|investor\s+relations)")
_CONTACT = re.compile(
    r"[\w.\-+]+@[\w\-]+\.[\w.\-]+|\(\d{3}\)\s*\d{3}|(?<!\d)\d{3}[-.\s]\d{3}[-.\s]\d{4}")


def _trim_tail(paras):
    """Drop the trailing boilerplate block (About/contact), if any."""
    n = len(paras)
    cut = n
    start = max(int(n * 0.5), 1)
    for i in range(n - 1, start - 1, -1):
        p = paras[i]
        first_line = p.split("\n", 1)[0].strip()
        if (_TAIL.match(first_line) and len(first_line) <= 80) or \
                (_CONTACT.search(p) and len(p) < 300):
            cut = i
    return paras[:cut] if cut > 0 else paras


def _ladder(n, spread, q):
    if q == 0:
        return 0.03
    if n == 0:
        return 0.15 + 0.05 * spread
    if n == 1:
        return 0.30 + 0.05 * spread
    if n == 2:
        return 0.60 + 0.08 * spread
    return 0.88 + 0.12 * spread


def _blank_spans(p, spans):
    chars = list(p)
    for s, e in spans:
        for i in range(s, min(e, len(chars))):
            chars[i] = " "
    return "".join(chars)


def score(text: str) -> float:
    try:
        t = _normalize(text)
        if not t or not t.strip():
            return 0.0
        paras = _paragraphs(t)
        paras = _trim_tail(paras)
        if not paras:
            return 0.03

        if len(paras) == 1:
            # Single-blob page: positional windows around the spans instead.
            blob = paras[0]
            spans = _quote_spans(blob)
            if not spans:
                return 0.03
            keys = set()
            for s, e in spans:
                win = blob[max(0, s - 140):s] + " " + blob[e:e + 140]
                k = _speaker_key(win)
                if k:
                    keys.add(k)
            L = max(len(blob), 1)
            pos = [s / L for s, _ in spans]
            spread = (max(pos) - min(pos)) if len(pos) >= 2 else 0.0
            n = min(len(keys), len(spans))
            return float(min(1.0, max(0.0, _ladder(n, spread, len(spans)))))

        # Paragraph mode.
        span_map = [_quote_spans(p) for p in paras]
        qidx = [i for i, sp in enumerate(span_map) if sp]
        if not qidx:
            return 0.03
        qset = set(qidx)
        keys = set()
        for i in qidx:
            frag = _blank_spans(paras[i], span_map[i])
            k = _speaker_key(frag)
            if k is None and (i + 1) < len(paras) and (i + 1) not in qset:
                k = _speaker_key(paras[i + 1][:140])
            if k:
                keys.add(k)
        denom = max(len(paras) - 1, 1)
        pos = [i / denom for i in qidx]
        spread = (max(pos) - min(pos)) if len(pos) >= 2 else 0.0
        n = min(len(keys), len(qidx))
        return float(min(1.0, max(0.0, _ladder(n, spread, len(qidx)))))
    except Exception:
        return 0.5
