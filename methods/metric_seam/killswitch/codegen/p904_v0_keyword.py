"""p904_v0_keyword -- Voice diversity (surface/lexical heuristic).

Criterion: the release features direct quotations from multiple distinct
named people, each clearly attributed.

Approach: normalize mojibake punctuation to ASCII quotes -> count paired
double-quote spans that look like real quotations (flat regex over the whole
text) -> flat regex scan for speech-verb + capitalized-name attribution
patterns anywhere in the text -> dedup speakers by surname -> map the number
of distinct quoted voices onto a fixed score ladder.
"""

import re


def _seq(*cps):
    return "".join(chr(c) for c in cps)


# --------------------------------------------------------------- mojibake
# UTF-8 bytes of curly punctuation re-read as cp1252 (or latin-1). Built via
# chr() codepoints because several members are INVISIBLE control characters
# (e.g. the U+009D tail of a mojibake closing double quote). Order matters:
# 3-char sequences first (incl. mojibake dashes whose THIRD char is itself a
# genuine curly quote, so they must be consumed before the standalone
# curly-quote entries), then bare 2-char stubs (scrapers often strip the
# invisible tail), then NBSP mojibake, then genuine curly punctuation.
_MOJIBAKE = [
    # cp1252 renderings: a-circumflex + euro + tail
    (_seq(0xE2, 0x20AC, 0x0153), '"'),    # left double quote (oe tail)
    (_seq(0xE2, 0x20AC, 0x009D), '"'),    # right double quote (invisible tail)
    (_seq(0xE2, 0x20AC, 0x2122), "'"),    # apostrophe (TM tail)
    (_seq(0xE2, 0x20AC, 0x02DC), "'"),    # left single quote (tilde tail)
    (_seq(0xE2, 0x20AC, 0x201C), "-"),    # en dash (tail is U+201C itself)
    (_seq(0xE2, 0x20AC, 0x201D), "--"),   # em dash (tail is U+201D itself)
    (_seq(0xE2, 0x20AC, 0x00A6), "..."),  # ellipsis (broken-bar tail)
    # latin-1 renderings: C1 control chars in 2nd/3rd position
    (_seq(0xE2, 0x80, 0x9C), '"'),
    (_seq(0xE2, 0x80, 0x9D), '"'),
    (_seq(0xE2, 0x80, 0x99), "'"),
    (_seq(0xE2, 0x80, 0x98), "'"),
    (_seq(0xE2, 0x80, 0x93), "-"),
    (_seq(0xE2, 0x80, 0x94), "--"),
    (_seq(0xE2, 0x80, 0xA6), "..."),
    # bare 2-char stubs left when the invisible tail was stripped
    (_seq(0xE2, 0x20AC), '"'),
    (_seq(0xE2, 0x80), '"'),
    # non-breaking-space mojibake: A-circumflex before nbsp
    (_seq(0xC2, 0xA0), " "),
    (_seq(0xC2), ""),
    # genuine curly punctuation -> ASCII (must come AFTER mojibake entries)
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

# Two-to-four capitalized tokens, optional honorific/office prefix.
_NAME = (r"(?:" + _TITLE + r"\s+)?"
         r"[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-\.]+){1,3}")

_P_AFTER = re.compile(r"\b" + _VERBS + r"\s+(?:that\s+)?(" + _NAME + r")")
_P_BEFORE = re.compile(r"(" + _NAME + r")\s*,?[^.!?\n\"]{0,60}?\s" + _VERBS + r"\b")
_P_LONE_BEFORE = re.compile(
    r"(?:^|[\"'.!?;:]\s*)([A-Z][a-zA-Z'\-]{2,})\s+" + _VERBS + r"\b",
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
    """Reduce a matched name to a dedup key (surname), or None if bogus."""
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


def _speaker_keys(text):
    keys = set()
    for pat in (_P_AFTER, _P_BEFORE, _P_LONE_BEFORE, _P_LONE_AFTER):
        for m in pat.finditer(text):
            k = _key_from_name(m.group(1))
            if k:
                keys.add(k)
    return keys


def _quote_spans(text):
    """Paired double-quote spans that look like real quotations."""
    spans = []
    for m in re.finditer(r'"([^"]{15,600})"', text):
        inner = m.group(1)
        if " " in inner.strip() and re.search(r"[A-Za-z]", inner):
            spans.append((m.start(), m.end()))
    return spans


def score(text: str) -> float:
    try:
        t = _normalize(text)
        if not t or not t.strip():
            return 0.0
        q = len(_quote_spans(t))
        if q == 0:
            return 0.05
        n = min(len(_speaker_keys(t)), q)
        if n == 0:
            s = 0.20
        elif n == 1:
            s = 0.35
        elif n == 2:
            s = 0.65
        else:
            s = 0.90 + 0.05 * (n - 3)
        return float(min(1.0, max(0.0, s)))
    except Exception:
        return 0.5
