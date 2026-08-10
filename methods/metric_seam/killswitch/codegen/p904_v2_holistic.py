"""p904_v2_holistic -- Voice diversity (composite of weak signals).

Criterion: the release features direct quotations from multiple distinct
named people, each clearly attributed.

Approach: normalize mojibake, then blend five weak signals -- (1) number of
quotation spans, (2) distinct attributed speaker count from windows around
each span (dominant weight), (3) diversity of speaker ROLE categories
(executive vs. customer/partner vs. public official), (4) canonical
quote-attribution punctuation patterns, (5) positional spread of quotes
across the document -- then damp the total by navigation-chrome/non-release
indicators.
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


def _keys_in(fragment):
    keys = set()
    for pat in (_P_AFTER, _P_BEFORE, _P_LONE_BEFORE, _P_LONE_AFTER):
        for m in pat.finditer(fragment):
            k = _key_from_name(m.group(1))
            if k:
                keys.add(k)
    return keys


# ------------------------------------------------------------ weak signals
def _quote_spans(text):
    spans = []
    for m in re.finditer(r'"([^"]{15,600})"', text):
        inner = m.group(1)
        if " " in inner.strip() and re.search(r"[A-Za-z]", inner):
            spans.append((m.start(), m.end()))
    return spans


_ROLE_EXEC = re.compile(
    r"(?i)\b(ceo|cto|cfo|coo|cio|chief\s+\w+\s+officer|chief\s+executive|"
    r"president|founder|co-?founder|chairman|chairwoman|chairperson|"
    r"executive|vice\s+president|vp|general\s+manager|head\s+of|"
    r"spokesman|spokeswoman|spokesperson|managing\s+director|director)\b")
_ROLE_EXTERNAL = re.compile(
    r"(?i)\b(customer|client|user|subscriber|patient|resident|partner|"
    r"member|fan|student|teacher|parent|analyst|consultant|professor|"
    r"researcher|expert|advocate|neighbor|shopper|driver|rider|attendee|"
    r"participant|beneficiary)\b")
_ROLE_OFFICIAL = re.compile(
    r"(?i)\b(mayor|governor|senator|congressman|congresswoman|"
    r"representative|councilman|councilwoman|councillor|commissioner|"
    r"secretary|minister|official|sheriff|superintendent|chancellor|"
    r"ambassador|legislator|regulator|police)\b")

_PAT1 = re.compile(r'[,.!?]"\s+' + _VERBS + r"\s+[A-Z]")
_PAT2 = re.compile(r'[,.!?]"\s+[A-Z][a-zA-Z\'\-]*\s+' + _VERBS + r"\b")
_PAT3 = re.compile(r"\b" + _VERBS + r'[,:]?\s*"[A-Z]')

_CHROME = re.compile(
    r"(?i)\b(cookies?|subscribe|sign\s+in|log\s+in|newsletter|"
    r"privacy\s+policy|all\s+rights\s+reserved|terms\s+of\s+(?:use|service)|"
    r"related\s+articles|read\s+more|share\s+this|advertisement)\b")


def score(text: str) -> float:
    try:
        t = _normalize(text)
        if not t or not t.strip():
            return 0.0
        L = max(len(t), 1)
        spans = _quote_spans(t)
        q = len(spans)

        # Signal 2 (dominant): distinct attributed speakers near quote spans,
        # with role categories read only from windows that yielded a speaker.
        keys = set()
        cats = set()
        for s, e in spans:
            win = t[max(0, s - 130):s] + " " + t[e:e + 110]
            got = _keys_in(win)
            if got:
                keys.update(got)
                if _ROLE_EXEC.search(win):
                    cats.add("exec")
                if _ROLE_EXTERNAL.search(win):
                    cats.add("external")
                if _ROLE_OFFICIAL.search(win):
                    cats.add("official")
        n = min(len(keys), q) if q else 0

        s_q = min(q, 4) / 4.0
        s_n = 0.0 if n == 0 else (0.33 if n == 1 else (0.67 if n == 2 else 1.0))
        s_roles = min(len(cats), 3) / 3.0
        pat_hits = (len(_PAT1.findall(t)) + len(_PAT2.findall(t))
                    + len(_PAT3.findall(t)))
        s_pat = min(pat_hits, 3) / 3.0
        if q >= 2:
            starts = [s for s, _ in spans]
            s_spread = (max(starts) - min(starts)) / L
        else:
            s_spread = 0.0

        total = (0.15 * s_q + 0.45 * s_n + 0.15 * s_roles
                 + 0.10 * s_pat + 0.15 * s_spread)

        # Non-release / navigation-chrome damping.
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        mult = 1.0
        if len(lines) >= 8:
            short = sum(1 for ln in lines if len(ln) < 35)
            if short / len(lines) > 0.6:
                mult -= 0.2
        mult -= 0.04 * min(len(_CHROME.findall(t)), 5)

        return float(min(1.0, max(0.0, total * mult)))
    except Exception:
        return 0.5
