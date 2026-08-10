"""p904_h0 -- Voice diversity (hybrid: quote-anchored attribution + LLM speaker list).

Criterion: direct quotations from multiple DISTINCT named people, each clearly
attributed. 0 voices -> low, 1 spokesperson -> low, 3+ -> high.

Improvements over v0_keyword baseline (train rho 0.722):
1. Attribution must be ADJACENT to a real quote span (the baseline counted
   quoted spans and speech-verb+name patterns independently anywhere in the
   text, so legalese pages full of quoted defined terms like "I AGREE"
   scored 0.9 while the judge gave 0.25).
2. One LLM field lists who is actually quoted in the FULL document. Code
   keeps the predicate: the field vetoes regex false positives (extractor
   says NONE but regex found "speakers") and rescues regex false negatives
   (mojibake-destroyed quote marks, quotes outside regex reach).
3. Quote spans must look like speech (length, word count, case) before they
   count for anything.

Deterministic; stdlib re only + provided ops; never raises (-> 0.5).
"""

import re

LLM_FIELDS = {
    "quoted_people": ("List the distinct full names of people directly quoted "
                      "in quotation marks in this document, comma-separated; "
                      "answer NONE if nobody is quoted."),
}

# --------------------------------------------------------------- mojibake
# Fallback normalization applied AFTER ops.normalize (harmless if already
# clean). Covers cp1252/latin-1 curly-quote mojibake, bare stubs, HTML
# entities, genuine curly punctuation.
_MOJIBAKE = [
    ("â€œ", '"'), ("â€", '"'),
    ("â€™", "'"), ("â€˜", "'"),
    ("â€“", "-"), ("â€”", "--"),
    ("â€¦", "..."),
    ("â", '"'), ("â", '"'),
    ("â", "'"), ("â", "'"),
    ("â", "-"), ("â", "--"),
    ("â¦", "..."),
    ("â€", '"'), ("â", '"'),
    ("Â ", " "), ("Â", ""),
    ("&quot;", '"'), ("&#34;", '"'), ("&#39;", "'"), ("&amp;", "&"),
    ("“", '"'), ("”", '"'), ("„", '"'),
    ("‘", "'"), ("’", "'"),
    ("«", '"'), ("»", '"'),
    ("—", "--"), ("–", "-"), (" ", " "),
]


def _extra_norm(text):
    for bad, good in _MOJIBAKE:
        if bad in text:
            text = text.replace(bad, good)
    return text


# ------------------------------------------------------------ vocabulary
_TITLES = {"dr", "mr", "ms", "mrs", "prof", "professor", "sen", "senator",
           "rep", "gov", "governor", "gen", "col", "capt", "sgt", "rev",
           "rabbi", "imam", "bishop", "mayor", "judge", "president",
           "secretary", "commissioner", "chairman", "ceo", "chief",
           "director", "officer", "founder", "spokesman", "spokeswoman",
           "spokesperson"}

_STOP = {"the", "a", "an", "this", "that", "these", "those", "it", "he",
         "she", "they", "we", "i", "you", "who", "which", "what", "when",
         "where", "why", "how", "and", "but", "or", "nor", "so", "yet",
         "for", "in", "on", "at", "by", "as", "if", "of", "to", "from",
         "with", "our", "your", "their", "his", "her", "its", "my", "one",
         "all", "some", "also", "then", "there", "here", "not", "no", "yes",
         "new", "more", "most", "many", "much", "both", "each", "per",
         "via", "am", "pm", "et", "pt", "gmt", "est", "pst", "edt", "pdt",
         "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
         "sunday", "january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december",
         "source", "contact", "email", "phone", "follow", "share", "tweet",
         "read", "learn", "click", "view", "see", "visit", "search", "skip",
         "login", "register", "sign", "home", "none"}

_ORG_WORDS = {"inc", "corp", "corporation", "company", "companies", "group",
              "bank", "university", "college", "school", "foundation",
              "association", "institute", "institutes", "department",
              "agency", "commission", "committee", "council", "ministry",
              "administration", "authority", "board", "fund", "capital",
              "partners", "holdings", "systems", "technologies", "solutions",
              "communications", "media", "network", "networks", "news",
              "newswire", "prnewswire", "reuters", "bloomberg", "press",
              "journal", "times", "post", "today", "report", "reports",
              "research", "statement", "release", "america", "americas",
              "international", "global", "worldwide", "limited", "ltd",
              "llc", "llp", "plc", "co", "team", "study", "survey",
              "organization", "organisation", "center", "centre", "office",
              "services", "service", "management", "investments", "security",
              "securities", "exchange", "nasdaq", "nyse", "policy",
              "policies", "terms", "conditions", "privacy", "agreement",
              "notice", "disclaimer", "act", "law", "website", "site",
              "page", "form", "review", "article", "story", "editor",
              "author", "staff"}

# Speech verbs that mark a quote attribution.
_V = (r"(?:said|says|stated|states|added|adds|noted|notes|commented|"
      r"comments|remarked|remarks|explained|explains|continued|concluded|"
      r"emphasized|emphasised|wrote|writes|observed|argued|argues)")
# Extended set allowed only in the name-BEFORE-quote pattern (quote-gated).
_V2 = (r"(?:said|says|stated|states|added|adds|noted|notes|commented|"
       r"comments|remarked|remarks|explained|explains|continued|concluded|"
       r"emphasized|emphasised|wrote|writes|told|called|warned|urged|"
       r"described|declared|argued)")
# Verbs allowed right after "quote," Name ___ .
_V3 = (r"(?:said|says|added|adds|told|noted|stated|explained|wrote|"
       r"continued|concluded|remarked|commented)")

_NAME = (r"((?:[A-Z][a-zA-Z'\-]+\.?\s+)?"
         r"[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-\.]{1,25}){0,3})")
_NAME_SHORT = r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2})"

# P1: closing quote, then verb, then name:   ...," said Jane Doe, CEO ...
_P1 = re.compile(r'"[\s,.\-]{0,3}' + _V +
                 r"\s+(?:that\s+)?(?:by\s+)?" + _NAME)
_P1B = re.compile(r'"[\s,.\-]{0,3}according\s+to\s+' + _NAME)
# P2: name (+<=70 chars of apposition), verb, then a quote opens nearby:
#     Hock Tan, President and CEO of Broadcom, stated, "We have heard..."
_P2 = re.compile(_NAME + r'[^".!?\n]{0,70}?\b' + _V2 +
                 r'\b[^"\n]{0,30}?[:,]?\s*"([^"]{15,1200})"')
# P3: closing quote, then short name, then verb:   ...," Stetson said.
_P3 = re.compile(r'"[\s,.\-]{0,3}' + _NAME_SHORT + r"\s+" + _V3 + r"\b")
# P5: interview/headline style:   Stephon Gilmore: "I just want..."
_P5 = re.compile(r"([A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+"
                 r"(?:\s+[A-Z][a-zA-Z'\-]+)?)\s*:\s*\"([^\"]{15,600})\"")
# P6: testimonial style:   "Best product ever." -- Jane Doe, customer
_P6 = re.compile(r'"\s*-{1,2}\s*' + _NAME)
# Paraphrase attribution (no quote required) -- weak tie-break signal only.
_P_PARA = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-zA-Z'\-]+)\s+" + _V2 + r"\b")

_Q_SPAN = re.compile(r'"([^"\n]{25,600})"')


def _surname_key(name):
    """Reduce a matched name to a dedup key; None if it isn't person-like."""
    toks = [t for t in re.split(r"[\s\.]+", name) if t]
    while toks and toks[0].lower().strip(".'-") in (_TITLES | _STOP):
        toks.pop(0)
    while toks and toks[-1].lower().strip(".'-") in (_TITLES | _STOP):
        toks.pop()
    if not toks:
        return None
    last = toks[-1]
    if len(last) > 1 and last.isupper():          # acronym / ticker
        return None
    key = last.lower().strip("'-")
    if len(key) < 2 or key in _ORG_WORDS or key in _STOP:
        return None
    return key


def _speechy(span):
    """Does a quoted span look like actual human speech?"""
    if not span or span[0].isspace() or span[-1].isspace():
        return False          # inter-quote junk from unpaired quote marks
    words = span.split()
    if len(words) < 5:
        return False
    letters = [c for c in span if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    if upper / float(len(letters)) > 0.5:          # legalese / nav shouting
        return False
    return any(c.islower() for c in span)


def _count_code_speakers(t):
    """Distinct surnames attributed adjacent to quote marks."""
    speakers = set()
    for pat in (_P1, _P1B, _P3, _P6):
        for m in pat.finditer(t):
            k = _surname_key(m.group(1))
            if k:
                speakers.add(k)
    for m in _P2.finditer(t):
        if _speechy(m.group(2)):
            k = _surname_key(m.group(1))
            if k:
                speakers.add(k)
    for m in _P5.finditer(t):
        if _speechy(m.group(2)):
            k = _surname_key(m.group(1))
            if k:
                speakers.add(k)
    return speakers


def _parse_llm_names(ans):
    """Parse the extractor's name list -> (distinct count, said_none).

    said_none=True only for an explicit empty/NONE answer; a rambling
    answer that parses to zero names returns (0, False) and is treated
    as uninformative rather than as a veto.
    """
    if not ans:
        return 0, True
    a = ans.strip().strip('."\' ')
    if not a or a.lower() in ("none", "no", "n/a", "na", "nobody", "-"):
        return 0, True
    if a.lower().startswith("none"):
        return 0, True
    keys = set()
    for part in re.split(r"[,;\n]|\band\b|&", a):
        part = part.strip(" .\"'()")
        if not part:
            continue
        toks = part.split()
        if len(toks) > 5:                          # not a name, a sentence
            continue
        caps = [w for w in toks
                if re.match(r"^[A-Z][a-zA-Z'\-\.]*$", w)]
        if not caps:
            continue
        k = _surname_key(" ".join(caps))
        if k:
            keys.add(k)
    return min(len(keys), 5), False


_LADDER = {0: 0.05, 1: 0.30, 2: 0.62, 3: 0.90}


def score(text, extracted, ops):
    try:
        try:
            t = ops.normalize(text)
            if not isinstance(t, str) or not t:
                t = text
        except Exception:
            t = text
        t = _extra_norm(t)

        speakers = _count_code_speakers(t)
        n_code = len(speakers)

        good_spans = sum(1 for m in _Q_SPAN.finditer(t)
                         if _speechy(m.group(1)))

        # ---- combine with LLM field (predicate stays in code) ----
        n = n_code
        vetoed = False
        raw = extracted.get("quoted_people") if isinstance(extracted, dict) \
            else None
        if raw is not None:
            n_llm, said_none = _parse_llm_names(
                raw if isinstance(raw, str) else str(raw))
            if said_none:
                # Extractor saw the full doc and found nobody quoted:
                # regex hits are likely quoted-jargon false positives.
                if n_code >= 1:
                    vetoed = True
                n = 0
            elif n_llm == 0:
                n = n_code            # uninformative answer: trust the code
            elif n_code == 0:
                # Regex blind (mojibake / unusual layout): partial trust,
                # but only if quote marks actually survive in the text.
                n = min(n_llm, 2) if t.count('"') >= 2 else 1
            else:
                # Round-half-up average of the two witnesses.
                n = (n_code + n_llm + 1) // 2

        if vetoed:
            # Hedge: extractor may be wrong; scale mildly with code count.
            s = 0.15 + 0.05 * min(n_code - 1, 3)
        else:
            s = _LADDER.get(n, 0.95)

        # Gentle low-band tie-breakers (judge gives ~0.1-0.2 to pages with
        # quote-ish content or person+verb mentions but no attributed quote).
        if n == 0 and not vetoed:
            if good_spans >= 1:
                s += 0.08
            para = False
            for m in _P_PARA.finditer(t):
                if _surname_key(m.group(1)):
                    para = True
                    break
            if para:
                s += 0.04

        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
