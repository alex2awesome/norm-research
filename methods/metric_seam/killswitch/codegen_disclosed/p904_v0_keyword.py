"""p904 v0 -- Voice diversity: surface/lexical heuristic.

Criterion: the release features direct quotations from multiple distinct
named people, each clearly attributed. No quotes / a single spokesperson
scores low; three or more distinct quoted voices scores high.

Approach (purely lexical, whole-text):
  1. Repair mojibake (a-hat-euro sequences) and normalize all quote glyphs
     to ASCII '"' / "'".
  2. Extract double-quoted spans that look like real speech (length- and
     content-filtered so nav tokens like "facebook" don't count).
  3. In a small window before/after each span, look for an attribution
     verb (said/says/added/noted/...) plus a capitalized person name.
  4. Count DISTINCT people (deduped on first-initial + last name) that are
     attached to at least one quote; map that count monotonically to [0,1].
"""

import re
from collections import OrderedDict

# --- mojibake repair (ordered: longer sequences first) -----------------
_MOJIBAKE = [
    ("â€œ", '"'),   # a-hat-euro-oe  -> left curly quote
    ("â€", '"'),   # -> right curly quote
    ("â€™", "'"),   # -> right single quote / apostrophe
    ("â€˜", "'"),   # -> left single quote
    ("â€”", "-"),   # -> em dash
    ("â€“", "-"),   # -> en dash
    ("â€¦", "..."), # -> ellipsis
    ("â€", '"'),         # bare a-hat-euro -> curly quote (per corpus note)
    ("Â ", " "),         # A-hat + nbsp
    ("Â", ""),                # stray A-hat
]

_ENTITIES = [
    ("&quot;", '"'), ("&amp;", "&"), ("&#39;", "'"), ("&apos;", "'"),
    ("&nbsp;", " "), ("&ldquo;", '"'), ("&rdquo;", '"'),
    ("&lsquo;", "'"), ("&rsquo;", "'"), ("&gt;", ">"), ("&lt;", "<"),
]

_CURLY = [
    ("“", '"'), ("”", '"'), ("„", '"'), ("«", '"'),
    ("»", '"'), ("‘", "'"), ("’", "'"), ("ʼ", "'"),
    (" ", " "),
]


def _normalize(text):
    for a, b in _MOJIBAKE:
        text = text.replace(a, b)
    for a, b in _ENTITIES:
        text = text.replace(a, b)
    for a, b in _CURLY:
        text = text.replace(a, b)
    return text


# --- quoted-span extraction --------------------------------------------
_QUOTE_SPAN = re.compile(r'"([^"]{20,1500})"')


def _speechlike(span):
    # real speech: has spaces, lowercase letters, and at least 4 words
    if " " not in span:
        return False
    if not re.search(r"[a-z]", span):
        return False
    return len(span.split()) >= 4


# --- attribution / name extraction -------------------------------------
_VERBS = (
    r"(?:said|says|stated|states|added|adds|noted|notes|commented|comments|"
    r"explained|explains|remarked|remarks|continued|concluded|emphasized|"
    r"emphasised|told|wrote|according\s+to)"
)

_HONORIFIC = r"(?:Dr|Mr|Ms|Mrs|Prof|Professor|Sen|Rep|Gov|Mayor|Sir)\.?\s+"

# 2-4 capitalized tokens (middle initial allowed)
_NAME_CORE = (
    r"([A-Z][a-zA-Z'\-]+(?:\s+(?:[A-Z]\.|[A-Z][a-zA-Z'\-]+|van|von|de|der|"
    r"da|del|la))?\s+[A-Z][a-zA-Z'\-]+)"
)

_VERB_THEN_NAME = re.compile(
    _VERBS + r"[\s:,]{1,4}(?:" + _HONORIFIC + r")?" + _NAME_CORE)
_NAME_THEN_VERB = re.compile(
    r"(?:" + _HONORIFIC + r")?" + _NAME_CORE + r"[\s,]{1,3}" + _VERBS)

_FIRST_TOKEN_STOP = {
    "the", "a", "an", "in", "on", "at", "as", "and", "but", "or", "for",
    "with", "from", "by", "to", "of", "that", "this", "it", "he", "she",
    "they", "we", "i", "you", "chief", "senior", "executive", "vice",
    "president", "director", "manager", "general", "global", "new",
    "north", "south", "east", "west", "company", "group", "one", "our",
    "these", "those", "his", "her", "their", "its", "not", "also",
}

_LAST_TOKEN_STOP = {
    "company", "inc", "corp", "corporation", "group", "llc", "ltd",
    "officer", "president", "director", "manager", "university",
    "institute", "health", "bank", "capital", "street", "avenue",
    "association", "department", "committee", "council", "fund",
    "partners", "technologies", "systems", "solutions", "york", "angeles",
    "francisco", "media", "relations", "communications", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "today", "release",
}


def _person_key(raw):
    toks = [t for t in re.split(r"\s+", raw.strip()) if t]
    toks = [t for t in toks if not re.match(r"^[A-Z]\.$", t)]  # drop initials
    if len(toks) < 2:
        return None
    first, last = toks[0], toks[-1]
    if first.lower() in _FIRST_TOKEN_STOP or last.lower() in _LAST_TOKEN_STOP:
        return None
    if first.isupper() and len(first) > 3:   # ALL-CAPS chrome
        return None
    return (first[0].lower(), last.lower())


def _voices_near(context):
    keys = []
    for m in _VERB_THEN_NAME.finditer(context):
        k = _person_key(m.group(1))
        if k:
            keys.append(k)
    for m in _NAME_THEN_VERB.finditer(context):
        k = _person_key(m.group(1))
        if k:
            keys.append(k)
    return keys


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _normalize(text)

        spans = []
        for m in _QUOTE_SPAN.finditer(t):
            if _speechlike(m.group(1)):
                spans.append((m.start(), m.end()))

        if not spans:
            return 0.0

        voices = OrderedDict()
        for s, e in spans:
            before = t[max(0, s - 160):s + 1]
            after = t[e - 1:e + 160]
            for k in _voices_near(before) + _voices_near(after):
                voices[k] = True

        n = len(voices)
        if n == 0:
            # quotes exist but nobody clearly attributed
            return 0.15
        if n == 1:
            # single spokesperson; tiny bump if they are quoted repeatedly
            return 0.4 if len(spans) < 3 else 0.45
        if n == 2:
            return 0.7
        return 1.0
    except Exception:
        return 0.5
