"""p904 v2 -- Voice diversity: holistic composite of weak signals.

Criterion: direct quotations from multiple distinct named people, each
clearly attributed. No quotes / single spokesperson -> low; 3+ distinct
quoted voices -> high.

Approach: blend several weak signals into one composite score.
  A. distinct attributed people (dominant signal, saturates at 3)
  B. role diversity -- do the attributions span different role families
     (executive vs. customer vs. partner vs. official vs. expert)?
  C. attribution completeness -- fraction of quoted spans with a nearby
     attribution pattern
  D. quote volume -- number of speech-like quoted spans (capped)
  E. positional spread -- quotes in more than one region of the document
The composite is anchored so 0 quotes -> ~0, one well-attributed
spokesperson -> mid-low, and 3+ distinct voices with role diversity -> ~1.
"""

import re
import math
from collections import OrderedDict

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


# --- quoted spans -------------------------------------------------------
_QUOTE_SPAN = re.compile(r'"([^"]{20,1500})"')


def _speechlike(span):
    return (" " in span and re.search(r"[a-z]", span)
            and len(span.split()) >= 4)


# --- attribution --------------------------------------------------------
_VERBS = (
    r"(?:said|says|stated|states|added|adds|noted|notes|commented|"
    r"comments|explained|explains|remarked|remarks|continued|concluded|"
    r"emphasized|emphasised|told|according\s+to)"
)
_NAME = (
    r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z]\.)?(?:\s+(?:van|von|de|der|da|del|la))?"
    r"\s+[A-Z][a-zA-Z'\-]+)"
)
_HON = r"(?:Dr|Mr|Ms|Mrs|Prof|Professor|Sen|Rep|Gov|Mayor|Sir)\.?\s+"

_VERB_NAME = re.compile(_VERBS + r"[\s:,]{1,4}(?:" + _HON + r")?" + _NAME)
_NAME_VERB = re.compile(r"(?:" + _HON + r")?" + _NAME + r"[\s,]{1,3}" + _VERBS)

_BAD_FIRST = {
    "the", "a", "an", "in", "on", "at", "and", "but", "for", "with",
    "from", "by", "of", "chief", "senior", "executive", "vice",
    "president", "general", "global", "new", "our", "this", "that",
    "he", "she", "it", "they", "we", "north", "south", "east", "west",
}
_BAD_LAST = {
    "company", "inc", "corp", "corporation", "group", "llc", "ltd",
    "officer", "president", "director", "manager", "university",
    "institute", "health", "bank", "capital", "association",
    "department", "committee", "council", "partners", "technologies",
    "systems", "solutions", "release", "statement", "report", "study",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december", "today",
}

# role families around an attribution -> diversity of voice TYPES
_ROLE_FAMILIES = [
    ("executive", re.compile(
        r"\b(CEO|CFO|CTO|COO|CMO|chief\s+\w+\s+officer|president|founder|"
        r"co-founder|chairman|chairwoman|chair|managing\s+director|"
        r"executive|(?:senior\s+)?vice\s+president|VP|general\s+manager|"
        r"head\s+of)\b", re.IGNORECASE)),
    ("customer", re.compile(
        r"\b(customer|client|user|patient|resident|member|shopper|"
        r"subscriber|homeowner|guest)\b", re.IGNORECASE)),
    ("partner", re.compile(
        r"\b(partner|collaborator|investor|sponsor|supplier|distributor|"
        r"alliance)\b", re.IGNORECASE)),
    ("official", re.compile(
        r"\b(mayor|governor|senator|congress(?:man|woman)|minister|"
        r"secretary|commissioner|councilman|councilwoman|official|"
        r"spokesperson|spokesman|spokeswoman|superintendent|sheriff)\b",
        re.IGNORECASE)),
    ("expert", re.compile(
        r"\b(professor|researcher|scientist|analyst|economist|physician|"
        r"doctor|Dr\.|expert|author|historian|engineer)\b", re.IGNORECASE)),
]


def _person_key(raw):
    raw = raw.strip()
    toks = [t for t in re.split(r"\s+", raw)
            if t and not re.match(r"^[A-Z]\.$", t)
            and t.lower() not in ("van", "von", "de", "der", "da", "del", "la")]
    if len(toks) < 2:
        return None
    first, last = toks[0], toks[-1]
    if first.lower() in _BAD_FIRST or last.lower() in _BAD_LAST:
        return None
    if first.isupper() and len(first) > 3:
        return None
    return (first[0].lower(), last.lower())


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        t = _normalize(text)
        n_chars = max(1, len(t))

        spans = [(m.start(), m.end()) for m in _QUOTE_SPAN.finditer(t)
                 if _speechlike(m.group(1))]
        if not spans:
            return 0.0

        voices = OrderedDict()          # person key -> True
        roles = set()                   # role families seen at attributions
        attributed_spans = 0

        for s, e in spans:
            before = t[max(0, s - 160):s + 1]
            after = t[e - 1:e + 200]
            found = []
            for ctx in (before, after):
                for m in _VERB_NAME.finditer(ctx):
                    found.append(m.group(1))
                for m in _NAME_VERB.finditer(ctx):
                    found.append(m.group(1))
            keyed = [k for k in (_person_key(r) for r in found) if k]
            if keyed:
                attributed_spans += 1
                for k in keyed:
                    voices[k] = True
                # roles: look only in attribution context, not quote body
                for fam, pat in _ROLE_FAMILIES:
                    if pat.search(after) or pat.search(before):
                        roles.add(fam)

        n_voices = len(voices)
        n_spans = len(spans)

        # A. distinct voices (dominant, saturates at 3)
        sig_voices = min(n_voices, 3) / 3.0
        # B. role diversity (saturates at 3 families)
        sig_roles = min(len(roles), 3) / 3.0
        # C. attribution completeness
        sig_attr = attributed_spans / n_spans
        # D. quote volume (log-saturating at ~5 spans)
        sig_vol = min(1.0, math.log1p(n_spans) / math.log1p(5))
        # E. positional spread of quoted spans across doc thirds
        thirds = {min(2, int(3 * s / n_chars)) for s, _ in spans}
        sig_spread = (len(thirds) - 1) / 2.0

        composite = (0.55 * sig_voices + 0.15 * sig_roles +
                     0.12 * sig_attr + 0.08 * sig_vol + 0.10 * sig_spread)

        # anchor the floor: quotes present but zero attributed voices
        # should stay clearly low regardless of volume/spread
        if n_voices == 0:
            composite = min(composite, 0.15)

        return max(0.0, min(1.0, composite))
    except Exception:
        return 0.5
