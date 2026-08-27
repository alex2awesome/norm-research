"""p905 v0 -- Authentic authorship, surface/lexical heuristic.

Criterion: the release reads as if written by someone with genuine
familiarity with the company and subject matter (concrete, specific,
coherent) rather than assembled from generic template marketing language.

Approach (pure keyword/regex densities):
  score = 0.5 + 0.48 * specificity_density - 0.48 * generic_buzzword_density
where specificity counts money figures, percents, dates, measured units,
version numbers, executive titles, capitalized name pairs and acronyms,
and genericness counts a broad bank of template-marketing phrases.

Contract: score(text) -> float in [0, 1]; higher = more authentic.
Deterministic; imports limited to re/math/statistics/collections.
"""

import re

# ----------------------------------------------------------------------
# Mojibake / unicode normalization (cp1252-misdecoded UTF-8 punctuation).
# All non-ASCII is written as \u escapes so the invisible U+009D is exact.
# ----------------------------------------------------------------------
_MOJIBAKE = (
    ("â€œ", '"'),   # a-circ + euro + oe   -> left double quote
    ("â€", '"'),   # a-circ + euro + 9D   -> right double quote
    ("â€™", "'"),   # a-circ + euro + TM   -> apostrophe
    ("â€˜", "'"),   # a-circ + euro + tilde-> left single quote
    ("â€“", "-"),   # misdecoded en dash
    ("â€”", "-"),   # misdecoded em dash
    ("â€¦", "..."), # misdecoded ellipsis
    ("â€¢", "*"),   # misdecoded bullet
    ("Â ", " "),         # A-circ + nbsp        -> space
)

_LEFTOVER_MOJI = re.compile("â€.")


def _normalize(text):
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    text = _LEFTOVER_MOJI.sub("'", text)        # any remaining misdecoded pair
    text = text.replace("Â", "")           # stray A-circumflex
    text = text.replace(" ", " ")          # real non-breaking space
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("[...]", "\n\n")        # elided-middle marker
    return text


# ----------------------------------------------------------------------
# Generic template-marketing phrase bank (broad, lexical)
# ----------------------------------------------------------------------
_GENERIC_PHRASES = [
    r"industry[\s-]+leading", r"world[\s-]+class", r"cutting[\s-]+edge",
    r"state[\s-]+of[\s-]+the[\s-]+art", r"best[\s-]+in[\s-]+class",
    r"next[\s-]+generation",
    r"leading\s+(?:provider|supplier|manufacturer|developer|source|brand)",
    r"premier\s+(?:provider|source|destination|choice)",
    r"innovative\s+solutions?", r"comprehensive\s+(?:solutions?|suite|range)",
    r"end[\s-]+to[\s-]+end", r"one[\s-]+stop(?:[\s-]+shop)?", r"turn[\s-]?key",
    r"mission[\s-]+critical", r"game[\s-]+chang(?:ing|ers?)",
    r"seamless(?:ly)?", r"leverag(?:e|es|ing)", r"synerg(?:y|ies|istic)",
    r"unparalleled", r"unmatched", r"unrivall?ed", r"second\s+to\s+none",
    r"revolutionar(?:y|ies)", r"revolutioniz(?:e|es|ing)", r"groundbreaking",
    r"paradigm", r"empower(?:s|ing|ment)?", r"transformative", r"disruptive",
    r"customer[\s-]+centric", r"client[\s-]+centric", r"value[\s-]+added",
    r"core\s+competenc(?:y|ies)", r"thought\s+leader(?:ship)?",
    r"commitment\s+to\s+(?:excellence|quality|innovation)",
    r"dedicated\s+to\s+(?:providing|delivering|serving)",
    r"striv(?:e|es|ing)\s+to", r"passionate\s+about",
    r"pride\s+(?:ourselves|itself|themselves)", r"proven\s+track\s+record",
    r"trusted\s+(?:partner|name|source|advisor)",
    r"wide\s+(?:range|variety|array)\s+of", r"broad\s+(?:range|variety|array)\s+of",
    r"full\s+(?:suite|range|spectrum)\s+of", r"vast\s+experience",
    r"highly\s+experienced", r"uniquely\s+positioned", r"at\s+the\s+forefront",
    r"sets?\s+(?:us|them|it)\s+apart", r"exceed(?:s|ing)?\s+(?:customer\s+)?expectations",
    r"world[\s-]+renowned", r"market[\s-]+leading", r"award[\s-]+winning",
    r"holistic", r"robust", r"scalable", r"visionary", r"excellence",
    r"innovative", r"top[\s-]+notch", r"first[\s-]+rate", r"unwavering",
    r"utmost", r"high(?:est)?[\s-]+quality",
    r"needs\s+of\s+(?:our|their)\s+(?:customers|clients)",
    r"exceptional\s+(?:service|quality|value|results)",
    r"is\s+(?:pleased|proud|excited|thrilled|delighted)\s+to\s+announce",
]
_GENERIC_RE = re.compile("|".join(_GENERIC_PHRASES), re.IGNORECASE)

# ----------------------------------------------------------------------
# Specificity markers
# ----------------------------------------------------------------------
_MONEY_RE = re.compile(
    r"(?:[$£€]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s*(?:million|billion|trillion)\b)",
    re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?"
    r"|dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*(?:19|20)\d{2})?\b",
    re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_VERSION_RE = re.compile(r"\b[vV]?\d+\.\d+(?:\.\d+)*\b")
_UNITS_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s?(?:mg|kg|km|mph|kwh?|mwh?|gwh?|ghz|mhz|gb|tb|mb"
    r"|nm|mm|cm|acres?|employees|staff|customers|clients|users|subscribers"
    r"|patients|students|stores|branches|locations|offices|countries|states"
    r"|cities|markets|patents|units|vehicles|devices|seats|rooms|miles"
    r"|hectares|tons?|tonnes|square\s+(?:feet|meters|metres)|sq\.?\s?ft\.?)\b",
    re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_TITLE_RE = re.compile(
    r"\b(?:CEO|CTO|CFO|COO|CMO|CIO|Chief\s+\w+\s+Officer|President"
    r"|Vice\s+President|Executive\s+Director|Managing\s+Director"
    r"|Co-?[Ff]ounder|Founder|Chairman|Chairwoman|Director\s+of|Head\s+of"
    r"|Professor|Dr\.)\b")
_CAP_PAIR_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")

# Contact-block noise stripped before counting numbers as "specific"
_URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\(?\+?\d{1,3}\)?[\s.-]?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Boilerplate-tail keywords (contact / About blocks sit near document END);
# text from the earliest such line-initial marker in the final 45% is cut.
_TAIL_CUT_RE = re.compile(
    r"^\s*(?:about\s+\S|media\s+contacts?|press\s+contacts?|contacts?\s*:"
    r"|for\s+(?:more|further|additional)\s+information|source\s*:|###"
    r"|copyright|all\s+rights\s+reserved|media\s+inquiries"
    r"|investor\s+relations|follow\s+us)",
    re.IGNORECASE | re.MULTILINE)

# Function-word rate distinguishes prose from navigation chrome; the
# capitalization-derived signals only count in prose-like text.
_PROSE_RE = re.compile(
    r"\b(?:the|of|and|a|an|to|in|for|with|on|at|by|from|as|is|are|was"
    r"|were|that|this|it|its|has|have|will)\b",
    re.IGNORECASE)


def score(text: str) -> float:
    try:
        if not text or not text.strip():
            return 0.5
        t = _normalize(text)

        # cut the trailing contact/About boilerplate block (final 45% only)
        cutoff = int(0.55 * len(t))
        for m in _TAIL_CUT_RE.finditer(t):
            if m.start() >= cutoff:
                t = t[:m.start()]
                break

        # strip URLs / emails / phone numbers so contact chrome does not
        # masquerade as content specificity
        t = _URL_RE.sub(" ", t)
        t = _EMAIL_RE.sub(" ", t)
        t = _PHONE_RE.sub(" ", t)

        n_words = len(_WORD_RE.findall(t))
        if n_words < 8:
            return 0.5

        buzz = len(_GENERIC_RE.findall(t))

        strong = (len(_MONEY_RE.findall(t)) + len(_PERCENT_RE.findall(t))
                  + len(_DATE_RE.findall(t)) + len(_UNITS_RE.findall(t))
                  + len(_TITLE_RE.findall(t)))
        medium = len(_VERSION_RE.findall(t)) + len(_YEAR_RE.findall(t))
        plain_numbers = min(len(_NUMBER_RE.findall(t)), 15)
        cap_pairs = min(len(_CAP_PAIR_RE.findall(t)), 12)
        acronyms = min(len(_ACRONYM_RE.findall(t)), 8)

        # gate capitalization-derived signals on prose-likeness so that
        # Title-Case navigation menus do not read as proper-noun density
        prose_ratio = len(_PROSE_RE.findall(t)) / float(n_words)
        cap_factor = max(0.0, min(1.0, (prose_ratio - 0.10) / 0.15))

        spec_units = (1.0 * strong + 0.7 * medium + 0.3 * plain_numbers
                      + cap_factor * (0.25 * cap_pairs + 0.2 * acronyms))

        spec_rate = 100.0 * spec_units / n_words   # per 100 words
        buzz_rate = 100.0 * buzz / n_words

        specificity = min(1.0, spec_rate / 6.0)
        generic_pressure = min(1.0, buzz_rate / 4.5)

        val = 0.5 + 0.48 * specificity - 0.48 * generic_pressure
        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.5
