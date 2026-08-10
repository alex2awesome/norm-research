"""p907 v0 (keyword) — surface/lexical heuristic for 'Comprehensive detail'.

Criterion: the release provides substantial informative detail — background,
specifics, supporting facts, context. Fuller releases score high; terse stubs
score low.

Approach: purely lexical. Count detail-bearing surface tokens — overall word
volume, numeric figures (money / percents / counts), calendar references,
elaboration & context phrases ("in addition", "according to", "founded", ...),
and quotation marks (supporting quotes). Each raw count is squashed through a
saturating transform 1 - exp(-x/k) and the pieces are combined as a weighted
sum. No positional or structural information is used.
"""
import math
import re

# --- cleanup: mojibake, HTML entities, truncation marker -----------------
_REPLACEMENTS = [
    ("â€œ", '"'),   # â€œ  -> left curly double quote
    ("â€", '"'),   # â€\x9d -> right curly double quote
    ("â€™", "'"),   # â€™ -> right curly apostrophe
    ("â€˜", "'"),   # â€˜ -> left curly apostrophe
    ("â€“", "-"),   # â€“ -> en dash
    ("â€”", "-"),   # â€” -> em dash
    ("â€¦", "..."), # â€¦ -> ellipsis
    ("â€", '"'),         # bare â€ remnant -> quote
    ("Â ", " "),         # Â + nbsp -> space
    ("Â ", " "),              # Â + space -> space
    ("Â", ""),                # stray Â
    (" ", " "),               # real nbsp
    ("&amp;", "&"), ("&gt;", ">"), ("&lt;", "<"),
    ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
    ("[...]", " "),                # truncation marker: not content
]


def _clean(text):
    for a, b in _REPLACEMENTS:
        text = text.replace(a, b)
    return text


# --- lexical detectors ----------------------------------------------------
_WORD = re.compile(r"[A-Za-z']+")
_NUMBER = re.compile(r"(?<![\w.])(?:\$\s?|€\s?|£\s?)?\d[\d,]*(?:\.\d+)?\s?%?")
_MONTH = re.compile(
    r"\b(?:january|february|march|april|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\b\.?",
    re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_QUOTE_MARK = re.compile(r'"')

# phrases that signal elaboration, background, or supporting specifics
_DETAIL_PHRASES = [
    "according to", "in addition", "additionally", "furthermore", "moreover",
    "for example", "for instance", "such as", "including", "as well as",
    "more than", "approximately", "an estimated", "compared to", "compared with",
    "as part of", "in order to", "designed to", "based in", "headquartered",
    "founded", "established in", "million", "billion", "percent",
    "background", "history", "previously", "since its", "over the past",
    "led by", "in partnership with", "in collaboration with",
]
_DETAIL_RE = re.compile("|".join(re.escape(p) for p in _DETAIL_PHRASES),
                        re.IGNORECASE)


def _sat(x, k):
    """Saturating map: 0 -> 0, ~k -> 0.63, large -> 1."""
    return 1.0 - math.exp(-float(x) / float(k))


def score(text: str) -> float:
    try:
        t = _clean("" if text is None else str(text))

        n_words = len(_WORD.findall(t))
        n_numbers = len(_NUMBER.findall(t))
        n_dates = len(_MONTH.findall(t)) + len(_YEAR.findall(t))
        n_phrases = len(_DETAIL_RE.findall(t))
        n_quotes = len(_QUOTE_MARK.findall(t)) // 2   # quote-mark pairs

        s = (0.35 * _sat(n_words, 500)     # sheer informative volume
             + 0.20 * _sat(n_numbers, 12)  # concrete figures
             + 0.15 * _sat(n_dates, 5)     # temporal specifics
             + 0.20 * _sat(n_phrases, 8)   # elaboration / context markers
             + 0.10 * _sat(n_quotes, 3))   # supporting quotations

        return float(max(0.0, min(1.0, s)))
    except Exception:
        return 0.5
