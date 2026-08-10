"""p907 v2 (holistic) — composite weak-signal score for 'Comprehensive detail'.

Criterion: the release provides substantial informative detail — background,
specifics, supporting facts, context. Fuller releases score high; terse stubs
score low.

Approach: combine eight weak signals, each mapped to [0,1], into a weighted
composite. Volume (log-scaled prose word count), prose share (running text vs
nav chrome), numeric-fact density, temporal specifics (distinct month/year
mentions), named-entity richness (distinct capitalized multi-word sequences),
quoted supporting material, elaboration connectives ("in addition",
"according to", ...), and lexical breadth (distinct word stems per sqrt of
tokens — fuller releases cover more ground). Signals are computed over
cleaned text (mojibake and HTML entities repaired, truncation marker '[...]'
removed) and, where relevant, restricted to prose-like lines so that link
lists and end-of-page contact boilerplate contribute little.
"""
import math
import re
from collections import Counter

# --- cleanup: mojibake, HTML entities, truncation marker -----------------
_E = "â€"                       # mojibake prefix 'â€'
_REPLACEMENTS = [
    (_E + "œ", '"'), (_E + "\x9d", '"'), (_E + "™", "'"),
    (_E + "˜", "'"), (_E + "“", "-"), (_E + "”", "-"),
    (_E + "¦", "..."), (_E, '"'),
    ("Â\xa0", " "), ("Â ", " "), ("Â", ""), ("\xa0", " "),
    ("&amp;", "&"), ("&gt;", ">"), ("&lt;", "<"),
    ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
    ("[...]", "\n"),
]


def _clean(text):
    for a, b in _REPLACEMENTS:
        text = text.replace(a, b)
    return text


_WORD = re.compile(r"[A-Za-z']+")
_SENT_END = re.compile(r"[.!?][\"')\]]?(?:\s|$)")
_NUMBER = re.compile(r"(?<![\w.])(?:\$\s?|€\s?|£\s?)?\d[\d,]*(?:\.\d+)?\s?%?")
_MONTH = re.compile(
    r"\b(?:january|february|march|april|june|july|august|september|"
    r"october|november|december)\b", re.IGNORECASE)  # 'may' excluded: modal verb
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_ENTITY = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_QUOTED = re.compile(r'"[^"\n]{25,400}"')
_CONNECT = re.compile(
    r"\b(?:in addition|additionally|furthermore|moreover|according to|"
    r"for example|for instance|such as|including|as well as|as part of|"
    r"more than|approximately|compared (?:to|with)|founded|headquartered|"
    r"based in|previously|since)\b", re.IGNORECASE)


def _prose_lines(t):
    out = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            continue
        n = len(_WORD.findall(s))
        if n >= 30 or (n >= 12 and _SENT_END.search(s)):
            out.append(s)
    return out


def _sat(x, k):
    return 1.0 - math.exp(-float(x) / float(k))


def _loglen(n, lo=80.0, hi=1600.0):
    """0 at <=lo words, 1 at >=hi words, log-linear between."""
    if n <= lo:
        return 0.0
    return min(1.0, (math.log(n) - math.log(lo)) / (math.log(hi) - math.log(lo)))


def score(text: str) -> float:
    try:
        t = _clean("" if text is None else str(text))
        all_words = [w.lower() for w in _WORD.findall(t)]
        n_all = len(all_words)
        if n_all == 0:
            return 0.0

        prose = _prose_lines(t)
        prose_text = "\n".join(prose)
        prose_words = _WORD.findall(prose_text)
        n_prose = len(prose_words)

        # 1) volume: log-scaled prose word count
        s_len = _loglen(n_prose)
        # 2) prose share: running text vs chrome
        s_share = n_prose / n_all
        # 3) numeric-fact density (per 100 prose words)
        dens = 100.0 * len(_NUMBER.findall(prose_text)) / max(1, n_prose)
        s_num = _sat(dens, 1.5)
        # 4) temporal specifics: distinct month names + years
        n_time = (len({m.group(0).lower() for m in _MONTH.finditer(t)})
                  + len({m.group(0) for m in _YEAR.finditer(t)}))
        s_time = _sat(n_time, 3)
        # 5) named-entity richness: distinct capitalized multi-word runs
        ents = {m.group(0) for m in _ENTITY.finditer(prose_text)}
        s_ent = _sat(len(ents), 8)
        # 6) quoted supporting material
        s_quote = _sat(len(_QUOTED.findall(prose_text)), 2)
        # 7) elaboration connectives
        s_conn = _sat(len(_CONNECT.findall(prose_text)), 6)
        # 8) lexical breadth: distinct stems per sqrt tokens (length-robust)
        stems = Counter(w[:6] for w in prose_words if len(w) > 2)
        breadth = len(stems) / math.sqrt(max(1, n_prose))
        s_vocab = max(0.0, min(1.0, (breadth - 4.0) / 8.0))

        s = (0.22 * s_len + 0.12 * s_share + 0.14 * s_num + 0.10 * s_time
             + 0.12 * s_ent + 0.09 * s_quote + 0.11 * s_conn + 0.10 * s_vocab)
        return float(max(0.0, min(1.0, s)))
    except Exception:
        return 0.5
