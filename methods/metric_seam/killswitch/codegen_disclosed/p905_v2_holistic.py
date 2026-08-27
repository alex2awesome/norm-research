"""p905_v2_holistic -- Authentic authorship, composite of weak signals.

Criterion: the release reads as written by someone with genuine familiarity
with the company and subject matter -- concrete, specific, internally
coherent -- rather than generic template marketing language that could
describe any company.

Approach: combine seven weak signals, each mapped to [0,1]:
  1. specificity  - density of numbers, money, dates, tickers, units
  2. buzzword     - density of template-marketing cliches (penalty)
  3. prose share  - fraction of text in real sentences vs chrome/labels
  4. attribution  - named human quotes ("... said Jane Doe, CEO")
  5. coherence    - one dominant proper noun recurring across the document
  6. diversity    - type-token ratio over a fixed prose window
  7. rhythm       - sentence-length variation typical of authored prose
Weighted sum, clipped to [0,1].
"""

import re
import statistics
from collections import Counter

# --- mojibake / entity cleanup (scraped-corpus hazard) ----------------------
_MOJI = [
    ("â€œ", '"'), ("â€\x9d", '"'), ("â€\x99", "'"), ("â€\x98", "'"),
    ("â€“", "-"), ("â€”", "-"), ("â€¦", "..."), ("â€¢", "*"),
    ("Â ", " "), ("Â", ""),
    ("&amp;", "&"), ("&nbsp;", " "), ("&quot;", '"'),
    ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">"),
]
_MOJI_LEFTOVER = re.compile("â€.")

_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|"
          r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)")

_SPEC_PATS = [
    (re.compile(r"[$€£¥]\s?\d"), 1.5),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent)\b", re.I), 1.5),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:million|billion|trillion)\b", re.I), 1.5),
    (re.compile(_MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b"), 2.0),
    (re.compile(r"\((?:NYSE|NASDAQ|AMEX|TSX[V]?|LSE|OTC\w*|Euronext)\s*:\s*[A-Z.\-]{1,8}\)"), 3.0),
    (re.compile(r"\b\d+(?:[,.]\d+)*\s?(?:mg|kg|km|mm|cm|nm|ml|mw|gw|kwh?|mhz|ghz|"
                r"gb|tb|mph|acres|employees|patients|customers|stores|"
                r"countries|locations|units)\b", re.I), 1.5),
    (re.compile(r"\b(?:19|20)\d{2}\b"), 0.5),
]

_BUZZ = re.compile(
    r"""\b(?:
        world[-\s]class | industry[-\s]leading | cutting[-\s]edge |
        state[-\s]of[-\s]the[-\s]art | best[-\s]in[-\s]class | market[-\s]leading |
        leading\s+(?:provider|supplier|manufacturer|producer|developer|innovator)s? |
        (?:global|world)\s+leader | premier\s+(?:provider|supplier|source) |
        trusted\s+partner | one[-\s]stop | turnkey | seamless(?:ly)? |
        synerg(?:y|ies|istic) | leverag(?:e|es|ing) | empower(?:s|ing|ment)? |
        unlock(?:s|ing)? | unparalleled | unrivall?ed | unmatched |
        next[-\s]generation | game[-\s]chang(?:er|ing) | revolutioniz\w* |
        revolutionary | disruptive | innovative\s+solutions? | holistic |
        value[-\s]added | mission[-\s]critical | end[-\s]to[-\s]end |
        customer[-\s]centric | results[-\s]driven |
        wide\s+(?:range|variety|array)\s+of | broad\s+(?:range|variety|array)\s+of |
        full\s+(?:suite|range)\s+of | comprehensive\s+(?:suite|range|portfolio) |
        tailored\s+solutions? | exceed\w*\s+expectations |
        commitment\s+to\s+excellence | passion(?:ate)?\s+(?:about|for) |
        dedicated\s+to\s+(?:providing|delivering|helping) | striv(?:e|es|ing)\s+to
    )\b""",
    re.IGNORECASE | re.VERBOSE)

_ATTR = re.compile(
    r'["\']\s*,?\s*said\b'
    r"|\bsaid\s+[A-Z][a-z]+\s+[A-Z][a-z]+"
    r"|\b[A-Z][a-z]+\s+[A-Z][a-z]+\s*,\s*(?:the\s+)?(?:CEO|CFO|CTO|COO|President|"
    r"Chief|Vice\s+President|VP|Director|Founder|Head\s+of|General\s+Manager)\b")

_STOP_CAPS = {"The", "This", "That", "These", "Those", "And", "But", "For",
              "With", "From", "About", "Our", "Your", "Their", "New", "More",
              "All", "Are", "Was", "Has", "Have", "Will", "Can", "May", "Not",
              "One", "Two", "How", "What", "When", "Where", "Why", "Who",
              "Home", "Contact", "News", "Login", "Search", "Read", "Skip",
              "Menu", "Privacy", "Terms", "Cookie", "Cookies", "Learn"}


def _normalize(text):
    for bad, good in _MOJI:
        text = text.replace(bad, good)
    text = _MOJI_LEFTOVER.sub("'", text)
    text = text.replace("[...]", "\n")
    return text


def _sat(x, k):
    """Saturating map: 0 -> 0, k -> 0.5, inf -> 1."""
    return x / (x + k) if x > 0 else 0.0


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or len(text.strip()) < 100:
            return 0.5
        t = _normalize(text)
        n = len(t)
        words = re.findall(r"[A-Za-z']+", t)
        n_words = len(words)
        if n_words < 40:
            return 0.5

        # 1. specificity density (weighted hits per 1000 words)
        spec_hits = 0.0
        for pat, w in _SPEC_PATS:
            spec_hits += w * len(pat.findall(t))
        s_spec = _sat(spec_hits / n_words * 1000.0, 18.0)

        # 2. buzzword density (penalty)
        s_buzz = _sat(len(_BUZZ.findall(t)) / n_words * 1000.0, 6.0)

        # 3. prose share: chars in sentence-like lines vs all nonblank chars
        prose_lines, total_c, prose_c = [], 0, 0
        for ln in t.split("\n"):
            s = ln.strip()
            if not s:
                continue
            total_c += len(s)
            if len(s.split()) >= 12 and re.search(r"[.!?]", s):
                prose_c += len(s)
                prose_lines.append(s)
        s_prose = prose_c / float(total_c) if total_c else 0.0

        # 4. named attribution (count-saturated)
        s_attr = _sat(float(len(_ATTR.findall(t))), 1.0)

        # 5. entity coherence: dominant proper noun spread across the doc
        cap_hits = [(m.start(), m.group(0)) for m in
                    re.finditer(r"\b[A-Z][A-Za-z]{2,}\b", t)
                    if m.group(0) not in _STOP_CAPS]
        s_coher = 0.0
        if cap_hits:
            c = Counter(w for _, w in cap_hits)
            top_word, top_n = max(sorted(c.items()), key=lambda kv: kv[1])
            if top_n >= 3:
                pos = [p for p, w in cap_hits if w == top_word]
                thirds = {int(3 * p / (n + 1)) for p in pos}
                s_coher = (min(top_n, 8) / 8.0) * (len(thirds) / 3.0)

        # 6. lexical diversity on a fixed prose window (length-controlled)
        prose_tokens = [w.lower() for w in
                        re.findall(r"[A-Za-z']+", " ".join(prose_lines))][:400]
        s_ttr = 0.0
        if len(prose_tokens) >= 100:
            ttr = len(set(prose_tokens)) / float(len(prose_tokens))
            # generic template text ~0.35, specific authored text ~0.55+
            s_ttr = max(0.0, min(1.0, (ttr - 0.30) / 0.30))

        # 7. sentence-length rhythm within prose
        sents = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(prose_lines))
                 if len(s.split()) >= 3]
        s_var = 0.0
        if len(sents) >= 4:
            s_var = min(1.0, statistics.pstdev([len(s.split()) for s in sents]) / 9.0)

        val = (0.04
               + 0.22 * s_spec
               + 0.20 * s_prose
               + 0.13 * s_attr
               + 0.13 * s_coher
               + 0.10 * s_ttr
               + 0.08 * s_var
               - 0.30 * s_buzz
               + 0.10)          # centering offset so mixed docs sit mid-scale
        return max(0.0, min(1.0, float(val)))
    except Exception:
        return 0.5
