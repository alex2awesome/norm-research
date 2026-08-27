"""p905_v0_keyword -- Authentic authorship, surface/lexical heuristic.

Criterion: the release reads as written by someone with genuine familiarity
with the company and subject matter (concrete, specific, coherent) rather
than generic template marketing language.

Approach (lexical only): reward the density of concrete-specificity markers
(money amounts, percentages, quantities with units, full dates, stock
tickers, named-person attributions, model/product codes) and penalize the
density of generic marketing buzzwords/cliches ("world-class",
"industry-leading", "seamless", ...). Densities are computed per 1000 words,
passed through saturating transforms, and combined around a 0.5 midpoint.
"""

import re

# ---------------------------------------------------------------------------
# Mojibake / HTML-entity normalization (scraped-corpus hazard)
# ---------------------------------------------------------------------------
_MOJI = [
    ("â€œ", '"'),   # curly left double quote
    ("â€", '"'),   # curly right double quote
    ("â€", "'"),   # curly apostrophe
    ("â€", "'"),   # curly left single quote
    ("â€“", "-"),   # en dash
    ("â€”", "-"),   # em dash
    ("â€¦", "..."),  # ellipsis
    ("â€¢", "*"),   # bullet
    ("Â ", " "),          # non-breaking space
    ("Â", ""),                 # stray A-circumflex
    ("&amp;", "&"), ("&nbsp;", " "), ("&quot;", '"'),
    ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">"),
]
_MOJI_LEFTOVER = re.compile("â€.")  # any remaining a-hat euro pair


def _normalize(text):
    for bad, good in _MOJI:
        text = text.replace(bad, good)
    text = _MOJI_LEFTOVER.sub("'", text)
    text = text.replace("[...]", " ")  # truncation marker
    return text


# ---------------------------------------------------------------------------
# Generic template-marketing buzzwords / cliches (penalty)
# ---------------------------------------------------------------------------
_BUZZ = re.compile(
    r"""\b(?:
        world[-\s]class | industry[-\s]leading | cutting[-\s]edge |
        state[-\s]of[-\s]the[-\s]art | best[-\s]in[-\s]class |
        market[-\s]leading | best[-\s]of[-\s]breed |
        leading\s+(?:provider|supplier|manufacturer|producer|developer|innovator)s? |
        (?:global|world)\s+leader | premier\s+(?:provider|supplier|source) |
        trusted\s+partner | one[-\s]stop(?:\s+shop)? | turnkey |
        seamless(?:ly)? | synerg(?:y|ies|istic) | leverag(?:e|es|ing) |
        empower(?:s|ing|ment)? | unlock(?:s|ing)? |
        unparalleled | unrivall?ed | unmatched | second\s+to\s+none |
        next[-\s]generation | game[-\s]chang(?:er|ing) |
        revolutioniz\w* | revolutionary | disruptive | paradigm |
        innovative\s+solutions? | holistic | value[-\s]added |
        mission[-\s]critical | end[-\s]to[-\s]end | customer[-\s]centric |
        results[-\s]driven | solutions?\s+provider |
        wide\s+(?:range|variety|array)\s+of | broad\s+(?:range|variety|array)\s+of |
        full\s+(?:suite|range)\s+of | comprehensive\s+(?:suite|range|portfolio) |
        tailored\s+solutions? | exceed\w*\s+(?:your\s+)?expectations |
        commitment\s+to\s+excellence | passion(?:ate)?\s+(?:about|for) |
        dedicated\s+to\s+(?:providing|delivering|helping) |
        striv(?:e|es|ing)\s+to | committed\s+to\s+(?:providing|delivering|excellence)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Concrete-specificity markers (reward), with weights
# ---------------------------------------------------------------------------
_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|"
          r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)")
_SPEC = [
    (re.compile(r"[$€£¥]\s?\d"), 1.5),                       # money
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent)\b", re.I), 1.5),          # percentages
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:million|billion|trillion)\b", re.I), 1.5),
    (re.compile(r"\b\d+(?:[,.]\d+)*\s?(?:mg|kg|km|mm|cm|nm|ml|mw|gw|kwh?|mhz|ghz|"
                r"gb|tb|mph|acres|employees|patients|customers|stores|"
                r"countries|locations|units|square\s+feet)\b", re.I), 1.5),  # quantities+units
    (re.compile(_MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b"), 2.0),  # full dates
    (re.compile(r"\((?:NYSE|NASDAQ|AMEX|TSX[V]?|LSE|OTC\w*|Euronext)\s*:\s*[A-Z.\-]{1,8}\)"), 3.0),
    (re.compile(r"\bsaid\s+[A-Z][a-z]+\s+[A-Z][a-z]+"), 2.5),               # "said Jane Doe"
    (re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s*,\s*(?:the\s+)?(?:CEO|CFO|CTO|COO|"
                r"President|Chief|Vice\s+President|VP|Director|Founder|"
                r"Head\s+of|General\s+Manager)\b"), 2.5),                    # "Jane Doe, CEO"
    (re.compile(r"\b(?:Phase\s+(?:I{1,3}|[123])|FDA|patent(?:ed|s)?\s+No\.?|"
                r"ISO\s?\d{4,5})\b"), 1.5),                                  # regulatory/technical
    (re.compile(r"\b[A-Z]{2,}-?\d+\w*\b"), 0.75),                            # model/product codes
    (re.compile(r"\b(?:19|20)\d{2}\b"), 0.5),                                # bare years
]


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or len(text.strip()) < 80:
            return 0.5
        t = _normalize(text)

        words = re.findall(r"[A-Za-z']+", t)
        n_words = len(words)
        if n_words < 40:
            return 0.5

        spec = 0.0
        for pat, w in _SPEC:
            spec += w * len(pat.findall(t))
        buzz = float(len(_BUZZ.findall(t)))

        # densities per 1000 words, then saturating transforms
        spec_d = spec / n_words * 1000.0
        buzz_d = buzz / n_words * 1000.0
        s_sat = spec_d / (spec_d + 18.0)   # ~18 weighted hits/1000w -> 0.5
        b_sat = buzz_d / (buzz_d + 6.0)    # ~6 buzz hits/1000w -> 0.5

        val = 0.5 + 0.55 * s_sat - 0.55 * b_sat
        return max(0.0, min(1.0, float(val)))
    except Exception:
        return 0.5
