"""p903 v0 -- Corpus distinctiveness, surface/lexical heuristic.

Criterion: the release is distinctive relative to the collection (not
near-duplicate template/boilerplate recycled across many announcements).
Single-document proxy: text stuffed with wire-service cliches and standard
boilerplate phrases ("global leader", "pleased to announce", forward-looking
statements, media-contact blocks) reads like thousands of other releases and
scores LOW; text with few cliches and concrete, specific detail (numbers,
dates, mid-sentence proper nouns) scores HIGH.

Contract: score(text) -> float in [0, 1]; deterministic; stdlib re/collections
only; returns 0.5 on empty input or unexpected error.
"""

import re
from collections import Counter

# Mojibake / typographic normalization (longest sequences first).
_MOJIBAKE = (
    ("â€œ", '"'),    # mangled left double quote
    ("â€", '"'),    # mangled right double quote (U+009D end)
    ("â€™", "'"),    # mangled apostrophe
    ("â€“", "-"),    # mangled en dash
    ("â€”", "-"),    # mangled em dash
    ("â€¦", "..."),  # mangled ellipsis
    ("â€", '"'),          # leftover pair (invisible byte stripped)
    ("Â ", " "),          # mangled non-breaking space
    ("Â", ""),                 # stray A-circumflex from mangled nbsp
    (" ", " "),                # real non-breaking space
    ("“", '"'), ("”", '"'),
    ("‘", "'"), ("’", "'"),
    ("–", "-"), ("—", "-"),
    ("…", "..."),
    ("[...]", " "),                 # corpus elision marker, not content
)

# Multi-word cliches / boilerplate phrases (matched on lowercased text).
_PHRASES = (
    ("for immediate release", 2.0),
    ("forward-looking statements", 2.0),
    ("forward looking statements", 2.0),
    ("safe harbor", 2.0),
    ("actual results may differ", 2.0),
    ("actual results could differ", 2.0),
    ("risks and uncertainties", 2.0),
    ("no obligation to update", 2.0),
    ("undue reliance", 2.0),
    ("media contact", 2.0),
    ("press contact", 2.0),
    ("investor relations", 2.0),
    ("this press release", 1.5),
    ("global leader", 1.5),
    ("world leader", 1.5),
    ("market leader", 1.5),
    ("industry leader", 1.5),
    ("leading provider", 1.5),
    ("leading supplier", 1.5),
    ("leading global", 1.5),
    ("premier provider", 1.5),
    ("solutions provider", 1.5),
    ("trusted partner", 1.5),
    ("pleased to announce", 1.5),
    ("proud to announce", 1.5),
    ("excited to announce", 1.5),
    ("thrilled to announce", 1.5),
    ("happy to announce", 1.5),
    ("innovative solutions", 1.5),
    ("innovative products", 1.5),
    ("for more information", 1.5),
    ("to learn more", 1.5),
    ("is pleased to", 1.0),
    ("is proud to", 1.0),
    ("is excited to", 1.0),
    ("announced today", 1.0),
    ("today announced", 1.0),
    ("strategic partnership", 1.0),
    ("strategic alliance", 1.0),
    ("uniquely positioned", 1.0),
    ("well positioned to", 1.0),
    ("poised to", 1.0),
    ("committed to providing", 1.0),
    ("commitment to excellence", 1.0),
    ("underscores our commitment", 1.0),
    ("testament to", 1.0),
    ("state of the art", 1.0),
    ("best in class", 1.0),
    ("one-stop shop", 1.0),
    ("game changer", 1.0),
    ("drive growth", 1.0),
    ("driving growth", 1.0),
    ("unlock value", 1.0),
    ("core competencies", 1.0),
    ("proven track record", 1.0),
    ("track record of", 1.0),
    ("look forward to", 1.0),
    ("looks forward to", 1.0),
    ("looking forward to", 1.0),
    ("learn more about", 1.0),
    ("headquartered in", 1.0),
    ("wholly owned subsidiary", 1.0),
    ("customary closing conditions", 1.0),
    ("meet the growing demand", 1.0),
    ("growing demand for", 1.0),
    ("broad portfolio", 1.0),
    ("comprehensive suite", 1.0),
    ("suite of solutions", 1.0),
    ("wide range of", 1.0),
    ("delighted to", 1.0),
    ("honored to", 1.0),
    ("passionate about", 1.0),
    ("dedicated to", 1.0),
    ("across the globe", 1.0),
    ("around the globe", 1.0),
    ("about us", 1.0),
)

# Single-token cliches (hyphenated compounds tokenize as one token).
_WORDS = (
    ("world-class", 1.5), ("state-of-the-art", 1.5), ("cutting-edge", 1.5),
    ("best-in-class", 1.5), ("industry-leading", 1.5), ("market-leading", 1.5),
    ("next-generation", 1.0), ("award-winning", 1.0), ("end-to-end", 1.0),
    ("value-added", 1.0), ("customer-centric", 1.0), ("game-changing", 1.0),
    ("mission-critical", 1.0), ("well-positioned", 1.0),
    ("seamless", 1.0), ("seamlessly", 1.0), ("robust", 1.0),
    ("leverage", 1.0), ("leverages", 1.0), ("leveraging", 1.0),
    ("synergy", 1.0), ("synergies", 1.0),
    ("empower", 1.0), ("empowers", 1.0), ("empowering", 1.0),
    ("unparalleled", 1.0), ("unmatched", 1.0), ("unrivaled", 1.0),
    ("revolutionary", 1.0), ("revolutionize", 1.0), ("groundbreaking", 1.0),
    ("turnkey", 1.0),
    ("innovative", 0.5), ("innovation", 0.5), ("flagship", 0.5),
    ("renowned", 0.5), ("stakeholders", 0.5), ("scalable", 0.5),
    ("holistic", 0.5), ("impactful", 0.5),
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}")
_NUM_RE = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
_MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b")


def _normalize(t):
    for bad, good in _MOJIBAKE:
        t = t.replace(bad, good)
    return t


def _midsentence_propn(t):
    """Capitalized words that are not sentence-initial, counted only on
    mixed-case lines (ALL-CAPS header lines like FOR IMMEDIATE RELEASE are
    formatting, not specificity)."""
    count = 0
    for line in t.split("\n"):
        s = line.strip()
        if not s or not re.search(r"[a-z]", s):
            continue
        for sent in re.split(r"[.!?;:]\s+", s):
            toks = re.findall(r"[A-Za-z][A-Za-z'-]*", sent)
            for tk in toks[1:]:
                if re.match(r"^[A-Z][a-z]", tk):
                    count += 1
    return count


def score(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        t = _normalize(text[:200000])
        low = re.sub(r"[ \t]+", " ", t.lower())
        words = re.findall(r"[a-z][a-z'-]*", low)
        n = len(words)
        if n < 5:
            return 0.5

        # --- boilerplate / cliche load ---
        cliche = 0.0
        for phrase, w in _PHRASES:
            c = low.count(phrase)
            if c:
                cliche += w * c
        wc = Counter(words)
        for word, w in _WORDS:
            c = wc.get(word, 0)
            if c:
                cliche += w * c
        boiler = min(1.0, (100.0 * cliche / n) / 5.0)

        # --- specificity (contact chrome stripped so phones/emails/urls
        #     in boilerplate blocks don't count as concrete detail) ---
        stripped = _EMAIL_RE.sub(" ", t)
        stripped = _URL_RE.sub(" ", stripped)
        stripped = _PHONE_RE.sub(" ", stripped)
        n_nums = len(_NUM_RE.findall(stripped))
        n_months = len(_MONTH_RE.findall(stripped.lower()))
        n_propn = _midsentence_propn(stripped)
        spec_count = n_nums + 0.5 * n_months + 0.5 * n_propn
        s_spec = min(1.0, (100.0 * spec_count / n) / 8.0)

        raw = 0.18 + 0.64 * (1.0 - boiler) + 0.18 * s_spec
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
