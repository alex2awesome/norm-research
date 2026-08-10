"""p903 v2 -- Corpus distinctiveness, composite of weak signals.

Criterion: the release is distinctive relative to the collection (not
near-duplicate template/boilerplate recycled across many announcements).
Single-document proxy blending seven weak signals:
  1. cliche/boilerplate phrase density (dominant; recycled wire language)
  2. template-slot census (header, dateline/wire tag, About, contacts, FLS)
  3. concrete-specificity density (numbers/dates/mid-sentence proper nouns,
     with emails/urls/phones stripped first so contact chrome doesn't count)
  4. lexical diversity (root type-token ratio)
  5. hapax fraction (one-off vocabulary suggests one-off content)
  6. internal 4-gram repetition (self-repetition marks template stamping)
  7. sentence-length variation (formulaic templates are metronomic)
Signals 4-7 are tiebreakers with small weights; a boilerplate-stuffed page
can still have diverse vocabulary, so cliches + slots carry the majority.

Contract: score(text) -> float in [0, 1]; deterministic; stdlib re/math/
statistics/collections only; returns 0.5 on empty input or unexpected error.
"""

import re
import math
import statistics
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

_PHRASES = (
    "for immediate release", "forward-looking statements",
    "forward looking statements", "safe harbor", "risks and uncertainties",
    "actual results may differ", "actual results could differ",
    "no obligation to update", "media contact", "press contact",
    "investor relations", "this press release", "global leader",
    "leading provider", "pleased to announce", "proud to announce",
    "excited to announce", "thrilled to announce", "today announced",
    "announced today", "strategic partnership", "strategic alliance",
    "uniquely positioned", "commitment to excellence", "testament to",
    "for more information", "to learn more", "headquartered in",
    "look forward to", "looks forward to", "dedicated to", "drive growth",
    "trusted partner", "wide range of", "track record of",
)
_WORDS = frozenset((
    "world-class", "state-of-the-art", "cutting-edge", "best-in-class",
    "industry-leading", "market-leading", "award-winning", "next-generation",
    "end-to-end", "game-changing", "mission-critical", "seamless",
    "seamlessly", "robust", "leverage", "leverages", "leveraging",
    "synergy", "synergies", "empower", "empowers", "empowering",
    "unparalleled", "unmatched", "unrivaled", "revolutionary",
    "revolutionize", "groundbreaking", "innovative", "turnkey",
))

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}")
_NUM_RE = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
_MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b")
_WIRE_RE = re.compile(
    r"prnewswire|business ?wire|globe ?newswire|accesswire|newsfile|"
    r"marketwired|einpresswire", re.IGNORECASE)
_TICKER_RE = re.compile(
    r"\((?:nasdaq|nyse|otc|otcqb|otcqx|tsx|tsxv|lse|asx|amex)[^)]{0,25}\)",
    re.IGNORECASE)
_ABOUT_RE = re.compile(r"\n\s*about\s+[A-Z0-9\"']", re.IGNORECASE)
_FLS_RE = re.compile(
    r"forward-?looking statements|safe harbor|"
    r"actual results (?:may|could) differ", re.IGNORECASE)
_ENDMARK_RE = re.compile(r"(?:^|\n)\s*(?:###|-\s*30\s*-|source[:\s])",
                         re.IGNORECASE)


def _normalize(t):
    for bad, good in _MOJIBAKE:
        t = t.replace(bad, good)
    return t


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _midsentence_propn(t):
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
        t = _normalize(text[:150000])
        low = re.sub(r"[ \t]+", " ", t.lower())
        words = re.findall(r"[a-z][a-z'-]*", low)
        n = len(words)
        if n < 5:
            return 0.5
        tlen = float(len(t)) or 1.0

        # 1. cliche density (dominant, inverted)
        hits = float(sum(low.count(p) for p in _PHRASES))
        wc = Counter(words)
        hits += sum(c for w, c in wc.items() if w in _WORDS)
        s_cliche = 1.0 - min(1.0, (100.0 * hits / n) / 5.0)

        # 2. template-slot census (inverted)
        slots = 0
        if "for immediate release" in low:
            slots += 1
        if _WIRE_RE.search(t) or _TICKER_RE.search(t):
            slots += 1
        m = _ABOUT_RE.search(t)
        if m and m.start() > 0.30 * tlen:
            slots += 1
        tail40 = t[int(0.60 * tlen):]
        if (_EMAIL_RE.search(tail40) and
                (_PHONE_RE.search(tail40) or
                 re.search(r"\bcontacts?\b", tail40, re.IGNORECASE))):
            slots += 1
        mf = _FLS_RE.search(t)
        if mf and mf.start() > 0.40 * tlen:
            slots += 1
        if _ENDMARK_RE.search(t[int(0.80 * tlen):]):
            slots += 1
        s_slots = 1.0 - min(1.0, slots / 5.0)

        # 3. concrete specificity (contact chrome stripped first)
        stripped = _EMAIL_RE.sub(" ", t)
        stripped = _URL_RE.sub(" ", stripped)
        stripped = _PHONE_RE.sub(" ", stripped)
        spec = (len(_NUM_RE.findall(stripped))
                + 0.5 * len(_MONTH_RE.findall(stripped.lower()))
                + 0.5 * _midsentence_propn(stripped))
        s_spec = min(1.0, (100.0 * spec / n) / 8.0)

        # 4. lexical diversity: root TTR
        rttr = len(set(words)) / math.sqrt(n)
        s_ttr = _clamp01((rttr - 3.5) / 5.0)

        # 5. hapax fraction
        hapax = sum(1 for c in wc.values() if c == 1) / float(len(wc))
        s_hap = _clamp01((hapax - 0.35) / 0.40)

        # 6. internal 4-gram repetition (inverted)
        grams = list(zip(words, words[1:], words[2:], words[3:]))[:20000]
        if len(grams) >= 8:
            gc = Counter(grams)
            rep = sum(c - 1 for c in gc.values() if c > 1) / float(len(grams))
            s_rep = 1.0 - min(1.0, rep / 0.12)
        else:
            s_rep = 0.5

        # 7. sentence-length variation (guard n<2 explicitly)
        sents = [s for s in re.split(r"[.!?]+\s", t)
                 if len(s.split()) >= 3]
        if len(sents) >= 3:
            lens = [float(len(s.split())) for s in sents]
            mu = statistics.fmean(lens)
            cv = (statistics.pstdev(lens) / mu) if mu > 0 else 0.0
            s_var = _clamp01(cv / 0.55)
        else:
            s_var = 0.5

        total = (0.30 * s_cliche + 0.25 * s_slots + 0.15 * s_spec
                 + 0.08 * s_ttr + 0.07 * s_hap + 0.07 * s_rep
                 + 0.08 * s_var)
        return _clamp01(total)
    except Exception:
        return 0.5
