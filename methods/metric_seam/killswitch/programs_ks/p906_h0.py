"""p906 h0 -- Persuasive cadence (hybrid channel, deliberately simple).

Criterion: prose rhythm builds momentum across paragraphs; sentence-length
variation, pacing, and transitions sustain attention from headline through
closing boilerplate.

DESIGN NOTE (read before "improving" this): the pack reports
judge_reliability = -0.019 and attenuation_ceiling = 0.0 for this criterion.
The judge channel is statistically indistinguishable from noise, and the 30
train examples contain near-duplicate documents with opposite labels (two
S&P Market Intelligence article pages at 0.15 vs 0.90; nav-chrome pages at
0.15 vs 0.95). There is no learnable residual: any complex fit to train is
overfitting to noise. This module therefore keeps a small, criterion-faithful
cadence composite computed on the PROSE portion of the document only:

  s_var    sentence-length coefficient of variation inside a healthy band
  s_alt    long/short alternation between adjacent sentences
  s_trans  transition/connective density per sentence (saturating)
  s_punct  rhythm punctuation (commas/semicolons/colons/dashes) per sentence

damped by the prose fraction (nav/chrome-heavy scrapes cannot exhibit
persuasive cadence). No LLM fields are declared: with a noise-level judge
channel, extraction cost buys nothing.

score(text, extracted, ops) -> float in [0.0, 1.0]; deterministic;
returns 0.5 on any unexpected error.
"""

import re
import math
import statistics

LLM_FIELDS = {}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_RHYTHM_PUNCT_RE = re.compile(r"[,;:]|\s-\s|--|–|—")

_TRANSITION_RE = re.compile(
    r"\b(?:however|moreover|furthermore|additionally|consequently|therefore"
    r"|meanwhile|ultimately|finally|notably|importantly|indeed|similarly"
    r"|likewise|nonetheless|nevertheless|in addition|as a result"
    r"|at the same time|in turn|for example|for instance|in fact"
    r"|building on|looking ahead|going forward|to that end|this means"
    r"|since then|first|second|third|next|then)\b",
    re.IGNORECASE,
)

# Lines that are almost certainly site chrome / navigation, not prose.
_CHROME_LINE_RE = re.compile(
    r"^\s*(?:home|about(?:\s+us)?|contact(?:\s+us)?|news(?:room)?|products?"
    r"|services|solutions|pricing|blog|careers|events|resources|support"
    r"|sign\s?in|log\s?in|register|subscribe|search|menu|skip to (?:main )?content"
    r"|privacy policy|terms(?:\s+of\s+(?:use|service))?|cookie[s]?(?:\s+\w+)?"
    r"|follow us|share(?:\s+this)?|site map|©.*|copyright.*)\s*$",
    re.IGNORECASE,
)


def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _band(x, lo, hi, soft):
    """1.0 inside [lo, hi], decaying linearly to 0 over `soft` outside."""
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return _clamp01(1.0 - d / max(soft, 1e-9))


def _prose_lines(text):
    """Split into lines; classify each as prose or chrome.

    Prose line: reasonably long, ends (or internally contains) sentence
    punctuation, has a decent lowercase-word share. Chrome line: short,
    unpunctuated, nav-vocabulary, or link-list shaped.
    """
    prose, prose_words, total_words = [], 0, 0
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        words = _WORD_RE.findall(line)
        total_words += len(words)
        if len(words) < 6:
            continue
        if _CHROME_LINE_RE.match(line):
            continue
        # Link-list / menu shape: many words, almost no sentence punctuation
        # AND high fraction of capitalized tokens.
        caps = sum(1 for w in words if w[:1].isupper())
        has_sent_punct = bool(re.search(r"[.!?]", line))
        if not has_sent_punct and caps / max(len(words), 1) > 0.6:
            continue
        prose.append(line)
        prose_words += len(words)
    frac = prose_words / max(total_words, 1)
    return prose, frac


def _sentences(prose_text):
    sents = []
    for chunk in _SENT_SPLIT_RE.split(prose_text):
        n = len(_WORD_RE.findall(chunk))
        if n >= 3:
            sents.append((chunk, n))
    return sents


def score(text, extracted, ops):
    try:
        if not text or not isinstance(text, str):
            return 0.5
        try:
            norm = ops.normalize(text)
            if not isinstance(norm, str) or not norm.strip():
                norm = text
        except Exception:
            norm = text

        prose, prose_frac = _prose_lines(norm)
        body = " ".join(prose)
        sents = _sentences(body)

        # Tiny or chrome-only pages: no cadence to speak of.
        if len(sents) < 5:
            return _clamp01(0.15 + 0.2 * prose_frac)

        lengths = [n for _, n in sents]

        # s_var: sentence-length variation, healthy CV band.
        mean_len = statistics.fmean(lengths)
        cv = (statistics.pstdev(lengths) / mean_len) if mean_len > 0 else 0.0
        s_var = _band(cv, 0.35, 0.75, 0.35)

        # s_alt: adjacent long/short alternation around the median.
        med = statistics.median(lengths)
        signs = [1 if n > med else -1 for n in lengths]
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        s_alt = _clamp01(flips / max(len(lengths) - 1, 1) / 0.6)

        # s_trans: connective density per sentence, saturating at ~0.35.
        n_trans = len(_TRANSITION_RE.findall(body))
        s_trans = _clamp01((n_trans / len(sents)) / 0.35)

        # s_punct: rhythm punctuation per sentence, healthy band ~0.8-2.5.
        n_punct = len(_RHYTHM_PUNCT_RE.findall(body))
        s_punct = _band(n_punct / len(sents), 0.8, 2.5, 1.5)

        # s_len: readable mean sentence length band (10-28 words).
        s_len = _band(mean_len, 10.0, 28.0, 12.0)

        base = (0.28 * s_var + 0.17 * s_alt + 0.20 * s_trans
                + 0.17 * s_punct + 0.18 * s_len)

        # Damp by prose fraction: chrome-heavy scrapes can't carry cadence.
        damp = 0.35 + 0.65 * _clamp01(prose_frac / 0.6)
        return _clamp01(base * damp)
    except Exception:
        return 0.5
