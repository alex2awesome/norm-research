"""Hybrid metric channel for a180: Earned payoff and climax.

Construct: climactic payoffs that follow from prior setup/causality and feel
deserved. This is tacit; keyword proxies are near-zero (baseline rho=0.062).
Strategy: delegate the tacit judgment to one compact categorical LLM field,
plus one evidence field (a quote of the setup detail the ending pays off)
that CODE VERIFIES against the early text. Code keeps the predicate:
band mapping, quote verification, author-meta penalties, ending-shape
signals (punchy landing vs. trailing-off), and a lexical callback backstop.
"""

import re
from collections import Counter

LLM_FIELDS = {
    "climax_earned": (
        "Does the ending deliver a climactic payoff that earlier setup earns? "
        "Answer exactly one word: EARNED, PARTIAL, UNEARNED, or NONE."
    ),
    "setup_callback": (
        "Quote under 12 words from the story's first half that the ending "
        "pays off or calls back to; else NONE."
    ),
}

# --- code-side pattern banks -------------------------------------------------

# Author self-deprecation / apology meta-notes: strong low-quality marker
# (present in 0.0-band: "sorry for bad formatting", "first time poster",
# "Don't tear me apart pls", "Still crap", "wrote this on my phone").
_APOLOGY_PATS = [
    r"sorry for (?:the |my )?(?:bad|poor|terrible|awful|crappy)",
    r"first time post",
    r"long time lurker",
    r"don'?t tear me apart",
    r"wrote this on my phone",
    r"isn'?t going to be great",
    r"still crap",
    r"plot holes,? i know",
    r"sorry for (?:bad )?formatting",
    r"been a (?:few|couple(?: of)?) years since i'?ve written",
]

# Self-promo / gratitude meta-notes: mild marker (appears in 0.3-band; a bare
# signature link also appears in one 0.8 story, so keep this penalty light).
_PROMO_PATS = [
    r"thank(?:s| you) for the (?:gold|gild)",
    r"my first (?:ever )?gild",
    r"feel free to check out",
    r"check out r/",
    r"if you enjoyed my (?:story|writing)",
    r"posting more of my writing",
    r"any support helps",
]

# Lines to strip from the tail before analyzing the true ending.
_JUNK_LINE_PAT = re.compile(
    r"(?:^\s*edit\s*\d*\s*:)|(?:^\s*\**\s*part \d+)|(?:r/\w+)|(?:https?://)"
    r"|(?:thank(?:s| you))|(?:sorry for)|(?:\[wp\])",
    re.IGNORECASE,
)

_STOP = frozenset(
    "the and that with this from were their there would could should about "
    "which when where what while have been they them then than into onto "
    "your his her its our was are had has did does not you all one two "
    "just like over under after before because through still very much "
    "some more most other another every each said says asked replied "
    "back down here even only never always people going think know "
    "began around toward looked seemed little things thing".split()
)


def _words(s):
    return re.findall(r"[a-z']+", s.lower())


def _content_tokens(s, min_len=4):
    return [w for w in _words(s) if len(w) >= min_len and w not in _STOP]


def _parse_band(answer):
    """Map the climax_earned answer to a base value. Priority order matters:
    'unearned' contains 'earned', so test it first. The harness converts an
    extractor answer of NONE to "", so "" means 'no climax found' (low),
    not 'abstained'."""
    try:
        a = str(answer or "").lower()
    except Exception:
        return 0.5
    if not a.strip():
        return 0.15  # extractor said NONE: no climactic payoff found
    if re.search(r"\bunearned\b|\bnot\s+(?:\w+\s+)?earned\b", a):
        return 0.18
    if re.search(r"\bpartial", a):
        return 0.50
    if re.search(r"\bnone\b|\bno\b", a):
        return 0.12
    if re.search(r"\bearned\b|\byes\b", a):
        return 0.90
    return 0.5  # unparseable: neutral


def _verify_quote(quote, early_text):
    """Return payoff-evidence value: 1.0 if the quoted setup detail is
    actually present in the early text (lenient token match), 0.5 if the
    extractor asserted a callback we cannot verify, 0.0 if none claimed."""
    try:
        q = str(quote or "").strip()
    except Exception:
        return 0.0
    if not q or re.fullmatch(r"(?i)\W*none\W*", q):
        return 0.0
    toks = _content_tokens(q)
    if not toks:
        return 0.5
    early = set(_content_tokens(early_text))
    frac = sum(1 for t in toks if t in early) / float(len(toks))
    need = 0.5 if len(toks) >= 3 else 0.99
    return 1.0 if frac >= need else 0.5


def _is_junk_line(line):
    if not line:
        return True
    if not re.search(r"\w", line):
        return True  # separator-only lines: ---, ____, ***
    if re.fullmatch(r"(?:&(?:amp;)?nbsp;|\s)+", line):
        return True
    if len(line) < 220 and _JUNK_LINE_PAT.search(line):
        return True
    if len(line) < 220 and any(re.search(p, line.lower()) for p in _APOLOGY_PATS):
        return True
    return False


def _strip_tail_junk(text):
    """Drop trailing meta lines (edits, plugs, links, thank-yous, separators)
    so ending analysis sees the story's real final beat."""
    lines = text.rstrip().split("\n")
    while lines and _is_junk_line(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines)


def _ending_shape(text):
    """+ for a short punchy landed final beat; - for trailing-off endings."""
    body = _strip_tail_junk(text)
    if not body.strip():
        return 0.0
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if re.search(r"\w", p)]
    if not paras:
        return 0.0
    last = paras[-1]
    n_words = len(_words(last))
    adj = 0.0
    if re.search(r"(?:\.\.\.|…)\s*[\"'”’]?\s*$", last):
        adj -= 0.05  # trails off with ellipsis: unresolved
    elif not re.search(r"[.!?\"'”’]\s*$", last):
        adj -= 0.04  # no terminal punctuation: abandoned mid-beat
    elif n_words <= 14 and len(_words(body)) >= 120:
        adj += 0.04  # short punchy landing after a real story
    return adj


def _meta_penalty(text_lc):
    pen = 0.0
    for pat in _APOLOGY_PATS:
        if re.search(pat, text_lc):
            pen += 0.10
    for pat in _PROMO_PATS:
        if re.search(pat, text_lc):
            pen += 0.04
    return min(pen, 0.15)


def _lexical_callback(text):
    """Backstop when the LLM evidence field is empty: does a distinctive
    early token recur in the final segment (Chekhov's-gun surface echo)?"""
    toks = _content_tokens(text, min_len=5)
    if len(toks) < 80:
        return 0.0
    counts = Counter(toks)
    common = {w for w, _ in counts.most_common(25)}
    head = set(toks[: max(1, len(toks) // 4)])
    tail = set(toks[-max(1, len(toks) // 8):])
    echo = [w for w in head & tail if w not in common and counts[w] <= 4]
    return 0.03 if echo else 0.0


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        try:
            t = ops.normalize(text)
            if not isinstance(t, str) or not t.strip():
                t = text
        except Exception:
            t = text

        band = _parse_band(extracted.get("climax_earned", ""))

        w = _words(t)
        cut = int(len(w) * 0.6)
        early_text = " ".join(w[:cut]) if cut > 0 else t
        evidence = _verify_quote(extracted.get("setup_callback", ""), early_text)
        if evidence == 0.0 and _lexical_callback(t) > 0.0:
            evidence = 0.35  # code-side surface-echo backstop

        s = 0.16 + 0.55 * band + 0.16 * evidence
        s += _ending_shape(t)
        s -= _meta_penalty(t.lower())
        if len(w) < 120:
            s -= 0.03  # joke-length posts rarely build an earned climax

        return float(max(0.0, min(1.0, s)))
    except Exception:
        return 0.5
