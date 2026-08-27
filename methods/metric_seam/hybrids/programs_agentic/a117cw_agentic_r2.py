"""Hybrid metric channel for a117: Causal coherence and earned outcomes. (r2)

Design rationale
----------------
Quantitative diagnosis (train, n=150) before writing this candidate:
  - causal_verdict ALONE (h0's primary field, mapped NO/PARTLY/YES -> 0.05/0.55/0.95)
    already reproduces rho ~0.56 of the judge's ranking on its own; it is doing almost
    all of h0's work (h0 full = 0.573). causal_verdict=YES lands 105/150 (70%) of train
    items in one bucket, and within that bucket judge scores still span the FULL 0.0-1.0
    range. A wide sweep for code-only signal inside that bucket (length, paragraph/
    sentence structure, dialogue density, lexical callback between setup and payoff,
    temporal-jump markers, reflection language, quote-balance, TF-IDF corpus similarity)
    found essentially nothing above rho~0.18 -- a first attempt at fixing this (a117_h1.py,
    which replaced BOTH h0 fields with resolution_mode/earned_link) tanked train rho to
    0.19 because brand-new field names are served as None during this harness's TRAIN
    iteration, and the fallback path was too weak. A second attempt here (keeping only
    causal_verdict + a new field) also underperformed h0 once actually measured through
    this harness, and a further check showed that breaking causal_verdict's large tied
    blocks with any of the WEAK code signals above (even lexicographically, never
    crossing a verdict bucket) tends to cost pooled Spearman rather than gain it -- with
    heavy ties, average-rank already acts as a good "no-information" hedge, and a noisy
    tiebreak signal can do worse than that hedge in aggregate.
  - h0's second field, uncaused_turn, is real (cached, verifiable here) and contributes
    real incremental signal: it is only ~+0.02 rho on top of causal_verdict alone under
    h0's original mapping, but two refinements to it verified in this diagnosis push
    higher: (a) when the extractor's named "uncaused_turn" uses "sudden(ly)" framing, the
    judge score is reliably worse (mean 0.27, n=10) than when it names a turn without that
    framing (mean 0.35, n=48) -- the extractor's own wording distinguishes a flagged
    non-sequitur from a flagged-but-arguably-earned reveal; (b) for many high-judge
    absurdist/gag pieces the "uncaused turn" the extractor names IS the story's earned
    punchline ("the cat is actually a hoover" -> judge 1.0; "crabs seeded life in the
    galaxy" -> judge 1.0) rather than a coherence violation, so a flat penalty for ANY
    named turn is too blunt; the "sudden" framing split is a real, code-cheap way to
    separate these two cases using only the text already returned by this field.

  So this candidate keeps BOTH of h0's real fields (causal_verdict, uncaused_turn) --
  reweighted toward a jointly-optimized point found by grid search on this diagnosis
  (verdict still dominant, turn and a modest code prior contributing) -- and layers in
  the "sudden"-framing split on uncaused_turn's own text as the one new discriminating
  signal, since it is verifiable on THIS harness right now (not a bet on a field that
  arrives as None). This is deliberately the conservative "add one verified signal to
  real code structure" move rather than a repeat of h1/the first r2 draft's higher-
  variance "swap in an unverified field" move.
"""

import re

LLM_FIELDS = {
    "causal_verdict": (
        "Do the story's later events and ending follow causally from causes "
        "established earlier, with no unexplained leaps? Answer only YES, PARTLY, or NO."
    ),
    "uncaused_turn": (
        "Name one major plot turn AFTER the opening premise that happens with no "
        "prior cause or setup; if every turn is set up earlier, answer NONE."
    ),
}

_FIRST_WORD = re.compile(r"[^A-Za-z]*([A-Za-z]+)")
_PARTLY = re.compile(r"\b(partly|partially|somewhat|mostly|mixed|kind of|sort of)\b", re.I)
_YES = re.compile(r"\byes\b", re.I)
_NO = re.compile(r"\b(no|nope)\b", re.I)
_PARTLY_WORDS = {"partly", "partially", "somewhat", "mostly", "mixed"}
_WORD = re.compile(r"[A-Za-z']+")

# Ex-machina / rambling-narrator markers (density signal, per 1000 words). Widened
# from h0's list to help recover some of what dropping uncaused_turn costs in code.
_LEAP_MARKERS = (
    "turns out", "turned out", "out of nowhere", "all of a sudden",
    "for some reason", "by pure accident", "by some absurd", "somehow",
    "wait,", "wait!", "woah", "whoa,", "little did", "unbeknownst",
    "coincidentally", "by chance", "as luck would have it", "just so happened",
    "for no reason", "without warning", "out of the blue", "as if by magic",
)

_NONE_LIKE = re.compile(r"^\W*(none|n/?a|nothing|no)\b", re.I)
_SUDDEN = re.compile(r"\bsudden(ly)?\b", re.I)


def _clamp(x):
    return max(0.0, min(1.0, x))


def _verdict_value(raw):
    """Map the causal_verdict answer to [0,1]; None if unusable."""
    raw = (raw or "").strip()
    if not raw:
        return None
    m = _FIRST_WORD.match(raw)
    first = m.group(1).lower() if m else ""
    if first == "yes":
        return 0.95
    if first in ("no", "nope"):
        return 0.05
    if first in _PARTLY_WORDS:
        return 0.55
    if _PARTLY.search(raw):
        return 0.55
    if _YES.search(raw) and not _NO.search(raw):
        return 0.95
    if _NO.search(raw) and not _YES.search(raw):
        return 0.05
    return None


def _turn_value(raw):
    """Map the uncaused_turn answer to [0,1]. Empty/NONE == no violation found ->
    coherent. Non-empty is split by whether the extractor's OWN wording frames the
    turn as "sudden(ly)" -- verified on train to mark a worse subgroup (mean judge
    0.27 vs 0.35 for non-"sudden" named turns) -- vs a named turn without that framing,
    which is often actually the story's earned reveal/punchline rather than a flaw."""
    raw = (raw or "").strip()
    if not raw or _NONE_LIKE.match(raw):
        return 0.90
    if _SUDDEN.search(raw):
        return 0.10
    return 0.32


def _code_prior(t):
    """Length runway to build a causal chain, minus ex-machina marker density."""
    words = _WORD.findall(t)
    nw = len(words)
    ramp = _clamp((nw - 120) / 260.0)  # 0 at <=120 words, 1 at >=380
    tl = t.lower()
    leaps = sum(tl.count(mk) for mk in _LEAP_MARKERS)
    per_kw = leaps / max(1.0, nw / 1000.0)
    leak = _clamp(per_kw / 8.0)
    return _clamp(ramp * (1.0 - 0.65 * leak))


def score(text, extracted, ops):
    try:
        t = text or ""
        try:
            t = ops.normalize(t)
        except Exception:
            pass

        prior = _code_prior(t)

        ext = extracted or {}
        v = _verdict_value(ext.get("causal_verdict", ""))
        u = _turn_value(ext.get("uncaused_turn", ""))

        if v is None:
            # Extractor unusable on the main field: lean on the secondary field and
            # the code prior, same fallback shape as h0.
            return _clamp(0.35 * u + 0.45 * prior + 0.10)
        return _clamp(0.55 * v + 0.20 * u + 0.25 * prior)
    except Exception:
        return 0.5
