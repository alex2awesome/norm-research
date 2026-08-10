"""Hybrid metric channel for a117: Causal coherence and earned outcomes.

Design rationale
----------------
h0 asked a single binary-flavored question ("does everything follow causally,
YES/PARTLY/NO?") plus a hunt for an explicit "uncaused turn". Both fields
collapsed to lenient defaults (YES / NONE) across every train residual cell
h0 got badly wrong -- the extractor is happy to certify LOCAL, sentence-to-
sentence logical flow ("she said X, so he did Y") even when the story's
actual RESOLUTION never depended on anything established earlier. That is
a different, harder-to-detect failure than an overt non-sequitur: a plan
that succeeds exactly as declared with no complication, a punchline that
resolves nothing because the piece was never building toward a resolution,
a coincidence/convenient memory-flood that unlocks the ending, a vignette
that stops before its own premise pays off, or a claimed causal link to a
real/famous event that is asserted rather than earned. Locally-coherent
sentences plus one of those patterns is exactly the shape of the h0 misses:
h0 always saw YES/"" and scored ~0.9-0.95 while the judge scored 0.2-0.3.

So h1 replaces the binary "is there a broken link" question with a graded,
harder-to-default question that forces the extractor to name its own
evidence rather than merely fail to find a violation:
  - `resolution_mode` asks HOW the ending arrives, with EARNED as only one
    of four options (vs. CONVENIENT / JOKE / UNRESOLVED) -- a lenient
    extractor now has to actively pick EARNED rather than just fail to spot
    a red flag.
  - `earned_link` asks the extractor to point to the SPECIFIC early detail
    the ending depends on. Requiring positive, concrete evidence for
    "earned" is much harder to satisfy by default than declaring the
    absence of an "uncaused turn" ever existed.
This is the same general causal-earned-ness construct as h0, not a set of
rules keyed to any excerpt; the code prior (length runway + ex-machina
marker density) is kept as a small hedge against extractor collapse, same
as h0, just down-weighted since it was never the source of the h0 misses.
"""

import re

LLM_FIELDS = {
    "resolution_mode": (
        "How does this story's ending/resolution actually arrive? Answer with "
        "exactly one word: EARNED (it grows out of complications and details "
        "established earlier), CONVENIENT (it relies on coincidence, luck, or a "
        "character's plan/guess working out exactly as declared with no real "
        "complication), JOKE (a punchline or gag with no real stakes being "
        "resolved), or UNRESOLVED (the piece stops or trails off before its own "
        "setup pays off)."
    ),
    "earned_link": (
        "Name ONE specific detail or event from the FIRST HALF of the story that "
        "the ending directly and necessarily depends on (not just precedes); if "
        "the ending would play out the same without anything established "
        "earlier, answer NONE."
    ),
}

_WORD = re.compile(r"[A-Za-z']+")
_NONE_LIKE = re.compile(r"^\W*(none|n/?a|nothing|no|unsure|unclear)\b", re.I)

# Ex-machina / rambling-narrator markers (density signal, per 1000 words) --
# kept from h0 as a small hedge, not the primary lever.
_LEAP_MARKERS = (
    "turns out", "turned out", "out of nowhere", "all of a sudden",
    "for some reason", "by pure accident", "by some absurd", "somehow",
    "against all logic", "against all odds", "wait,", "wait!", "woah", "whoa,",
)

_MODE_VALUES = {
    "unresolved": 0.30,
    "convenient": 0.32,
    "joke": 0.20,
    "earned": 0.92,
}
# Check order matters only in that none of these tokens are substrings of
# one another, so a simple first-match-wins scan over this order is safe.
_MODE_ORDER = ("unresolved", "convenient", "joke", "earned")


def _clamp(x):
    return max(0.0, min(1.0, x))


def _resolution_value(raw):
    """Map the resolution_mode answer to [0,1]; None if unusable."""
    raw = (raw or "").strip().lower()
    if not raw:
        return None
    for key in _MODE_ORDER:
        if key in raw:
            return _MODE_VALUES[key]
    return None


def _earned_link_value(raw):
    """Map the earned_link answer to [0,1]. Empty/NONE == no concrete link found."""
    raw = (raw or "").strip()
    if not raw or _NONE_LIKE.match(raw):
        return 0.20  # no positive evidence the ending depends on early setup
    if len(raw) < 3:
        return 0.20
    return 0.85  # extractor pointed to a specific, concrete earlier dependency


def score(text, extracted, ops):
    try:
        t = text or ""
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        words = _WORD.findall(t)
        nw = len(words)

        # ---- code prior: enough runway to build a causal chain, minus
        # ---- ex-machina marker density (non-sequitur signature). Small
        # ---- hedge only; kept from h0 since it was never implicated in
        # ---- the residual misses.
        ramp = _clamp((nw - 120) / 260.0)  # 0 at <=120 words, 1 at >=380
        tl = t.lower()
        leaps = sum(tl.count(mk) for mk in _LEAP_MARKERS)
        per_kw = leaps / max(1.0, nw / 1000.0)
        leak = _clamp(per_kw / 8.0)
        prior = _clamp(ramp * (1.0 - 0.6 * leak))

        ext = extracted or {}
        rv = _resolution_value(ext.get("resolution_mode", ""))
        el = _earned_link_value(ext.get("earned_link", ""))

        if rv is None:
            # Main field unusable: lean on the secondary field and the
            # code prior, same fallback shape as h0.
            return _clamp(0.35 * el + 0.45 * prior + 0.10)
        return _clamp(0.55 * rv + 0.20 * el + 0.25 * prior)
    except Exception:
        return 0.5
