"""Hybrid metric channel for a117: Causal coherence and earned outcomes
(BUDGET-4 extension).

Base design rationale (unchanged from h0): the v1_structure baseline
(connective + pronoun density) is near-zero (rho=0.045) because the judge
responds to whether LATER plot developments are SET UP by earlier ones
(earned twists, causal chains), not to surface connectives. That is a
comprehension-level construct, so the load-bearing signal comes from two
narrow LLM extraction fields (causal_verdict, uncaused_turn) with the
predicate kept in code; a small code prior (length ramp + ex-machina
marker density) breaks ties and hedges against extractor collapse.

Budget-4 extension (blind design -- no eval signal used, chosen on
construct grounds): the full criterion text is "Events proceed by clear
cause and effect with probable/necessary links; the throughline is logical
and resolutions arise from established causes -- no implausible leaps."
h0's causal_verdict is a single GLOBAL YES/PARTLY/NO read on "no
unexplained leaps" anywhere in the story, and uncaused_turn names at most
ONE unexplained turn; neither field distinguishes (a) whether the ENDING
specifically -- not just some mid-story beat -- is causally earned, which
is what "resolutions arise from established causes" singles out, or (b)
whether stated causal links are actually PROBABLE/NECESSARY (well-
motivated) versus merely PRESENT-BUT-CONTRIVED (a character acts a certain
way "because the plot needs it," which counts as "no leap" under
causal_verdict's coarse reading yet still violates the criterion's
"probable/necessary links" clause). Two new thin LLM fields target exactly
those two gaps:
  - resolution_setup: a direct read on whether the STORY'S ENDING
    specifically is caused by something established earlier, vs an
    unexplained coincidence/convenience -- the criterion's own singled-out
    case, distinct from a global mid-story leap check.
  - link_plausibility: whether the causal links driving the story, even
    when technically "explained," are well-motivated/believable rather
    than arbitrary or forced -- the "probable/necessary" clause, which is
    a strength-of-motivation judgment orthogonal to mere leap-detection.
Both fields degrade to a no-op when absent from `extracted` (budget < 4),
so this program reproduces h0's exact score whenever only the original 2
fields are served.
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
    "resolution_setup": (
        "Is the story's ENDING caused by something established earlier in the "
        "story, or by unexplained coincidence/convenience? Answer SETUP, "
        "COINCIDENCE, or UNCLEAR."
    ),
    "link_plausibility": (
        "Are the causal links between events believable and well-motivated, or "
        "arbitrary/forced even though technically explained? Answer PLAUSIBLE, "
        "FORCED, or NA."
    ),
}

_FIRST_WORD = re.compile(r"[^A-Za-z]*([A-Za-z]+)")
_PARTLY = re.compile(r"\b(partly|partially|somewhat|mostly|mixed|kind of|sort of)\b", re.I)
_YES = re.compile(r"\byes\b", re.I)
_NO = re.compile(r"\b(no|nope)\b", re.I)
_NONE_LIKE = re.compile(r"^\W*(none|n/?a|nothing|no)\b", re.I)

# Ex-machina / rambling-narrator markers (density signal, judged per 1000 words).
_LEAP_MARKERS = (
    "turns out", "turned out", "out of nowhere", "all of a sudden",
    "for some reason", "by pure accident", "by some absurd", "somehow",
    "wait,", "wait!", "woah", "whoa,",
)

_PARTLY_WORDS = {"partly", "partially", "somewhat", "mostly", "mixed"}


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
    # Fall back to whole-answer search; check hedges before polar terms.
    if _PARTLY.search(raw):
        return 0.55
    if _YES.search(raw) and not _NO.search(raw):
        return 0.95
    if _NO.search(raw) and not _YES.search(raw):
        return 0.05
    return None


def _uncaused_value(raw):
    """Map the uncaused_turn answer to [0,1]. Empty string == extractor said NONE."""
    raw = (raw or "").strip()
    if not raw or _NONE_LIKE.match(raw):
        return 0.9  # no unexplained turn found -> coherent
    return 0.15     # extractor named a concrete uncaused turn


def _budget4_adjustment(ext):
    """Bounded additive nudge from the two new budget-4 fields. Returns 0.0
    when both fields are absent (budget < 4), so callers degrade to h0
    behavior exactly."""
    res = (ext.get("resolution_setup", "") or "").strip().upper()
    link = (ext.get("link_plausibility", "") or "").strip().upper()

    adj = 0.0
    if res == "SETUP":
        adj += 0.08
    elif res == "COINCIDENCE":
        adj -= 0.15

    if link == "FORCED":
        adj -= 0.10
    elif link == "PLAUSIBLE":
        adj += 0.05

    return adj


def score(text, extracted, ops):
    try:
        t = text or ""
        try:
            t = ops.normalize(t)
        except Exception:
            pass
        words = re.findall(r"[A-Za-z']+", t)
        nw = len(words)

        # ---- code prior: enough runway to build a causal chain, minus
        # ---- ex-machina marker density (non-sequitur signature).
        ramp = _clamp((nw - 120) / 260.0)  # 0 at <=120 words, 1 at >=380
        tl = t.lower()
        leaps = sum(tl.count(mk) for mk in _LEAP_MARKERS)
        per_kw = leaps / max(1.0, nw / 1000.0)
        leak = _clamp(per_kw / 8.0)
        prior = _clamp(ramp * (1.0 - 0.6 * leak))

        ext = extracted or {}
        v = _verdict_value(ext.get("causal_verdict", ""))
        u = _uncaused_value(ext.get("uncaused_turn", ""))
        adj = _budget4_adjustment(ext)

        if v is None:
            # Extractor unusable on the main field: lean on the secondary
            # field and the code prior.
            return _clamp(0.35 * u + 0.45 * prior + 0.10 + adj)
        return _clamp(0.55 * v + 0.20 * u + 0.25 * prior + adj)
    except Exception:
        return 0.5
