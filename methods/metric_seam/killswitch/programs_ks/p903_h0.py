"""p903 h0 -- Corpus distinctiveness, retrieval-evidence hybrid.

Criterion: the release's content is distinctive relative to the OTHER
documents in this collection -- not near-duplicate template/boilerplate
recycled across many similar announcements.

Why the baseline fails (train rho 0.183): distinctiveness is a
CORPUS-RELATIVE property. A single-document lexical program cannot see
whether this text is recycled across many scraped pages (publisher nav
chrome, terms-of-use blocks, state-by-state template releases). In the
train pack, judge-0 documents are exactly those with many near-copies in
the corpus (Coca-Cola nav pages, Moody's ToS, GlobeNewswire footers, the
CVS per-state "time delay safes" series), while judge-1 documents are
one-of-a-kind in the corpus even when they are boilerplate-heavy wire
releases or oddball pages (a unique Rite Aid error page scores 1.0).

Predicate (kept in code): distinctiveness = strictly decreasing function
of TF-IDF similarity to the document's nearest corpus neighbours, via the
EVIDENCE op ops.retrieve_similar. IDF naturally discounts corpus-wide
wire-service chrome (PR Newswire/Cision footers shared by hundreds of
docs) while preserving similarity within publisher template families --
matching the judge, who tolerates wire chrome on distinctive releases.

Implementation notes:
- RAW text is passed to retrieve_similar so that the self-hit (the doc
  itself, similarity ~1.0) stays intact and can be dropped reliably.
  Exactly ONE hit with similarity >= SELF_SIM_THRESHOLD is dropped; a
  second ~1.0 hit is a genuine exact duplicate and correctly drives the
  score to ~0.
- s_eff blends nearest-neighbour similarity (near-duplicate detection)
  with the mean of the top-3 (recycled across MANY announcements).
- The logistic map is strictly monotone (no clipping ties); calibration
  constants do not affect Spearman, only readability of the scores.
- No LLM_FIELDS: a per-document extractor cannot observe corpus
  frequency, so no extraction earns its keep here.
- Never raises: any failure falls back to 0.5.
"""

import math

LLM_FIELDS = {}

# A hit at least this similar is assumed to be the document itself
# (retrieve_similar is called with exclude_id=None; score() never sees
# the datapoint_id). Only the single best such hit is discarded.
_SELF_SIM_THRESHOLD = 0.995

# Blend weights: nearest neighbour vs mean of top-3 neighbours.
_W_MAX = 0.6
_W_MEAN3 = 0.4

# Strictly decreasing logistic map from effective similarity to score.
_MID = 0.42   # similarity at which score = 0.5
_SCALE = 0.11  # softness of the transition


def _neighbor_sims(text, ops):
    """Return non-self neighbour similarities, sorted descending, or None."""
    hits = None
    try:
        hits = ops.retrieve_similar(text, k=6, exclude_id=None)
    except Exception:
        try:
            hits = ops.retrieve_similar(text)
        except Exception:
            return None
    if not hits:
        return None
    sims = []
    for h in hits:
        try:
            s = float(h[0])
        except Exception:
            continue
        if math.isnan(s) or math.isinf(s):
            continue
        # Clamp defensively; TF-IDF cosine should already be in [0, 1].
        sims.append(min(max(s, 0.0), 1.0))
    if not sims:
        return None
    sims.sort(reverse=True)
    # Drop exactly one presumed self-hit.
    if sims[0] >= _SELF_SIM_THRESHOLD:
        sims = sims[1:]
    if not sims:
        # Retrieval worked and found ONLY the document itself: no other
        # corpus doc shares meaningful TF-IDF overlap -> maximally
        # distinctive, not missing evidence.
        return [0.0]
    return sims


def score(text, extracted, ops):
    try:
        if not text or not isinstance(text, str) or not text.strip():
            return 0.5
        # Guard against degenerate near-empty scrapes.
        if len(text.strip()) < 40:
            return 0.5

        sims = _neighbor_sims(text, ops)
        if sims is None:
            return 0.5  # no corpus evidence available -> neutral

        s1 = sims[0]
        top3 = sims[:3]
        mean3 = sum(top3) / len(top3)
        s_eff = _W_MAX * s1 + _W_MEAN3 * mean3

        # Strictly decreasing, never saturating to exact 0/1 ties.
        val = 1.0 / (1.0 + math.exp((s_eff - _MID) / _SCALE))
        return min(max(val, 0.0), 1.0)
    except Exception:
        return 0.5
