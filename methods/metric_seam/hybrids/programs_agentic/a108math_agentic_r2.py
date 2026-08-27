"""Hybrid metric channel for a108: "Novelty and creative contribution" (Math SE).

r2 revision of h0/h1.

Diagnosis, from directly analyzing all 148 cached TRAIN items (not just the
30-item pack excerpt) against the cached "creative_trick" field extraction:

  - Judge scores are extremely compressed: 76/148 train items score exactly
    0.0, mean judge given creative_trick present = 0.076, mean given absent
    = 0.015. Presence alone (any non-NONE extraction) already correlates at
    rho=0.48 with the judge ranking over the full population -- most of
    h0/h1's rho comes from this one bit, not from either program's depth
    modeling.
  - h1 tried to fix h0's "any named trick scores ~0.55+" overcrediting by
    adding a SECOND LLM field asking the extractor to verdict the named
    trick STANDARD vs UNUSUAL. Checked outputs/metric_seam_pilot/tasks/math/
    field_results.jsonl directly: that field ("technique_novelty") has ZERO
    cached records -- it was declared but never actually run through the
    extractor, so during agentic-compile TRAIN iteration `extracted` always
    serves it empty and h1's "unusual" branch is dead code on this signal;
    every present-trick item silently falls into h1's low-credit "other"
    bucket regardless of the item's true novelty. (h1 still beats h0 on
    TRAIN, 0.584 vs 0.573, purely because that low-credit bucket happens to
    be less inflated than h0's flat 0.55 weight -- not because the intended
    STANDARD/UNUSUAL distinction is doing any real work yet.)
  - Directly measuring h0's own code-side "structural depth" formula
    (display-eq count, case markers, connective density, notation
    diversity) against judge score: rho=0.31 pooled over ALL train items,
    but rho=0.49 restricted to just the 83 items where creative_trick is
    non-empty. I.e. derivation depth is a MUCH sharper predictor of "is
    this technique actually novel" once you already know a technique was
    named -- exactly the pack's warning to decouple presence from quality,
    but the fix is a GATE, not an additive blend. A small per-feature sweep
    on the present-trick subset (Spearman rho, tanh-normalized): raw answer
    length .46, display-eq count .44, per-eq token complexity .40, notation
    diversity .40, connective density .29, case markers ~0 (dropped). A
    finer weight sweep on this present-trick subset settled near length
    .35 / display-eq .20 / token-complexity .20 / diversity .10 /
    connective .15. This gated combination reaches ~0.68 pooled TRAIN rho
    by itself, with NO reliance on the never-extracted 2nd field. Also
    checked the trick-ABSENT subset directly (65 items): raw answer length
    still correlates rho=.46 with judge there too (e.g. a long, literature-
    citing answer connecting Bessel-function zeros to Riemann-zeta zeros,
    judged 0.20, whose extraction came back empty) -- so the absent branch
    keeps a small length-only term rather than being a flat floor.

Design: (1) gate ALL structural-depth scoring on creative_trick being
present and not self-negating (some cached extractions already embed their
own disqualification, e.g. "Irrelevant connection to..." -- checked for
this pattern in code rather than trusting field 2); (2) reweight depth
toward the empirically strongest components; (3) keep h1's STANDARD/UNUSUAL
field as a neutral-by-default post-hoc refinement -- it contributes nothing
extra during TRAIN iteration (verified: always absent here, so this
collapses to the structural-gated score above, which is the actual
measured 0.68), but if it DOES get extracted for the real cert run it can
only sharpen the presence/quality split further in the direction the pack
asks for, never regress a clear structural read into a false positive.
"""

import re
import math

LLM_FIELDS = {
    "creative_trick": (
        "In <=10 words, name any distinctive trick, unusual connection, "
        "counterexample, or creative construction this answer genuinely and "
        "correctly uses beyond routinely applying a standard definition/"
        "theorem/method; reply NONE if it is a standard/routine application, "
        "or if the described element is irrelevant, incorrect, or does not "
        "actually address the problem."
    ),
    "technique_novelty": (
        "Consider only the trick you just named (skip if none). Would a "
        "mathematician experienced in the relevant subfield regard it as a "
        "standard, commonly-taught technique they would recognize "
        "immediately (reply STANDARD), or as a genuinely unusual, "
        "surprising, or inventive move rarely seen in routine coursework "
        "(reply UNUSUAL)? Reply NA if no trick applies or it is irrelevant."
    ),
}

_SELF_NEGATING_MARKERS = (
    "irrelevant", "incorrect", "not correct", "off-topic", "off topic",
    "unrelated", "does not address", "doesn't address", "misunderstand",
    "not applicable", "n/a", "wrong approach",
)


def _safe_call(fn, default):
    try:
        return fn()
    except Exception:
        return default


def _is_none_like(s: str) -> bool:
    s2 = re.sub(r"[^a-z]", "", (s or "").lower())
    return s2 in ("", "none", "na", "notapplicable", "no", "nothing", "null", "noneofthese")


def _self_negating(s: str) -> bool:
    low = (s or "").lower()
    return any(m in low for m in _SELF_NEGATING_MARKERS)


def _answer_span(t: str) -> str:
    m = re.search(r"\banswer\s*:", t, flags=re.IGNORECASE)
    return t[m.end():] if m else t


def _classify_novelty(s: str) -> str:
    """Classify the (post-hoc-only, empty during TRAIN iteration) technique_novelty
    extraction into 'standard' / 'unusual' / 'unknown'. Conservative: anything not
    clearly and unambiguously flagged falls to 'unknown' (neutral, no adjustment)."""
    s2 = re.sub(r"[^a-z]", "", (s or "").lower())
    if not s2:
        return "unknown"
    unusual_markers = ("unusual", "uncommon", "novel", "inventive", "surprising", "rare", "creative", "unexpected")
    standard_markers = ("standard", "routine", "common", "textbook", "typical", "usual", "ordinary")
    has_unusual = any(m in s2 for m in unusual_markers)
    has_standard = any(m in s2 for m in standard_markers)
    if has_unusual and not has_standard:
        return "unusual"
    if has_standard and not has_unusual:
        return "standard"
    return "unknown"


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = _safe_call(lambda: ops.normalize(text), text)
        ans = _answer_span(t)

        trick = ((extracted or {}).get("creative_trick") or "").strip()
        trick_present = bool(trick) and not _is_none_like(trick) and not _self_negating(trick)

        if not trick_present:
            # Matches the near-zero unconditional judge mean when no genuine
            # technique is grounded by the extractor (0.015 measured on
            # train) -- but NOT flat: checked the 65 trick-absent train
            # items directly, and raw answer length alone still correlates
            # rho=0.46 with judge score even in this subset (e.g. a long,
            # literature-citing answer connecting Bessel-function zeros to
            # Riemann-zeta zeros judged 0.20 despite an empty extraction --
            # the extractor missed a diffuse, discursive contribution that
            # isn't one crisp "trick"). A small length-only contribution
            # here (no other structural component transfers into this
            # subset) captures that without disturbing the strong
            # presence-based ranking (all other absent-trick components
            # tested near-zero or negative correlation here, so length only).
            depth_len = math.tanh(len(ans) / 1500.0)
            return max(0.0, min(1.0, 0.02 + 0.30 * depth_len))

        # ---- structural depth of the ANSWER, GATED on trick presence ----
        # (necessary-but-not-sufficient signal on its own; empirically much
        # sharper once conditioned on a genuinely named technique -- see
        # module docstring)
        ans_spans = _safe_call(lambda: ops.extract_math_spans(ans), [])
        n_display_ans = 0
        try:
            n_display_ans = sum(1 for kind, _ in ans_spans if kind == "display")
        except Exception:
            n_display_ans = 0

        avg_tokens = 0.0
        try:
            _n_disp, _n_inl, _n_num, avg_tokens = ops.equation_stats(ans)
            avg_tokens = avg_tokens or 0.0
        except Exception:
            avg_tokens = 0.0

        skeleton = _safe_call(lambda: ops.proof_skeleton(ans), {})
        n_connective = 0
        if isinstance(skeleton, dict):
            n_connective = len(skeleton.get("connective", []) or [])

        notation = _safe_call(lambda: ops.notation_census(ans), {})
        diversity = len(notation) if isinstance(notation, dict) else 0

        depth = (
            0.35 * math.tanh(len(ans) / 1500.0)
            + 0.20 * math.tanh(n_display_ans / 6.0)
            + 0.20 * math.tanh(avg_tokens / 20.0)
            + 0.10 * math.tanh(diversity / 15.0)
            + 0.15 * math.tanh(n_connective / 6.0)
        )
        depth = max(0.0, min(1.0, depth))

        # ---- optional post-hoc refinement: never populated during TRAIN
        # iteration (verified 0 cached records for this field name), so this
        # is a no-op ("unknown" -> depth unchanged) on the signal measured
        # here. At final cert time, if extracted, it can only sharpen the
        # presence/quality split further in the direction the pack asks for.
        verdict = _classify_novelty((extracted or {}).get("technique_novelty") or "")
        if verdict == "standard":
            depth *= 0.35
        elif verdict == "unusual":
            depth = min(1.0, depth * 1.15 + 0.15)

        base = 0.05
        out = base + 0.90 * depth
        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5
