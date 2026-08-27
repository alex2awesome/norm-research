"""Hybrid metric channel for a108: "Novelty and creative contribution" (Math SE).

h1 revision of h0.

What went wrong in h0 (general failure mode, not a per-example patch):
h0's single LLM field ("creative_trick") asks an extractor to *name* any
distinctive trick/connection/construction beyond routine application of a
standard method. In practice the extractor is happy to name *something* for
almost any non-trivial answer -- because nearly every math answer contains
SOME identifiable manipulation (a substitution, a bound, a symmetry, a
special-case check, a known identity) that can be described in a few words.
Naming a technique is not the same as the technique being genuinely novel:
"bound by total variation to get convergence", "treat a scalar as a 1x1
matrix to transpose it", "split into cases containing/not containing an
element", "rewrite a series via the log form of arctan" are all textbook,
commonly-taught moves that a mathematician in the relevant subfield would
recognize immediately -- yet h0 scored every one of these as if it were
strong evidence of novelty, because its scoring function treated *any*
non-NONE extraction as near-maximal signal (0.55 base weight, saturating by
word count alone, with no check on whether the named thing was actually
uncommon). That is the "presence vs quality" trap recurring one level up:
h0 avoided lexical-keyword presence-vs-quality in the code, but reintroduced
the same trap through the LLM field (extraction-presence vs genuine-novelty
quality).

Design change: decouple "a technique was named" from "the technique is
actually unusual" by asking the extractor an explicit second, narrow
question that forces a standard-vs-unusual judgment (with an explicit
irrelevant/incorrect escape hatch, since a named "trick" that is actually
off-topic or wrong should not earn credit either). The code side then makes
the LLM's *novelty verdict* -- not the mere presence of an extraction -- the
gate on the large novelty bonus. A named-but-standard (or irrelevant/
unclear) technique now earns only a small bump above baseline, matching the
judge's heavily low-compressed distribution described in the pack. The
structural/derivation-depth signal from h0 is kept as-is (necessary, not
sufficient, evidence) since it was not the identified source of the
worst-divergence failures.
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


def _safe_call(fn, default):
    try:
        return fn()
    except Exception:
        return default


def _is_none_like(s: str) -> bool:
    s2 = re.sub(r"[^a-z]", "", (s or "").lower())
    return s2 in ("", "none", "na", "notapplicable", "no", "nothing", "null", "noneofthese")


def _answer_span(t: str) -> str:
    m = re.search(r"\banswer\s*:", t, flags=re.IGNORECASE)
    return t[m.end():] if m else t


def _classify_novelty(s: str) -> str:
    """Classify the technique_novelty extraction into 'unusual' or 'other'.

    Conservative by construction: anything not clearly and unambiguously
    flagged as unusual (standard, NA, unclear, empty, contradictory) falls
    back to 'other', matching the judge's heavy compression toward the low
    end described in the pack (almost nothing counts as genuinely novel).
    """
    s2 = re.sub(r"[^a-z]", "", (s or "").lower())
    if not s2:
        return "other"
    unusual_markers = ("unusual", "uncommon", "novel", "inventive", "surprising", "rare", "creative", "unexpected")
    standard_markers = ("standard", "routine", "common", "textbook", "typical", "usual", "ordinary")
    has_unusual = any(m in s2 for m in unusual_markers)
    has_standard = any(m in s2 for m in standard_markers)
    if has_unusual and not has_standard:
        return "unusual"
    return "other"


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = _safe_call(lambda: ops.normalize(text), text)
        ans = _answer_span(t)

        # ---- LLM signal: thick-input grounding for "is this actually creative" ----
        trick = ((extracted or {}).get("creative_trick") or "").strip()
        novelty_raw = ((extracted or {}).get("technique_novelty") or "").strip()
        trick_present = bool(trick) and not _is_none_like(trick)

        llm_signal = 0.0
        if trick_present:
            verdict = _classify_novelty(novelty_raw)
            if verdict == "unusual":
                n_words = len(trick.split())
                llm_signal = min(1.0, 0.5 + 0.05 * min(n_words, 9))
            else:
                # A technique was named but the extractor did not affirm it
                # is genuinely unusual (standard / NA / unclear / irrelevant
                # all land here). Give a small bump for having *some*
                # identifiable structure, but do not let naming alone imply
                # novelty -- this is the direct fix for the failure mode.
                llm_signal = 0.12

        # ---- code-side structural depth of the ANSWER (not the question) ----
        # Kept as h0: necessary-but-not-sufficient evidence of derivation
        # depth; not the source of the identified worst-divergence failures.
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
        n_case = 0
        n_connective = 0
        if isinstance(skeleton, dict):
            n_case = len(skeleton.get("case", []) or [])
            n_connective = len(skeleton.get("connective", []) or [])

        notation = _safe_call(lambda: ops.notation_census(ans), {})
        diversity = len(notation) if isinstance(notation, dict) else 0

        depth = (
            0.40 * math.tanh(n_display_ans / 6.0)
            + 0.25 * math.tanh(n_case / 2.0)
            + 0.15 * math.tanh(n_connective / 6.0)
            + 0.20 * math.tanh(diversity / 15.0)
        )
        depth = max(0.0, min(1.0, depth))

        complexity = max(0.0, min(1.0, math.tanh(avg_tokens / 20.0)))

        structural = 0.6 * depth + 0.4 * complexity  # in [0,1]

        base = 0.05
        out = base + 0.55 * llm_signal + 0.15 * structural
        return max(0.0, min(1.0, out))
    except Exception:
        return 0.5
