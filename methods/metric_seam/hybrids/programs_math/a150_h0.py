"""Hybrid metric for a150: Completeness and accuracy of hypotheses and statements.

Rationale (see reply for full writeup): the v0 keyword baseline (rho=0.079) fails because
"assume/suppose/condition" words are ambient in Math-SE prose (mostly in the *question*)
and are not a proxy for whether the *answer* precisely states/fixes hypotheses. Reading the
train residuals, the judge rewards (a) answers that actually engage the exact question with
a chain of justified steps (vs. short deflections, off-topic non-sequiturs, or "clearly"/"etc"
hand-waves) and (b) answers that explicitly name and correct a missing, incorrect, or
extraneous hypothesis/condition/notation -- exactly the criterion's own language. Both facets
are THICK judgments (semantic engagement-with-the-question, explicit-correction-detection) that
a regex over the raw text can't reliably see, so they are the two LLM_FIELDS. Everything else
(logical-scaffolding density, answer substantiveness, hedge-word penalty) stays in code using
the math-aware ops so the predicate remains deterministic and auditable.
"""

import re

LLM_FIELDS = {
    "hyp_fix": (
        "In <=15 words: the missing, incorrect, or extraneous hypothesis, condition, or "
        "notation that this ANSWER explicitly identifies and fixes/corrects; else NONE."
    ),
    "engagement": (
        "One word: does the ANSWER directly solve the exact question asked (ONTOPIC), "
        "partly engage without finishing (PARTIAL), or deflect / address a different "
        "problem (OFFTOPIC)?"
    ),
}

_HEDGE_RE = re.compile(
    r"\b(clearly|obviously|trivially|it is easy to see|of course)\b", re.IGNORECASE
)


def _answer_part(t):
    try:
        low = t.lower()
        idx = low.rfind("\nanswer:")
        if idx == -1:
            idx = low.find("answer:")
        return t[idx:] if idx != -1 else t
    except Exception:
        return t


def _safe(fn, default):
    try:
        val = fn()
        return val if val is not None else default
    except Exception:
        return default


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = _safe(lambda: ops.normalize(text), text)
        ans = _answer_part(t)
        if len(ans.strip()) < 5:
            ans = t

        n_sent, mean_wps, frac_long = _safe(lambda: ops.sent_stats(ans), (1, 10.0, 0.0))
        n_sent = max(int(n_sent or 1), 1)

        skeleton = _safe(lambda: ops.proof_skeleton(ans), {})
        n_connective = len(skeleton.get("connective", []) or [])
        n_case = len(skeleton.get("case", []) or [])
        n_definition = len(skeleton.get("definition", []) or [])
        n_theorem = len(skeleton.get("theorem", []) or [])
        logical_hits = n_connective + n_case + n_definition + n_theorem

        # density of logical scaffolding (connectives/cases/definitions/theorem-refs)
        # per sentence, saturating so a handful of hits already earns most of the credit
        density = logical_hits / n_sent
        base = density / (density + 1.5)
        base = 0.15 + 0.55 * base  # maps to [0.15, 0.70)

        n_display, n_inline, n_numbered, avg_tok = _safe(
            lambda: ops.equation_stats(ans), (0, 0, 0, 0.0)
        )
        has_math = (n_display + n_inline) > 0

        # substantiveness gate: a 1-2 sentence non-mathematical reply (deflection,
        # "see example 3", rhetorical question-back) cannot demonstrate precise
        # hypothesis handling regardless of any accidental connective hit
        if n_sent <= 2 and not has_math:
            base = min(base, 0.25)

        # unjustified-leap / "unlicensed generality" penalty
        hedges = len(_HEDGE_RE.findall(ans))
        if hedges:
            base -= min(0.10, 0.04 * hedges)

        # THICK grounding: explicit correction of a hypothesis/condition/notation is
        # the core of this criterion and code cannot reliably detect it
        hyp_fix = (extracted.get("hyp_fix") or "").strip()
        if hyp_fix and hyp_fix.upper() != "NONE":
            base += 0.20

        # THICK grounding: on-topic engagement vs. deflection/off-topic non-sequitur
        # explains most of the bottom of the train distribution
        engagement = (extracted.get("engagement") or "").strip().upper()
        if "OFF" in engagement:
            base = min(base, 0.20)
        elif "PARTIAL" in engagement:
            base -= 0.08

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
