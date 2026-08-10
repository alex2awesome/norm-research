"""Hybrid metric channel for a108: "Novelty and creative contribution" (Math SE).

Rationale (from studying the 30 train examples): judge scores are heavily
compressed near the low end (0.0-0.3) -- almost no Math-SE answer counts as
"novel" in a research sense, but the judge still discriminates: routine
definition-application / straight verification of the OP's own work scores
0.0 (even when long/technical, e.g. re-deriving a connection 1-form the OP
already set up, or "your proof is correct, here's the standard alternative
method"); answers that introduce a genuine trick, symmetry/substitution
argument, counterexample, multi-case constructive solution, or a connection
to a different area of math (Bessel zeros <-> Riemann zeta, literature
citation) score 0.05-0.3. Keyword presence ("novel","new","contribution"...)
is a near-noise proxy (baseline_train_rho=0.115) -- e.g. "new group" in a
routine semidirect-product answer triggers the keyword baseline but the
judge scores it 0.0. That is exactly the "presence vs quality" trap the pack
warns about, so the predicate here does NOT use lexical keyword matching.

Design: the code side measures DERIVATION DEPTH restricted to the ANSWER
span only (case-analysis markers, distinct display equations, connective
density, notation diversity via the math-aware ops) -- this is a necessary
but not sufficient signal (some long/complex answers are just careful
verification, not creative). The one LLM field grounds the genuinely tacit
part of the judgment: whether the answer actually contains a distinctive
trick/counterexample/cross-topic connection, which a parser cannot see.
"""

import re
import math

LLM_FIELDS = {
    "creative_trick": (
        "In <=10 words, name any distinctive trick, unusual connection, "
        "counterexample, or creative construction this answer uses beyond "
        "routinely applying a standard definition/theorem/method; reply "
        "NONE if it is just a standard direct application."
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


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = _safe_call(lambda: ops.normalize(text), text)
        ans = _answer_span(t)

        # ---- LLM signal: thick-input grounding for "is this actually creative" ----
        trick = ((extracted or {}).get("creative_trick") or "").strip()
        llm_signal = 0.0
        if trick and not _is_none_like(trick):
            n_words = len(trick.split())
            llm_signal = min(1.0, 0.55 + 0.05 * min(n_words, 9))

        # ---- code-side structural depth of the ANSWER (not the question) ----
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
