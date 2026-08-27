"""
Hybrid metric for a180: "Notation conventions and consistency" (Math StackExchange).

Rationale (see improver_pack a180.json train residuals):
  - The v0 keyword baseline (regex hits on "notation"/"denote"/"standard"/"convention")
    gets train rho = 0.034, essentially noise: whether an answer TALKS ABOUT notation
    is a weak proxy for whether its notation is actually good.
  - What separates low-judge from high-judge answers in the 30 train examples is mostly
    the PARSED hygiene/consistency of the LaTeX itself: unmatched delimiters, mixed
    \\left/\\right, brace imbalance, \\begin/\\end mismatch (e.g. d03265's clunky
    "\\pi_1(U,x_0)\\space/\\space N_1" quotient notation, ad hoc \\space hacks) and, most
    diagnostically, the SAME function/operator rendered two incompatible ways in one
    answer (e.g. d02378 mixes bare "ln x"/"lnx" with "\\ln x" -- judge_score 0.25 despite
    a clean-looking amount of math). Clean, consistent, standard-command usage (d01462,
    d01742, d00734, d01866, d03818, d04903, d02293: all judge 1.0, all *silent* about
    notation-as-topic) scores high with zero explicit "documentation" language, so the
    predicate must be structural/parse-based, not lexical.
  - A minority of high-scoring answers introduce genuinely nonstandard notation but
    explicitly gloss it (d01778 packs constants into named A,B; d01064 names a
    "synthetic" construction; d04167 explains what <.,.> means) -- exactly the
    "document nonstandard notation" clause of the criterion. That's a semantic
    judgment (is this explanation actually present and is it about a symbol the LM
    reads?) that a parser can't reliably make, so it's delegated to one LLM_FIELDS
    extractor used only as a modest positive nudge, plus one extractor for the
    complementary failure (LM explicitly spots an unresolved same-symbol clash) as a
    modest negative nudge. Both are thin, targeted, and bounded so they cannot dominate
    the code predicate or run away on any one example.
"""

import re
from collections import Counter

LLM_FIELDS = {
    "notation_defined": (
        "Quote (<=8 words) the symbol/term the answer explicitly defines, introduces, "
        "or explains the meaning of (e.g. 'let X denote...', 'here <.,.> means...'); "
        "else answer NONE."
    ),
    "notation_clash": (
        "Name one symbol this answer uses inconsistently for two different meanings "
        "or two incompatible spellings (e.g. plain 'ln' vs '\\ln'); else answer NONE."
    ),
}

_BARE_FUNCS = (
    "sin", "cos", "tan", "cot", "sec", "csc",
    "ln", "log", "lim", "det", "exp", "sup", "inf",
    "min", "max", "gcd", "lcm", "ker", "dim", "deg", "arg", "mod",
)


def _safe_call(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:
        return default


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = _safe_call(getattr(ops, "normalize", None), text, default=text) or text

        # ---- 1. delimiter / typesetting hygiene (objective, code-only) ----
        dh = _safe_call(getattr(ops, "delimiter_health", None), t, default={}) or {}
        try:
            dh_total = sum(v for v in dh.values() if isinstance(v, (int, float)))
        except Exception:
            dh_total = 0

        spans = _safe_call(getattr(ops, "extract_math_spans", None), t, default=[]) or []
        n_spans = max(len(spans), 1)

        dh_density = dh_total / n_spans
        hygiene = 1.0 / (1.0 + 1.5 * dh_density)

        # ---- 2. same-symbol-two-ways consistency (code-only, via notation_census) ----
        census = _safe_call(getattr(ops, "notation_census", None), t, default={}) or {}
        clashes = 0
        checked = 0
        for fn in _BARE_FUNCS:
            bare_ct = census.get(fn, 0) if isinstance(census, dict) else 0
            cmd_ct = census.get("\\" + fn, 0) if isinstance(census, dict) else 0
            if bare_ct or cmd_ct:
                checked += 1
                if bare_ct and cmd_ct:
                    clashes += 1
        consistency = 1.0 if checked == 0 else 1.0 - (clashes / checked)

        # ---- 3. mild pathology guard on equation structure ----
        stats = _safe_call(getattr(ops, "equation_stats", None), t, default=None)
        complexity_penalty = 1.0
        if stats:
            try:
                n_display, n_inline, n_numbered, avg_tokens = stats
                if avg_tokens and avg_tokens > 60:
                    complexity_penalty = 0.85
            except Exception:
                pass

        code_score = (0.55 * hygiene + 0.45 * consistency) * complexity_penalty
        code_score = max(0.0, min(1.0, code_score))

        # ---- 4. thin LLM-grounded nudges (thick judgments a parser can't see) ----
        defined = (extracted.get("notation_defined", "") or "").strip()
        clash = (extracted.get("notation_clash", "") or "").strip()

        bonus = 0.0
        if defined and defined.upper() != "NONE":
            bonus += 0.08
        if clash and clash.upper() != "NONE":
            bonus -= 0.12

        final = code_score + bonus
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
