"""
Hybrid metric for a180 (BUDGET-4 extension): "Notation conventions and
consistency" (Math StackExchange).

Base rationale (see improver_pack a180.json train residuals; unchanged from h0):
  - The v0 keyword baseline (regex hits on "notation"/"denote"/"standard"/"convention")
    gets train rho = 0.034, essentially noise: whether an answer TALKS ABOUT notation
    is a weak proxy for whether its notation is actually good.
  - What separates low-judge from high-judge answers in the 30 train examples is mostly
    the PARSED hygiene/consistency of the LaTeX itself: unmatched delimiters, mixed
    \\left/\\right, brace imbalance, \\begin/\\end mismatch, ad hoc \\space hacks, and,
    most diagnostically, the SAME function/operator rendered two incompatible ways in
    one answer. Clean, consistent, standard-command usage scores high with zero explicit
    "documentation" language, so the predicate must be structural/parse-based, not
    lexical.
  - A minority of high-scoring answers introduce genuinely nonstandard notation but
    explicitly gloss it -- exactly the "document nonstandard notation" clause of the
    criterion. That's a semantic judgment a parser can't reliably make, so h0 delegates
    it to two thin LLM extractors (notation_defined / notation_clash) used as bounded
    nudges on top of the code predicate.

Budget-4 extension (blind design -- no eval signal used, chosen on construct grounds):
  The full criterion text is: "Adopt clear, standard notation (including house-style
  choices), document any nonstandard or abusive notation, and use it consistently;
  choose conventional symbols (e.g., x vs juxtaposition, duals/pairings) and standard
  typographic conventions (e.g., italics for variables)." h0's code+LLM combo covers
  CONSISTENCY (parse-based clash detection) and DOCUMENTATION (notation_defined /
  notation_clash), but nothing in h0 evaluates whether the notation CHOSEN is itself
  conventional (independent of whether it's used consistently or defined) or whether
  standard TYPOGRAPHIC style is followed. Two new thin LLM fields target exactly those
  two remaining sub-clauses of the criterion:
    - nonstandard_choice: names a symbol whose CHOICE (not its consistency) departs
      from mathematical convention -- a house-style symbol used perfectly consistently
      throughout still violates "choose conventional symbols" unless it is documented,
      so the predicate cross-checks this field against notation_defined before
      penalizing (undocumented nonstandard choice is penalized; documented nonstandard
      choice, which the criterion explicitly permits, is not).
    - typography_ok: a direct categorical read on the "standard typographic
      conventions (e.g., italics for variables)" clause, which the code predicate has
      no parse-level access to (LaTeX's default math-mode italics vs explicit
      \\mathrm{}/\\text{} wrapping is a rendering-semantics judgment, not a structural
      count).
  Both fields degrade to a no-op when absent from `extracted` (budget < 4), so this
  program reproduces h0's exact score whenever only the original 2 fields are served.
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
    "nonstandard_choice": (
        "Name ONE common math object/operation (e.g. multiplication, inner product, "
        "set difference, duals/pairings) whose SYMBOL here is not the standard "
        "mathematical convention, in <=8 words; else answer NONE."
    ),
    "typography_ok": (
        "Are variables/constants in this answer typeset in the standard style "
        "(italic variables, upright named functions/operators)? Answer YES, NO, or NA."
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
        ex = extracted or {}
        defined = (ex.get("notation_defined", "") or "").strip()
        clash = (ex.get("notation_clash", "") or "").strip()

        bonus = 0.0
        if defined and defined.upper() != "NONE":
            bonus += 0.08
        if clash and clash.upper() != "NONE":
            bonus -= 0.12

        # ---- 5. budget-4 nudges: symbol-choice conventionality + typography ----
        # Missing fields (budget < 4) leave nonstd/typo as "" -> no branch fires ->
        # bonus2 stays 0.0, so this program reproduces h0 exactly at budget 2.
        nonstd = (ex.get("nonstandard_choice", "") or "").strip()
        typo = (ex.get("typography_ok", "") or "").strip().upper()

        bonus2 = 0.0
        has_nonstd = bool(nonstd) and nonstd.upper() != "NONE"
        if has_nonstd:
            if defined and defined.upper() != "NONE":
                # Documented nonstandard choice: the criterion explicitly permits
                # this ("document any nonstandard or abusive notation"), so no
                # penalty -- but no extra bonus either, to avoid double-rewarding
                # the same evidence notation_defined already credited above.
                pass
            else:
                bonus2 -= 0.10
        if typo == "NO":
            bonus2 -= 0.08
        elif typo == "YES":
            bonus2 += 0.03

        final = code_score + bonus + bonus2
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
