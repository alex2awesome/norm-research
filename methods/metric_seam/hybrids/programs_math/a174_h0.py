"""
Hybrid metric channel for a174: "Equation display/layout and alignment"
(Math StackExchange answers).

Rationale (see docstring at bottom of file for the fuller version):
The keyword baseline (count \\begin{align/equation/...}) gets rho=0.273 because
env *presence* is a weak proxy: many high-scoring answers use zero display
environments (the math is simple/definitional and inline is correct), and some
low-scoring answers *do* contain display blocks that are messy, unaligned, or
just a single overlong un-broken line. The judged quality is really about
MATCH between derivation complexity and layout choice:
  - simple content decoratively left inline -> good (no violation)
  - a genuine multi-step derivation properly boxed into an aligned/numbered
    display env (align/gather/multline/split/eqnarray/cases, or explicit
    \\tag/\\label) -> good
  - a genuine multi-step derivation crammed into long inline spans, or into a
    flat un-aligned multi-line $$...$$ block, or an overlong single-line
    display that can't "fit within page width" -> bad
  - multi-step reasoning that never gets any display treatment at all (e.g.
    proof-by-cases entirely narrated inline, or numeric results laid out as a
    plain-text table) is a blind spot for pure LaTeX-span parsing, so we use
    one LLM field purely as thick-input grounding for "was there a real
    multi-step derivation here" and keep the actual scoring predicate in code.
Broken markup (delimiter_health) is a separate, additive penalty.
"""

import re

LLM_FIELDS = {
    "derivation_steps": (
        "In a few words, does this answer's math involve ZERO calculation, "
        "ONE formula, or MULTIPLE (2+) sequential derivation/algebra steps?"
    ),
}

_ALIGN_ENVS = ("align", "gather", "multline", "split", "eqnarray", "flalign", "cases")

_LONG_INLINE_TOKENS = 20     # an inline span this token-long is a display candidate
_LONG_FLAT_TOKENS = 40       # a display span this long w/o an internal break is likely too wide
_LONG_CHARS_NO_BREAK = 200   # char-based backstop for the same check
_MIN_CASE_MARKERS = 2        # >=2 "case" markers signals a case-based derivation


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def score(text: str, extracted: dict, ops) -> float:
    try:
        spans = _safe(lambda: ops.extract_math_spans(text), []) or []
        if not isinstance(spans, list):
            spans = []
        inline_spans = [c for (k, c) in spans if k == "inline" and isinstance(c, str)]
        display_spans = [c for (k, c) in spans if k == "display" and isinstance(c, str)]
        n_display_spans = len(display_spans)

        def tok_len(content):
            toks = _safe(lambda: ops.latex_tokens(content), [])
            try:
                return len(toks)
            except Exception:
                return 0

        # --- inline spans: flag ones that are really crammed derivations ---
        overloaded_inline = 0
        for c in inline_spans:
            tl = tok_len(c)
            crammed_eq = c.count("=") >= 2 and len(c) > 60
            if tl >= _LONG_INLINE_TOKENS or crammed_eq:
                overloaded_inline += 1

        # --- display spans: classify structure quality ---
        good_display = 0
        ok_display = 0
        unaligned_stack = 0
        flat_long_display = 0
        for c in display_spans:
            has_env = any(("\\begin{%s" % e) in c for e in _ALIGN_ENVS)
            has_amp = "&" in c
            has_tag = ("\\tag" in c) or ("\\label" in c)
            n_breaks = c.count("\\\\")
            tl = tok_len(c)
            if has_env or has_amp or has_tag:
                good_display += 1
            elif n_breaks >= 1:
                unaligned_stack += 1
            elif tl >= _LONG_FLAT_TOKENS or len(c) >= _LONG_CHARS_NO_BREAK:
                flat_long_display += 1
            else:
                ok_display += 1

        good_units = good_display + ok_display
        bad_units = overloaded_inline + unaligned_stack + flat_long_display

        # --- case-based derivations that never got boxed into any display ---
        skel = _safe(lambda: ops.proof_skeleton(text), {}) or {}
        n_cases = 0
        try:
            n_cases = len(skel.get("case", []))
        except Exception:
            n_cases = 0
        missing_case_structure = (n_cases >= _MIN_CASE_MARKERS and n_display_spans == 0)
        if missing_case_structure:
            bad_units += 1

        # --- LLM cross-check: real multi-step reasoning the LaTeX parse can't see
        #     (e.g. narrated case-splits, plain-text result tables) ---
        hint = (extracted.get("derivation_steps") or "").strip().lower()
        llm_multi = bool(re.search(r"multi|\b(2|two|three|four|five|several)\b", hint))
        llm_none = (hint == "") or bool(re.search(r"\bzero\b|\bnone\b|\bno calc", hint))

        if llm_multi and n_display_spans == 0 and not missing_case_structure:
            bad_units += 1
        if llm_none and not llm_multi:
            # dense but purely definitional/notational content: don't penalize
            # code-side "complexity" flags that are just heavy notation.
            bad_units = max(0, bad_units - 1)

        structure = 1.0 if bad_units <= 0 else good_units / float(good_units + bad_units)

        # --- reference bonus: numbered equations that are actually referenced ---
        eqstats = _safe(lambda: ops.equation_stats(text), None)
        n_numbered = 0
        if isinstance(eqstats, (tuple, list)) and len(eqstats) >= 3:
            try:
                n_numbered = int(eqstats[2] or 0)
            except Exception:
                n_numbered = 0

        prose = text
        for c in display_spans + inline_spans:
            prose = _safe(lambda c=c: prose.replace(c, " ", 1), prose)
        refs = len(_safe(
            lambda: re.findall(r"\beq(?:uation)?\.?\s*\(?\d+\)?|\\eqref|\\ref\{", prose, flags=re.IGNORECASE),
            [],
        ))
        bonus = 0.0
        if n_numbered > 0 and refs > 0:
            bonus += 0.05
        if llm_multi and good_display > 0 and bad_units == 0:
            bonus += 0.05

        # --- broken-markup penalty (delimiter health), key-name agnostic ---
        dhealth = _safe(lambda: ops.delimiter_health(text), {}) or {}
        breakage = 0
        try:
            for _, v in dhealth.items():
                if isinstance(v, (int, float)) and v:
                    breakage += min(int(v), 3)
        except Exception:
            breakage = 0
        penalty = min(0.30, 0.05 * breakage)

        raw = 0.15 + 0.75 * structure + bonus - penalty
        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
