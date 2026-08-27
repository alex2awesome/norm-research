"""
a156: Uniform and proper substitution and variable discipline.

Rationale (see pack a156.json): the v0 keyword baseline (train rho=0.074) fires on
words like "substitution"/"proviso" regardless of whether the step is actually
valid -- a textbook keyword-as-proxy failure. Reading the 30 train examples, the
judge's low scores line up with concrete, checkable-in-principle defects: a
variable silently changes referent mid-derivation (e.g. e^{iz} becomes e^{z}),
two maps that must stay distinct collapse onto the same subscript, a step is
justified by "obviously"/"clearly" instead of stating a domain/sign proviso, or
notation is switched (phi -> Phi) without comment. That is exactly a THICK-INPUT
judgment (does this substitution actually respect the theorem's distinct-variable
side condition?) that a regex cannot make but a full-text reader can. The code
side is restricted to what a parser genuinely owns: LaTeX well-formedness
(the criterion's own word "well-formed") and simple symbol-form duplication as a
weak, non-semantic consistency check.
"""

import re

LLM_FIELDS = {
    "subst_issue": (
        "Name the clearest spot where a variable, subscript, or symbol silently "
        "changes meaning, or a substitution/side-condition is asserted without "
        "justification (e.g. 'obviously'/'clearly'); else say NONE."
    ),
}


def _safe_dict_sum(d):
    total = 0.0
    try:
        for v in d.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total += v
    except Exception:
        pass
    return total


_NEGATIONS = {
    "", "none", "none.", "no issue", "no issues", "n/a", "na",
    "none found", "no problem", "no problems", "not applicable",
}


def _is_negation(issue_lower: str) -> bool:
    if issue_lower in _NEGATIONS:
        return True
    if issue_lower.startswith("none"):
        return True
    if "no issue" in issue_lower or "not applicable" in issue_lower:
        return True
    return False


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        try:
            t = ops.normalize(text)
        except Exception:
            t = text

        # ---- code predicate 1: LaTeX/math well-formedness ("well-formed" is
        # literally in the criterion description) ----
        try:
            spans = ops.extract_math_spans(t)
        except Exception:
            spans = []
        n_spans = len(spans) if spans else 0

        try:
            dh = ops.delimiter_health(t)
        except Exception:
            dh = {}
        defect_total = _safe_dict_sum(dh)
        defect_rate = defect_total / max(n_spans, 1)
        wellformed = 1.0 / (1.0 + 3.0 * defect_rate)

        # ---- code predicate 2: weak, non-semantic notation-consistency check --
        # same function rendered two ways (bare word vs command) in one doc is a
        # mild, purely-mechanical symptom of the kind of notational sloppiness
        # the criterion cares about (NOT a claim of correctness/incorrectness).
        try:
            census = ops.notation_census(t)
        except Exception:
            census = {}
        fn_names = ("sin", "cos", "tan", "cot", "sec", "csc", "log", "ln",
                    "exp", "lim", "max", "min", "det", "ker", "sup", "inf", "arg")
        checked = 0
        dual = 0
        for fn in fn_names:
            bare = census.get(fn, 0) or 0
            cmd = census.get("\\" + fn, 0) or 0
            if bare > 0 or cmd > 0:
                checked += 1
                if bare > 0 and cmd > 0:
                    dual += 1
        consistency = 1.0 - (dual / checked) if checked else 1.0

        # ---- code predicate 3: mild substance-engagement floor (spans exist
        # at all) -- capped low weight, this is deliberately NOT a length proxy
        try:
            n_display, n_inline, n_numbered, avg_tokens = ops.equation_stats(t)
        except Exception:
            n_display, n_inline, n_numbered, avg_tokens = (0, 0, 0, 0.0)
        engagement = min(1.0, (n_display + n_inline) / 6.0)

        code_score = 0.5 * wellformed + 0.25 * consistency + 0.25 * engagement
        code_score = max(0.0, min(1.0, code_score))

        # ---- LLM predicate: the tacit judgment a parser cannot make ----
        raw_issue = ""
        try:
            raw_issue = (extracted.get("subst_issue") or "").strip()
        except Exception:
            raw_issue = ""
        issue_found = not _is_negation(raw_issue.lower())

        llm_component = 0.15 if issue_found else 0.85

        final = 0.55 * llm_component + 0.45 * code_score
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
