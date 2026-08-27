"""
Hybrid metric channel for a0: "Sentence-level grammar and parsing clarity"
(Math StackExchange answers).

Design rationale (see rationale in final report). Core idea: the naive
structural baseline (bracket balance / avg-sentence-char-length / capital-start
ratio) got only rho=0.311 because it treats LaTeX-heavy math text the same as
plain prose. Reading the 30 train examples, the judge rewards documents that
(a) read as connected EXPOSITORY prose around the math (transition words like
"First/Then/Note that/Hence"), not bare chains of equations with nothing
joining them, (b) are typeset/authored cleanly (no unbalanced $, no naked
LaTeX commands leaking outside math, no runs of "??" / "!!"), and (c) are free
of concrete grammar mistakes / typos (missing words, wrong prepositions,
misspelled terms) and unclear pronoun/modifier placement -- which a regex
cannot reliably detect, hence pushed into two LLM_FIELDS.
"""

import re
import math

LLM_FIELDS = {
    "grammar_flaw": (
        "Quote verbatim the single clearest grammar mistake, typo, or "
        "missing/wrong word in this text, else answer NONE."
    ),
    "awkward_phrasing": (
        "Quote one sentence with unclear pronoun reference, ambiguous "
        "wording, or a misplaced modifier, else answer NONE."
    ),
}

_REPEAT_PUNCT_RE = re.compile(r"[!?]{2,}")
_MATH_STRIP_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]{1,400}\$", re.S)
_CAP_ERROR_RE = re.compile(r"[.!?]\s+[a-z]")
_LOWER_I_RE = re.compile(r"(?<![A-Za-z0-9])i(?![A-Za-z0-9])")


def _safe_call(fn, default):
    try:
        val = fn()
        return val if val is not None else default
    except Exception:
        return default


def _clip01(x):
    if x != x:  # nan guard
        return 0.5
    return max(0.0, min(1.0, x))


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5

        t = _safe_call(lambda: ops.normalize(text), text)
        if not t or len(t.strip()) < 20:
            return 0.5

        # --- sentence stats (computation op) ---
        n_sent, mean_words_per_sent, frac_long_words = _safe_call(
            lambda: ops.sent_stats(t), (1, 15.0, 0.15)
        )
        n_sent = max(1, int(n_sent) if n_sent else 1)

        # --- math-aware equation stats ---
        n_display, n_inline, n_numbered, avg_tokens = _safe_call(
            lambda: ops.equation_stats(t), (0, 0, 0, 0.0)
        )
        n_display = n_display or 0
        n_inline = n_inline or 0

        # --- proof/exposition skeleton: connective density ---
        skeleton = _safe_call(lambda: ops.proof_skeleton(t), {})
        n_connective = 0
        try:
            n_connective = len(skeleton.get("connective", []) or [])
        except Exception:
            n_connective = 0
        connective_density = n_connective / float(n_sent)
        s_connective = _clip01(connective_density / 0.35)

        # --- delimiter / LaTeX authoring health (aggregate, key-name agnostic) ---
        dh = _safe_call(lambda: ops.delimiter_health(t), {})
        try:
            malform_total = sum(v for v in dh.values() if isinstance(v, (int, float)))
        except Exception:
            malform_total = 0
        chars_per_unit = max(len(t) / 600.0, 1.0)
        malform_rate = malform_total / chars_per_unit
        s_delim = _clip01(1.0 - 0.20 * malform_rate)

        # --- punctuation abuse ("????", "!!!") ---
        n_repeat_punct = len(_REPEAT_PUNCT_RE.findall(t))
        s_punct = _clip01(1.0 - 0.30 * n_repeat_punct)

        # --- crude prose isolation for cheap proofreading checks ---
        prose = _MATH_STRIP_RE.sub(" MATH ", t)

        n_cap_err = len(_CAP_ERROR_RE.findall(prose))
        # normalize by sentence count so long docs aren't unfairly punished
        cap_err_rate = n_cap_err / float(n_sent)
        s_cap = _clip01(1.0 - 0.5 * cap_err_rate)

        n_lower_i = len(_LOWER_I_RE.findall(prose))
        lower_i_rate = n_lower_i / float(n_sent)
        s_pronoun_i = _clip01(1.0 - 0.5 * lower_i_rate)

        # --- prose-to-math balance: penalize bare equation chains ---
        math_spans = n_display + n_inline
        ratio = math_spans / float(n_sent)
        if ratio <= 1.5:
            s_mathdensity = 1.0
        elif ratio >= 6.0:
            s_mathdensity = 0.0
        else:
            s_mathdensity = 1.0 - (ratio - 1.5) / 4.5
        s_mathdensity = _clip01(s_mathdensity)

        code_score = (
            0.22 * s_connective
            + 0.18 * s_delim
            + 0.14 * s_punct
            + 0.14 * s_cap
            + 0.10 * s_pronoun_i
            + 0.22 * s_mathdensity
        )

        # --- LLM-grounded tacit judgments a parser can't see ---
        # Real word-level grammar errors ("for solve this we have find...")
        # and unclear pronoun/modifier placement are exactly the failures a
        # regex/LaTeX-aware parser is blind to, so they carry real weight.
        llm_penalty = 0.0
        gf = (extracted or {}).get("grammar_flaw", "") or ""
        if gf.strip():
            llm_penalty += 0.30
        ap = (extracted or {}).get("awkward_phrasing", "") or ""
        if ap.strip():
            llm_penalty += 0.15

        final = code_score - llm_penalty
        return _clip01(final)
    except Exception:
        return 0.5
