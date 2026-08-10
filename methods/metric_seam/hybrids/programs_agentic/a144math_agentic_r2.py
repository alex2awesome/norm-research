"""a144 agentic r2: Existential proofs: explicit witnesses and scope.

Builds on h0 (train rho 0.564) / h1 (train rho 0.588). Both lean on two LLM
fields ("witness": short witness text or NONE, "completed": yes/no) plus
generic structural counts (proof_skeleton categories, equation_stats,
extract_math_spans, delimiter_health). Running h1 as a candidate through the
harness surfaces a fairly consistent failure mode in its worst residuals:

  OVERpredicted (cand >> judge): items where "completed"=Yes fires on an
  answer that is really restating/discussing the question's setup (lots of
  inline math, "denote"/"is called"-style definition words) without ever
  binding a scope word to a concrete constructed object, OR where the
  extracted "witness" text is quoting an ABANDONED attempt ("I stopped early
  because it did not work out...") that the current hedge list doesn't catch.

  UNDERpredicted (cand << judge): items where "completed"=No fires on an
  answer that in fact ends in a definite equation/boxed result (a real
  witness), so h0/h1's flat negative swing on "no" drags a genuinely complete
  answer down; and short, terse-but-correct answers (e.g. "7/2 is rational
  because ...") where code-structural signals are near zero by construction
  (too short to have display equations/case markers), so the LLM-axis
  contribution needs to be able to carry the item most of the way to a high
  score on its own rather than being capped at a small additive bonus.

Three targeted, general (non-excerpt-specific) changes on top of h1:

  1. NEW code signal `_binding_count`: a scope word (let/set/define/choose/
     take/fix) immediately followed, within the same clause, by an
     assignment/relation operator (=, :=, \\in, \\mapsto, >, <, \\leq, \\geq,
     \\subset...). This is the code-detectable analogue of "introduce the
     witness with correct scope and naming" -- stronger than counting scope
     words and equation-spans separately (h0/h1's approach), since it
     requires the CONJUNCTION of a scope word actually binding to a value.

  2. NEW code signal `_precise_exists_count` and `_ends_with_equation`:
     "there exist(s) ... such that" / "\\exists ... :" phrasing (precise
     quantifier structure, not just bare "exists" counts), and whether the
     answer's final ~220 chars land on a math span containing "=" or
     \\boxed (a register-agnostic "ends on a definite conclusion" signal
     that does not depend on the noisy LLM "completed" field).

  3. Rebalanced LLM axis: when the code-side completion evidence
     (`_ends_with_equation`, a binding, or a qed marker) CONTRADICTS
     completed="no", the penalty is nearly zeroed instead of h1's flat
     softened -0.11 (code evidence outranks the noisy label). When
     completed="yes" the bonus no longer requires reinforcement from other
     signals, since a short-but-genuine answer (e.g. "7/2 is rational...")
     has no other structural signal to lean on. An LLM witness with NO
     supporting scope/quantifier apparatus anywhere in the answer (the
     "N/q(0)" appears in an algebra manipulation with no existential framing
     at all) is downweighted, since it is likely the extractor pointing at
     an incidental quantity rather than a genuine existential witness.
     Hedge presence disables the code-only witness fallback (a leftover
     display equation from an abandoned attempt should not count as a
     "concrete witness delivered").
"""

import re

LLM_FIELDS = {
    "witness": "In <=10 words, the specific value/function/set the answer constructs as the witness, or NONE if none is given",
    "completed": "Answer only yes or no: does the answer finish the derivation to a definite conclusion rather than stop at a hint or sketch",
}

_ANSWER_MARK = re.compile(r"\bAnswer\s*:", re.IGNORECASE)

_HEDGE_PATS = [
    re.compile(r"should (give|work|prove|do the trick)", re.IGNORECASE),
    re.compile(r"i(?:'m| am) not (?:really )?sure", re.IGNORECASE),
    re.compile(r"i don't (?:see|know) how", re.IGNORECASE),
    re.compile(r"left (?:to|as) (?:the reader|an exercise)", re.IGNORECASE),
    re.compile(r"without success", re.IGNORECASE),
    re.compile(r"did(?:n't| not) work out", re.IGNORECASE),
    re.compile(r"stopped early", re.IGNORECASE),
    re.compile(r"no idea", re.IGNORECASE),
]

_QUANT_PATS = [
    re.compile(r"there\s+exists?", re.IGNORECASE),
    re.compile(r"\\exists"),
    re.compile(r"for\s+(all|each|every)\b", re.IGNORECASE),
    re.compile(r"for\s+some\b", re.IGNORECASE),
    re.compile(r"\\forall"),
    re.compile(r"\barbitrary\b", re.IGNORECASE),
    re.compile(r"\bfix\b", re.IGNORECASE),
]

_UNIQ_PATS = [
    re.compile(r"\bunique(ness)?\b", re.IGNORECASE),
    re.compile(r"exactly one", re.IGNORECASE),
    re.compile(r"at most one", re.IGNORECASE),
    re.compile(r"is the only", re.IGNORECASE),
]

_SCOPE_PATS = [
    re.compile(r"\blet\b", re.IGNORECASE),
    re.compile(r"\bdefine\b", re.IGNORECASE),
    re.compile(r"\bdenote\b", re.IGNORECASE),
    re.compile(r"\bset\b", re.IGNORECASE),
    re.compile(r"\bchoose\b", re.IGNORECASE),
    re.compile(r"\btake\b", re.IGNORECASE),
    re.compile(r"\bfix\b", re.IGNORECASE),
    re.compile(r"\bsuppose\b", re.IGNORECASE),
    re.compile(r"\bassume\b", re.IGNORECASE),
    re.compile(r"\bconsider\b", re.IGNORECASE),
    re.compile(r":="),
    re.compile(r"\\mapsto"),
]

_RESOLVE_PATS = [
    re.compile(r"\bconclude[sd]?\b", re.IGNORECASE),
    re.compile(r"it follows that", re.IGNORECASE),
    re.compile(r"which (shows|proves)", re.IGNORECASE),
    re.compile(r"we deduce", re.IGNORECASE),
]

# NEW: scope word immediately binding to a value within the same clause --
# the code-detectable analogue of "introduce the witness with correct scope".
_BINDING_PAT = re.compile(
    r"\b(?:let|set|define|choose|take|fix)\b[^.?!\n]{0,60}?"
    r"(?:=|:=|\\in\b|\\mapsto|\\subseteq?\b|\\geq|\\leq|\\ge\b|\\le\b|>|<)",
    re.IGNORECASE,
)

# NEW: precise existential phrasing, not just a bare "exists" count.
_PRECISE_EXISTS_PATS = [
    re.compile(r"there\s+exists?\b[^.?!\n]{0,80}?\bsuch\s+that\b", re.IGNORECASE),
    re.compile(r"\\exists[^.?!\n]{0,60}?(?::|\\text\{such that\})"),
]

_TAIL_CHARS = 220

# NEW (round 2): does the LLM witness text actually name a CONCRETE
# value/formula ("=", a digit, ":=", "\\mapsto", "\\to") rather than an
# abstract description ("the orthogonal set {n,s,t}", "a set of
# probabilities (a,b,c,d)")? In the round-1 residuals, concrete-formula
# witnesses reliably co-occur with high judge scores; abstract-noun-phrase
# witnesses reliably over-predict even when the LLM also says completed=Yes.
_CONCRETE_WIT_PAT = re.compile(r"=|\d|:=|\\mapsto|\\to\b")

# round 3: relation operators counted as "this math span states a conclusion"
# -- broadened past bare "=" to inequalities/set-relations, since existence
# proofs often conclude with a bound (trace(A^TA) <= ...) not an equality.
_RELATION_PAT = re.compile(
    r"=|<|>|\\le\b|\\leq|\\ge\b|\\geq|\\ne\b|\\neq|\\subset|\\subseteq|\\in\b"
)


def _safe_call(fn, default, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _clip01(x):
    return max(0.0, min(1.0, x))


def _ends_with_equation(ans):
    """Answer's tail lands on a definite math conclusion: a boxed result or
    an '=' inside a math delimiter near the very end of the text."""
    tail = ans.strip()[-_TAIL_CHARS:]
    if "\\boxed" in tail:
        return True
    if ("$" in tail or "\\]" in tail or "\\)" in tail) and _RELATION_PAT.search(tail):
        return True
    return False


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0

        norm = _safe_call(ops.normalize, text, text)
        if not isinstance(norm, str) or not norm.strip():
            norm = text

        m = _ANSWER_MARK.search(norm)
        ans = norm[m.end():] if m else norm
        if not ans.strip():
            ans = norm

        # ---- code-only structural signals (Answer-only text) ----
        skel = _safe_call(ops.proof_skeleton, {}, ans)
        if not isinstance(skel, dict):
            skel = {}

        def cat_count(name):
            try:
                return len(skel.get(name, []) or [])
            except Exception:
                return 0

        n_def = cat_count("definition")
        n_qed = cat_count("qed")
        n_case = cat_count("case")
        n_thm = cat_count("theorem")
        n_conn = cat_count("connective")
        n_proof = cat_count("proof")
        structure = n_case + n_thm + min(n_conn, 5) + min(n_proof, 3)

        eq_stats = _safe_call(ops.equation_stats, (0, 0, 0, 0.0), ans)
        try:
            n_display, n_inline, n_numbered, avg_tokens = eq_stats
        except Exception:
            n_display, n_inline, n_numbered, avg_tokens = (0, 0, 0, 0.0)

        spans = _safe_call(ops.extract_math_spans, [], ans)
        if not isinstance(spans, list):
            spans = []
        eq_spans = 0
        boxed = False
        for item in spans:
            try:
                kind, content = item
            except Exception:
                continue
            if not isinstance(content, str):
                continue
            if "\\boxed" in content:
                boxed = True
            # round 3: an existence/uniqueness conclusion is often an
            # INEQUALITY or set-relation, not just an equality (e.g. a trace
            # upper bound) -- count any relation operator, not only "=".
            if _RELATION_PAT.search(content) and len(content.strip()) > 1:
                eq_spans += 1

        dh = _safe_call(ops.delimiter_health, {}, ans)
        bad = 0
        if isinstance(dh, dict):
            for v in dh.values():
                if isinstance(v, (int, float)):
                    bad += v

        hedges = sum(len(p.findall(ans)) for p in _HEDGE_PATS)
        quant = sum(len(p.findall(ans)) for p in _QUANT_PATS)
        uniq = sum(len(p.findall(ans)) for p in _UNIQ_PATS)
        n_scope = n_def + sum(len(p.findall(ans)) for p in _SCOPE_PATS)
        n_resolve = n_qed + n_conn + sum(len(p.findall(ans)) for p in _RESOLVE_PATS)

        n_bindings = len(_BINDING_PAT.findall(ans))
        n_precise_exists = sum(len(p.findall(ans)) for p in _PRECISE_EXISTS_PATS)
        ends_eq = _ends_with_equation(ans)

        has_concrete_math_code = (eq_spans >= 1) or boxed or (n_numbered >= 1) or (n_bindings >= 1)
        # a hedge means any leftover equations are from an abandoned attempt,
        # not a delivered witness -- disable the code-only witness fallback.
        has_concrete_math_code = has_concrete_math_code and hedges == 0

        # ---- structural axis (secondary; code evidence of proof craft) ----
        struct = 0.0
        struct += 0.08 * min(1.0, n_scope / 4.0)
        struct += 0.07 * min(1.0, structure / 4.0)
        struct += 0.08 * min(1.0, (eq_spans + (1 if boxed else 0)) / 4.0)
        struct += 0.04 * min(1.0, (n_display + n_numbered) / 2.0)
        struct += 0.05 * min(1.0, quant / 3.0)
        struct += 0.03 * min(1.0, uniq / 2.0)
        struct += 0.09 * min(1.0, n_bindings / 2.0)
        struct += 0.05 * min(1.0, n_precise_exists / 2.0)
        struct += 0.06 * (1.0 if (ends_eq and hedges == 0) else 0.0)
        struct -= 0.10 * min(1.0, hedges / 2.0)
        struct -= 0.04 * min(1.0, bad / 5.0)

        # ---- LLM-grounded axis (primary; witness + completion semantics) ----
        extracted = extracted or {}
        witness_txt = str(extracted.get("witness", "") or "").strip()
        has_witness_llm = bool(witness_txt) and witness_txt.strip().upper() not in ("NONE", "N/A", "NA")
        has_witness = has_witness_llm or has_concrete_math_code

        support = n_bindings + quant + n_precise_exists
        witness_concrete = bool(has_witness_llm and _CONCRETE_WIT_PAT.search(witness_txt))
        witness_bonus = 0.0
        if has_witness_llm and witness_concrete:
            witness_bonus = 0.27 if support > 0 else 0.15
        elif has_witness_llm:
            # round 3: softened from round 2's 0.12/0.06 -- an abstract-noun
            # witness (a named map/set with no "=") still reliably beats no
            # witness at all (d04994/d00671/d03004 are genuine high-judge
            # items with exactly this register); only a mild discount vs the
            # concrete-formula case is warranted, not a >2x cut.
            witness_bonus = 0.22 if support > 0 else 0.12
        elif has_concrete_math_code:
            # derivation-chain evidence: more relation-spans (a full chain of
            # equations, not one leftover formula) is stronger code-only
            # witness evidence when there's no LLM witness/scope word to lean
            # on (e.g. direct symbolic-substitution answers with no "let").
            # Smooth in eq_spans rather than a single threshold.
            witness_bonus = min(0.18, 0.08 + 0.02 * eq_spans)
        else:
            witness_bonus = -0.05

        completed_txt = str(extracted.get("completed", "") or "").strip().lower()
        code_completed = (ends_eq or n_qed >= 1 or n_bindings >= 1) and hedges == 0
        if completed_txt.startswith("y"):
            # round 3: flat bonus restored (round 2's concreteness-gated
            # 0.20/0.09 split regressed train rho -- too many genuine
            # completions have an empty/abstract witness field).
            comp_adj = 0.20
        elif completed_txt.startswith("n"):
            comp_adj = -0.03 if code_completed else -0.13
        else:
            comp_adj = 0.08 if code_completed else -0.03
        if hedges >= 2:
            # TWO OR MORE co-occurring hedge phrases (e.g. "stopped early" +
            # "did not work out") is strong evidence of an abandoned attempt
            # that outranks a "completed=Yes" label the extractor got wrong;
            # a single incidental match is weaker evidence and, per round-4
            # residuals, was over-triggering on answers that hedge mildly
            # while still reaching a genuine conclusion -- require 2+, matching
            # the threshold the structural axis already uses for a full
            # hedge penalty (min(1, hedges/2)).
            comp_adj = min(comp_adj, -0.05)

        final = 0.05 + struct + witness_bonus + comp_adj
        return _clip01(final)
    except Exception:
        return 0.5
