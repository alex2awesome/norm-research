"""
a144 h1: Existential proofs: explicit witnesses and scope.

h0 (train rho 0.564) already separates clearly-punting answers (an
"Observation 1/2/3..."-chain that never resolves, "left as an exercise", etc.)
from real derivations reasonably well -- its zero-score train anchors are all
genuine punts and h0 matches the judge there. Its systematic miss is on the
OTHER end: answers that DO construct and use a witness but are phrased in a
register h0's signals don't reward:

  - "Hint:"-prefixed answers that nonetheless justify every step (fix eps,
    choose M/N, get delta, conclude) -- judge high, h0 near zero.
  - Socratic/guided-question derivations ("are these linearly independent?
    how many are there?") that still hand the reader an exact construction
    (e.g. a spanning-forest basis) -- judge high, h0 near zero.
  - direct symbolic manipulation (coordinate substitution, epsilon-delta
    chains, an explicit coordinate-change function) that never says
    "let"/"there exists" but plainly builds and uses a concrete object.
  - counterexample constructions (an explicit traceless block matrix) that
    the witness-extractor field missed because it isn't phrased as
    "the answer is X".

The common root cause: h0 leans hard on a single LLM binary ("completed":
yes/no, +0.15/-0.20 swing) and a single short LLM field ("witness") as the
ONLY source of witness evidence. Both are useful but noisy, forced-choice
reads of a genuinely hard judgment (h0's own notes concede "Hint:"-prefixed
answers span the full quality range) -- reliable enough to reward a clear
"yes", but frequently wrong in the *negative* direction on hedge/Socratic/
computational answers. This h1, as a GENERAL (not excerpt-specific) fix:

  1. Detects a concrete witness two ways -- the LLM field OR a code-side
     check for a defining equation/boxed/numbered result in the (Answer-only)
     math spans -- so a computed counterexample or coordinate-change formula
     counts even when the LLM field returns empty.
  2. Broadens the code-side scope/definition vocabulary a little (adds
     consider/suppose/assume and ":="/"\\mapsto" as symbolic
     object-introduction markers) since these are standard, generic
     proof-writing moves, not specific to any one excerpt.
  3. Softens (does not remove or conditionally override) the LLM "completed"
     penalty -- the negative swing is roughly halved. A flat, register-
     agnostic softening is lower-risk than trying to have code decide when to
     trust a "no", since code genuinely cannot always tell a verified
     derivation from a confident-but-unverified assertion.
  4. Keeps everything else close to h0 (answer-only scoring, hedge penalty,
     quantifier/uniqueness bonuses, delimiter-health penalty) since h0's
     zero-score train anchors already match the judge there.
"""

import re


LLM_FIELDS = {
    "witness": (
        "In <=15 words, the specific value, function, set, or counterexample "
        "the answer exhibits, computes, or constructs to satisfy or refute "
        "the existence claim, or NONE if none is given"
    ),
    "completed": (
        "Answer yes or no: does the answer reach a definite, justified "
        "conclusion (even if phrased as a hint, informal computation, or "
        "step-by-step questions), rather than stopping at an unresolved "
        "sketch or abandoning the attempt"
    ),
}

_ANSWER_MARK = re.compile(r"\bAnswer\s*:", re.IGNORECASE)

_HEDGE_PATS = [
    re.compile(r"should (give|work|prove|do the trick)", re.IGNORECASE),
    re.compile(r"i(?:'m| am) not (?:really )?sure", re.IGNORECASE),
    re.compile(r"i don't (?:see|know) how", re.IGNORECASE),
    re.compile(r"left (?:to|as) (?:the reader|an exercise)", re.IGNORECASE),
    re.compile(r"without success", re.IGNORECASE),
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

# Scope / "introduce an object" markers -- broadened a little from h0's list
# with a few more standard proof-writing verbs and symbolic-definition
# notations, not keyed to any particular excerpt.
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

# Resolution/conclusion markers supplementing ops.proof_skeleton's
# "connective"/"qed" categories with a few more generic closing phrases.
_RESOLVE_PATS = [
    re.compile(r"\bconclude[sd]?\b", re.IGNORECASE),
    re.compile(r"it follows that", re.IGNORECASE),
    re.compile(r"which (shows|proves)", re.IGNORECASE),
    re.compile(r"we deduce", re.IGNORECASE),
]


def _safe_call(fn, default, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _clip01(x):
    return max(0.0, min(1.0, x))


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
            if "=" in content and len(content.strip()) > 1:
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

        # Concrete mathematical object: a defining equation, a boxed result,
        # or a numbered/displayed equation -- register-agnostic evidence that
        # SOME specific value/function/set appears, independent of whether the
        # LLM's short "witness" field happened to name it.
        has_concrete_math = (eq_spans >= 1) or boxed or (n_numbered >= 1)

        raw = 0.05
        raw += 0.14 * min(1.0, n_def / 3.0)
        raw += 0.10 * min(1.0, n_qed / 2.0)
        raw += 0.07 * min(1.0, structure / 4.0)
        raw += 0.14 * min(1.0, (eq_spans + (1 if boxed else 0)) / 3.0)
        raw += 0.07 * min(1.0, (n_display + n_numbered) / 2.0)
        raw += 0.06 * min(1.0, n_scope / 4.0)
        raw += 0.05 * min(1.0, n_resolve / 3.0)
        raw += 0.06 * min(1.0, quant / 3.0)
        raw += 0.04 * min(1.0, uniq / 2.0)
        raw -= 0.08 * min(1.0, hedges / 2.0)
        raw -= 0.04 * min(1.0, bad / 5.0)

        # ---- LLM-grounded signal: witness presence + hint-vs-completed ----
        extracted = extracted or {}
        witness_txt = str(extracted.get("witness", "") or "").strip()
        has_witness_llm = bool(witness_txt) and witness_txt.strip().upper() not in ("NONE", "N/A", "NA")
        has_witness = has_witness_llm or has_concrete_math

        completed_txt = str(extracted.get("completed", "") or "").strip().lower()
        if completed_txt.startswith("y"):
            comp_adj = 0.12
        elif completed_txt.startswith("n"):
            # Softened vs h0's -0.20: the "no" read is frequently triggered by
            # surface register (a "Hint:" prefix, Socratic questions, informal
            # computation) rather than genuine non-completion, per the residual
            # evidence -- but it is still somewhat informative on average, so
            # we keep a real (just smaller) penalty rather than discarding it.
            comp_adj = -0.11
        else:
            comp_adj = 0.0

        wit_adj = 0.12 if has_witness else -0.04

        final = raw + wit_adj + comp_adj
        return _clip01(final)
    except Exception:
        return 0.5
