"""a12 Precision/rigor in proofs - hybrid (LaTeX-aware parsing + LLM gap-detection).

Rationale (see improver pack a12.json known_failure_pattern): raw keyword counts
(proof/lemma/hence/therefore/quantifier/equation-presence) are a weak proxy for
rigor (baseline_train_rho=0.123) because presence of proof-y words does not mean
the reasoning is actually closed. The judge appears to reward answers whose
argument is fully justified end-to-end (no "clearly"/"it is easy to see"/"I'll
leave the rest to you" gaps, no truncated asymptotic hand-waving via repeated
\\cdots), with clean LaTeX, and it heavily punishes non-answers (pointers to
external resources, off-topic tangents, purely discursive prose with no derived
claim) regardless of how much surface math/keyword-scaffolding surrounds them.
Two LLM fields externalize exactly the judgment a parser cannot make: whether a
specific step is unjustified, and whether anything concrete was actually derived.
"""

import re

LLM_FIELDS = {
    "unjustified_step": (
        "Name ONE specific claim or step in the ANSWER that is stated without "
        "proof/justification (or that is left for the reader to finish); "
        "say NONE if every step is justified and the argument is complete."
    ),
    "concrete_result": (
        "State in <=15 words the specific fact/equation/claim the ANSWER "
        "actually derives or proves about THIS question; say NONE if it "
        "derives nothing concrete (e.g. it only redirects, hints vaguely, "
        "or is off-topic)."
    ),
}

_HANDWAVE_RE = re.compile(
    r"\b(clearly|obvious(?:ly)?|trivial(?:ly)?|it is easy to see|it's easy to see|"
    r"as is well[- ]known|evidently|self[- ]evident)\b",
    re.IGNORECASE,
)
_ELLIPSIS_RE = re.compile(r"(\\cdots|\\ldots|\\dots|\.\.\.|…)")
_ANSWER_SPLIT_RE = re.compile(r"\n\s*answer\s*:\s*", re.IGNORECASE)

_STRUCT_CATS = ("theorem", "definition", "proof", "qed", "case", "connective")


def _none_str(s):
    """True if an LLM field value should be treated as absent/NONE."""
    if not s:
        return True
    s2 = s.strip().strip(".").strip().lower()
    return s2 in ("", "none", "n/a", "na", "null")


def _split_answer(t):
    parts = _ANSWER_SPLIT_RE.split(t)
    return parts[-1] if len(parts) >= 2 and parts[-1].strip() else t


def _safe_call(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:
        return default


def score(text: str, extracted: dict, ops) -> float:
    try:
        extracted = extracted or {}
        raw = text or ""
        if len(raw.strip()) < 15:
            return 0.1

        t = _safe_call(ops.normalize, raw, default=raw) or raw
        ans = _split_answer(t)
        has_elision = "[...]" in t

        n_words = max(1, len(ans.split()))

        # ---- math density (on the ANSWER only, not the question) ----
        n_disp, n_inl, n_num, avg_tok = _safe_call(
            ops.equation_stats, ans, default=(0, 0, 0, 0.0)
        ) or (0, 0, 0, 0.0)
        n_spans = n_disp + n_inl
        math_present = n_spans > 0

        # ---- notation precision: penalize malformed LaTeX in the answer ----
        dh = _safe_call(ops.delimiter_health, ans, default={}) or {}
        try:
            issues = sum(v for v in dh.values() if isinstance(v, (int, float)))
        except Exception:
            issues = 0
        issue_rate = issues / max(1, n_spans if n_spans else 1)
        if has_elision:
            issue_rate *= 0.5  # elisions can spuriously break delimiter parity
        notation_score = max(0.0, 1.0 - min(1.0, issue_rate))

        # ---- proof/argument structure (presence + closure), kept WEAK ----
        ps = _safe_call(ops.proof_skeleton, ans, default={}) or {}
        cats_present = sum(1 for k in _STRUCT_CATS if ps.get(k))
        presence_score = min(1.0, cats_present / 3.0)

        closure_bonus = 0.0
        total_len = max(1, len(ans))
        last_offsets = []
        for cat in ("connective", "qed"):
            for item in ps.get(cat) or []:
                try:
                    last_offsets.append(item[0])
                except Exception:
                    pass
        if last_offsets and max(last_offsets) >= 0.7 * total_len:
            closure_bonus = 1.0

        case_bonus = 1.0 if len(ps.get("case") or []) >= 2 else 0.0

        structure_score = min(1.0, 0.6 * presence_score + 0.4 * closure_bonus + 0.2 * case_bonus)

        # ---- hand-wave penalty: skipped justification / truncated expansions ----
        handwave_hits = len(_HANDWAVE_RE.findall(ans))
        ellipsis_hits = len(_ELLIPSIS_RE.findall(ans))
        handwave_rate = (handwave_hits + max(0, ellipsis_hits - 2) * 0.5) / max(1.0, n_words / 60.0)
        handwave_score = max(0.0, 1.0 - min(1.0, handwave_rate))

        # ---- grounded signals from the LLM extractor ----
        unjustified = extracted.get("unjustified_step", "")
        justification_score = 0.15 if not _none_str(unjustified) else 1.0

        concrete = extracted.get("concrete_result", "")
        substantive = not _none_str(concrete)

        thin_answer = (n_words < 12) and not math_present

        base = (
            0.45 * justification_score
            + 0.25 * notation_score
            + 0.15 * structure_score
            + 0.15 * handwave_score
        )

        if not substantive:
            base = min(base, 0.18)
        if thin_answer:
            base = min(base, 0.15)

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
