"""
Hybrid metric channel for a84: "Aesthetic virtues and elegance" (Math StackExchange answers).

Rationale (see train-set residual study):
  - The judge rewards DECISIVE, insight-carrying answers -- often short, invoking a named
    theorem/law/identity, or a minimal surprising trick -- and punishes (a) answers that
    dodge the question (meta-commentary, counter-questions, "see elsewhere") and
    (b) answers that are technically correct but "merely intricate": long runs of bare
    LaTeX with almost no narrating language, or a single bulky/unwieldy closed form
    (giant polynomial, dumped numeric table). Two near-identical short answers can land
    at opposite ends of the judge scale depending on whether the trick is a genuine
    "aha" or a pedestrian restatement -- that distinction is NOT visible to a parser, so
    it is delegated to the two LLM fields below. Everything else (engagement tier,
    equation-dump / bulk-formula detection, case-bashing, LaTeX hygiene) stays in code
    over the LaTeX-aware ops, since keyword-vocabulary presence ("elegant", "deep", ...)
    is a known-bad proxy here (baseline_train_rho 0.122 from exactly that approach).
"""
import re

LLM_FIELDS = {
    "engagement": (
        "Reply with exactly one word: 'solves' if the answer reaches a definitive "
        "conclusion to the question, 'partial' if it is an incomplete hint/outline/sketch, "
        "or 'deflects' if it avoids answering (asks a question back, says it can't help, "
        "or just points elsewhere without content)."
    ),
    "key_move": (
        "Name ONLY a specific named theorem, law, or classical identity the answer "
        "explicitly invokes as its key step (e.g. 'AM-GM inequality', 'Schur's Lemma', "
        "'Gauss-Bonnet theorem'). If it only uses generic algebra/computation with no "
        "named theorem, reply NONE."
    ),
}

_TIER_BASE = {"deflects": 0.10, "partial": 0.45, "solves": 0.60}
_INSIGHT_BONUS = {"deflects": 0.0, "partial": 0.08, "solves": 0.22}
_BAD_KM = {"", "NONE", "N/A", "NA", "NONE.", "NO", "NOTHING", "NONE,"}


def _tier_from_engagement(raw: str) -> str:
    s = (raw or "").strip().lower()
    if any(k in s for k in ("deflect", "avoid", "redirect", "no answer",
                             "doesn't", "does not", "can't help", "cannot help")):
        return "deflects"
    if any(k in s for k in ("partial", "hint", "sketch", "outline", "incomplete")):
        return "partial"
    if any(k in s for k in ("solv", "complete", "defin", "resolv", "yes")):
        return "solves"
    return "partial"  # unrecognized / empty -> conservative middle, never a tier extreme


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.5
        t = ops.normalize(text)

        # Corpus convention is "Question: ... \n\n Answer: ..."; judge only cares about
        # the answer, so isolate it and run all structural ops on that portion only.
        m = re.search(r"\bAnswer\s*:\s*", t)
        atext = t[m.end():].strip() if m else t.strip()
        if not atext:
            atext = t.strip()

        eng_raw, km_raw = "", ""
        try:
            eng_raw = extracted.get("engagement", "") or ""
            km_raw = extracted.get("key_move", "") or ""
        except Exception:
            pass

        tier = _tier_from_engagement(eng_raw)
        km = km_raw.strip()
        has_insight = km.upper().rstrip(".") not in _BAD_KM and len(km) > 1

        val = _TIER_BASE[tier]
        if has_insight:
            val += _INSIGHT_BONUS[tier]

        # --- deterministic structural signals, computed on the answer only ---
        try:
            n_disp, n_inl, n_num, avg_tok = ops.equation_stats(atext)
        except Exception:
            n_disp, n_inl, n_num, avg_tok = (0, 0, 0, 0.0)
        n_spans = (n_disp or 0) + (n_inl or 0)

        try:
            skel = ops.proof_skeleton(atext)
        except Exception:
            skel = {}
        n_conn = len(skel.get("connective", []) or [])
        n_case = len(skel.get("case", []) or [])

        clarity = 1.0
        # "merely intricate without insight": long run of math, almost no narration
        # (thus/hence/since/recall/...) connecting the steps.
        if n_spans >= 4:
            conn_ratio = n_conn / float(n_spans + 1)
            clarity *= min(1.0, 0.35 + 1.6 * conn_ratio)
        # a single bulky/unwieldy closed form (giant polynomial, dumped table row) drags
        # the mean token length per span up sharply; penalize smoothly past a threshold.
        if avg_tok and avg_tok > 30:
            clarity *= max(0.35, 1.0 - (avg_tok - 30) / 70.0)

        val *= clarity

        # case-bashing penalty (structural marker, not a literal keyword count).
        val -= min(0.15, 0.04 * n_case)

        # malformed/unbalanced LaTeX cuts against the "crisp / clean" aesthetic virtue.
        try:
            dh = ops.delimiter_health(atext) or {}
            sloppy = sum(v for v in dh.values() if isinstance(v, (int, float)))
        except Exception:
            sloppy = 0
        if sloppy and sloppy > 2:
            val -= 0.05

        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
