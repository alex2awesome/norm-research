"""a72 hybrid: Hypothesis Plausibility and Logical Argument.

Construct: ~1.0 = a hypothesis/claim is stated and the argument connecting evidence to
conclusion uses clear logical connectives (because/since/therefore/thus) with no apparent
gap or contradiction; ~0.5 = a hypothesis is present but the argument chain is thin (few
connectives) or partially hedged; ~0.0 = no hypothesis is stated, or the LLM flags a clear
logical gap/non-sequitur in the reasoning.

INPUT = abstract/excerpt only. Code sees: density of logical-connective markers per sentence
(via ops.sent_stats), contradiction/hedge-marker density. Code CANNOT judge whether the
hypothesis is semantically plausible or whether the argument actually holds together —
LLM_FIELDS carry the hypothesis text and a direct flaw judgment that regex can't reach.
"""
import re

LLM_FIELDS = {
    "hypothesis": (
        "In <=25 words, the paper's core hypothesis or main claim as stated. Answer NONE if "
        "no hypothesis or claim is stated."
    ),
    "argument_flaw": (
        "One word: does the excerpt's reasoning contain a clear logical gap, contradiction, "
        "or non-sequitur? 'yes', 'no', or 'unclear'."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_CONNECTIVE_RE = re.compile(
    r"\b(because|since|therefore|thus|hence|consequently|as a result|given that|"
    r"it follows that|due to|which implies|leads to)\b", re.I)
_CONTRA_RE = re.compile(r"\b(however|contradict|nonetheless|although|despite|surprisingly)\b", re.I)
_HEDGE_RE = re.compile(r"\b(may|might|could|suggest|possibly|perhaps|seems?|appears? to)\b", re.I)


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _code_score(text, extracted, ops):
    hyp = extracted.get("hypothesis")
    has_hyp = 0.0 if _is_none(hyp) else 1.0
    n_sent = 1
    try:
        n_sent, _, _ = ops.sent_stats(text) if (ops and text) else (1, 0.0, 0.0)
    except Exception:
        pass
    n_sent = max(1, n_sent)
    conn_density = len(_CONNECTIVE_RE.findall(text)) / n_sent
    contra_density = len(_CONTRA_RE.findall(text)) / n_sent
    hedge_density = len(_HEDGE_RE.findall(text)) / n_sent
    structure = min(1.0, conn_density * 3.0)
    s = 0.45 * has_hyp + 0.40 * structure
    if contra_density > 0.3:
        s -= 0.10
    if hedge_density > 0.4:
        s -= 0.05
    return max(0.0, min(1.0, s))


def _llm_score(extracted):
    hyp = extracted.get("hypothesis")
    if _is_none(hyp):
        return 0.15
    flaw = (extracted.get("argument_flaw") or "").strip().lower()
    return {"no": 0.9, "unclear": 0.5, "yes": 0.15}.get(flaw, 0.5)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        final = 0.55 * _code_score(t, extracted, ops) + 0.45 * _llm_score(extracted)
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
