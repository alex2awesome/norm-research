"""a14 hybrid: Replication, Robustness, and Uncertainty.

Criterion: experiments should report replicates, demonstrate statistical
significance, propagate uncertainty, and quantify variation across runs.

CODE sees uncertainty/significance VOCABULARY density (std, CI, error bars,
p-values, ...) as a weak structural proxy (weight 0.35) -- weak because this
vocabulary frequently appears when a paper only CITES another work's
uncertainty reporting rather than reporting its own. The two LLM fields
carry the thick-input grounding needed to attribute the measure to the
paper's OWN result: the uncertainty measure actually reported for THIS
paper's own main result, and the number of independent runs/seeds used for
it. Both are checked GROUNDED against the document text before being
credited; code maps ungrounded/NONE answers to zero rather than trusting
them at face value.
"""
import re

LLM_FIELDS = {
    "own_uncertainty_measure": (
        "State in <=10 words this paper's own uncertainty measure "
        "(std/CI/error bars), or NONE."
    ),
    "num_replicates": (
        "State the number of independent runs/seeds used for this "
        "paper's own result, or NONE."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_UNCERTAINTY_REG = re.compile(
    r"\b(std|standard deviation|confidence interval|error bars?|variance)\b|±|\+/-|\+-",
    re.I,
)
_SIG_REG = re.compile(
    r"\b(p\s*[<=]\s*0?\.\d+|p-value|significant(?:ly)?|t-test|wilcoxon|anova|bootstrap)\b",
    re.I,
)
_REPLICATE_NUM_REG = re.compile(
    r"\b(\d+)\s*(?:runs?|seeds?|trials?|replicates?|repetitions?)\b", re.I
)
_STOP = {"the", "a", "an", "and", "or", "but", "for", "with", "that", "this",
         "its", "was", "were", "is", "are", "of", "to", "in", "on", "at", "as"}


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _content_words(s):
    return [w for w in re.findall(r"[A-Za-z']+", s.lower()) if len(w) >= 3 and w not in _STOP]


def _grounded(val, text_lower):
    if _is_none(val):
        return False
    words = _content_words(val)
    if not words:
        digits = re.findall(r"\d+", val)
        return any(d in text_lower for d in digits)
    hits = sum(1 for w in words if w in text_lower)
    return hits / len(words) >= 0.4


def _structural(t):
    s = 0.0
    if _UNCERTAINTY_REG.search(t):
        s += 0.4
    if _SIG_REG.search(t):
        s += 0.3
    if _REPLICATE_NUM_REG.search(t):
        s += 0.3
    return min(1.0, s)


def _replicate_credit(val):
    if _is_none(val):
        return 0.0
    nums = [int(d) for d in re.findall(r"\d+", val)]
    if not nums:
        return 0.3  # stated non-numerically (e.g. "multiple") -- weak credit
    n = max(nums)
    if n <= 1:
        return 0.2
    return min(1.0, 0.2 + 0.2 * min(n, 5))  # saturates near n=5


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        tl = t.lower()
        extracted = extracted or {}

        structural = _structural(tl)

        measure = extracted.get("own_uncertainty_measure")
        n_rep = extracted.get("num_replicates")
        measure_credit = 1.0 if _grounded(measure, tl) else 0.0
        rep_credit = _replicate_credit(n_rep) if _grounded(n_rep, tl) else 0.0

        llm_score = 0.5 * measure_credit + 0.5 * rep_credit
        final = 0.35 * structural + 0.65 * llm_score
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
