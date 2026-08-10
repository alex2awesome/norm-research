"""a5 hybrid: code ranks weight-language ("great/some/little weight to X") assigned to non-examining/state-agency vs examining/treating sources; LLM fields carry the comparison verdict and whether the credited opinion is stale relative to later evidence."""
import re

LLM_FIELDS = {
    "weight_comparison": (
        "Did the ALJ give a non-examining/state-agency opinion equal or "
        "greater weight than an examining or treating source's opinion?"
    ),
    "stale_evidence": (
        "Name any significant new medical evidence (worsening, surgery, "
        "imaging) after the credited opinion's review date, else NONE."
    ),
}

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


_WEIGHT_ASSIGN = re.compile(
    r"\b(great|significant|substantial|considerable|controlling|full|greater|"
    r"some|partial|moderate|little|minimal|no|reduced|less|small)\s+weight\s+"
    r"(?:was\s+given\s+)?to\s+([^,;.]{3,90})", re.I)

_RANK = {
    "great": 2, "significant": 2, "substantial": 2, "considerable": 2,
    "controlling": 2, "full": 2, "greater": 2,
    "some": 1, "partial": 1, "moderate": 1,
    "little": 0, "minimal": 0, "no": 0, "reduced": 0, "less": 0, "small": 0,
}

_NONEXAM_RE = re.compile(r"state.?agency|non-?examin", re.I)
_EXAM_RE = re.compile(r"treating|examining|consultative|\bdr\.\s*\w+", re.I)


def _weight_signal(t):
    """Best rank found for a non-examining source vs. an examining/treating
    source in explicit weight-assignment language; returns a 0..1 term where
    higher means the ALJ favored (or exclusively mentioned) the non-examining
    source."""
    best_non = best_exam = None
    for word, src in _WEIGHT_ASSIGN.findall(t):
        r = _RANK.get(word.lower())
        if r is None:
            continue
        if _NONEXAM_RE.search(src):
            best_non = r if best_non is None else max(best_non, r)
        elif _EXAM_RE.search(src):
            best_exam = r if best_exam is None else max(best_exam, r)
    if best_non is None:
        return 0.0
    if best_exam is None:
        return 0.15 * (best_non + 1)
    diff = best_non - best_exam
    return max(0.0, min(1.0, (diff + 2) / 4.0))


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""
        ex = extracted if isinstance(extracted, dict) else {}

        w_sig = _weight_signal(t)
        base = 0.2 + 0.3 * w_sig

        wc = str(ex.get("weight_comparison") or "").lower()
        if re.search(r"\b(greater|equal|more|same|yes)\b", wc):
            base += 0.25
        elif re.search(r"\b(less|lower|no|not)\b", wc):
            base -= 0.15

        if not _is_none(ex.get("stale_evidence")):
            base += 0.25

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
