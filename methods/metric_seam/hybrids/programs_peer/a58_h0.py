"""a58 hybrid: Patient-Reported Outcome (PRO) Instruments.

Criterion: PRO instruments should be justified, with mode equivalence
established between electronic and paper versions.

CODE owns a hard GATE: if no PRO-instrument vocabulary (patient-reported
outcome, PROM, or a named instrument like PHQ-9/EQ-5D/SF-36) appears
anywhere in the document, the criterion cannot be exhibited and the score
is a deterministic 0.0 -- no LLM judgment needed to decide that. When the
gate opens, the two LLM fields supply the thick-input judgment regex cannot
make: whether the instrument choice is actually justified for the study
population, and whether electronic/paper mode-equivalence evidence is
given. Both are credited only if GROUNDED against the text; code owns the
predicate rather than trusting the LLM's own framing.
"""
import re

LLM_FIELDS = {
    "pro_justification": (
        "State in <=15 words why the PRO instrument was chosen/validated "
        "for this population, or NONE."
    ),
    "mode_equivalence_evidence": (
        "State in <=12 words any evidence electronic and paper PRO "
        "versions score equivalently, or NONE."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_PRO_REG = re.compile(
    r"\bpatient[- ]reported outcomes?\b|\bproms?\b|\bpro instruments?\b", re.I
)
_NAMED_INSTR_REG = re.compile(
    r"\b(phq-9|eq-5d|sf-36|hrqol|epro|e-pro|promis)\b", re.I
)
_STOP = {"the", "a", "an", "and", "or", "but", "for", "with", "that", "this",
         "its", "was", "were", "is", "are", "of", "to", "in", "on", "at", "as"}


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _content_words(s):
    return [w for w in re.findall(r"[A-Za-z']+", s.lower()) if len(w) >= 4 and w not in _STOP]


def _grounded(val, text_lower):
    if _is_none(val):
        return False
    words = _content_words(val)
    if not words:
        return False
    hits = sum(1 for w in words if w in text_lower)
    return hits / len(words) >= 0.4


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        tl = t.lower()
        extracted = extracted or {}

        if not (_PRO_REG.search(tl) or _NAMED_INSTR_REG.search(tl)):
            return 0.0

        just = extracted.get("pro_justification")
        mode = extracted.get("mode_equivalence_evidence")
        just_credit = 1.0 if _grounded(just, tl) else 0.0
        mode_credit = 1.0 if _grounded(mode, tl) else 0.0

        named_bonus = 0.1 if _NAMED_INSTR_REG.search(tl) else 0.0
        final = 0.5 * just_credit + 0.4 * mode_credit + named_bonus
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
