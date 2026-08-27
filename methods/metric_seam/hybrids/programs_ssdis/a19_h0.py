"""a19 hybrid: code regex flags an RFC "simple/routine" limitation and an explicit reasoning-level-3 mention as a fallback (no DOT registry available); two LLM fields quote the RFC limitation and the cited job/level for a conjunctive decision table."""

# Criterion is a conjunction: the RFC restricts the claimant to simple/
# routine/unskilled work AND a cited step-5 DOT occupation requires GED
# Reasoning Level 3 -- a facial mismatch courts treat as a live conflict
# under SSR 00-4p. Higher score = stronger evidence BOTH halves hold.
#
# Design: the module has no DOT code -> reasoning-level registry (ops does
# not provide one, and hardcoding an occupation-name lookup risks asserting
# unverified vocational facts), so reasoning-level detection is TEXT-
# EXPLICIT only -- either a literal "Reasoning Level 3"/"R3" mention, or an
# LLM field that reads the decision/VE-testimony discussion (this corpus's
# narratives, being remand-basis case facts, tend to state the level
# explicitly when it is actually litigated). The RFC's simple/routine
# language is straightforward boilerplate a regex catches reliably. Code
# owns both predicates' presence/absence tables and multiplies them
# (AND semantics); LLM fields carry the thicker, paraphrase-tolerant reads
# and are trusted over the regex hedge when present.
import re

LLM_FIELDS = {
    "rfc_mental_limit": (
        "Quote the RFC's mental-limitation phrase restricting the claimant "
        "to simple/routine/unskilled work, or NONE if the RFC has no such "
        "restriction."
    ),
    "job_reasoning_level": (
        "State the GED Reasoning Level (e.g. 'Level 3') of a cited step-5 "
        "DOT job if stated, or name the job itself; NONE if no step-5 job "
        "or reasoning level is discussed."
    ),
}

_SIMPLE_RFC = re.compile(
    r"\bsimple\b[^.]{0,20}(?:routine|repetitive|tasks|instructions)|"
    r"\broutine\b[^.]{0,20}(?:simple|tasks)|\bunskilled\b", re.I)
_R3_EXPLICIT = re.compile(
    r"reasoning\s+level\s*3|\blevel\s*3\b[^.]{0,15}reasoning|"
    r"ged\s+reasoning[^.]{0,10}3|\bR3\b", re.I)
_LEVEL3_CUE = re.compile(r"\b3\b|level\s*3|reasoning\s*3", re.I)

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""

        code_simple = bool(_SIMPLE_RFC.search(t))
        code_r3 = bool(_R3_EXPLICIT.search(t))

        ex = extracted if isinstance(extracted, dict) else {}
        rfc_raw = str(ex.get("rfc_mental_limit") or "").strip()
        level_raw = str(ex.get("job_reasoning_level") or "").strip()

        if _is_none(rfc_raw):
            simple_signal = 0.1 if code_simple else 0.0
        else:
            simple_signal = 1.0

        if _is_none(level_raw):
            r3_signal = 0.1 if code_r3 else 0.0
        elif _LEVEL3_CUE.search(level_raw):
            r3_signal = 1.0
        else:
            r3_signal = 0.5

        final = simple_signal * r3_signal
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
