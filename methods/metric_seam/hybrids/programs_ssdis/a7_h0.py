"""a7 hybrid: code textually detects a facial reasoning-level/simple-work mismatch and conflict-resolution language; LLM fields carry the specific RFC-vs-job mismatch and whether the ALJ actually probed and resolved it."""
import re

LLM_FIELDS = {
    "rfc_job_mismatch": (
        "State the specific mismatch between the RFC limitation and a "
        "cited job's DOT requirements, else NONE."
    ),
    "conflict_addressed": (
        "Did the ALJ ask about and resolve any DOT conflict for the "
        "cited jobs?"
    ),
}

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


_LEVEL_WORDS = {"one": 1, "1": 1, "two": 2, "2": 2, "three": 3, "3": 3, "four": 4, "4": 4}
_RL_RE = re.compile(r"reasoning level (?:of )?(\w+)", re.I)
_SIMPLE_RE = re.compile(r"\bsimple\b|\broutine\b", re.I)
_RESOLVE_RE = re.compile(
    r"no (?:apparent )?conflict|resolved the conflict|explained (?:the )?conflict|"
    r"consistent with the (?:DOT|Dictionary)|"
    r"asked (?:the )?(?:VE|vocational expert) (?:whether|if)", re.I)
_UNRESOLVED_RE = re.compile(
    r"fail(?:ed|ure) to (?:resolve|address|identify) (?:the )?conflict|"
    r"never (?:asked|resolved|addressed)|did not (?:ask|resolve|address)|"
    r"unresolved conflict|apparent conflict", re.I)


def _max_level(t):
    best = 0
    for m in _RL_RE.finditer(t):
        v = _LEVEL_WORDS.get(m.group(1).lower())
        if v:
            best = max(best, v)
    return best


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""
        ex = extracted if isinstance(extracted, dict) else {}

        level = _max_level(t)
        facial = level >= 3 and bool(_SIMPLE_RE.search(t))
        resolved = bool(_RESOLVE_RE.search(t))
        unresolved = bool(_UNRESOLVED_RE.search(t))

        base = 0.15
        if facial:
            base += 0.25
        if unresolved:
            base += 0.2
        if resolved and not unresolved:
            base -= 0.1

        if not _is_none(ex.get("rfc_job_mismatch")):
            base += 0.25

        addressed = str(ex.get("conflict_addressed") or "").lower()
        if re.search(r"\byes\b|resolved|addressed|explained", addressed):
            base -= 0.35
        elif re.search(r"\bno\b|not addressed|never", addressed):
            base += 0.2

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
