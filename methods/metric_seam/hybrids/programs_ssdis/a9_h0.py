"""a9 hybrid: code detects pro-se/unrepresented status and duty-to-develop phrasing mechanically; LLM fields carry the represented-status verdict and the specific evidentiary gap, which code cannot identify from keywords alone."""
import re

LLM_FIELDS = {
    "represented_status": (
        "Was the claimant unrepresented (pro se) at the ALJ hearing? "
        "Answer yes, no, or unclear."
    ),
    "evidence_gap": (
        "State the specific evidentiary gap alleged, e.g. missing records "
        "or no consultative exam, else NONE."
    ),
}

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


_UNREP_RE = re.compile(
    r"\bpro se\b|unrepresented|without (?:the assistance of\s+)?an? attorney|"
    r"no attorney|represent(?:ed|ing) (?:himself|herself|themselves)", re.I)
_DUTY_LANG_RE = re.compile(
    r"fail(?:ed|ure)? to (?:properly )?develop the record|duty to develop|"
    r"full and fair record|(?:failed to|did not) (?:obtain|order) a consultative exam|"
    r"missing (?:treatment )?records|gap in the (?:medical )?(?:record|evidence)", re.I)
_COMPLETE_RE = re.compile(
    r"record (?:was|is) complete|fully developed record|adequate record", re.I)


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""
        ex = extracted if isinstance(extracted, dict) else {}

        unrep_code = bool(_UNREP_RE.search(t))
        duty_code = bool(_DUTY_LANG_RE.search(t))
        complete_code = bool(_COMPLETE_RE.search(t))

        rep = str(ex.get("represented_status") or "").lower()
        unrep_field = bool(re.search(r"unrep|pro se|\byes\b", rep))

        base = 0.15
        if unrep_code or unrep_field:
            base += 0.25  # heightened duty when claimant is unrepresented
        if duty_code:
            base += 0.2

        gap = ex.get("evidence_gap")
        if not _is_none(gap):
            base += 0.3
        elif complete_code:
            base -= 0.1

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
