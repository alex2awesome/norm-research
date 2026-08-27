"""a6 hybrid: code detects the disapproved credibility-boilerplate phrase mechanically (a thin, textbook signal); the LLM field carries whether the surrounding text supplies individualized reasons, which code cannot judge on its own."""
import re

LLM_FIELDS = {
    "individualized_reasons": (
        "Quote or summarize any specific record-grounded reason the ALJ "
        "gave for discounting symptoms beyond the template phrase, else NONE."
    ),
}

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


_BOILERPLATE_RE = re.compile(
    r"not entirely consistent with|not entirely credible|"
    r"not credible to the extent|"
    r"inconsistent with the (?:above )?residual functional capacity", re.I)


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""
        ex = extracted if isinstance(extracted, dict) else {}

        hits = len(_BOILERPLATE_RE.findall(t))
        if hits == 0:
            # the criterion is specifically about this template phrase --
            # if it never appears, this is a weak remand basis by definition
            return 0.12

        base = 0.65 + min(0.1, 0.03 * (hits - 1))
        if not _is_none(ex.get("individualized_reasons")):
            base -= 0.45  # boilerplate followed by real reasons -> harmless
        else:
            base += 0.2  # boilerplate with no visible individualized reasons

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
