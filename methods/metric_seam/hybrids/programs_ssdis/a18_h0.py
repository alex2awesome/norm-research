"""a18 hybrid: code regex extracts national job-number figures as a fallback and runs a ratio-to-floor decision table like a14's SGA arithmetic; one LLM field carries the disambiguated NATIONAL (not regional) job figures."""

# Criterion tests the total cited national job numbers across step-5
# occupations against the "significant numbers" floor courts apply (no
# bright line in the case law; commonly-cited figures cluster well below
# 25,000 and well above ~1,000). Higher score = the cited total sits more
# comfortably above a documented, best-effort floor.
#
# Design: job-number figures are a pure extractable quantity, but a step-5
# narrative typically cites SEVERAL numbers -- national totals for each
# occupation, but also regional/state figures the VE distinguished, and
# sometimes unrelated numbers (past relevant work statistics). A bare regex
# scan for "<number> jobs" can't reliably tell national from regional, so an
# LLM field is used for that disambiguation and summed in code; a parallel
# regex fallback (excluding numbers near "regional"/"state"/"local" cues)
# is kept as a hedge and is the sole signal when the field is missing.
import re

LLM_FIELDS = {
    "job_numbers_national": (
        "List each cited step-5 occupation's NATIONAL (not regional/state/"
        "local) job-number figure, comma-separated numerals (e.g. '32000, "
        "15000'); NONE if no national job numbers are cited."
    ),
}

_JOB_NUM = re.compile(
    r"(?:approximately\s+|about\s+)?([\d][\d,]{1,9})\s*(?:such\s+)?jobs?\b",
    re.I)
_REGIONAL_CUE = re.compile(
    r"\bregion(?:al|ally)?\b|\bstate\s+economy\b|\blocal(?:ly)?\b", re.I)
_NUM_TOKEN = re.compile(r"[\d,]+")

# Best-effort "significant numbers" floor reference point -- courts have no
# bright line, but figures well below this are routinely challenged and
# figures well above it are routinely upheld; treated as approximate.
_FLOOR = 15000

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


def _code_numbers(t):
    out = []
    for m in _JOB_NUM.finditer(t):
        window = t[max(0, m.start() - 50): m.end() + 20]
        if _REGIONAL_CUE.search(window):
            continue
        try:
            n = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if 10 <= n <= 5_000_000:
            out.append(n)
    return out


def _llm_numbers(raw):
    out = []
    for tok in _NUM_TOKEN.findall(raw):
        digits = tok.replace(",", "")
        if digits.isdigit():
            n = int(digits)
            if 10 <= n <= 5_000_000:
                out.append(n)
    return out


def _ratio_score(total):
    ratio = total / _FLOOR
    if ratio >= 1.0:
        return min(1.0, 0.85 + 0.15 * min(ratio - 1.0, 1.0))
    return 0.75 * ratio


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""

        code_total = sum(_code_numbers(t))

        ex = extracted if isinstance(extracted, dict) else {}
        raw = str(ex.get("job_numbers_national") or "").strip()

        if _is_none(raw):
            final = 0.75 * _ratio_score(code_total)  # field found nothing: trust it, keep a code hedge
        else:
            llm_total = sum(_llm_numbers(raw))
            final = 0.75 * _ratio_score(llm_total) + 0.25 * _ratio_score(code_total)

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
