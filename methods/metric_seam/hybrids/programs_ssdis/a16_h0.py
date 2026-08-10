"""a16 hybrid: code embeds the SSA Med-Voc grid age ladder as a lookup table and does age arithmetic from date cues as a fallback; two LLM fields carry the claimant's stated age and the ALJ's own age-category label."""

# Criterion buckets the claimant's chronological age at the ALJ decision (or
# DLI) into the Medical-Vocational Guidelines' age categories (20 C.F.R.
# Sec. 404.1563): younger individual (18-49), closely approaching advanced
# age (50-54), advanced age (55-59), closely approaching retirement age
# (60+). These categories are ordinal and matter directionally for the
# grids (older claimants get more favorable rules), so -- absent any other
# natural [0,1] embedding for an ordinal bucket -- higher score = older
# bucket. This directionality is a documented judgment call, not a stated
# fact from the criterion description itself.
#
# Design: age is a pure arithmetic quantity (decision/DLI year minus birth
# year) that code should own once the two dates are located, but a narrative
# usually contains MANY dates (onset, filing, hearing, prior decisions) that
# a bare ops.extract_dates() scan can't attribute to "born" vs "decided" --
# so one LLM field carries the claimant's stated/computable age directly.
# A second field carries the ALJ's own category label when the decision
# states one verbatim (a strong, doctrinally-anchored cross-check); code
# blends the two and falls back to a same-document date-cue search, then to
# a bare category-phrase search, when the fields are missing.
import re

LLM_FIELDS = {
    "claimant_age": (
        "State the claimant's exact age in years at the ALJ decision (or "
        "DLI if earlier), as stated or computable; NONE if not determinable."
    ),
    "alj_age_category": (
        "Quote the ALJ's own stated age-category label for the claimant "
        "(e.g. 'closely approaching advanced age', 'advanced age'), else "
        "NONE if the decision states no such label."
    ),
}

_AGE_NUM = re.compile(r"\b(1[89]|[2-6]\d|70)\b")
_YEAR = re.compile(r"\b(19[3-9]\d|20[0-2]\d)\b")
_BORN_CUE = re.compile(r"date of birth|born on|born in|\bd\.?o\.?b\.?\b", re.I)
_DECISION_CUE = re.compile(
    r"alj'?s?\s+decision\s+(?:dated|of)|decision\s+dated|"
    r"issued\s+(?:his|her|its)\s+decision", re.I)

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


def _age_to_score(age):
    return max(0.0, min(1.0, (age - 18) / 49.0))


def _category_anchor(s):
    sl = (s or "").lower()
    if "closely approaching retirement age" in sl or "approaching retirement age" in sl:
        return 0.90
    if "closely approaching advanced age" in sl:
        return 0.45
    if "advanced age" in sl:
        return 0.70
    if "younger individual" in sl:
        return 0.15
    return None


def _first_year_after(t, cue_re):
    m = cue_re.search(t)
    if not m:
        return None
    tail = t[m.end(): m.end() + 30]
    ym = _YEAR.search(tail)
    return int(ym.group(0)) if ym else None


def _code_fallback(t):
    birth_year = _first_year_after(t, _BORN_CUE)
    decision_year = _first_year_after(t, _DECISION_CUE)
    if birth_year and decision_year and decision_year > birth_year:
        age = decision_year - birth_year
        if 18 <= age <= 90:
            return _age_to_score(min(age, 67))
    anchor = _category_anchor(t)
    return anchor if anchor is not None else 0.5


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""

        ex = extracted if isinstance(extracted, dict) else {}
        age_raw = str(ex.get("claimant_age") or "").strip()
        cat_raw = str(ex.get("alj_age_category") or "").strip()

        age_score = None
        if not _is_none(age_raw):
            m = _AGE_NUM.search(age_raw)
            if m:
                age_score = _age_to_score(min(int(m.group(0)), 67))

        cat_score = None if _is_none(cat_raw) else _category_anchor(cat_raw)

        if age_score is not None and cat_score is not None:
            final = 0.7 * age_score + 0.3 * cat_score
        elif age_score is not None:
            final = age_score
        elif cat_score is not None:
            final = cat_score
        else:
            final = _code_fallback(t)

        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
