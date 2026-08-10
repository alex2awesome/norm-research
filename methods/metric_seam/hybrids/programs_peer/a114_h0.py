"""a114 hybrid: Sample, Population and Inclusion Criteria.

Construct: ~1.0 = an explicit sample size (N=...) AND explicit inclusion/exclusion or
sampling-method criteria are both stated; ~0.5 = only one of the two is present (N stated but
no criteria, or criteria stated but N vague/absent); ~0.0 = no sample-size or population
language at all.

INPUT = abstract/excerpt only. Code sees: numeric N-pattern density and sampling/criteria
vocabulary (recruited, excluded, randomly sampled, inclusion criteria). Code CANNOT tell
whether the stated N is GROUNDED in the actual study population described (vs. an unrelated
number, e.g. a hyperparameter) — LLM_FIELDS extract the specific N and criteria text so code
can cross-check the digits against the document.
"""
import re

LLM_FIELDS = {
    "sample_size": (
        "The stated sample size or N used in the study, with unit (e.g. '500 users', "
        "'N=1200 papers'). Answer NONE if no sample size is stated."
    ),
    "inclusion_criteria": (
        "In <=20 words, any stated inclusion, exclusion, or sampling-method criteria for "
        "participants or data. Answer NONE if none stated."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_N_RE = re.compile(
    r"\bN\s*=\s*\d[\d,\.]*\b|\b\d[\d,\.]*\s*(?:participants|subjects|users|annotators|"
    r"papers|documents|samples|respondents)\b", re.I)
_CRITERIA_RE = re.compile(
    r"\b(inclusion criteria|exclusion criteria|exclu(?:de|ded|sion)|recruit(?:ed|ment)?|"
    r"randomly sampl(?:ed|ing)|stratifi|convenience sampl|eligib(?:le|ility)|population)\b", re.I)
_NUM_RE = re.compile(r"\b\d[\d,\.]*\b")


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _code_score(text, extracted):
    has_n_pattern = bool(_N_RE.search(text))
    has_criteria_vocab = bool(_CRITERIA_RE.search(text))

    size = extracted.get("sample_size")
    crit = extracted.get("inclusion_criteria")
    grounded_size = False
    if not _is_none(size):
        nums = _NUM_RE.findall(size)
        grounded_size = any(n in text for n in nums) if nums else False
    has_size = 0.0 if _is_none(size) else (1.0 if grounded_size else 0.5)
    has_crit = 0.0 if _is_none(crit) else 1.0

    s = (0.35 * has_size + 0.35 * has_crit
         + 0.15 * (1.0 if has_n_pattern else 0.0)
         + 0.15 * (1.0 if has_criteria_vocab else 0.0))
    return max(0.0, min(1.0, s))


def _llm_score(extracted):
    size = extracted.get("sample_size")
    crit = extracted.get("inclusion_criteria")
    s = (0.5 if not _is_none(size) else 0.0) + (0.5 if not _is_none(crit) else 0.0)
    return s


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        final = 0.6 * _code_score(t, extracted) + 0.4 * _llm_score(extracted)
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
