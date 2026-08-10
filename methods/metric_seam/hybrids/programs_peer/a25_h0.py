"""a25 Impact & Dissemination: code scores dissemination/impact lexicon density plus a
concrete-artifact regex (github/zenodo/huggingface/DOI/URL); LLM fields ground the actual
stated release/archiving plan and target audience, since code can't tell a concrete promise
from vague boilerplate impact language."""
import re

LLM_FIELDS = {
    "dissemination_plan": "State the concrete dissemination, release, or archiving plan mentioned, in <=15 words, or NONE if none given.",
    "impact_audience": "Name the specific audience or use-case the work's impact targets, in <=10 words, or NONE if unstated.",
}

_NONE_STRINGS = {"", "none", "n/a", "na", "unknown", "not stated", "not present",
                 "no evidence", "not applicable", "not specified", "not mentioned", "unclear"}

_DISSEM_PAT = re.compile(
    r"\b(will (?:be )?releas\w*|open[- ]source|publicly available|dissemin\w+|archiv\w+|"
    r"host(?:ed|ing)?|maintained|outreach|will (?:be )?deploy\w*|make (?:it |this )?available)\b", re.I)
_IMPACT_PAT = re.compile(
    r"\b(impact|benefit\w*|downstream|stakeholders?|practitioners?|policy ?makers?|"
    r"real[- ]world (?:use|application)|broader impact)\b", re.I)
_CONCRETE_PAT = re.compile(r"\b(github\.com|zenodo|huggingface|doi\.org|https?://)\b", re.I)


def _clean(v):
    if not v:
        return ""
    v = str(v).strip()
    if v.lower().strip(".") in _NONE_STRINGS:
        return ""
    return v


def score(text: str, extracted: dict, ops) -> float:
    try:
        raw = text or ""
        try:
            t = ops.normalize(raw)
        except Exception:
            t = raw
        if len(t) < 20:
            return 0.5
        tl = t.lower()
        n_words = max(1, len(t.split()))
        scale = 500.0 / n_words

        dissem_d = len(_DISSEM_PAT.findall(tl)) * scale
        impact_d = len(_IMPACT_PAT.findall(tl)) * scale
        concrete_hit = bool(_CONCRETE_PAT.search(tl))

        lex = (0.3
               + 0.2 * min(1.5, dissem_d)
               + 0.15 * min(1.5, impact_d)
               + (0.15 if concrete_hit else 0.0))

        extracted = extracted or {}
        plan_field = _clean(extracted.get("dissemination_plan", ""))
        aud_field = _clean(extracted.get("impact_audience", ""))

        field_adj = 0.0
        if plan_field:
            field_adj += 0.15
        if aud_field:
            field_adj += 0.1

        s = lex + field_adj
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
