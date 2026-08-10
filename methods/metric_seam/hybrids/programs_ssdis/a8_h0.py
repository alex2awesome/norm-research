"""a8 hybrid: code parses cited job-number figures and step-5/challenge boilerplate mechanically; LLM fields carry whether the claimant's job-number challenge got a substantive response and whether cited occupations are obsolete/eroded."""
import re

LLM_FIELDS = {
    "job_number_challenge": (
        "Did the claimant challenge the VE's job-number methodology, and "
        "how did the VE or ALJ respond?"
    ),
    "occupations_obsolete": (
        "Are the cited occupations described as obsolete or inconsistent "
        "with the claimant's RFC, else NONE?"
    ),
}

_NONE_VALUES = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", ""}


def _is_none(val):
    if val is None:
        return True
    return str(val).strip().lower().strip(". ") in _NONE_VALUES


_JOBNUM_RE = re.compile(r"([\d][\d,]{2,})\s*(?:jobs|positions)", re.I)
_CHALLENGE_RE = re.compile(
    r"challeng(?:e|ed|es|ing) the (?:VE'?s?|vocational expert'?s?)?\s*"
    r"(?:job[- ]number|reliability|methodology|data)|"
    r"refused to (?:provide|produce)|"
    r"no (?:underlying )?data (?:or|nor) (?:source|methodology)", re.I)
_OBSOLETE_RE = re.compile(
    r"obsolete|no longer (?:exist|performed)|eroded by the RFC|"
    r"inconsistent with (?:the )?RFC", re.I)
_NO_STEP5_RE = re.compile(
    r"fail(?:ed|ure) to make a step[- ]?five finding|no step[- ]?five finding", re.I)


def _total_jobs(t):
    total = 0
    for m in _JOBNUM_RE.finditer(t):
        try:
            total += int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return total


def score(text: str, extracted: dict, ops) -> float:
    try:
        try:
            t = ops.normalize(text) if text else ""
        except Exception:
            t = text or ""
        ex = extracted if isinstance(extracted, dict) else {}

        total = _total_jobs(t)
        base = 0.3
        if total:
            if total < 1000:
                base += 0.25
            elif total >= 10000:
                base -= 0.2
        else:
            base += 0.1  # no explicit job-number figure in the narrative

        if _NO_STEP5_RE.search(t):
            base += 0.35
        if _CHALLENGE_RE.search(t) or _OBSOLETE_RE.search(t):
            base += 0.1

        chal = ex.get("job_number_challenge")
        if not _is_none(chal):
            c = str(chal).lower()
            if re.search(r"refus|no data|did not|failed|no response|no methodology", c):
                base += 0.25
            elif re.search(r"explain|provided|respond|address", c):
                base -= 0.1

        if not _is_none(ex.get("occupations_obsolete")):
            base += 0.25

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5
