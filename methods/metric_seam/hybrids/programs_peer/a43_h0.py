"""a43 hybrid: Submission Completeness.

Criterion: submissions should provide all required information, supporting
reasons, and required sections with substantial content.

CODE owns two regex-legible structural signals: count of distinct standard
section headers detected in the excerpt, and presence of explicit
placeholder/TBD markers (a direct completeness violation), blended with a
length credit. A raw header count is itself a proxy that can overstate
completeness if a section is titled but empty, so the two LLM fields supply
the thick-input check: which standard sections actually carry SUBSTANTIAL
content (not just a header), and any explicitly missing/placeholder content
named in the excerpt. The sections-present answer is credited only for
names that are both on the canonical list and GROUNDED in the text; the
missing-content flag is applied as a direct code-side penalty rather than
trusted blindly.
"""
import re

LLM_FIELDS = {
    "missing_signal": (
        "Name in <=12 words any explicitly missing, placeholder, or "
        "incomplete section, or NONE."
    ),
    "sections_present": (
        "List (comma-separated) standard paper sections with substantial "
        "content in this excerpt, or NONE."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_HEADER_REG = re.compile(
    r"\b(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"experiments?|results?|discussion|conclusions?|limitations|references|appendix)\b",
    re.I,
)
_PLACEHOLDER_REG = re.compile(
    r"\b(tbd|to be (?:added|determined)|placeholder|omitted for (?:space|brevity)|lorem ipsum)\b"
    r"|\[missing\]|\[todo\]",
    re.I,
)
_CANON_SECTIONS = {
    "abstract", "introduction", "related work", "background", "method",
    "methods", "methodology", "experiments", "experiment", "results",
    "result", "discussion", "conclusion", "conclusions", "limitations",
    "references", "appendix",
}


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _structural(t):
    n_headers = len(set(m.lower() for m in _HEADER_REG.findall(t)))
    has_placeholder = bool(_PLACEHOLDER_REG.search(t))
    n_words = len(t.split())
    length_credit = min(1.0, n_words / 600.0)  # a few-thousand-char excerpt ~ 500-700 words
    base = 0.6 * min(1.0, n_headers / 3.0) + 0.4 * length_credit
    if has_placeholder:
        base -= 0.3
    return max(0.0, min(1.0, base))


def _sections_credit(val, text_lower):
    if _is_none(val):
        return 0.0
    n = 0
    for part in re.split(r"[,;]", val):
        name = part.strip().lower()
        if name in _CANON_SECTIONS and name in text_lower:
            n += 1
    return min(1.0, n / 3.0)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        tl = t.lower()
        extracted = extracted or {}

        structural = _structural(t)

        missing = extracted.get("missing_signal")
        sections = extracted.get("sections_present")
        missing_penalty = 0.4 if not _is_none(missing) else 0.0
        sections_credit = _sections_credit(sections, tl)

        final = 0.4 * structural + 0.6 * sections_credit - missing_penalty
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
