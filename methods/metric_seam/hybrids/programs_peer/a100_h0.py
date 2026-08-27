"""a100 hybrid: Licensing and Open Access of Outputs.

Construct: ~1.0 = a specific license is named AND a concrete artifact (code/data/model) is
stated as released, ideally with a URL; ~0.5 = release is stated but the license is unnamed,
or a license is named but release is vague/conditional ("upon request", "upon acceptance");
~0.0 = no license or release language at all.

INPUT = abstract/excerpt only. Code sees: named-license keyword matches, release-verb density,
and URL-like tokens (github/huggingface/zenodo) as concrete-release evidence. Code CANNOT
reliably tell WHICH artifact (code vs data vs model) is being released or whether a vague
"available upon request" really counts as open access — LLM_FIELDS carry that judgment.
"""
import re

LLM_FIELDS = {
    "license_name": (
        "The name of any software or data license mentioned (e.g. MIT, Apache-2.0, "
        "CC-BY-4.0). Answer NONE if no license is named."
    ),
    "release_artifact": (
        "What artifact is stated as being released or made available: 'code', 'data', "
        "'model', 'multiple', or 'none'."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_LICENSE_RE = re.compile(
    r"\b(MIT|Apache(?:-2\.0)?|GPL(?:v[23])?|BSD(?:-[23]-Clause)?|CC[- ]BY(?:-(?:SA|NC|ND))*"
    r"(?:-4\.0)?|CC0|public domain|proprietary licen[sc]e)\b", re.I)
_RELEASE_RE = re.compile(
    r"\b(release[ds]?|open[- ]source(?:d)?|publicly available|available at|"
    r"will be (?:released|made available)|github\.com|huggingface\.co|zenodo\.org)\b", re.I)
_URL_RE = re.compile(r"\b(github\.com|huggingface\.co|zenodo\.org|osf\.io)\S*", re.I)
_VAGUE_RE = re.compile(r"\bupon (?:request|acceptance)\b", re.I)


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _code_score(text):
    lic_hit = bool(_LICENSE_RE.search(text))
    rel_hit = bool(_RELEASE_RE.search(text))
    url_hit = bool(_URL_RE.search(text))
    vague = bool(_VAGUE_RE.search(text))
    s = 0.0
    if rel_hit:
        s += 0.35
    if lic_hit:
        s += 0.35
    if url_hit:
        s += 0.20
    if vague and not url_hit:
        s -= 0.15
    return max(0.0, min(1.0, s))


def _llm_score(extracted):
    lic = extracted.get("license_name")
    art = (extracted.get("release_artifact") or "").strip().lower()
    s = 0.5 if not _is_none(lic) else 0.0
    s += {"code": 0.35, "data": 0.3, "model": 0.3, "multiple": 0.5, "none": 0.0}.get(art, 0.0)
    return max(0.0, min(1.0, s))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        final = 0.5 * _code_score(t) + 0.5 * _llm_score(extracted)
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
