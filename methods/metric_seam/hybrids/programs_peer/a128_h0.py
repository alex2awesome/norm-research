"""a128 hybrid: Strengths, Weaknesses and Limitations.

Construct: ~1.0 = the excerpt states both a specific strength/advantage AND a specific
limitation/weakness (balanced self-assessment); ~0.5 = only one side is present (pure
strength-selling, or an isolated limitation with no strength framing); ~0.0 = neither
strengths nor limitations are discussed.

INPUT = abstract/excerpt only. Code sees: limitation-marker and strength-marker vocabulary
density and whether both co-occur (balance signal). Code CANNOT tell whether a flagged
"limitation" is actually SPECIFIC (a concrete named weakness) vs. generic boilerplate
("future work will explore more") — LLM_FIELDS carry the extracted specific claims so code
can check them against a genericness pattern.
"""
import re

LLM_FIELDS = {
    "limitation": (
        "In <=25 words, a specific limitation or weakness of the work stated in the text. "
        "Answer NONE if no limitation is stated."
    ),
    "strength_claim": (
        "In <=20 words, a specific strength or advantage claimed about the work. Answer "
        "NONE if none is stated."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_LIMIT_RE = re.compile(
    r"\b(limitation|weakness|drawback|shortcoming|does not|fails? to|cannot|"
    r"struggles? (?:to|with)|future work|caveat)\b", re.I)
_STRENGTH_RE = re.compile(
    r"\b(strength|advantage|outperform|effective|novel|robust|superior|"
    r"state[- ]of[- ]the[- ]art)\b", re.I)
_GENERIC_RE = re.compile(
    r"\b(future work|leave (?:this|it) (?:for|to) future|more (?:research|work) is needed)\b", re.I)


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _code_score(text, extracted):
    limit_hits = len(_LIMIT_RE.findall(text))
    strength_hits = len(_STRENGTH_RE.findall(text))

    lim = extracted.get("limitation")
    strn = extracted.get("strength_claim")
    has_lim = 0.0 if _is_none(lim) else 1.0
    has_strn = 0.0 if _is_none(strn) else 1.0
    generic = bool(_GENERIC_RE.search(lim)) if not _is_none(lim) else False
    balance_bonus = 0.2 if (has_lim and has_strn) else 0.0

    s = (0.35 * has_lim + 0.25 * has_strn + balance_bonus
         + 0.1 * min(1.0, (limit_hits + strength_hits) / 4.0))
    if generic:
        s -= 0.15
    return max(0.0, min(1.0, s))


def _llm_score(extracted):
    lim = extracted.get("limitation")
    strn = extracted.get("strength_claim")
    return (0.5 if not _is_none(lim) else 0.0) + (0.5 if not _is_none(strn) else 0.0)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        final = 0.6 * _code_score(t, extracted) + 0.4 * _llm_score(extracted)
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
