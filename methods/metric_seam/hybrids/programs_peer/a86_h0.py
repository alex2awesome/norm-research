"""a86 hybrid: Scale Awareness in Theory and Methods.

Construct: ~1.0 = the work explicitly names a scale dimension (dataset/model/spatial/temporal
size) and states how theory or method adapts to or is evaluated across that scale (multiple
magnitude-different operating points, an asymptotic argument, a scale-dependent bound); ~0.5 =
scale is mentioned but only as a single fixed operating point or passing caveat; ~0.0 = no
scale language at all.

INPUT = abstract/excerpt only. Code sees: density of scale-vocabulary and whether two or more
distinct magnitude numbers appear (a cheap proxy for "tested across scale"). Code CANNOT tell
WHICH dimension is being scaled or whether the accounting claim is substantive — LLM_FIELDS
name the dimension and the specific claim for a grounding check.
"""
import re

LLM_FIELDS = {
    "scale_dimension": (
        "In <=10 words, the scale dimension the work addresses (e.g. dataset size, model "
        "size, spatial or temporal scale). Answer NONE if no scale dimension is mentioned."
    ),
    "scale_claim": (
        "In <=20 words, how the method or theory explicitly accounts for or varies across "
        "that scale. Answer NONE if not stated."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_SCALE_RE = re.compile(
    r"\b(scale|scales|scaling|large-scale|small-scale|multi-scale|magnitude|at scale|"
    r"across scales|orders? of magnitude)\b", re.I)
_NUM_RE = re.compile(r"\b\d[\d,\.]*(?:[kKmMbB]|%)?\b")


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _code_score(text, extracted):
    scale_hits = len(_SCALE_RE.findall(text))
    nums = set(_NUM_RE.findall(text))
    multi_num = 1.0 if len(nums) >= 2 else (0.4 if len(nums) == 1 else 0.0)
    vocab = min(1.0, scale_hits / 3.0)

    dim = extracted.get("scale_dimension")
    grounded = 0.0
    if not _is_none(dim):
        toks = re.findall(r"[A-Za-z]{4,}", dim)
        grounded = 1.0 if any(w.lower() in text.lower() for w in toks) else 0.3
    has_dim = 0.0 if _is_none(dim) else 1.0

    s = 0.35 * vocab + 0.15 * multi_num + 0.35 * has_dim * (0.5 + 0.5 * grounded) / 1.0
    return max(0.0, min(1.0, s))


def _llm_score(extracted):
    dim = extracted.get("scale_dimension")
    claim = extracted.get("scale_claim")
    if _is_none(dim):
        return 0.05
    return 0.5 + (0.5 if not _is_none(claim) else 0.0)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        final = 0.55 * _code_score(t, extracted) + 0.45 * _llm_score(extracted)
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
