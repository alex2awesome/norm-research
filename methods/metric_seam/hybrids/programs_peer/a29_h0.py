"""a29 hybrid: Theoretical Grounding.

Criterion: work should have a sound theoretical basis with clearly stated
theoretical claims and derived analytical solutions.

CODE sees a weak structural proxy: presence of theorem/proof vocabulary and
equation markup (weight 0.35) -- weak because a paper can NAME a theorem
without ever stating or deriving it (the classic keyword-presence-as-proxy
failure). The two LLM fields carry the thick-input judgment code cannot
make: the actual theoretical claim as stated (credited only if GROUNDED
against the text), and whether the excerpt shows the derivation as
complete, partial, or absent. Code owns the predicate by mapping the
completeness label to a fixed credit scale rather than trusting a raw
score from the LLM.
"""
import re

LLM_FIELDS = {
    "theoretical_claim": (
        "State in <=15 words the paper's main theorem or theoretical "
        "claim, or NONE."
    ),
    "derivation_completeness": (
        "One word: complete, partial, or none -- is the claim's "
        "derivation/proof fully shown?"
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_THEORY_MARKER_REG = re.compile(
    r"\b(theorem|lemma|corollary|proposition|proof|derive[sd]?|derivation)\b", re.I
)
_EQUATION_REG = re.compile(
    r"\$[^$]{2,}\$|\\begin\{(?:equation|align)\}|\bq\.?e\.?d\.?\b|∎", re.I
)
_STOP = {"the", "a", "an", "and", "or", "but", "for", "with", "that", "this",
         "its", "was", "were", "is", "are", "of", "to", "in", "on", "at", "as"}


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _content_words(s):
    return [w for w in re.findall(r"[A-Za-z']+", s.lower()) if len(w) >= 4 and w not in _STOP]


def _grounded(quote, text_lower):
    if _is_none(quote):
        return False
    words = _content_words(quote)
    if not words:
        return False
    hits = sum(1 for w in words if w in text_lower)
    return hits / len(words) >= 0.4


def _structural(t):
    n_marker = len(_THEORY_MARKER_REG.findall(t))
    has_eq = bool(_EQUATION_REG.search(t))
    return min(1.0, 0.6 * min(n_marker, 3) / 3.0 + 0.4 * (1 if has_eq else 0))


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        tl = t.lower()
        extracted = extracted or {}

        structural = _structural(tl)

        claim = extracted.get("theoretical_claim")
        comp = extracted.get("derivation_completeness")
        claim_credit = 1.0 if _grounded(claim, tl) else 0.0
        comp_map = {"complete": 1.0, "partial": 0.5, "none": 0.0}
        comp_val = comp_map.get(str(comp).strip().lower(), 0.0) if not _is_none(comp) else 0.0

        llm_score = 0.5 * claim_credit + 0.5 * comp_val
        final = 0.35 * structural + 0.65 * llm_score
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
