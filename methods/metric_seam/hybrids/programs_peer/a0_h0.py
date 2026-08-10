"""a0 hybrid: Methodological Rigor and Soundness.

Criterion: research design, methodology, and analytical choices should be
appropriate, sound, justified, and free of fundamental errors.

CODE owns two regex-legible structural proxies: density of rationale-connective
phrases ("because", "in order to", "to control for", ...) and density of
validation-practice vocabulary (baseline, ablation, held-out, significance
test, ...). Per the lesson that KEYWORD PRESENCE IS OFTEN A PROXY FOR JUDGED
QUALITY (not the quality itself), these structural counts are capped at 40%
of the score and combined with two LLM fields carrying the thick-input
grounding regex cannot do: (1) the actual stated justification for the
paper's main methodological choice, credited only if GROUNDED (its content
words must appear in the document, guarding against hallucination), and
(2) any self-evident, unaddressed methodological flaw the excerpt reveals,
used as a direct code-side penalty. Code owns the predicate throughout:
NONE/ungrounded LLM answers map to zero credit rather than being trusted.
"""
import re

LLM_FIELDS = {
    "justification_quote": (
        "Quote in <=12 words the paper's stated reason for its main "
        "methodological choice, or NONE."
    ),
    "unaddressed_flaw": (
        "Name in <=10 words a methodological flaw or threat to validity "
        "left unaddressed, or NONE."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_RATIONALE_REG = re.compile(
    r"\b(because|in order to|so as to|to ensure|to avoid|we chose|we opted|"
    r"rather than|unlike|in contrast to|to account for|to control for|to mitigate)\b",
    re.I,
)
_VALIDATION_REG = re.compile(
    r"\b(baseline|ablation|control(?:led)?\s+(?:for|group)|sensitivity analysis|"
    r"cross[- ]validation|held[- ]out|significance test|robustness check)\b",
    re.I,
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
    return hits / len(words) >= 0.5


def _structural(t):
    n_rat = len(_RATIONALE_REG.findall(t))
    n_val = len(_VALIDATION_REG.findall(t))
    return min(1.0, 0.5 * min(n_rat, 3) / 3.0 + 0.5 * min(n_val, 3) / 3.0)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        tl = t.lower()
        extracted = extracted or {}

        structural = _structural(tl)

        quote = extracted.get("justification_quote")
        flaw = extracted.get("unaddressed_flaw")
        just_credit = 1.0 if _grounded(quote, tl) else 0.0
        flaw_penalty = 0.35 if _grounded(flaw, tl) else 0.0

        final = 0.4 * structural + 0.6 * just_credit - flaw_penalty
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
