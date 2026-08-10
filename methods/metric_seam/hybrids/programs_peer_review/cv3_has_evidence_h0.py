"""cv3 hybrid: Do we have evidence for this claim? (evidence presence + type-match vs body)

Construct: ~1.0 = the body contains evidence of the RIGHT type for the claim (empirical claim ->
numbers/results; theoretical -> proof/bound markers) and references figures/tables; ~0.5 =
evidence present but type-weak (qualitative) or no figure/table refs; ~0.0 = claim made but body
has no matching evidence.

Unlike a25 (which checks type-alignment at the abstract level only), cv3 checks the BODY for
actual evidence presence: does a results/experiments section exist with numbers (empirical) or
proof language (theoretical), and are there figure/table references? The LLM labels the claim's
evidence TYPE from the abstract; CODE verifies the body carries that type of evidence.

INPUT split: extraction (evidence_type) from ABSTRACT; score() verifies evidence presence in BODY.
"""
import re

LLM_FIELDS = {
    "main_claim": (
        "In <=30 words, the abstract's main empirical or theoretical RESULT claim. Answer NONE if none."
    ),
    "evidence_type": (
        "One word: the TYPE of evidence the claim needs: 'empirical' (experiments/numbers), "
        "'theoretical' (proof/theorem/bound), 'qualitative' (examples/intuition), or 'none'."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_NUM_RE = re.compile(r"\d+\.\d+|\d+")
_PROOF_RE = re.compile(r"\b(prove|proof|theorem|lemma|proposition|guarantee|bound|converg|optimal|lemma)\b", re.I)
_FIGTBL_RE = re.compile(r"\b(?:figure|fig\.?|table|tab\.?|equation|eq\.?)\s*\d", re.I)
_RESULTSECTION_RE = re.compile(r"\b(result|experiment|evaluation|finding|ablation)\b", re.I)

def _code_score(body, extracted):
    body = body or ""
    evid = (extracted.get("evidence_type") or "").strip().lower()
    claim = extracted.get("main_claim")
    if not isinstance(claim, str) or claim.strip().lower() in _NONE:
        return 0.15

    has_num = bool(_NUM_RE.search(body))
    has_proof = bool(_PROOF_RE.search(body))
    has_figtbl = bool(_FIGTBL_RE.search(body))
    has_resultsec = bool(_RESULTSECTION_RE.search(body))
    body_len = len(body)

    # type-match: does the body carry evidence of the claimed type?
    if evid == "empirical":
        type_match = 1.0 if has_num else 0.3
    elif evid == "theoretical":
        type_match = 1.0 if has_proof else 0.3
    elif evid == "qualitative":
        type_match = 0.55
    else:  # none / unknown
        type_match = 0.2

    # substance: results section + non-trivial body + figure/table
    substance = 0.0
    if has_resultsec: substance += 0.3
    if body_len > 1500: substance += 0.25
    if has_figtbl: substance += 0.2
    substance = min(0.6, substance)

    return max(0.0, min(1.0, 0.5 * type_match + substance))

def _llm_score(extracted):
    evid = (extracted.get("evidence_type") or "").strip().lower()
    claim = extracted.get("main_claim")
    if not isinstance(claim, str) or claim.strip().lower() in _NONE:
        return 0.1
    return {"empirical": 0.6, "theoretical": 0.65, "qualitative": 0.4, "none": 0.1}.get(evid, 0.25)

def score(text: str, extracted: dict, ops) -> float:
    try:
        body = ops.normalize(text) if (text and ops) else (text or "")
        return max(0.0, min(1.0, 0.7 * _code_score(body, extracted) + 0.3 * _llm_score(extracted)))
    except Exception:
        return 0.5
