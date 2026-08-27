"""a25 hybrid: Claim–evidence alignment and causal caution.

Construct: ~1.0 = abstract makes an empirical/theoretical claim that is backed by matching
evidence of the right type (empirical claim -> numbers/results; theoretical -> proof/bound) and
uses appropriate causal caution; ~0.5 = claim and evidence partially aligned or hedged; ~0.0 =
claim made with no supporting evidence, or strong causal language with only weak evidence.

INPUT = abstract only. Code sees: claim-evidence TYPE consistency (does an 'empirical' claim have
a number? a 'theoretical' claim have a proof/bound marker?), hedging/causal-caution register, and
superlative over-claiming ("best/state-of-the-art" with no number). Retrieval fit is weak here
(could compare claim magnitude to method-family neighbors — deferred).
"""
import re

LLM_FIELDS = {
    "main_claim": (
        "In <=30 words, the paper's main empirical or theoretical RESULT claim as stated in the "
        "abstract (the headline finding). Answer NONE if no result claim is made."
    ),
    "evidence_type": (
        "One word for the TYPE of evidence given for the main claim: 'empirical' (experiments, "
        "benchmarks, numbers), 'theoretical' (proof, theorem, bound, guarantee), 'qualitative' "
        "(examples, intuition, case study), or 'none'."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_NUM_RE = re.compile(r"\b\d[\d,\.]*\b")
_PROOF_RE = re.compile(r"\b(prove|proof|theorem|lemma|guarantee|bound|converg|optimal)\b", re.I)
_HEDGE_REG = re.compile(r"\b(may|might|could|suggest|potential|possibly|appears? to|seems?|likely)\b", re.I)
_OVERCLAIM_REG = re.compile(r"\b(best|state[- ]of[- ]the[- ]art|sota|superior|outperform|unprecedented)\b", re.I)


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _alignment(claim, evid, text):
    """How well does the evidence TYPE match what's actually present?"""
    if _is_none(claim):
        return 0.2
    evid = (evid or "").strip().lower()
    has_num = bool(_NUM_RE.search(claim)) or bool(_NUM_RE.search(text or ""))
    has_proof = bool(_PROOF_RE.search(claim)) or bool(_PROOF_RE.search(text or ""))
    if evid == "empirical":
        return 1.0 if has_num else 0.4
    if evid == "theoretical":
        return 1.0 if has_proof else 0.4
    if evid == "qualitative":
        return 0.55
    return 0.2  # 'none'


def _code_score(text, extracted):
    t = text or ""
    claim = extracted.get("main_claim")
    evid = extracted.get("evidence_type")
    align = _alignment(claim, evid, t)
    hedge = bool(_HEDGE_REG.search(claim)) if not _is_none(claim) else False
    over = bool(_OVERCLAIM_REG.search(claim)) if not _is_none(claim) else False
    has_num = bool(_NUM_RE.search(claim)) if not _is_none(claim) else False
    # alignment is the core (0.6); a number present adds (0.2); overclaim-without-number docks (0.2)
    s = 0.6 * align + 0.2 * (1 if has_num else 0)
    if over and not has_num:
        s -= 0.2
    if hedge:
        s += 0.05  # appropriate caution is good, small bonus
    return max(0.0, min(1.0, s))


def _llm_score(extracted):
    """LLM contributes the evidence-type label directly."""
    evid = (extracted.get("evidence_type") or "").strip().lower()
    claim = extracted.get("main_claim")
    if _is_none(claim):
        return 0.15
    return {"empirical": 0.7, "theoretical": 0.75, "qualitative": 0.45, "none": 0.1}.get(evid, 0.3)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        return max(0.0, min(1.0, 0.65 * _code_score(t, extracted) + 0.35 * _llm_score(extracted)))
    except Exception:
        return 0.5
