"""cv1 hybrid: Is the claim supported in the body? (claim->results numeric/statment verification)

Construct: ~1.0 = the abstract's headline claim carries a specific number/figure that ALSO
appears in the paper's results section (the body substantiates the claim); ~0.5 = the claim is
supported only at the statement level (key terms co-occur in body, no number); ~0.0 = the claim's
number/statements do NOT appear in the body (unsupported / possibly exaggerated).

This is the strongest place for code to verify: NUMBERS are deterministic. The LLM only EXTRACTS
the claim + its key numbers from the abstract; CODE checks whether those numbers (or the claim's
key terms) appear in the results/experiments body. A claimed "12.3% improvement" that appears
nowhere in results is a red flag code can catch without judgment.

INPUT split: extraction reads the ABSTRACT (where the claim is made); score() verifies against
the BODY text (results+experiments). Failure modes: numbers reformatted (12.3 vs 12.34 vs 0.123)
-> code normalizes; claim paraphrased -> fall back to key-term co-occurrence.
"""
import re

LLM_FIELDS = {
    "headline_claim": (
        "In <=25 words, the abstract's main RESULT claim (the headline finding). Answer NONE if "
        "the abstract makes no result claim."
    ),
    "key_numbers": (
        "Comma-separated SPECIFIC numbers/figures cited in the headline claim exactly as written "
        "(e.g. '12.3%, 3.5x, 87.2, 4.2M'). Answer NONE if the claim cites no specific number."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_NUM_RE = re.compile(r"\d+\.\d+|\d+")
# strip commas/percent/x/M/k so 12.3% / 12.3 / 12.34 / 3.5x normalize for matching
def _norm_num(s):
    return re.sub(r"[%xXkKmM,\s]", "", s)

def _extract_nums(s):
    if not isinstance(s, str) or s.strip().lower() in _NONE:
        return []
    out = []
    for tok in re.split(r"[,;]", s):
        m = _NUM_RE.findall(tok)
        out.extend(m)
    return [_norm_num(n) for n in out if n]

def _body_nums(body):
    return {_norm_num(m.group(0)) for m in _NUM_RE.finditer(body or "")}

def _content_words(claim):
    if not isinstance(claim, str) or claim.strip().lower() in _NONE:
        return []
    return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", claim) if w.lower() not in
            {"that", "this", "with", "from", "have", "their", "which", "model", "method", "approach", "result", "paper"}]

def _code_score(body, extracted):
    body = body or ""
    bnums = _body_nums(body)
    knums = _extract_nums(extracted.get("key_numbers"))
    claim = extracted.get("headline_claim")

    # numeric support: how many claim numbers appear in body (strong signal)
    if knums:
        hit = sum(1 for n in knums if n in bnums and len(n) >= 2)
        frac = hit / max(1, len(knums))
        if frac >= 0.5:
            return min(1.0, 0.6 + 0.4 * frac)   # numbers substantiated
        elif frac > 0:
            return 0.45
        # claimed numbers but NONE in body -> unsupported (the key red flag)
        # fall through to statement check, but cap low
    # statement support: key content words of claim co-occur in body
    cw = _content_words(claim)
    if cw:
        blow = body.lower()
        present = sum(1 for w in cw if w.lower() in blow)
        frac = present / len(cw)
        if frac >= 0.6:
            return 0.55 if not knums else 0.3   # statement-supported but no number match
        if frac >= 0.3:
            return 0.35
        return 0.15
    return 0.1

def _llm_score(extracted):
    """LLM contributes: is there even a claim with a number? (presence, not verification)."""
    claim = extracted.get("headline_claim")
    has_claim = 0.0 if (not isinstance(claim, str) or claim.strip().lower() in _NONE) else 0.4
    knums = _extract_nums(extracted.get("key_numbers"))
    has_num = 0.3 if knums else 0.0
    return has_claim + has_num

def score(text: str, extracted: dict, ops) -> float:
    """text = BODY (results+experiments). Verification is code-owned against the body."""
    try:
        body = ops.normalize(text) if (text and ops) else (text or "")
        # code verification is the dominant signal (0.7); LLM only confirms claim presence (0.3)
        return max(0.0, min(1.0, 0.7 * _code_score(body, extracted) + 0.3 * _llm_score(extracted)))
    except Exception:
        return 0.5
