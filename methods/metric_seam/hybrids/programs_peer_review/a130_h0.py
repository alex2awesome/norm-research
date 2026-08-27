"""a130 hybrid: Novelty and significance of the contribution.

Construct: ~1.0 = abstract claims a specific, novel, significant contribution (names what is
new, ideally with a quantified impact); ~0.5 = a contribution is claimed but vague or purely
incremental ("we improve X"); ~0.0 = no novelty/significance claim at all.

INPUT = abstract only. Code sees: the novelty register (first/novel/new vs improve/extend),
specificity (method-name or number in the claim), hedging, superlative/impact register.
Code CANNOT see (without retrieval): whether the novelty claim is TRUE vs prior art.
Cross-check: LLM-extracted novelty_claim must be grounded — its method names must appear in
the abstract; an ungrounded/hallucinated claim scores low. `ops.retrieve_similar` docks a
near-duplicate (cheap internal prior-art proxy); external FAISS is the tier-2 upgrade.
"""
import re

LLM_FIELDS = {
    "novelty_claim": (
        "In <=25 words, the novelty claim of the paper (what is new/original) as stated in the "
        "abstract. Answer NONE if the abstract makes no novelty claim."
    ),
    "significance_claim": (
        "In <=20 words, the stated significance, impact, or benefit of the contribution "
        "(e.g. what it enables, how much better). Answer NONE if none is stated."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_NOVELTY_REG = re.compile(r"\b(first|novel|new|propose|introduce|pioneer|previously (?:not|no))\b", re.I)
_INCREMENT_REG = re.compile(r"\b(improve|extends?|faster|better|outperform|superior|build(?:s)? on)\b", re.I)
_HEDGE_REG = re.compile(r"\b(may|might|could|suggest|potential|possibly|appears? to)\b", re.I)
_IMPACT_REG = re.compile(r"\b(significant|important|impact|enable|critical|substantial|state[- ]of[- ]the[- ]art|sota)\b", re.I)
_METHODNUM_REG = re.compile(r"\b\d[\d,\.]*\b|[A-Z][A-Za-z0-9-]{2,}")


def _is_none(v):
    return not isinstance(v, str) or v.strip().lower().strip(". ") in _NONE


def _register(claim):
    """novelty register score: explicit-novel=1, novel+incremental=0.6, only-incremental=0.3, neither=0.4."""
    if _is_none(claim):
        return 0.0
    n, i = bool(_NOVELTY_REG.search(claim)), bool(_INCREMENT_REG.search(claim))
    if n and not i: return 1.0
    if n and i: return 0.6
    if i: return 0.3
    return 0.4


def _code_score(text, extracted):
    t = (text or "").lower()
    nov, sig = extracted.get("novelty_claim"), extracted.get("significance_claim")
    reg = _register(nov)
    # specificity: does the novelty claim contain a method-name or number?
    spec = 0.0 if _is_none(nov) else (1.0 if _METHODNUM_REG.search(nov) else 0.5)
    # grounding: at least one capitalized token from the claim should appear in the abstract
    grounded = 0.0
    if not _is_none(nov):
        toks = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", nov) if w[0].isupper()]
        grounded = 1.0 if any(w.lower() in t for w in toks) else 0.3
    hedge = bool(_HEDGE_REG.search(nov)) if not _is_none(nov) else False
    impact = bool(_IMPACT_REG.search(sig)) if not _is_none(sig) else False
    has_sig = 0.0 if _is_none(sig) else 0.6

    s = 0.35 * reg + 0.20 * spec + 0.20 * grounded + 0.15 * has_sig + (0.10 if impact else 0.0)
    if hedge:
        s -= 0.08
    return max(0.0, min(1.0, s))


def _llm_score(extracted):
    nov, sig = extracted.get("novelty_claim"), extracted.get("significance_claim")
    has_nov = 0.0 if _is_none(nov) else 0.6
    has_sig = 0.0 if _is_none(sig) else 0.4
    return has_nov + has_sig


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        final = 0.60 * _code_score(t, extracted) + 0.40 * _llm_score(extracted)
        try:
            nbrs = ops.retrieve_similar(t, k=1) if ops and getattr(ops, "_vec", None) is not None else []
            if nbrs and nbrs[0][0] >= 0.85:
                final -= 0.15
        except Exception:
            pass
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
