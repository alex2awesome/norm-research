"""a163 hybrid: Positioning within and comparison to prior work (novelty + baselines).

Construct (from the rubric): a well-positioned paper names the prior work it builds on
AND the baselines/methods it compares against, and states a specific contribution that
is differentiated from that prior work. ~1.0 = explicit named baselines + named prior
work + a specific, non-hedged, novel contribution; ~0.5 = some positioning but vague
("we compare to existing methods", no names); ~0.0 = no positioning at all (abstract
names no prior work and no baselines).

INPUT = paper ABSTRACT only (median ~1300 chars). What code can legitimately see at
this granularity:
  - count and plausibility of NAMED baselines/prior-work the LLM extracts (the LLM is
    used only as a structured extractor; code owns the judgment of "is this specific?");
  - the novelty register of the contribution (first/novel/new vs improve/extend/faster);
  - specificity (does the contribution contain a named method or a number);
  - hedging of the contribution claim.
What code CANNOT see from an abstract (routed to retrieval tier, not this h0):
  - whether the named prior work is the RIGHT/comprehensive set (needs external DB);
  - whether the claimed differentiation actually holds (needs to read the prior work).
`ops.retrieve_similar` (internal TF-IDF over the corpus) gives a cheap dup/overlap signal;
an external arXiv/OpenReview FAISS index is the stronger tier-2 upgrade (see ladder note).

Baseline failure modes this h0 guards against (vs a naive keyword counter):
  - "we compare with existing methods" has the word "compare" but names nothing -> must
    score LOW (specificity gate on whether real names were extracted, not on the word);
  - "we improve ResNet" names a baseline but the contribution is incremental (improve/extend
    register) -> partial credit, not full;
  - LLM may hallucinate a baseline name not in the text -> code cross-checks each named
    baseline against the abstract text (must actually appear) before counting it.
"""
import re

LLM_FIELDS = {
    "claimed_contribution": (
        "In <=25 words, the paper's single main CLAIMED CONTRIBUTION or result as stated in "
        "the abstract. Answer NONE if the abstract states no contribution."
    ),
    "named_baselines": (
        "Comma-separated NAMED baseline or comparison methods explicitly mentioned in the "
        "abstract (e.g. ResNet, BERT, GPT-4, CLIP, human performance, state-of-the-art ONLY "
        "if paired with a name). Answer NONE if no specific baseline is named."
    ),
    "named_prior_work": (
        "Comma-separated specific prior works/methods cited BY NAME in the abstract as prior "
        "art the paper builds on or differs from. Answer NONE if none are named."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}

_NOVELTY_REG = re.compile(r"\b(first|novel|new|propose|introduce|present(?:s)? a|pioneer)\b", re.I)
_INCREMENT_REG = re.compile(r"\b(improve|extends?|faster|better|outperform|superior|build(?:s)? on|based on)\b", re.I)
_HEDGE_REG = re.compile(r"\b(may|might|could|suggest|potential|possibly|appears? to|seems?)\b", re.I)
_NUM_METHOD_REG = re.compile(r"\b\d[\d,\.]*\b|[A-Z][A-Za-z0-9-]*(?:Net|BERT|GPT|LM|GNN|Transformer|T5|Clip|Llama|Cod)\b")


def _split_names(val):
    if not isinstance(val, str):
        return []
    v = val.strip()
    if v.lower().strip(". ") in _NONE:
        return []
    return [p.strip() for p in re.split(r"[,;]", v) if p.strip() and p.strip().lower() not in _NONE]


def _specificity(contrib):
    """1.0 if contribution names a method or a number; 0.5 if generic-but-present; 0 if NONE."""
    if not isinstance(contrib, str) or contrib.strip().lower() in _NONE:
        return 0.0
    if _NUM_METHOD_REG.search(contrib):
        return 1.0
    return 0.5


def _novelty_register(contrib):
    """+ if novelty register, - if only incremental, 0 if neither/none."""
    if not isinstance(contrib, str) or contrib.strip().lower() in _NONE:
        return 0.0
    n = bool(_NOVELTY_REG.search(contrib))
    i = bool(_INCREMENT_REG.search(contrib))
    if n and not i:
        return 1.0
    if n and i:
        return 0.6
    if i:
        return 0.3
    return 0.4


def _code_score(text, extracted):
    """Code owns: are named baselines/prior-work REAL (in-text), and how specific/positioned."""
    t = text or ""
    bases = [b for b in _split_names(extracted.get("named_baselines")) if b and b.lower() in t.lower()]
    prior = [p for p in _split_names(extracted.get("named_prior_work")) if p and p.lower() in t.lower()]
    contrib = extracted.get("claimed_contribution")

    n_real = len(bases) + len(prior)
    # positioning breadth: 0 named -> 0.0; 1 -> 0.4; 2-3 -> 0.7; >=4 -> 0.9
    pos = 0.0 if n_real == 0 else (0.4 if n_real == 1 else (0.7 if n_real <= 3 else 0.9))

    spec = _specificity(contrib)
    reg = _novelty_register(contrib)
    hedge = bool(_HEDGE_REG.search(contrib)) if isinstance(contrib, str) else False

    # blend positioning (40%), specificity (25%), novelty register (25%), hedge penalty (10%)
    s = 0.40 * pos + 0.25 * spec + 0.25 * reg
    if hedge:
        s -= 0.10
    return max(0.0, min(1.0, s))


def _llm_score(extracted):
    """LLM contributes the positioning breadth directly (count of named items),
    independent of code's in-text cross-check — agreement = strong signal."""
    bases = _split_names(extracted.get("named_baselines"))
    prior = _split_names(extracted.get("named_prior_work"))
    n = len(bases) + len(prior)
    if n == 0:
        return 0.15
    if n == 1:
        return 0.45
    if n <= 3:
        return 0.7
    return 0.9


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text) if (text and ops) else (text or "")
        code = _code_score(t, extracted)
        llm = _llm_score(extracted)
        final = 0.55 * code + 0.45 * llm
        # tier-2 retrieval signal (internal TF-IDF dup/overlap): if a near-duplicate exists
        # in the corpus, dock novelty. Cheap, free, wired via ops.retrieve_similar.
        try:
            nbrs = ops.retrieve_similar(t, k=1) if ops and getattr(ops, "_vec", None) is not None else []
            if nbrs and nbrs[0][0] >= 0.85:
                final -= 0.15
        except Exception:
            pass
        return max(0.0, min(1.0, final))
    except Exception:
        return 0.5
