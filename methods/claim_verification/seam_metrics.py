"""Seam-disciplined claim/retrieval metrics (patterned after metric_seam + patents prior-art flow).

Each metric family has explicitly separated arms:
  CODE arm  — deterministic, no LLM: regex/count/retrieval-score computations.
  LLM arm   — judge calls (claim extraction, localize-then-verify) via claim_verification.core.
  HYBRID    — code features computed ON LLM outputs (e.g., retrieval stats over extracted claims).
  NULL twin — same pipeline with the evidence op nulled (shuffled/foreign evidence pool);
              the certified marginal of the evidence machinery = metric(real) - metric(null),
              per the metric_seam ops/NullOps pattern.

Metric families:
  F1 sourcing_code      (CODE)   — attribution-verb counts, quote counts, named-entity-source
                                   patterns, "according to" density, anonymous-source phrases.
  F2 claim_retrieval    (HYBRID) — per extracted claim, bge-style lexical retrieval over body
                                   passages: top1_overlap, mean_overlap, support_coverage
                                   (frac claims with a passage above threshold). CODE retrieval
                                   on LLM-extracted claims.
  F3 claim_support_llm  (LLM)    — FULL/PARTIAL/NONE verdicts (core.verify_claim), support_rate,
                                   evidence-type mix.
  F4 placebo twins      (NULL)   — F2/F3 with foreign evidence pools (evidence_shuffle).

Everything returns flat dicts; run_seam_battery() executes all arms on a dataframe.
"""
import re
import numpy as np

# ---------------- F1: CODE sourcing metrics (no LLM) ----------------
ATTRIB_VERBS = r"(said|says|stated|announced|according to|told|reported|confirmed|added|noted|claimed|argued|testified|wrote)"
ANON_SRC = r"(sources? (?:familiar|close to|with knowledge)|officials? (?:said|say|told)|people (?:familiar|briefed)|a (?:person|source) (?:who|with)|on condition of anonymity)"
DOC_SRC = r"(according to (?:a|the) (?:report|filing|study|survey|document|lawsuit|complaint|indictment|memo|analysis|data)|court (?:documents|filings|records)|the (?:report|study|survey|filing) (?:said|says|found|shows))"
NAMED_QUOTE = r'"[^"]{20,400}"\s*,?\s*(?:said|says)\s+[A-Z][a-z]+'
NAMED_QUOTE2 = r"[A-Z][a-z]+ [A-Z][a-z]+,\s+(?:the |a |an )?[a-z][\w \-]{3,40}(?:at|of|for)\s+[A-Z]"
TITLED_PERSON = r"\b(?:CEO|CFO|COO|president|director|professor|Dr\.|spokesperson|spokesman|spokeswoman|analyst|attorney|senator|representative|secretary|chief)\s+[A-Z][a-z]+"

def sourcing_code_metrics(text):
    t = text or ""
    words = max(len(t.split()), 1)
    m = {
        "c_attrib_verbs": len(re.findall(ATTRIB_VERBS, t, re.I)),
        "c_anon_sources": len(re.findall(ANON_SRC, t, re.I)),
        "c_doc_sources": len(re.findall(DOC_SRC, t, re.I)),
        "c_named_quotes": len(re.findall(NAMED_QUOTE, t)) + len(re.findall(NAMED_QUOTE2, t)),
        "c_titled_persons": len(re.findall(TITLED_PERSON, t)),
        "c_direct_quotes": t.count('"') // 2,
        "c_numbers": len(re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", t)),
        "c_percents": len(re.findall(r"\d+(?:\.\d+)?\s?%", t)),
        "c_dollars": len(re.findall(r"\$\s?\d", t)),
    }
    m["c_attrib_density"] = m["c_attrib_verbs"] / words * 1000
    m["c_quote_density"] = m["c_direct_quotes"] / words * 1000
    return m

# ---------------- F2: HYBRID claim retrieval (code retrieval on LLM claims) ----------------
_STOP = set("the a an and or but of to in on for with as by at is are was were be been it its this that from has have had not no".split())

def _toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP and len(w) > 2]

def _overlap(claim_toks, passage_toks):
    """Jaccard-ish containment: |claim ∩ passage| / |claim| — how much of the claim's content
    the passage covers (retrieval score, deterministic)."""
    if not claim_toks: return 0.0
    ps = set(passage_toks)
    return sum(1 for w in claim_toks if w in ps) / len(claim_toks)

def claim_retrieval_metrics(claims, passages, thresh=0.5):
    """claims: list[str] (LLM-extracted); passages: list[str] (code-built pool).
    All retrieval math is CODE (no LLM)."""
    if not claims or not passages:
        return {"r_top1_overlap": np.nan, "r_mean_top1": np.nan,
                "r_support_coverage": np.nan, "r_margin": np.nan}
    ptoks = [_toks(p) for p in passages]
    top1s, margins = [], []
    for c in claims:
        ct = _toks(c)
        scores = sorted((_overlap(ct, pt) for pt in ptoks), reverse=True)
        top1s.append(scores[0])
        margins.append(scores[0] - (scores[1] if len(scores) > 1 else 0.0))
    return {"r_top1_overlap": float(np.max(top1s)),
            "r_mean_top1": float(np.mean(top1s)),
            "r_support_coverage": float(np.mean([s >= thresh for s in top1s])),
            "r_margin": float(np.mean(margins))}

# ---------------- battery runner ----------------
def run_seam_battery(df, text_col, cfg, cache_path, workers=16, with_llm=True,
                     with_null=True, seed=0):
    """Returns dataframe-aligned list of flat metric dicts with arms prefixed:
    c_* (code), r_* (hybrid retrieval), s_* (LLM support), null_r_*/null_s_* (placebo twins)."""
    from .core import (Cache, split_head_body, make_passages, extract_claims,
                       verify_claim, doc_metrics)
    cache = Cache(cache_path)
    texts = df[text_col].fillna("").tolist()
    heads, pools = [], []
    for t in texts:
        h, b = split_head_body(t)
        heads.append(h); pools.append(make_passages(b))
    # foreign pools for the NULL twin (shift-by-one derangement on a permutation)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(texts))
    perm = [(p if p != i else (int(p) + 1) % len(texts)) for i, p in enumerate(perm)]
    null_pools = [pools[j] for j in perm]

    from concurrent.futures import ThreadPoolExecutor
    def work(i):
        out = {}
        out.update(sourcing_code_metrics(texts[i]))                      # F1 CODE
        claims = []
        if with_llm:
            try:
                claims = [c["claim"] for c in extract_claims(heads[i], cfg, cache)]
            except Exception:
                claims = []
        out["n_claims"] = float(len(claims))
        out.update(claim_retrieval_metrics(claims, pools[i]))            # F2 HYBRID
        if with_null:
            nu = claim_retrieval_metrics(claims, null_pools[i])          # F4 null twin (retrieval)
            out.update({("null_" + k): v for k, v in nu.items()})
        if with_llm and claims:
            try:
                vs = [verify_claim(c, pools[i], cfg, cache) for c in claims]  # F3 LLM
                full = sum(1 for v in vs if v["verdict"] == "FULL")
                out["s_support_rate"] = full / len(vs)
                out["s_partial_rate"] = sum(1 for v in vs if v["verdict"] == "PARTIAL") / len(vs)
                out["s_grounded_rate"] = sum(1 for v in vs if v.get("grounded")) / len(vs)
                if with_null:
                    nvs = [verify_claim(c, null_pools[i], cfg, cache) for c in claims]
                    out["null_s_support_rate"] = sum(1 for v in nvs if v["verdict"] == "FULL") / len(nvs)
            except Exception:
                pass
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(work, range(len(texts))))
