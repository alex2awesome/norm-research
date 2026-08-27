#!/usr/bin/env python3
"""Gold-bib SOFT readout: the binary SS102/SS103 checker under-fires (fired .273 matched
vs .000 mismatched) -> replace the binary verdict with a GRADED 0-10 anticipation score
and read out threshold-free discrimination AUC(matched vs mismatched). Resolution funnel
reused from run_gold_bib (all API lookups cached in oa_cache2.jsonl -> no S2 hammering).
Anchors: 2 synthetic pairs (self-anticipation=high, unrelated=low) injected blind.
Run on sk3: python -m methods.claim_verification.run_gold_bib_soft"""
import json, re, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from claim_verification.core import Cache, _post, _parse_json, _key
import claim_verification.run_gold_bib as gb
from claim_verification.run_gold_bib import (ROOT, PDF_DB, OUTD, review_ref_defs,
                                             resolve_gold)
from claim_verification.run_gold_openalex import GENERIC

# http_json caches FAILURES as None, which Cache.get can't distinguish from a miss ->
# re-running resolution replays every failed lookup through the 7-try/45s backoff ladder.
# Fail fast here (S2 fetcher is done; the key has the pool to itself now).
_orig_http_json = gb.http_json
gb.http_json = lambda url, cache, tries=7, sleep=1.0: _orig_http_json(
    url, cache, tries=2, sleep=0.5)

CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}

GRADED = """A paper CLAIMS:
"{claim}"

A reviewer says this was already done by PRIOR WORK:
Title: {title} ({year})
Abstract: {abstract}

On a 0-10 scale, how strongly does this prior work ANTICIPATE the claim?
  0-2  = unrelated or superficially similar topic only
  3-4  = same area, but the claimed contribution is clearly different
  5-6  = overlapping ideas; the claim is an incremental variant of the prior work
  7-8  = the prior work substantially discloses the claimed contribution
  9-10 = the prior work IS the claimed contribution (full anticipation)

Return ONLY JSON: {{"anticipation": <int 0-10>, "reason": "<one sentence>"}}"""

ANCHORS = [
    # self-anticipation: prior IS the claim -> expect >=8
    {"claim": "We introduce a contrastive language-image pretraining method that learns "
              "transferable visual representations from natural-language supervision.",
     "title": "Learning Transferable Visual Models From Natural Language Supervision",
     "year": 2021,
     "abstract": "We demonstrate that the simple pre-training task of predicting which "
                 "caption goes with which image is an efficient and scalable way to learn "
                 "image representations from scratch on 400 million image-text pairs, "
                 "enabling zero-shot transfer to downstream tasks.",
     "expect": "high"},
    # unrelated -> expect <=2
    {"claim": "We propose a graph neural network for molecular property prediction with "
              "E(3)-equivariant message passing.",
     "title": "Deep Residual Learning for Image Recognition",
     "year": 2016,
     "abstract": "We present a residual learning framework to ease the training of "
                 "networks that are substantially deeper than those used previously, "
                 "reformulating layers as learning residual functions.",
     "expect": "low"},
]

def grade(claim, g, cache):
    k = _key("graded_pa", CFG["model"], claim[:200], str(g["title"])[:120])
    hit = cache.get(k)
    if hit is not None: return hit
    raw = _post(CFG["base_url"], CFG["model"],
                GRADED.format(claim=claim[:400], title=g["title"], year=g.get("year"),
                              abstract=str(g.get("abstract", ""))[:1100]), max_tokens=160)
    obj = _parse_json(raw) or {}
    try: s = max(0, min(10, int(obj.get("anticipation", -1))))
    except Exception: s = -1
    out = {"score": s, "reason": str(obj.get("reason", ""))[:160]}
    cache.put(k, out)
    return out

def build_resolved(cache):
    nov = [json.loads(l) for l in open(f"{ROOT}/outputs/reviewer_flags/novelty_flags.jsonl")]
    named = [r for r in nov if r["flag"] and len(r.get("prior", "")) > 5
             and len(r.get("claim", "")) > 20]
    con = sqlite3.connect(PDF_DB)
    resolved = []
    for r in named:
        forum = r["paper"].replace("iclr_", "")
        citation = None
        marks = re.findall(r"\[(\d{1,2})\]", r["prior"]) or \
                re.findall(r"\[(\d{1,2})\]", r["sent"])
        if marks:
            row = con.execute("SELECT review_text FROM reviews WHERE paper_id=? AND "
                              "review_text LIKE ?", (forum, f"%{r['sent'][:80]}%")).fetchone()
            if row and row[0]:
                defs = review_ref_defs(row[0])
                for n_ in marks:
                    if n_ in defs: citation = defs[n_]; break
        if not citation:
            terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", r["prior"])
                     if t.lower() not in GENERIC][:5]
            if terms:
                srow = con.execute("SELECT sections FROM pdf_versions WHERE paper_id=? "
                                   "AND version=0", (forum,)).fetchone()
                if srow and srow[0]:
                    try: s = json.loads(srow[0])
                    except Exception: s = {}
                    refs = s.get("references") or ""
                    if len(refs) > 200:
                        low = refs.lower()
                        for t in sorted(terms, key=len, reverse=True):
                            i = low.find(t.lower())
                            if i >= 0:
                                citation = refs[max(0, i - 300):i + 450]; break
        if not citation: continue
        gold = resolve_gold(citation, r["prior"], cache)
        if gold:
            resolved.append({"paper": r["paper"], "claim": r["claim"],
                             "prior": r["prior"], **gold})
    return resolved

def main():
    cache = Cache(f"{OUTD}/oa_cache2.jsonl")
    gcache = Cache(f"{OUTD}/graded_soft_cache.jsonl")
    resolved = build_resolved(cache)
    print(f"[soft] resolved gold pairs: {len(resolved)}", flush=True)
    half = max(1, len(resolved) // 2)
    tasks = [("matched", x["claim"], x, x["paper"]) for x in resolved]
    tasks += [("mismatched", resolved[i]["claim"], resolved[(i + half) % len(resolved)],
               resolved[i]["paper"]) for i in range(len(resolved))]
    tasks += [("anchor_" + a["expect"], a["claim"], a, "anchor") for a in ANCHORS]
    rows = []
    def work(t):
        arm, claim, g, pid = t
        try: r = grade(claim, g, gcache)
        except Exception: return None
        return {"arm": arm, "paper": pid, "claim": claim[:100],
                "gold_title": str(g["title"])[:100], **r}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(work, tasks):
            if r and r["score"] >= 0: rows.append(r)
    for arm in sorted({r["arm"] for r in rows}):
        vs = [r["score"] for r in rows if r["arm"] == arm]
        print(f"  {arm:14} n={len(vs):3d} mean={np.mean(vs):.2f} median={np.median(vs):.1f} "
              f">=7: {np.mean([v >= 7 for v in vs]):.3f}  >=5: {np.mean([v >= 5 for v in vs]):.3f}", flush=True)
    m = [r["score"] for r in rows if r["arm"] == "matched"]
    mm = [r["score"] for r in rows if r["arm"] == "mismatched"]
    if m and mm:
        from sklearn.metrics import roc_auc_score
        y = [1] * len(m) + [0] * len(mm)
        a = roc_auc_score(y, m + mm)
        print(f"\n[soft] DISCRIMINATION AUC (matched vs mismatched, graded) = {a:.4f} "
              f"(binary checker fired .273 vs .000)", flush=True)
    with open(f"{OUTD}/gold_bib_soft_results.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    hi = sorted([r for r in rows if r["arm"] == "matched"], key=lambda r: -r["score"])[:4]
    for r in hi:
        print(f"  [{r['score']}] '{r['claim'][:70]}' <- {r['gold_title'][:60]}: {r['reason'][:90]}", flush=True)
    print("GOLD_BIB_SOFT_DONE", flush=True)

if __name__ == "__main__":
    main()
