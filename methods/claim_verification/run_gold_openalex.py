#!/usr/bin/env python3
"""TRUE gold arm: resolve reviewer-NAMED prior work via the OpenAlex search API (any venue —
fixes the 3-venue pool gap that broke the name-match arm), fetch the real abstract, and run
BOTH checkers against it: SS102 anticipation and SS103 delta.
GATE G2: resolution rate >= .50 of named priors AND fired-rate (ANTICIPATED or TRIVIAL_DELTA)
on disputed claims >> the 4.8% name-match arm.
Run on sk3: python -m methods.claim_verification.run_gold_openalex"""
import json, os, re, sys, time, urllib.parse, urllib.request, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np
from claim_verification.core import Cache
from claim_verification.run_check_v2 import pa_check
from claim_verification.run_delta_check import delta_check

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
GENERIC = {"model", "models", "method", "methods", "baseline", "baselines", "approach",
           "learning", "network", "networks", "work", "works", "literature", "et", "al"}

def name_query(prior):
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", prior)
    toks = [t for t in toks if t.lower() not in GENERIC]
    return " ".join(toks[:8]) if toks else None

def openalex_search(q, cache, tries=4):
    key = "oa:" + q
    hit = cache.get(key)
    if hit is not None: return hit
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(q)
           + "&per-page=3&mailto=alex2awesome@gmail.com"
           + "&filter=from_publication_date:2010-01-01,to_publication_date:2024-06-30")
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                res = json.loads(r.read()).get("results", [])
            break
        except Exception:
            if t == tries - 1: res = []
            time.sleep(5 * (t + 1))
    out = []
    for w in res:
        inv = w.get("abstract_inverted_index")
        if not inv: continue
        pos = {}
        for word, idxs in inv.items():
            for i in idxs: pos[i] = word
        abstract = " ".join(pos[i] for i in sorted(pos))[:1200]
        if len(abstract) > 200:
            out.append({"title": w.get("title", ""), "year": w.get("publication_year"),
                        "abstract": abstract})
    cache.put(key, out)
    time.sleep(0.15)
    return out

def main():
    nov = [json.loads(l) for l in open(f"{ROOT}/outputs/reviewer_flags/novelty_flags.jsonl")]
    named = [r for r in nov if r["flag"] and len(r.get("prior", "")) > 5
             and len(r.get("claim", "")) > 20]
    cache = Cache(f"{ROOT}/outputs/checks_v2/oa_cache.jsonl")
    resolvable, resolved, rows = 0, 0, []
    for r in named:
        q = name_query(r["prior"])
        if not q: continue
        resolvable += 1
        works = openalex_search(q, cache)
        if not works: continue
        resolved += 1
        gold = works[0]
        cands102 = [(str(gold["year"]), f"{gold['title']}. {gold['abstract']}")]
        cands103 = [f"{gold['title']} ({gold['year']}): {gold['abstract']}"]
        try:
            v102 = pa_check(r["claim"], 2024, cands102, cache)["claim_verdict"]
            v103 = delta_check(r["claim"], cands103, cache)["verdict"]
        except Exception:
            continue
        rows.append({"paper": r["paper"], "claim": r["claim"], "prior": r["prior"],
                     "gold_title": gold["title"], "gold_year": gold["year"],
                     "v102": v102, "v103": v103})
    res_rate = resolved / max(resolvable, 1)
    print(f"[oa] named priors resolvable: {resolvable}; RESOLVED via OpenAlex: {resolved} "
          f"({res_rate:.3f})", flush=True)
    def dist(key, vals):
        vs = [x[key] for x in rows]
        n = max(len(vs), 1)
        return {v: round(sum(1 for y in vs if y == v) / n, 3) for v in vals}
    print(f"  SS102 on gold: {dist('v102', ('ANTICIPATED','RELATED','CLEAR'))}", flush=True)
    print(f"  SS103 on gold: {dist('v103', ('TRIVIAL_DELTA','SUBSTANTIVE_DELTA','NO_OVERLAP','PARSE_FAIL'))}", flush=True)
    fired = np.mean([x["v102"] == "ANTICIPATED" or x["v103"] == "TRIVIAL_DELTA" for x in rows]) if rows else 0
    print(f"  fired (ANTICIPATED or TRIVIAL_DELTA) on disputed: {fired:.3f} "
          f"(name-match arm was .048)", flush=True)
    gate = res_rate >= 0.5 and fired >= 0.35
    print(f"  GATE G2 (resolution>=.50 & fired>=.35): {'PASS' if gate else 'FAIL'}", flush=True)
    with open(f"{ROOT}/outputs/checks_v2/gold_openalex_results.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print("\n  examples (fired on gold):", flush=True)
    k = 0
    for x in rows:
        if (x["v102"] == "ANTICIPATED" or x["v103"] == "TRIVIAL_DELTA") and k < 5:
            print(f"    {x['paper']}: disputes '{x['claim'][:80]}'\n      named: {x['prior'][:60]} "
                  f"-> resolved: {str(x['gold_title'])[:70]} ({x['gold_year']}) "
                  f"[{x['v102']}/{x['v103']}]", flush=True)
            k += 1
    print("GOLD_OPENALEX_DONE", flush=True)

if __name__ == "__main__":
    main()
