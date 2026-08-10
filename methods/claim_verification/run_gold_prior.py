#!/usr/bin/env python3
"""GOLD-REFERENCE arm for the prior-art checker (patents lesson: organic retrieval excluded
the gold ref 82%+ of the time — test whether the same pathology explains our novelty null).
For each confirmed reviewer novelty complaint that NAMES prior work:
  1. resolve the named prior in our abstract pools (FTS by name tokens) -> POOL COVERAGE
  2. where found: run the anticipation verdict with the gold candidate FORCE-APPENDED
     alongside BM25 candidates -> does the checker say ANTICIPATED given the right reference?
  3. compare vs the organic-retrieval verdict for the same disputed claim.
Run on sk3: python -m methods.claim_verification.run_gold_prior"""
import json, os, re, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np
from claim_verification.core import Cache, _post, _parse_json, _key
from claim_verification.run_check_v2 import PriorFTS, pa_check

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}

GENERIC = {"model", "models", "method", "methods", "baseline", "baselines", "approach",
           "learning", "network", "networks", "decoding", "generation", "work", "works"}

def name_terms(prior):
    """Searchable tokens from a reviewer-named prior work; None if unresolvable ([1,2] etc.)."""
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", prior)
    toks = [t for t in toks if t.lower() not in GENERIC and t.lower() not in
            ("et", "al", "the", "and", "with", "for", "using", "many", "several")]
    return toks[:6] if toks else None

def find_gold(fts_paths, terms):
    """Search both abstract FTS bases for the named prior work."""
    q = " ".join(f'"{t}"' for t in terms[:3])
    for path, col in fts_paths:
        try:
            con = sqlite3.connect(path)
            rows = con.execute(f"SELECT text, year FROM ab WHERE ab MATCH ? "
                               f"ORDER BY bm25(ab) LIMIT 3", (q,)).fetchall()
            con.close()
        except sqlite3.OperationalError:
            continue
        for txt, yr in rows:
            # accept if at least 2 name terms (or the single distinctive one) appear
            low = txt.lower()
            hits = sum(1 for t in terms if t.lower() in low)
            if hits >= min(2, len(terms)):
                return (str(yr)[:4], txt[:900])
    return None

def main():
    nov = [json.loads(l) for l in open(f"{ROOT}/outputs/reviewer_flags/novelty_flags.jsonl")]
    named = [r for r in nov if r["flag"] and len(r.get("prior", "")) > 5
             and len(r.get("claim", "")) > 20]
    fts_paths = [(os.path.join(EB, "peer_abstracts.sqlite"), "paper_id"),
                 (os.path.join(EB, "citation_abstracts.sqlite"), "pid")]
    pfts = PriorFTS()
    cache = Cache(f"{ROOT}/outputs/checks_v2/gold_cache.jsonl")
    resolvable, found, results = 0, 0, []
    for r in named:
        terms = name_terms(r["prior"])
        if not terms: continue
        resolvable += 1
        gold = find_gold(fts_paths, terms)
        if not gold: continue
        found += 1
        claim = r["claim"]
        organic = pfts.query(claim, 2024, k=5)
        arms = {}
        try:
            arms["organic"] = pa_check(claim, 2024, organic, cache)["claim_verdict"] if organic else None
            arms["gold_appended"] = pa_check(claim, 2024, ([gold] + organic)[:6], cache)["claim_verdict"]
            arms["gold_only"] = pa_check(claim, 2024, [gold], cache)["claim_verdict"]
        except Exception:
            continue
        # was gold already in organic?
        in_org = any(gold[1][:80] == c[1][:80] for c in organic)
        results.append({"paper": r["paper"], "claim": claim, "prior": r["prior"],
                        "gold_in_organic": in_org, **arms})
    print(f"[gold] named priors: {len(named)}; resolvable name-terms: {resolvable}; "
          f"FOUND in pool: {found} (coverage {found/max(resolvable,1):.3f})", flush=True)
    def rate(key):
        vs = [x[key] for x in results if x.get(key)]
        n = max(len(vs), 1)
        return {v: round(sum(1 for y in vs if y == v) / n, 3)
                for v in ("ANTICIPATED", "RELATED", "CLEAR")}, len(vs)
    for k in ("organic", "gold_appended", "gold_only"):
        d, n = rate(k)
        print(f"  {k:14} n={n:3d}  {d}", flush=True)
    print(f"  gold ref present in organic top-5: "
          f"{np.mean([x['gold_in_organic'] for x in results]):.3f}", flush=True)
    with open(f"{ROOT}/outputs/checks_v2/gold_prior_results.jsonl", "w") as f:
        for x in results: f.write(json.dumps(x) + "\n")
    print("\n  examples (gold-only ANTICIPATED):", flush=True)
    k = 0
    for x in results:
        if x.get("gold_only") == "ANTICIPATED" and k < 4:
            print(f"    {x['paper']}: disputes '{x['claim'][:90]}' | named: {x['prior'][:50]}", flush=True)
            k += 1
    print("GOLD_PRIOR_DONE", flush=True)

if __name__ == "__main__":
    main()
