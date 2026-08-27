#!/usr/bin/env python3
"""Claim-QUALITY A-layer (user q2: 'good, interesting, elegant' vs 'bad, boring, poorly worded').
Judges each doc's claims on 5 dimensions (0-2 each, guided JSON):
  specificity   — precise, checkable commitments vs vague generalities
  ambition      — scope/consequence of what is claimed vs incremental/trivial
  surprisingness— violates expectation / non-obvious vs routine
  falsifiability— could evidence disprove it vs unfalsifiable puffery
  elegance      — crisply worded, one clean assertion vs muddled/hedged
Doc metrics: mean per-dimension + claim_quality_mean. Runs per domain over extracted claims.
Run on sk3: python -m methods.claim_verification.run_claim_quality --domain peerintro|pr|newsfull [--n 800]"""
import argparse, json, os, sys, time
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
from claim_verification.core import Cache, _post, _parse_json, _key

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}
DIMS = ["specificity", "ambition", "surprisingness", "falsifiability", "elegance"]

PROMPT = """Rate the QUALITY of this claim from a {doc_kind} on five dimensions, each 0-2:
- specificity: 2 = precise checkable commitment (numbers, named mechanisms); 0 = vague generality
- ambition: 2 = consequential, broad-scope assertion; 0 = trivial or incremental
- surprisingness: 2 = non-obvious, violates expectations; 0 = routine, expected
- falsifiability: 2 = evidence could clearly disprove it; 0 = unfalsifiable puffery
- elegance: 2 = crisply worded single assertion; 0 = muddled, hedged, poorly worded

CLAIM: {claim}

Return ONLY JSON: {{"specificity": <0-2>, "ambition": <0-2>, "surprisingness": <0-2>, "falsifiability": <0-2>, "elegance": <0-2>}}"""

KIND = {"peerintro": "scientific paper", "pr": "press release", "newsfull": "news article"}

def judge_claim(claim, doc_kind, cache):
    k = _key("cq", CFG["model"], claim)
    hit = cache.get(k)
    if hit is not None: return hit
    raw = _post(CFG["base_url"], CFG["model"], PROMPT.format(doc_kind=doc_kind, claim=claim[:400]),
                max_tokens=120)
    obj = _parse_json(raw) or {}
    out = {}
    for d in DIMS:
        try: out[d] = max(0.0, min(2.0, float(obj.get(d))))
        except Exception: out[d] = None
    cache.put(k, out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(KIND))
    ap.add_argument("--n", type=int, default=800, help="max docs (0=all)")
    ap.add_argument("--ids-file", default=None, help="file with one doc_id per line; score exactly these")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    want = None
    if args.ids_file:
        want = {ln.strip() for ln in open(args.ids_file) if ln.strip()}
        args.n = 0  # ids-file overrides the sampling cap
    src = os.path.join(EB, f"claims_{args.domain}.jsonl")
    out_path = os.path.join(EB, f"claimquality_{args.domain}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try: done.add(json.loads(ln)["doc_id"])
            except Exception: pass
    docs = []
    for ln in open(src):
        try:
            r = json.loads(ln)
            if r.get("claims") and r["doc_id"] not in done and (want is None or str(r["doc_id"]) in want):
                docs.append((r["doc_id"], [c["claim"] if isinstance(c, dict) else str(c) for c in r["claims"]][:4]))
        except Exception: pass
    import random; random.Random(0).shuffle(docs)
    if args.n: docs = docs[:args.n]
    print(f"[cq-{args.domain}] {len(docs)} docs (done={len(done)})", flush=True)
    cache = Cache(os.path.join(EB, f"claimquality_cache_{args.domain}.jsonl"))
    lock = Lock(); fout = open(out_path, "a"); n = [0]
    def work(item):
        doc_id, claims = item
        try:
            scores = [judge_claim(c, KIND[args.domain], cache) for c in claims]
            m = {"doc_id": doc_id, "n_claims": len(claims)}
            for d in DIMS:
                vals = [s[d] for s in scores if s.get(d) is not None]
                m[f"cq_{d}"] = float(np.mean(vals)) if vals else None
            allv = [s[d] for s in scores for d in DIMS if s.get(d) is not None]
            m["cq_mean"] = float(np.mean(allv)) if allv else None
        except Exception as e:
            m = {"doc_id": doc_id, "err": str(e)[:80]}
        with lock:
            fout.write(json.dumps(m) + "\n"); fout.flush()
            n[0] += 1
            if n[0] % 200 == 0: print(f"[cq-{args.domain}] {n[0]}/{len(docs)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, docs))
    print(f"CQ_{args.domain.upper()}_DONE", flush=True)

if __name__ == "__main__":
    main()
