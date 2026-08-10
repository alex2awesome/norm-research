#!/usr/bin/env python3
"""Direct verifier validation, decoupled from the extractor: run verify_claim on the
reviewer-challenged claim paraphrases themselves (from run_reviewer_flags), against
their own paper's internals.
  arm A: challenged claims -> own paper internals   (expect NONE-heavy if verifier valid)
  arm B: our extracted claims, same papers          (baseline claim population)
  arm C: challenged claims -> MISMATCHED paper      (placebo; should be ~all NONE)
Run on sk3: python -m methods.claim_verification.run_verify_challenged"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
from claim_verification.core import Cache, verify_claim
from claim_verification.run_tiered_peer import PaperDB

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma", "doc_kind": "scientific paper"}

def main():
    chall = []   # (paper, claim_text)
    for ln in open(f"{ROOT}/outputs/reviewer_flags/flags.jsonl"):
        r = json.loads(ln)
        if r.get("flag") and len(r.get("claim", "")) > 25:
            chall.append((r["paper"], r["claim"]))
    papers = sorted({p for p, _ in chall})
    print(f"[vchal] {len(chall)} challenged claims over {len(papers)} papers", flush=True)
    ours = {}
    for ln in open(os.path.join(EB, "claims_peerintro.jsonl")):
        try:
            r = json.loads(ln)
            if str(r["doc_id"]) in papers and r.get("claims"):
                ours[str(r["doc_id"])] = [c["claim"] if isinstance(c, dict) else str(c)
                                          for c in r["claims"]][:4]
        except Exception: pass
    pdb = PaperDB()
    cache = Cache(f"{ROOT}/outputs/reviewer_flags/vchal_cache.jsonl")
    rng = np.random.default_rng(0)
    mism = {p: papers[(i + len(papers) // 2) % len(papers)] for i, p in enumerate(papers)}
    tasks = [("A_challenged", p, c, p) for p, c in chall]
    tasks += [("B_extracted", p, c, p) for p in papers for c in ours.get(p, [])]
    tasks += [("C_placebo", p, c, mism[p]) for p, c in chall]
    lock = Lock(); out = {"A_challenged": [], "B_extracted": [], "C_placebo": []}
    rows = []
    def work(t):
        arm, p, cl, pool_paper = t
        pool = pdb.internals(pool_paper.replace("iclr_", ""))
        if not pool: return
        try: v = verify_claim(cl, pool, CFG, cache)["verdict"]
        except Exception: return
        with lock:
            out[arm].append(v)
            rows.append({"arm": arm, "paper": p, "claim": cl, "verdict": v})
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(work, tasks))
    def dist(vs):
        n = max(len(vs), 1)
        return {k: round(sum(1 for v in vs if v == k) / n, 3) for k in ("FULL", "PARTIAL", "NONE")}
    for arm in ("A_challenged", "B_extracted", "C_placebo"):
        print(f"  {arm:14} n={len(out[arm]):4d}  {dist(out[arm])}", flush=True)
    # significance: NONE-rate A vs B (two-proportion permutation)
    a = np.array([v == "NONE" for v in out["A_challenged"]], float)
    b = np.array([v == "NONE" for v in out["B_extracted"]], float)
    if len(a) > 10 and len(b) > 10:
        obs = a.mean() - b.mean()
        pool_ = np.r_[a, b]; cnt = 0
        for _ in range(5000):
            rng.shuffle(pool_)
            if pool_[:len(a)].mean() - pool_[len(a):].mean() >= obs: cnt += 1
        print(f"  NONE-rate: challenged {a.mean():.3f} vs extracted {b.mean():.3f} "
              f"(diff {obs:+.3f}, perm p={cnt/5000:.4f})", flush=True)
    with open(f"{ROOT}/outputs/reviewer_flags/vchal_results.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    # show NONE examples from arm A (the "pipeline agrees with reviewer" cases)
    print("\n  examples where verifier AGREES with reviewer (NONE on challenged):", flush=True)
    k = 0
    for r in rows:
        if r["arm"] == "A_challenged" and r["verdict"] == "NONE" and k < 5:
            print(f"    {r['paper']}: {r['claim'][:130]}", flush=True); k += 1
    print("VCHAL_DONE", flush=True)

if __name__ == "__main__":
    main()
