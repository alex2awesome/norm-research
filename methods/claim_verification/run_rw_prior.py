#!/usr/bin/env python3
"""RELATED-WORK prior-art arm: reviewers' novelty ammunition is usually cited by the paper
itself. Check disputed novelty claims against the paper's OWN related-work section (any venue,
method-level descriptions) — no external retrieval needed.
  arm A: reviewer-disputed novel things (confirmed novelty complaints)   -> own related-work
  arm B: our extracted novelty-type claims from NON-flagged papers       -> own related-work
If A >> B on ANTICIPATED, the checker works once given the right evidence base.
Run on sk3: python -m methods.claim_verification.run_rw_prior"""
import json, os, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
from claim_verification.core import Cache
from claim_verification.evidence_api import chunk_passages
from claim_verification.run_check_v2 import pa_check

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
PDF_DB = f"{ROOT}/datasets/peer-review/peer_review_pdfs.db"

def rw_chunks(con, forum):
    row = con.execute("SELECT sections FROM pdf_versions WHERE paper_id=? AND version=0",
                      (forum,)).fetchone()
    if not row or not row[0]: return []
    try: s = json.loads(row[0])
    except Exception: return []
    rw = s.get("related work") or s.get("related works") or ""
    if len(rw) < 300: return []
    return [("", c) for c in chunk_passages(rw, words_per=130, max_passages=6)]

def main():
    nov = [json.loads(l) for l in open(f"{ROOT}/outputs/reviewer_flags/novelty_flags.jsonl")]
    conf = [r for r in nov if r["flag"] and len(r.get("claim", "")) > 20]
    flagged_papers = {r["paper"] for r in conf}
    # control: extracted novelty-type claims from non-flagged papers
    ctrl = []
    for ln in open(os.path.join(EB, "claims_v2_peer.jsonl")):
        r = json.loads(ln)
        if r["doc_id"] in flagged_papers: continue
        for c in r.get("claims", []):
            if c["type"] == "novelty": ctrl.append((r["doc_id"], c["claim"]))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ctrl), size=min(250, len(ctrl)), replace=False)
    tasks = [("A_disputed", r["paper"], r["claim"]) for r in conf]
    tasks += [("B_control", ctrl[i][0], ctrl[i][1]) for i in idx]
    print(f"[rw] arm A (disputed): {sum(1 for t in tasks if t[0]=='A_disputed')}; "
          f"arm B (control novelty claims): {sum(1 for t in tasks if t[0]=='B_control')}", flush=True)
    cache = Cache(f"{ROOT}/outputs/checks_v2/rw_cache.jsonl")
    import threading
    local = threading.local()
    lock = Lock(); out = {"A_disputed": [], "B_control": []}; rows = []
    def work(t):
        arm, pid, claim = t
        if getattr(local, "con", None) is None:
            local.con = sqlite3.connect(PDF_DB)
        cands = rw_chunks(local.con, pid.replace("iclr_", ""))
        if not cands: return
        try: r = pa_check(claim, 2024, cands, cache)
        except Exception: return
        with lock:
            out[arm].append(r["claim_verdict"])
            rows.append({"arm": arm, "paper": pid, "claim": claim,
                         "verdict": r["claim_verdict"],
                         "span": next((j["span"] for j in r["judgments"]
                                       if j["verdict"] == "ANTICIPATED" and j["span"]), "")})
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(work, tasks))
    for arm in ("A_disputed", "B_control"):
        vs = out[arm]; n = max(len(vs), 1)
        d = {k: round(sum(1 for v in vs if v == k) / n, 3)
             for k in ("ANTICIPATED", "RELATED", "CLEAR")}
        print(f"  {arm:12} n={len(vs):3d}  {d}", flush=True)
    a = np.array([v == "ANTICIPATED" for v in out["A_disputed"]], float)
    b = np.array([v == "ANTICIPATED" for v in out["B_control"]], float)
    if len(a) > 10 and len(b) > 10:
        obs = a.mean() - b.mean()
        pool = np.r_[a, b]; cnt = 0
        for _ in range(5000):
            rng.shuffle(pool)
            if pool[:len(a)].mean() - pool[len(a):].mean() >= obs: cnt += 1
        print(f"  ANTICIPATED: disputed {a.mean():.3f} vs control {b.mean():.3f} "
              f"(diff {obs:+.3f}, perm p={cnt/5000:.4f})", flush=True)
    with open(f"{ROOT}/outputs/checks_v2/rw_prior_results.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print("\n  examples (disputed + ANTICIPATED by own related-work):", flush=True)
    k = 0
    for x in rows:
        if x["arm"] == "A_disputed" and x["verdict"] == "ANTICIPATED" and k < 5:
            print(f"    {x['paper']}: {x['claim'][:100]}\n      span: {x['span'][:120]}", flush=True)
            k += 1
    print("RW_PRIOR_DONE", flush=True)

if __name__ == "__main__":
    main()
