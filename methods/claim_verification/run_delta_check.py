#!/usr/bin/env python3
"""SPOT-CHECK the SS103 fix: DELTA-graded prior-art verdict instead of single-ref disclosure.
For each claim + prior-work descriptions (paper's own related-work section), the checker must
STATE the delta (what the claim adds beyond prior work) and grade it TRIVIAL_DELTA /
SUBSTANTIVE_DELTA / NO_OVERLAP. This is the proposition reviewers actually assert
("incremental", "straightforward combination").
Arms (identical populations to run_rw_prior for comparability):
  A = reviewer-disputed novel things (confirmed novelty complaints)
  B = extracted novelty-type claims from non-flagged papers (control)
GATE G1: TRIVIAL_DELTA(A) - TRIVIAL_DELTA(B) >= +.15, perm p < .05
         (SS102 version scored +.097 p=.077 -> must beat it to justify expansion).
Run on sk3: python -m methods.claim_verification.run_delta_check"""
import json, os, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
from claim_verification.core import Cache, _post, _parse_json, _key
from claim_verification.evidence_api import chunk_passages

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
PDF_DB = f"{ROOT}/datasets/peer-review/peer_review_pdfs.db"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}

DELTA = """A machine-learning paper makes this CLAIM:
CLAIM: {claim}

Below are descriptions of PRIOR WORK (from the paper's own related-work discussion):
{cands}

Step 1 — state precisely what the claim adds BEYOND the prior work described (the delta).
Step 2 — grade the delta:
- "TRIVIAL_DELTA": the addition is a straightforward variation, combination, tuning, or
  application of the prior work to a new setting — the kind of increment a competent
  practitioner would consider obvious.
- "SUBSTANTIVE_DELTA": the addition is a genuinely new idea, mechanism, formulation, or
  capability not suggested by the prior work described.
- "NO_OVERLAP": the prior work described is not related enough to this claim to grade a delta.

Return ONLY JSON: {{"delta": "<one sentence: what the claim adds>", "verdict": "TRIVIAL_DELTA"|"SUBSTANTIVE_DELTA"|"NO_OVERLAP"}}"""

def delta_check(claim, cands, cache):
    k = _key("delta", CFG["model"], claim, "|".join(c[:60] for c in cands))
    hit = cache.get(k)
    if hit is not None: return hit
    ctext = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(cands))
    raw = _post(CFG["base_url"], CFG["model"], DELTA.format(claim=claim[:400], cands=ctext),
                max_tokens=280)
    obj = _parse_json(raw) or {}
    v = str(obj.get("verdict", "")).upper()
    if v not in ("TRIVIAL_DELTA", "SUBSTANTIVE_DELTA", "NO_OVERLAP"): v = "PARSE_FAIL"
    out = {"verdict": v, "delta": str(obj.get("delta", ""))[:250]}
    cache.put(k, out)
    return out

def rw_chunks(con, forum):
    row = con.execute("SELECT sections FROM pdf_versions WHERE paper_id=? AND version=0",
                      (forum,)).fetchone()
    if not row or not row[0]: return []
    try: s = json.loads(row[0])
    except Exception: return []
    rw = s.get("related work") or s.get("related works") or ""
    if len(rw) < 300: return []
    return chunk_passages(rw, words_per=130, max_passages=6)

def main():
    nov = [json.loads(l) for l in open(f"{ROOT}/outputs/reviewer_flags/novelty_flags.jsonl")]
    conf = [r for r in nov if r["flag"] and len(r.get("claim", "")) > 20]
    flagged_papers = {r["paper"] for r in conf}
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
    cache = Cache(f"{ROOT}/outputs/checks_v2/delta_cache.jsonl")
    import threading
    local = threading.local()
    lock = Lock(); out = {"A_disputed": [], "B_control": []}; rows = []
    def work(t):
        arm, pid, claim = t
        if getattr(local, "con", None) is None:
            local.con = sqlite3.connect(PDF_DB)
        cands = rw_chunks(local.con, pid.replace("iclr_", ""))
        if not cands: return
        try: r = delta_check(claim, cands, cache)
        except Exception: return
        with lock:
            out[arm].append(r["verdict"])
            rows.append({"arm": arm, "paper": pid, "claim": claim, **r})
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(work, tasks))
    for arm in ("A_disputed", "B_control"):
        vs = out[arm]; n = max(len(vs), 1)
        d = {k: round(sum(1 for v in vs if v == k) / n, 3)
             for k in ("TRIVIAL_DELTA", "SUBSTANTIVE_DELTA", "NO_OVERLAP", "PARSE_FAIL")}
        print(f"  {arm:12} n={len(vs):3d}  {d}", flush=True)
    a = np.array([v == "TRIVIAL_DELTA" for v in out["A_disputed"]], float)
    b = np.array([v == "TRIVIAL_DELTA" for v in out["B_control"]], float)
    obs = a.mean() - b.mean()
    pool = np.r_[a, b]; cnt = 0
    for _ in range(5000):
        rng.shuffle(pool)
        if pool[:len(a)].mean() - pool[len(a):].mean() >= obs: cnt += 1
    p = cnt / 5000
    print(f"  TRIVIAL_DELTA: disputed {a.mean():.3f} vs control {b.mean():.3f} "
          f"(diff {obs:+.3f}, perm p={p:.4f})", flush=True)
    gate = obs >= 0.15 and p < 0.05
    print(f"  GATE G1 (diff>=+.15 & p<.05): {'PASS' if gate else 'FAIL'}", flush=True)
    with open(f"{ROOT}/outputs/checks_v2/delta_results.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print("\n  examples disputed+TRIVIAL_DELTA:", flush=True)
    k = 0
    for x in rows:
        if x["arm"] == "A_disputed" and x["verdict"] == "TRIVIAL_DELTA" and k < 4:
            print(f"    {x['paper']}: {x['claim'][:95]}\n      delta: {x['delta'][:130]}", flush=True)
            k += 1
    print("  examples control+SUBSTANTIVE_DELTA:", flush=True)
    k = 0
    for x in rows:
        if x["arm"] == "B_control" and x["verdict"] == "SUBSTANTIVE_DELTA" and k < 3:
            print(f"    {x['paper']}: {x['claim'][:95]}\n      delta: {x['delta'][:130]}", flush=True)
            k += 1
    print("DELTA_CHECK_DONE", flush=True)

if __name__ == "__main__":
    main()
