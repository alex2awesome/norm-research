#!/usr/bin/env python3
"""Evidence-ADEQUACY verifier mode (construct fix candidate): the standard verifier measures
assertion-echo (paper body states the claim), but reviewers' "not supported" means the
EVIDENCE doesn't establish the claim at its scope. Three-way verdict:
  ESTABLISHED   — passages contain evidence (results/experiments/proofs) covering the claim's scope
  ASSERTED_ONLY — passages state/imply the claim but shown evidence doesn't cover its scope
  ABSENT        — passages neither state nor support the claim
Arms: A = reviewer-challenged claims (propositional only) vs B = our extracted claims (control).
Prediction if this fixes the construct: A piles into ASSERTED_ONLY, B into ESTABLISHED.
Run on sk3: python -m methods.claim_verification.run_adequacy_mode"""
import json, os, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
from claim_verification.core import Cache, _post, _parse_json, _key
from claim_verification.run_tiered_peer import PaperDB

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}

NONPROP = re.compile(
    r"(mentioned by the reviewer|as stated in|presented in the|summarized at|Remark \d|lines? \d"
    r"|^(the|certain|multiple) (claims?|assertions?|statements?|contributions?|motivations?)\b"
    r"|claims? (made|regarding)? ?(throughout|in the paper)$|of the (paper|work)$)", re.I)

ADEQ = """You are auditing a scientific paper. Below is a CLAIM the paper makes, and passages
from the paper's methods/experiments/results/conclusion sections.

CLAIM: {claim}

PASSAGES:
{passages}

Judge whether the presented evidence ESTABLISHES the claim at its stated scope:
- "ESTABLISHED": the passages contain empirical results, experiments, or proofs that actually
  cover the claim's full scope (generality, magnitude, comparisons claimed).
- "ASSERTED_ONLY": the passages state or restate the claim, but the evidence shown does not
  cover its scope (e.g., asserted without experiments, tested on narrower settings than
  claimed, magnitude or comparison not demonstrated).
- "ABSENT": the passages neither state nor support the claim.

Return ONLY JSON: {{"verdict": "ESTABLISHED"|"ASSERTED_ONLY"|"ABSENT", "reason": "<one sentence>"}}"""

def adeq(claim, pool, cache):
    k = _key("adeq", CFG["model"], claim, "|".join(pool)[:1500])
    hit = cache.get(k)
    if hit is not None: return hit
    ps = "\n---\n".join(p[:600] for p in pool[:12])
    raw = _post(CFG["base_url"], CFG["model"], ADEQ.format(claim=claim[:400], passages=ps),
                max_tokens=200)
    obj = _parse_json(raw) or {}
    v = str(obj.get("verdict", "")).upper()
    if v not in ("ESTABLISHED", "ASSERTED_ONLY", "ABSENT"): v = "PARSE_FAIL"
    out = {"verdict": v, "reason": str(obj.get("reason", ""))[:200]}
    cache.put(k, out)
    return out

def main():
    chall = []
    for ln in open(f"{ROOT}/outputs/reviewer_flags/flags.jsonl"):
        r = json.loads(ln)
        c = r.get("claim", "")
        if r.get("flag") and len(c) > 25 and not NONPROP.search(c):
            chall.append((r["paper"], c))
    papers = sorted({p for p, _ in chall})
    print(f"[adeq] {len(chall)} propositional challenged claims over {len(papers)} papers "
          f"(non-propositional filtered)", flush=True)
    ours = {}
    for ln in open(os.path.join(EB, "claims_peerintro.jsonl")):
        try:
            r = json.loads(ln)
            if str(r["doc_id"]) in papers and r.get("claims"):
                ours[str(r["doc_id"])] = [c["claim"] if isinstance(c, dict) else str(c)
                                          for c in r["claims"]][:4]
        except Exception: pass
    pdb = PaperDB()
    cache = Cache(f"{ROOT}/outputs/reviewer_flags/adeq_cache.jsonl")
    tasks = [("A_challenged", p, c) for p, c in chall]
    tasks += [("B_extracted", p, c) for p in papers for c in ours.get(p, [])]
    lock = Lock(); out = {"A_challenged": [], "B_extracted": []}; rows = []
    def work(t):
        arm, p, cl = t
        pool = pdb.internals(p.replace("iclr_", ""))
        if not pool: return
        try: r = adeq(cl, pool, cache)
        except Exception: return
        with lock:
            out[arm].append(r["verdict"])
            rows.append({"arm": arm, "paper": p, "claim": cl, **r})
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(work, tasks))
    keys = ("ESTABLISHED", "ASSERTED_ONLY", "ABSENT", "PARSE_FAIL")
    for arm in ("A_challenged", "B_extracted"):
        n = max(len(out[arm]), 1)
        d = {k: round(sum(1 for v in out[arm] if v == k) / n, 3) for k in keys}
        print(f"  {arm:14} n={len(out[arm]):4d}  {d}", flush=True)
    # the discriminative readout: NOT-ESTABLISHED rate, A vs B, perm test
    rng = np.random.default_rng(0)
    a = np.array([v != "ESTABLISHED" for v in out["A_challenged"] if v != "PARSE_FAIL"], float)
    b = np.array([v != "ESTABLISHED" for v in out["B_extracted"] if v != "PARSE_FAIL"], float)
    if len(a) > 10 and len(b) > 10:
        obs = a.mean() - b.mean()
        pool_ = np.r_[a, b]; cnt = 0
        for _ in range(5000):
            rng.shuffle(pool_)
            if pool_[:len(a)].mean() - pool_[len(a):].mean() >= obs: cnt += 1
        print(f"  NOT-ESTABLISHED: challenged {a.mean():.3f} vs extracted {b.mean():.3f} "
              f"(diff {obs:+.3f}, perm p={cnt/5000:.4f})", flush=True)
    with open(f"{ROOT}/outputs/reviewer_flags/adeq_results.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print("\n  sample ASSERTED_ONLY on challenged (reviewer-agreement cases):", flush=True)
    k = 0
    for r in rows:
        if r["arm"] == "A_challenged" and r["verdict"] == "ASSERTED_ONLY" and k < 5:
            print(f"    {r['paper']}: {r['claim'][:110]}\n      reason: {r['reason'][:130]}", flush=True)
            k += 1
    print("ADEQ_DONE", flush=True)

if __name__ == "__main__":
    main()
