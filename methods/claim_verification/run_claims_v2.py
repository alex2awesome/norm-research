#!/usr/bin/env python3
"""Typed claim extraction v2 — targets the claim populations reviewers actually dispute
(failure mode 1: our v1 top-4 intro contribution-claims missed 91% of reviewer-challenged
claims). Types: contribution / performance / novelty / assumption / scope / design_justification.
Input: abstract + introduction + method slice (assumptions & design choices live there).
Runs on the 600 tiered_peer papers. Output: evidence_bases/claims_v2_peer.jsonl
Run on sk3: python -m methods.claim_verification.run_claims_v2"""
import json, os, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import pandas as pd
from claim_verification.core import Cache, _post, _parse_json, _key

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
PDF_DB = f"{ROOT}/datasets/peer-review/peer_review_pdfs.db"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}
TYPES = ("contribution", "performance", "novelty", "assumption", "scope", "design_justification")

EXTRACT_V2 = """From this scientific paper text, extract the paper's key CLAIMS, each typed:
- "contribution": what the paper claims to contribute or do
- "performance": empirical superiority/improvement claims (e.g. "outperforms X", "surpasses state-of-the-art", "3x faster")
- "novelty": priority claims ("first to...", "novel...", "unlike all prior work...")
- "assumption": premises the work rests on, stated or implied ("we assume...", motivating facts taken as given)
- "scope": generalization claims ("generalizes to...", "robust across...", "works for any...")
- "design_justification": claims that a specific design choice is necessary, optimal, or justified

Extract up to 10 claims covering as many types as the text supports. Each claim must be a
short self-contained sentence (resolve pronouns and abbreviations).

TEXT:
{text}

Return ONLY JSON: {{"claims": [{{"claim": "<sentence>", "type": "<one of the six types>"}}, ...]}}"""

def main():
    t = pd.read_csv(f"{ROOT}/outputs/tiered_peer/tiered_metrics.csv")
    ids = t.id.astype(str).tolist()
    con = sqlite3.connect(PDF_DB)
    docs = []
    for pid in ids:
        forum = pid.replace("iclr_", "")
        row = con.execute("SELECT sections FROM pdf_versions WHERE paper_id=? AND version=0",
                          (forum,)).fetchone()
        if not row or not row[0]: continue
        try: s = json.loads(row[0])
        except Exception: continue
        txt = ((s.get("abstract") or "") + "\n\n" + (s.get("introduction") or "")[:4000]
               + "\n\n" + (s.get("method") or s.get("methods") or "")[:1500])
        if len(txt) > 500: docs.append((pid, txt[:7000]))
    print(f"[cv2] {len(docs)} papers with usable sections", flush=True)
    cache = Cache(os.path.join(EB, "claims_v2_cache.jsonl"))
    out_path = os.path.join(EB, "claims_v2_peer.jsonl")
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try: done.add(json.loads(ln)["doc_id"])
            except Exception: pass
    todo = [(p, x) for p, x in docs if p not in done]
    lock = Lock(); fout = open(out_path, "a"); n = [0]
    def work(item):
        pid, txt = item
        k = _key("cv2", CFG["model"], pid)
        hit = cache.get(k)
        if hit is None:
            try:
                raw = _post(CFG["base_url"], CFG["model"], EXTRACT_V2.format(text=txt),
                            max_tokens=900)
                obj = _parse_json(raw) or {}
                cl = [{"claim": str(c.get("claim", ""))[:400], "type": str(c.get("type", ""))}
                      for c in (obj.get("claims") or []) if isinstance(c, dict)]
                hit = [c for c in cl if c["type"] in TYPES and len(c["claim"]) > 20][:10]
                cache.put(k, hit)
            except Exception:
                hit = []
        with lock:
            fout.write(json.dumps({"doc_id": pid, "claims": hit}) + "\n"); fout.flush()
            n[0] += 1
            if n[0] % 100 == 0: print(f"[cv2] {n[0]}/{len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(work, todo))
    # type census
    from collections import Counter
    cnt = Counter(); per = []
    for ln in open(out_path):
        r = json.loads(ln); per.append(len(r["claims"]))
        for c in r["claims"]: cnt[c["type"]] += 1
    print(f"[cv2] type census: {dict(cnt)}; mean claims/paper "
          f"{sum(per)/max(len(per),1):.2f}", flush=True)
    print("CLAIMS_V2_DONE", flush=True)

if __name__ == "__main__":
    main()
