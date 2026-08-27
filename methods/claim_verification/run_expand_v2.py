#!/usr/bin/env python3
"""FULL-SET expansion of the validated v2 pipeline (gates: delta-checker G1 diff +.145
p=.0064 vs declared +.15/p<.05 — marginal-pass on effect, decisive on significance;
adequacy checker claim-level p=.011/.041; external gold arm G2 FAILED -> excluded).
Set: ALL 14,495 labeled ICLR 2024-25 papers with PDF sections + reviews.
Per paper (one worker = one paper, resumable):
  1. typed extraction (contribution/performance/novelty/assumption/scope/design_justification)
  2. adequacy check per claim vs internals (ESTABLISHED/ASSERTED_ONLY/ABSENT)
  3. rw-delta check (TRIVIAL/SUBSTANTIVE/NO_OVERLAP) for novelty+performance+contribution
     claims vs own related-work section
Then: flags mining (support + novelty complaints, anchored judges) over all their reviews.
Then: run_race_v2 (separate) for outcome + flags validation at scale.
Run on sk3: python -m methods.claim_verification.run_expand_v2 [--workers 32]"""
import argparse, json, os, re, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, local as thread_local
import numpy as np, pandas as pd
from claim_verification.core import Cache, _post, _parse_json, _key, _sentences
from claim_verification.run_claims_v2 import EXTRACT_V2, TYPES
from claim_verification.run_adequacy_mode import ADEQ
from claim_verification.run_delta_check import DELTA, delta_check
from claim_verification.evidence_api import chunk_passages
import claim_verification.run_reviewer_flags as SF
import claim_verification.run_novelty_flags as NF

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
PDF_DB = f"{ROOT}/datasets/peer-review/peer_review_pdfs.db"
OUT = f"{ROOT}/outputs/expand_v2"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma"}
PA_TYPES = ("novelty", "performance", "contribution")
INTERNAL = ["experiments", "results", "related work", "conclusion", "method", "methods"]

def get_sections(con, forum):
    row = con.execute("SELECT sections FROM pdf_versions WHERE paper_id=? AND version=0",
                      (forum,)).fetchone()
    if not row or not row[0]: return {}
    try: return json.loads(row[0])
    except Exception: return {}

def stage_papers(sample, ex_cache, ad_cache, dl_cache, workers):
    done = set()
    out_path = f"{OUT}/paper_checks.jsonl"
    if os.path.exists(out_path):
        for ln in open(out_path):
            try: done.add(json.loads(ln)["doc_id"])
            except Exception: pass
    todo = [r for r in sample.itertuples() if r.paper_id not in done]
    print(f"[exp] papers todo={len(todo)} done={len(done)}", flush=True)
    tl = thread_local()
    lock = Lock(); fout = open(out_path, "a"); n = [len(done)]
    def work(r):
        pid, forum = r.paper_id, r.forum
        if getattr(tl, "con", None) is None:
            tl.con = sqlite3.connect(PDF_DB)
        s = get_sections(tl.con, forum)
        txt = ((s.get("abstract") or "") + "\n\n" + (s.get("introduction") or "")[:4000]
               + "\n\n" + (s.get("method") or s.get("methods") or "")[:1500])
        row = {"doc_id": pid, "claims": []}
        if len(txt) > 500:
            k = _key("cv2", CFG["model"], pid)
            cl = ex_cache.get(k)
            if cl is None:
                try:
                    raw = _post(CFG["base_url"], CFG["model"],
                                EXTRACT_V2.format(text=txt[:7000]), max_tokens=900)
                    obj = _parse_json(raw) or {}
                    cl = [{"claim": str(c.get("claim", ""))[:400], "type": str(c.get("type", ""))}
                          for c in (obj.get("claims") or []) if isinstance(c, dict)]
                    cl = [c for c in cl if c["type"] in TYPES and len(c["claim"]) > 20][:10]
                    ex_cache.put(k, cl)
                except Exception:
                    cl = []
            internals_txt = "\n".join(str(s.get(x, "")) for x in INTERNAL if s.get(x))
            pool = chunk_passages(internals_txt, words_per=110, max_passages=14)
            rw = s.get("related work") or s.get("related works") or ""
            rw_c = chunk_passages(rw, words_per=130, max_passages=6) if len(rw) > 300 else []
            for c in cl:
                item = dict(c)
                if pool:
                    ka = _key("adeq", CFG["model"], c["claim"], "|".join(pool)[:1500])
                    hit = ad_cache.get(ka)
                    if hit is None:
                        try:
                            ps = "\n---\n".join(p[:600] for p in pool[:12])
                            raw = _post(CFG["base_url"], CFG["model"],
                                        ADEQ.format(claim=c["claim"][:400], passages=ps),
                                        max_tokens=200)
                            obj = _parse_json(raw) or {}
                            v = str(obj.get("verdict", "")).upper()
                            if v not in ("ESTABLISHED", "ASSERTED_ONLY", "ABSENT"): v = None
                            hit = {"verdict": v, "reason": ""}
                            ad_cache.put(ka, hit)
                        except Exception:
                            hit = {"verdict": None}
                    item["adequacy"] = hit.get("verdict")
                if rw_c and c["type"] in PA_TYPES:
                    try:
                        item["delta"] = delta_check(c["claim"], rw_c, dl_cache)["verdict"]
                    except Exception:
                        item["delta"] = None
                row["claims"].append(item)
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n[0] += 1
            if n[0] % 200 == 0: print(f"[exp] papers {n[0]}/{len(sample)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    print("EXPAND_PAPERS_DONE", flush=True)

def stage_flags(sample, workers):
    forums = {r.forum: r.paper_id for r in sample.itertuples()}
    con = sqlite3.connect(PDF_DB)
    revs = []
    for chunk in np.array_split(list(forums), 20):
        q = ",".join("?" * len(chunk))
        revs += con.execute(f"SELECT paper_id, review_text, is_meta_review FROM reviews "
                            f"WHERE paper_id IN ({q})", list(chunk)).fetchall()
    print(f"[exp-flags] {len(revs)} reviews", flush=True)
    for tag, mod, judge, cachef in (("support", SF, SF.judge_sent, "sup"),
                                    ("novelty", NF, NF.judge_sent, "nov")):
        out_path = f"{OUT}/flags_{tag}.jsonl"
        done_s = set()
        if os.path.exists(out_path):
            for ln in open(out_path):
                try: done_s.add(json.loads(ln)["sent"][:120])
                except Exception: pass
        cands = []
        for pid, txt, meta in revs:
            if not txt or meta: continue
            for s in _sentences(str(txt)):
                if 40 < len(s) < 700 and mod.CAND.search(s) and s[:120] not in done_s:
                    cands.append((forums[pid], s))
        cache = Cache(f"{OUT}/flags_{cachef}_cache.jsonl")
        lock = Lock(); fout = open(out_path, "a"); anchor_out = {}; n = [0]
        items = [("ANCHOR", s, lab) for s, lab in mod.ANCHORS] + [(p, s, None) for p, s in cands]
        def work(item):
            p, s, lab = item
            try: r = judge(s, cache)
            except Exception: return
            with lock:
                if p == "ANCHOR": anchor_out[s[:50]] = (r["flag"], lab)
                else:
                    fout.write(json.dumps({"paper": p, "sent": s, **r}) + "\n")
                    n[0] += 1
                    if n[0] % 1000 == 0:
                        print(f"[exp-flags-{tag}] {n[0]}/{len(cands)}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, items))
        fout.flush()
        ok = sum(1 for f_, l in anchor_out.values() if f_ == l)
        print(f"[exp-flags-{tag}] candidates={len(cands)} ANCHORS {ok}/{len(anchor_out)}", flush=True)
    print("EXPAND_FLAGS_DONE", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sp = f"{OUT}/sample.csv"
    if os.path.exists(sp):
        sample = pd.read_csv(sp)
    else:
        d = pd.read_csv(f"{ROOT}/datasets/peer-review/peer_review_modeling_dataset.csv.gz",
                        compression="gzip")
        d = d[d.paper_id.astype(str).str.startswith("iclr") & d.judgement.notna()].copy()
        d["forum"] = d.paper_id.astype(str).str.replace("iclr_", "", regex=False)
        con = sqlite3.connect(PDF_DB)
        pdf_ids = {r[0] for r in con.execute(
            "SELECT DISTINCT paper_id FROM pdf_versions WHERE version=0")}
        sample = d[d.forum.isin(pdf_ids)][["paper_id", "forum", "judgement", "year"]]
        sample.to_csv(sp, index=False)
    print(f"[exp] FULL SET n={len(sample)} accept={sample.judgement.mean():.3f}", flush=True)
    ex_cache = Cache(os.path.join(EB, "claims_v2_cache.jsonl"))          # reuse 600 done
    ad_cache = Cache(f"{ROOT}/outputs/checks_v2/cache.jsonl")            # reuse adequacy
    dl_cache = Cache(f"{ROOT}/outputs/checks_v2/delta_cache.jsonl")      # reuse deltas
    stage_papers(sample, ex_cache, ad_cache, dl_cache, args.workers)
    stage_flags(sample, args.workers)
    print("EXPAND_V2_ALL_DONE", flush=True)

if __name__ == "__main__":
    main()
