#!/usr/bin/env python3
"""Scale claim extraction (best GEPA prompt per domain, fallback seed) across all corpora on the
Gemma server. Streams results to datasets/evidence_bases/claims_{domain}.jsonl (resumable).
Domains + doc sets:
  pr    press_release_deconfounded.parquet          (72,315 full bodies)
  news  news_articles.csv rows with text>=800       (cap via --cap, default 60k, stratified head)
  peer  peer_review_modeling_dataset.csv.gz         (ICLR-primary texts)
Run on sk3: python -m methods.claim_verification.run_extract_all --domain pr [--cap N]"""
import argparse, csv, json, os, sys, time
sys.path.insert(0, "methods")
csv.field_size_limit(sys.maxsize)
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import pandas as pd
from claim_verification.core import Cache, split_head_body, extract_claims

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
OUT_DIR = os.path.join(ROOT, "datasets", "evidence_bases")
os.makedirs(OUT_DIR, exist_ok=True)
GEPA_RESULTS = os.path.join(ROOT, "outputs", "gepa_extract_results.jsonl")

def best_prompt(domain):
    try:
        for ln in open(GEPA_RESULTS):
            r = json.loads(ln)
            if r["domain"] == domain and r.get("best_prompt"):
                return r["best_prompt"]
    except Exception:
        pass
    return None  # core.extract_claims uses the seed CLAIM_EXTRACT

def load_docs(domain, cap):
    if domain == "pr":
        d = pd.read_parquet(f"{ROOT}/datasets/press-releases/press_release_deconfounded.parquet")
        return [(str(r.id), str(r.text)) for r in d.itertuples() if len(str(r.text)) > 400], "press release"
    if domain == "peer":
        d = pd.read_csv(f"{ROOT}/datasets/peer-review/peer_review_modeling_dataset.csv.gz", compression="gzip")
        return [(str(r.paper_id), str(r.text)) for r in d.itertuples() if len(str(r.text)) > 400], \
               "scientific paper (abstract and introduction)"
    if domain == "news":
        out = []
        for r in csv.DictReader(open(f"{ROOT}/datasets/press-releases/news_articles.csv")):
            t = (r.get("news_article_text") or "").strip()
            if len(t) >= 800:
                out.append((str(r["news_article_id"]), t))
                if cap and len(out) >= cap: break
        return out, "news article"
    raise ValueError(domain)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["pr", "news", "peer"])
    ap.add_argument("--cap", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    out_path = os.path.join(OUT_DIR, f"claims_{args.domain}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try: done.add(json.loads(ln)["doc_id"])
            except Exception: pass
    docs, doc_kind = load_docs(args.domain, args.cap)
    todo = [(i, t) for i, t in docs if i not in done]
    print(f"[extract-{args.domain}] todo={len(todo)} done={len(done)} kind={doc_kind}", flush=True)
    bp = best_prompt(args.domain)
    cfg = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma",
           "doc_kind": doc_kind, "max_claims": 5}
    if bp is not None:
        import claim_verification.core as core, claim_verification.prompts as P
        P.CLAIM_EXTRACT = bp  # patch in the GEPA winner
        print(f"[extract-{args.domain}] using GEPA-optimized prompt", flush=True)
    cache = Cache(os.path.join(OUT_DIR, f"claims_cache_{args.domain}.jsonl"))
    lock = Lock(); fout = open(out_path, "a"); n = [0, 0]
    def work(item):
        doc_id, text = item
        try:
            head, _ = split_head_body(text)
            claims = extract_claims(head, cfg, cache)
            row = {"doc_id": doc_id, "n_claims": len(claims), "claims": claims,
                   "ts": int(time.time())}
        except Exception as e:
            row = {"doc_id": doc_id, "err": str(e)[:80], "n_claims": 0, "claims": []}
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n[0] += 1; n[1] += bool(row["claims"])
            if n[0] % 500 == 0:
                print(f"[extract-{args.domain}] {n[0]}/{len(todo)} ok={n[1]}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print(f"[extract-{args.domain}] DONE {n[0]} ok={n[1]}", flush=True)
    print(f"EXTRACT_{args.domain.upper()}_DONE", flush=True)

if __name__ == "__main__":
    main()
