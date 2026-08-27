#!/usr/bin/env python3
"""Expanded-scope claim extraction:
  newsfull  — claims from FULL fetched articles (consolidated v2 shards -> body-extracted).
              Head = headline + first 3 sentences of body; extraction sees the real article.
  peerintro — claims from FULL INTRODUCTION sections (peer_review_pdfs.db, 49,052 papers).
Uses GEPA-optimized prompts (news/peer) via the same auto-patch as run_extract_all.
Run on sk3: python -m methods.claim_verification.run_extract_full --domain newsfull|peerintro"""
import argparse, json, os, sqlite3, sys, time, glob
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from claim_verification.core import Cache, extract_claims, _sentences
from claim_verification.evidence_api import clean_evidence_text

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
OUT_DIR = os.path.join(ROOT, "datasets", "evidence_bases")
FULLTEXT_GLOBS = [os.path.join(ROOT, "datasets/news-homepages/fulltext/fulltext_v2_shard*.jsonl"),
                  os.path.join(ROOT, "datasets/news-homepages/fulltext/fulltext.jsonl")]
BODY = os.path.join(ROOT, "datasets/news-homepages/fulltext/fulltext_body.jsonl")
PDF_DB = os.path.join(ROOT, "datasets/peer-review/peer_review_pdfs.db")
GEPA_RESULTS = os.path.join(ROOT, "outputs", "gepa_extract_results.jsonl")

def best_prompt(domain_key):
    try:
        for ln in open(GEPA_RESULTS):
            r = json.loads(ln)
            if r["domain"] == domain_key and r.get("best_prompt"):
                return r["best_prompt"]
    except Exception:
        pass
    return None

def load_newsfull():
    """Prefer verbatim bodies from reprint_body2; fall back to cleaned raw text."""
    bodies = {}
    if os.path.exists(BODY):
        for ln in open(BODY):
            try:
                r = json.loads(ln)
                if r.get("body"): bodies[r["url"]] = r["body"]
            except Exception: pass
    docs, seen = [], set()
    for g in FULLTEXT_GLOBS:
        for path in glob.glob(g):
            for ln in open(path):
                try: r = json.loads(ln)
                except Exception: continue
                u = r.get("url")
                if not u or u in seen or r.get("route") == "FAIL": continue
                seen.add(u)
                txt = bodies.get(u) or clean_evidence_text(r.get("text", ""))
                if len(txt) > 600:
                    docs.append((u, txt))
    return docs, "news article"

def load_peerintro():
    con = sqlite3.connect(PDF_DB)
    docs = []
    for pid, sections in con.execute("SELECT paper_id, sections FROM pdf_versions WHERE version=0"):
        try: s = json.loads(sections) if sections else {}
        except Exception: continue
        intro = (s.get("abstract", "") or "") + "\n\n" + (s.get("introduction", "") or "")
        if len(intro) > 600: docs.append((f"iclr_{pid}", intro[:12000]))
    return docs, "scientific paper (abstract and introduction)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["newsfull", "peerintro"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max_claims", type=int, default=6)
    args = ap.parse_args()
    gepa_key = {"newsfull": "news", "peerintro": "peer"}[args.domain]
    out_path = os.path.join(OUT_DIR, f"claims_{args.domain}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try: done.add(json.loads(ln)["doc_id"])
            except Exception: pass
    docs, doc_kind = (load_newsfull() if args.domain == "newsfull" else load_peerintro())
    todo = [(i, t) for i, t in docs if i not in done]
    print(f"[{args.domain}] todo={len(todo)} done={len(done)}", flush=True)
    bp = best_prompt(gepa_key)
    if bp is not None:
        import claim_verification.prompts as P
        P.CLAIM_EXTRACT = bp
        print(f"[{args.domain}] using GEPA prompt ({gepa_key})", flush=True)
    cfg = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma",
           "doc_kind": doc_kind, "max_claims": args.max_claims}
    cache = Cache(os.path.join(OUT_DIR, f"claims_cache_{args.domain}.jsonl"))
    lock = Lock(); fout = open(out_path, "a"); n = [0, 0]
    def work(item):
        doc_id, text = item
        try:
            # expanded scope: head = first ~5 sentences (headline+lede for news; abstract lead
            # for papers); the FULL text is what verification pools will use later.
            sents = _sentences(text)
            head = " ".join(sents[:5]) if sents else text[:1200]
            claims = extract_claims(head, cfg, cache)
            row = {"doc_id": doc_id, "n_claims": len(claims), "claims": claims, "ts": int(time.time())}
        except Exception as e:
            row = {"doc_id": doc_id, "err": str(e)[:80], "n_claims": 0, "claims": []}
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n[0] += 1; n[1] += bool(row["claims"])
            if n[0] % 500 == 0:
                print(f"[{args.domain}] {n[0]}/{len(todo)} ok={n[1]}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print(f"[{args.domain}] DONE {n[0]} ok={n[1]}", flush=True)
    print(f"EXTRACT_{args.domain.upper()}_DONE", flush=True)

if __name__ == "__main__":
    main()
