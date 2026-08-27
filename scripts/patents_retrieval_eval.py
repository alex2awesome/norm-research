#!/usr/bin/env python3
"""Retriever retrain support: leakage-safe split + honest pool-based recall eval (v3 vs v7).

The shipped recall number (20.6%) came from the 16.86M-doc global FAISS index; re-encoding that
for a new model is prohibitive. Instead we measure the v3-vs-v7 DELTA on a fixed held-out pool:
  pool  = the unique examiner-cited docs of the held-out test queries (each query's gold is in
          the pool; the other queries' golds are the distractors) -- a closed retrieval task.
  query = the rejected application's anchor claim text.
Same pool + same queries for both models -> the recall delta is the pure effect of retraining on
the corrected (claim-1, not claim-2) pairs. Pool doc text uses the CORRECTED pairs, i.e. the true
cited-doc representation a deployed index would hold.

  split:  python scripts/patents_retrieval_eval.py split
  eval :  CUDA_VISIBLE_DEVICES=N python scripts/patents_retrieval_eval.py eval --model <path> --tag v7
"""
import argparse, gzip, hashlib, json, os, sys
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
PAIRS = f"{PROC}/clean_102_pairs_v2.jsonl.gz"
TRAIN = f"{PROC}/clean_102_pairs_v2_train.jsonl.gz"
TEST = f"{PROC}/clean_102_pairs_v2_test.jsonl.gz"
EVAL_OUT = f"{BASE}/outputs/claimverify_paper/retrieval_eval.json"


def app_bucket(app):
    return int(hashlib.md5(str(app).encode()).hexdigest(), 16) % 10


def cmd_split(_):
    n = ntr = nte = 0
    with gzip.open(PAIRS, "rt") as f, gzip.open(TRAIN, "wt") as ftr, gzip.open(TEST, "wt") as fte:
        for ln in f:
            n += 1
            try:
                r = json.loads(ln)
            except Exception:
                continue
            # bucket 0 -> test (~10%), rest -> train; split by APP so no app leaks across
            if app_bucket(r.get("rejected_app_id", "")) == 0:
                fte.write(ln); nte += 1
            else:
                ftr.write(ln); ntr += 1
    print(f"[split] {n} pairs -> train {ntr} / test {nte}  ({TRAIN}, {TEST})", flush=True)


def load_test_pool(cap_queries):
    """Return (queries, pool_ids, pool_texts). One query per (app, gold_doc); pool = unique golds."""
    seen_q, queries = set(), []
    pool_text = {}
    with gzip.open(TEST, "rt") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            gold = r["positive_pgpub_id"]
            pool_text.setdefault(gold, r["positive_text"][:2000])
            qk = (r["rejected_app_id"], gold)
            if qk in seen_q:
                continue
            seen_q.add(qk)
            if r["anchor_text"].strip():
                queries.append({"anchor": r["anchor_text"][:2000], "gold": gold})
    # stable subsample of queries (never seeded-shuffle a growing list)
    queries.sort(key=lambda q: hashlib.md5((q["anchor"][:80] + q["gold"]).encode()).hexdigest())
    if cap_queries:
        queries = queries[:cap_queries]
    pool_ids = list(pool_text)
    return queries, pool_ids, [pool_text[i] for i in pool_ids]


def cmd_eval(a):
    from sentence_transformers import SentenceTransformer
    queries, pool_ids, pool_texts = load_test_pool(a.n_queries)
    gold_idx = {pid: i for i, pid in enumerate(pool_ids)}
    print(f"[eval:{a.tag}] {len(queries)} queries over pool of {len(pool_ids)} docs", flush=True)

    model = SentenceTransformer(a.model)
    model.max_seq_length = a.seq_len
    import torch
    if torch.cuda.is_available():
        model = model.to("cuda")
    P = model.encode(pool_texts, batch_size=a.batch, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    Q = model.encode([q["anchor"] for q in queries], batch_size=a.batch, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    sims = Q @ P.T  # cosine (normalized)
    ranks = []
    for i, q in enumerate(queries):
        gi = gold_idx[q["gold"]]
        # rank of gold = # docs strictly more similar than gold, +1
        ranks.append(int((sims[i] > sims[i, gi]).sum()) + 1)
    ranks = np.array(ranks)
    res = {"tag": a.tag, "model": a.model, "n_queries": len(queries), "pool": len(pool_ids),
           "R@1": float((ranks <= 1).mean()), "R@10": float((ranks <= 10).mean()),
           "R@50": float((ranks <= 50).mean()), "R@100": float((ranks <= 100).mean()),
           "MRR": float((1.0 / ranks).mean()), "median_rank": int(np.median(ranks))}
    print(f"[eval:{a.tag}] R@1={res['R@1']:.3f} R@10={res['R@10']:.3f} R@50={res['R@50']:.3f} "
          f"R@100={res['R@100']:.3f} MRR={res['MRR']:.4f} medrank={res['median_rank']}", flush=True)
    os.makedirs(os.path.dirname(EVAL_OUT), exist_ok=True)
    allres = {}
    if os.path.exists(EVAL_OUT):
        try:
            allres = json.load(open(EVAL_OUT))
        except Exception:
            allres = {}
    allres[a.tag] = res
    json.dump(allres, open(EVAL_OUT, "w"), indent=1)
    print(f"EVAL_{a.tag}_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--tag", required=True)
    e.add_argument("--n-queries", type=int, default=5000, dest="n_queries")
    e.add_argument("--batch", type=int, default=128)
    e.add_argument("--seq-len", type=int, default=256, dest="seq_len",
                   help="encode max_seq_length; v3 trained@256, v7 trained@512 -> compare at both")
    a = ap.parse_args()
    {"split": cmd_split, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
