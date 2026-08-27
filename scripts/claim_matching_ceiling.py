#!/usr/bin/env python3
"""Dense/frontier UPPER BOUND for claim-matching, on the same gold-vs-filler probe as the metric run.

Answers "what's the ceiling?" so T (tacit) = ceiling - A(articulated .573). Within-claim accuracy =
does the model score the examiner's gold span above the filler, per claim (the honest matching metric).

Arms:
  bge-m3-base        : untrained BAAI/bge-m3 cosine (element vs span) — LEAK-FREE unsupervised dense.
  v6a                : our production trained bi-encoder (bge-m3-anticipation-v6a) cosine — trained
                       CLAIM-MATCHING retriever ceiling (CAVEAT: trained on §102 disclosure pairs from
                       the same corpus -> possible train/test overlap -> optimistic).
  reranker-v3        : our trained cross-encoder (bge-reranker-anticipation-v3) — strongest trained
                       matching model (same overlap caveat).
Runs CPU by default (GPUs contended); ~1600 pairs. Same probe (first 800 claims by hash) as scoring.
  python scripts/claim_matching_ceiling.py
"""
import json, hashlib, collections
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = "/lfs/skampere3/0/alexspan/norm-research"
# v2 = multiple-gold fix. Published ceiling numbers (ceiling.json) were computed on v1;
# only 59/800 probe negatives changed, so v1 arms are comparable to within ~1 point.
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed_v2.jsonl"
MODELS = {"bge-m3-base": "BAAI/bge-m3",
          "v6a": f"{BASE}/models/bge-m3-anticipation-v6a"}
CROSS = {"reranker-v3": f"{BASE}/models/bge-reranker-anticipation-v3"}


def probe(n_claims=800):
    byu = collections.defaultdict(list)
    for ln in open(TESTBED):
        r = json.loads(ln); byu[r["uid"]].append(r)
    uids = [u for u, v in byu.items() if len(v) == 2 and {x["y"] for x in v} == {0, 1}]
    uids.sort(key=lambda u: hashlib.md5(f"probe::{u}".encode()).hexdigest())
    return [p for u in uids[:n_claims] for p in byu[u]]


def within_acc(pairs, score):
    byu = collections.defaultdict(dict)
    for p, s in zip(pairs, score):
        byu[p["uid"]][p["y"]] = s
    acc = n = 0
    for d in byu.values():
        if 1 in d and 0 in d:
            n += 1; acc += 1.0 if d[1] > d[0] else 0.5 if d[1] == d[0] else 0.0
    return acc / max(1, n), n


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 8e9 else "cpu"
    pairs = probe()
    y = np.array([p["y"] for p in pairs])
    els = [p["element"][:512] for p in pairs]
    sps = [p["span"][:512] for p in pairs]
    print(f"[ceiling] {len(pairs)} pairs ({len(pairs)//2} claims), device={dev}", flush=True)
    print(f"  (reference: lexical V=.516, articulated bank A=.573 within-claim)\n", flush=True)
    res = {}

    from sentence_transformers import SentenceTransformer, CrossEncoder
    for name, path in MODELS.items():
        try:
            m = SentenceTransformer(path, device=dev); m.max_seq_length = 512
            E = m.encode(els, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)
            S = m.encode(sps, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)
            cos = (E * S).sum(1)
            wc, n = within_acc(pairs, cos)
            auc = roc_auc_score(y, cos)
            res[name] = {"within": wc, "auc": auc}
            print(f"  {name:14s} within-claim={wc:.3f}  pooledAUC={auc:.3f}", flush=True)
            del m
        except Exception as e:
            print(f"  {name:14s} FAILED: {e}", flush=True)

    for name, path in CROSS.items():
        try:
            ce = CrossEncoder(path, device=dev, max_length=512)
            sc = ce.predict(list(zip(els, sps)), batch_size=32, show_progress_bar=False)
            wc, n = within_acc(pairs, sc)
            auc = roc_auc_score(y, sc)
            res[name] = {"within": wc, "auc": auc}
            print(f"  {name:14s} within-claim={wc:.3f}  pooledAUC={auc:.3f}  (trained cross-encoder)",
                  flush=True)
        except Exception as e:
            print(f"  {name:14s} FAILED: {e}", flush=True)

    json.dump(res, open(f"{BASE}/outputs/claim_matching/ceiling.json", "w"), indent=1)
    print("\n[read] bge-m3-base = leak-free dense ceiling; v6a/reranker = trained (possible overlap, "
          "optimistic). T_tacit ~ ceiling - .573.", flush=True)
    print("CEILING_DONE", flush=True)


if __name__ == "__main__":
    main()
