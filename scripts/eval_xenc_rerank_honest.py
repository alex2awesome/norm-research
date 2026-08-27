#!/usr/bin/env python3
"""Does xenc-v1 rerank beat the bi-encoder plateau? Honest within-doc eval.

Same clean protocol as eval_v6_within_doc_honest.py (round-2-only pairs,
postdate v6a's training; [[feedback-stable-hash-splits]]): v6a ranks the
gold doc's paragraphs for each claim element, then element-para-xenc-v1
(distilled from 71,685 Qwen-122B judge labels, test AUC 0.844) reranks.

Variants reported on the round-2-only subset (v6a honest MRR baseline 0.207):
  - v6a alone
  - xenc rerank of v6a top-10 / top-20
  - xenc full ranking (all paragraphs, no v6a gate)

Run on sk3, stacked on GPU 6 (bge-m3 encode + bge-reranker-base predict).
"""
import ast
import glob
import gzip
import json
import os

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE = os.path.expanduser("~/norm-research/datasets/patents/processed")
LEGACY_EXTR = f"{BASE}/oa_102_extractions.jsonl.gz"
TEST = f"{BASE}/training_pairs_v6_test.jsonl.gz"
SPECS = f"{BASE}/paragraph_keyed_specs.jsonl.gz"
V6A = os.path.expanduser("~/norm-research/models/bge-m3-anticipation-v6a")
XENC = os.path.expanduser("~/norm-research/models/element-para-xenc-v1")
TEXT_CAP = 1500

# ---- round-1 keys from the legacy extraction file ----
legacy_keys = set()
with gzip.open(LEGACY_EXTR, "rt") as f:
    for line in f:
        try:
            r = json.loads(line)
            ex = r.get("extraction")
            if isinstance(ex, str):
                ex = ast.literal_eval(ex)
            for e in ex or []:
                legacy_keys.add((str(r.get("app_id")),
                                 str(e.get("target_claim")),
                                 str(e.get("prior_art_pgpub_id")),
                                 (e.get("claim_element") or "")[:200]))
        except Exception:
            continue
print(f"legacy round-1 keys: {len(legacy_keys):,}", flush=True)

# ---- test split, paragraph-gold pairs, round-2-only ----
pairs = []
with gzip.open(TEST, "rt") as f:
    for line in f:
        r = json.loads(line)
        if r.get("positive_kind") != "paragraph":
            continue
        k = (str(r.get("anchor_app_id")), str(r.get("anchor_target_claim")),
             str(r.get("positive_pgpub_id")),
             (r.get("anchor_text") or "")[:200])
        if k not in legacy_keys:
            pairs.append(r)
print(f"round-2-only paragraph-gold pairs: {len(pairs):,}", flush=True)

# ---- specs for the docs we need ----
need = {str(p["positive_pgpub_id"]) for p in pairs}
specs = {}


def read_specs(fh):
    for line in fh:
        d = json.loads(line)
        if not d.get("_error") and d.get("paragraphs") \
                and str(d["pgpub_id"]) in need:
            specs[str(d["pgpub_id"])] = d["paragraphs"]


with gzip.open(SPECS, "rt") as f:
    read_specs(f)
for path in glob.glob(f"{BASE}/paragraph_keyed_specs_v2/*.jsonl"):
    with open(path) as f:
        read_specs(f)
print(f"specs loaded: {len(specs):,}/{len(need):,}", flush=True)

queries = [(p["anchor_text"][:TEXT_CAP], p["positive_key"],
            str(p["positive_pgpub_id"]))
           for p in pairs if str(p["positive_pgpub_id"]) in specs
           and p["positive_key"] in specs[str(p["positive_pgpub_id"])]
           and len(specs[str(p["positive_pgpub_id"])]) >= 5]
print(f"eval queries: {len(queries):,}", flush=True)

# ---- stage 1: v6a within-doc ranking ----
benc = SentenceTransformer(V6A, device="cuda")
Q = benc.encode([q[0] for q in queries], batch_size=256,
                convert_to_tensor=True, show_progress_bar=False,
                normalize_embeddings=True)
v6a_order = []  # per query: list of para keys, v6a-descending
for qi, (_, gold_key, doc) in enumerate(queries):
    keys = list(specs[doc].keys())
    txts = [specs[doc][k][:TEXT_CAP] for k in keys]
    P = benc.encode(txts, batch_size=256, convert_to_tensor=True,
                    show_progress_bar=False, normalize_embeddings=True)
    sims = (Q[qi:qi + 1] @ P.T).squeeze(0)
    order = torch.argsort(sims, descending=True).tolist()
    v6a_order.append([keys[i] for i in order])
del benc, Q
torch.cuda.empty_cache()
print("v6a ranking done", flush=True)

# ---- stage 2: xenc scores, one batched predict per query's candidate set ----
xenc = CrossEncoder(XENC, max_length=512, device="cuda")
xenc_scores = []  # per query: {para_key: score} over ALL paras
for qi, (qtext, gold_key, doc) in enumerate(queries):
    keys = list(specs[doc].keys())
    batch = [[qtext, specs[doc][k][:TEXT_CAP]] for k in keys]
    s = xenc.predict(batch, batch_size=256, show_progress_bar=False)
    xenc_scores.append(dict(zip(keys, [float(x) for x in s])))
    if (qi + 1) % 200 == 0:
        print(f"  xenc {qi + 1}/{len(queries)}", flush=True)
print("xenc scoring done", flush=True)


def report(name, ranked_keys_fn):
    rs = []
    for qi, (_, gold_key, doc) in enumerate(queries):
        ranked = ranked_keys_fn(qi)
        rank = next((j + 1 for j, k in enumerate(ranked) if k == gold_key),
                    None)
        if rank:
            rs.append(rank)
    rs = np.array(rs)
    print(f"{name:>28}: n={len(rs):,}  MRR={np.mean(1 / rs):.4f}  "
          f"top1={np.mean(rs <= 1):.3f}  top3={np.mean(rs <= 3):.3f}  "
          f"top10={np.mean(rs <= 10):.3f}", flush=True)


def rerank_topk(qi, k):
    head = v6a_order[qi][:k]
    tail = v6a_order[qi][k:]
    head = sorted(head, key=lambda p: -xenc_scores[qi][p])
    return head + tail


report("v6a alone (honest)", lambda qi: v6a_order[qi])
report("xenc rerank v6a top-10", lambda qi: rerank_topk(qi, 10))
report("xenc rerank v6a top-20", lambda qi: rerank_topk(qi, 20))
report("xenc full ranking", lambda qi: sorted(
    xenc_scores[qi], key=lambda p: -xenc_scores[qi][p]))
print("XENC-RERANK-EVAL-DONE", flush=True)
