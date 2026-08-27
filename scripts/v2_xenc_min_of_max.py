#!/usr/bin/env python3
"""V2 min_of_max with element-para-xenc-v1 as the judge (Qwen-122B swap test).

Scores the SAME v6a-retrieved (element, paragraph) pairs the Qwen judge saw
(v2_pairs.jsonl, top-3 paras/element) with the distilled cross-encoder
(test AUC 0.844 vs judge), then recomputes the min_of_max accept/reject AUC
with the exact recipe from v61_post_chain.sh.

Reference numbers (Qwen-122B judge): pooled=0.5743 within=0.5747 (n=3273).
If xenc ~matches, the 122B judge is replaceable by a 280M reranker and
xenc-gated re-retrieval over full specs becomes cheap to test.
"""
import json
import os

import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.metrics import roc_auc_score

T = os.path.expanduser(
    "~/norm-research/datasets/patents/processed/truecite_testbed_v1")
XENC = os.path.expanduser("~/norm-research/models/element-para-xenc-v1")
MAX_TEXT = 1500

rows = [json.loads(line) for line in open(f"{T}/v2_pairs.jsonl")]
print(f"element rows: {len(rows):,}", flush=True)

flat, idx = [], []  # idx[i] = (row_i, p_idx)
for ri, r in enumerate(rows):
    for pi, (_, _, text) in enumerate(r["paras"]):
        flat.append([r["element"][:MAX_TEXT], text[:MAX_TEXT]])
        idx.append((ri, pi))
print(f"pairs to score: {len(flat):,}", flush=True)

xenc = CrossEncoder(XENC, max_length=512, device="cuda")
scores = xenc.predict(flat, batch_size=512, show_progress_bar=True)
print("scoring done", flush=True)

# min_of_max, exact v61_post_chain.sh recipe (scores on 0-1 instead of 0-100)
el, fell = {}, {}
for (ri, pi), sc in zip(idx, scores):
    r = rows[ri]
    k = (r["app_id"], r["ifw"], r["claim_num"])
    fell[k] = r["fell_102"]
    el.setdefault(k, {}).setdefault(r["el_idx"], []).append(float(sc))
ys, ss, br = [], [], {}
for k, v in el.items():
    s = float(np.min([max(x) for x in v.values()]))
    ys.append(int(fell[k]))
    ss.append(s)
    br.setdefault(k[:2], []).append((int(fell[k]), s))
ys, ss = np.array(ys), np.array(ss)
wr = [roc_auc_score([y for y, _ in v], [x for _, x in v])
      for v in br.values() if len({y for y, _ in v}) == 2]
print(f"v6a+xenc-v1: min_of_max pooled={roc_auc_score(ys, ss):.4f} "
      f"within={np.mean(wr):.4f} (n={len(ys)})", flush=True)

# agreement with the Qwen judge on these very pairs (sanity, in-distribution)
judge = {}
for line in open(f"{T}/v2_pair_judgments.jsonl"):
    j = json.loads(line)
    if j["disclosed"] is None:
        continue
    judge[(j["app_id"], j["ifw"], j["claim_num"], j["el_idx"],
           j["p_idx"])] = int(j["disclosed"])
y_j, s_x = [], []
for (ri, pi), sc in zip(idx, scores):
    r = rows[ri]
    k = (r["app_id"], r["ifw"], r["claim_num"], r["el_idx"], pi)
    if k in judge:
        y_j.append(judge[k])
        s_x.append(float(sc))
print(f"pairwise vs judge: n={len(y_j):,} AUC={roc_auc_score(y_j, s_x):.4f} "
      f"agree@0.5={np.mean((np.array(s_x) > .5) == np.array(y_j)):.4f}",
      flush=True)
print("XENC-V2-SCORE-DONE", flush=True)
