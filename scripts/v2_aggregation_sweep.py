#!/usr/bin/env python3
"""Aggregation sweep for V2: is min-over-elements the right claim score?

All evidence-side levers are exhausted (judge swap null, 7x data null,
wide-K worse, p±1 windows worse). The remaining knob is the AGGREGATOR:
min-over-elements is brittle (oracle recall 27% — one weak element kills
the claim; also the n_elements artifact at app level). Sweep alternatives
on the SAME pairs using both judges' existing scores:

  per-element score e_i = max over its top-3 paras
  claim score = agg({e_i}) for agg in min / mean / median / softmin(T) /
                noisy-OR / frac(e_i>0.5) / second-min

Uses v2_pair_judgments.jsonl (Qwen) — no GPU needed.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

T = ("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/"
     "truecite_testbed_v1")

el, fell = {}, {}
for line in open(f"{T}/v2_pair_judgments.jsonl"):
    j = json.loads(line)
    if j["disclosed"] is None:
        continue
    sc = (j["confidence"] if j["disclosed"] else 100 - j["confidence"]) / 100
    k = (j["app_id"], j["ifw"], j["claim_num"])
    fell[k] = j["fell_102"]
    el.setdefault(k, {}).setdefault(j["el_idx"], []).append(sc)

claims = []
for k, v in el.items():
    es = np.array([max(x) for x in v.values()])
    claims.append((int(fell[k]), es, k[:2]))
print(f"claims: {len(claims):,}", flush=True)


def softmin(es, t):
    w = np.exp(-es / t)
    return float((es * w).sum() / w.sum())


AGGS = {
    "min (baseline)": lambda es: float(es.min()),
    "second-min": lambda es: float(np.sort(es)[1]) if len(es) > 1
    else float(es.min()),
    "mean": lambda es: float(es.mean()),
    "median": lambda es: float(np.median(es)),
    "softmin T=0.1": lambda es: softmin(es, 0.1),
    "softmin T=0.3": lambda es: softmin(es, 0.3),
    "noisy-OR(1-prod(1-e))": lambda es: float(1 - np.prod(1 - es)),
    "frac(e>0.5)": lambda es: float((es > 0.5).mean()),
    "frac(e>0.5)*mean": lambda es: float((es > 0.5).mean() * es.mean()),
}

for name, fn in AGGS.items():
    ys = np.array([y for y, _, _ in claims])
    ss = np.array([fn(es) for _, es, _ in claims])
    br = {}
    for (y, es, rk), s in zip(claims, ss):
        br.setdefault(rk, []).append((y, s))
    wr = [roc_auc_score([y for y, _ in v], [x for _, x in v])
          for v in br.values() if len({y for y, _ in v}) == 2]
    print(f"{name:>24}: pooled={roc_auc_score(ys, ss):.4f} "
          f"within={np.mean(wr):.4f}", flush=True)
print("AGG-SWEEP-DONE", flush=True)
