#!/usr/bin/env python3
"""Embed ALL peer-review forms (OpenAI te3-small, one batched pass — no GLM) and report exact
pair counts per similarity band -> exact GLM-4.7 call estimate for the (B) ambiguous-band path.
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from methods.metric_implementer.experiments.glm_cluster import load_forms, _openai_emb

keys, texts = load_forms("peer-review", n_forms=None)
print(f"peer-review forms: {len(keys)}", flush=True)
E = np.asarray(_openai_emb(texts, "text-embedding-3-small"))
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
n = len(keys)
print(f"all pairs: {n*(n-1)//2:,}", flush=True)
S = E @ E.T
iu = np.triu_indices(n, 1)
sims = S[iu]
print("\n pair counts at/above threshold  -> GLM-4.7 calls @15/batch:")
for thr in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
    c = int((sims >= thr).sum())
    print(f"  sim>={thr:.2f}: {c:>9,} pairs  ({c/15:>7,.0f} calls)")
for lo, hi in [(0.40, 0.55), (0.45, 0.60), (0.40, 0.60)]:
    amb = int(((sims >= lo) & (sims < hi)).sum())
    print(f"  ambiguous [{lo:.2f},{hi:.2f}): {amb:>9,} pairs  ({amb/15:>7,.0f} calls)")
