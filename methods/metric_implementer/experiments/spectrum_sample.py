#!/usr/bin/env python3
"""Dense-across-spectrum pair sample for band calibration: ~PER pairs per 0.05 sim bin from ALL
peer-review forms (so the high-sim transition is well-sampled, unlike the 500-pair set). Embeddings
cached to a dedicated dir (GLMCLUSTER_EMB_CACHE) so the 300-form caches are untouched.
Writes outputs/analyses/spectrum_pairs.jsonl.
"""
import os, sys, json, random, numpy as np
os.environ.setdefault("GLMCLUSTER_EMB_CACHE", "outputs/analyses/emb_all_cache")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from methods.metric_implementer.experiments.glm_cluster import load_forms, get_embeddings

OUT = "outputs/analyses/spectrum_pairs.jsonl"
BINW = 0.05
PER = 150

keys, texts = load_forms("peer-review", n_forms=None)
key2text = dict(zip(keys, texts))
E = np.asarray(get_embeddings(texts, keys, "peer-review", "openai", "text-embedding-3-small"), dtype=np.float32)
E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
n = len(keys)
S = E @ E.T
iu = np.triu_indices(n, 1)
sims = S[iu].astype(np.float32)
ka = np.array(keys)[iu[0]]
kb = np.array(keys)[iu[1]]
binidx = np.clip((sims / BINW).astype(int), 0, 19)
rng = random.Random(7)
picked = []
print(f"{n} forms, {len(sims):,} total pairs; sampling {PER}/bin:", flush=True)
for b in range(20):
    idxs = np.where(binidx == b)[0]
    if len(idxs) == 0:
        print(f"  [{b*BINW:.2f},{(b+1)*BINW:.2f}): 0", flush=True)
        continue
    sel = sorted(rng.sample(range(len(idxs)), min(PER, len(idxs))))
    picked.extend(int(x) for x in idxs[sel])
    print(f"  [{b*BINW:.2f},{(b+1)*BINW:.2f}): {len(idxs):>9,} avail -> {len(sel)} sampled", flush=True)

with open(OUT, "w") as f:
    for pid, s in enumerate(picked):
        a, b = ka[s], kb[s]
        f.write(json.dumps({"pid": pid, "key_a": a, "key_b": b,
                            "text_a": key2text.get(a, ""), "text_b": key2text.get(b, ""),
                            "sim": round(float(sims[s]), 3)}) + "\n")
print(f"wrote {len(picked)} pairs -> {OUT}", flush=True)
