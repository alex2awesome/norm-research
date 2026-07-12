#!/usr/bin/env python3
"""Sample rubric-form pairs for the arbiter, BANDED by OpenAI cosine similarity with high-sim
densification (true-same paraphrases are rare in this sparse space, so we oversample the high-sim
region where undermerging hides, while keeping low-sim pairs for the precision side).

Bands: >=0.60 (200), 0.45-0.60 (150), <0.45 (150) = 500 pairs from the 300 GLM-clustered forms.
Writes outputs/analyses/arbiter_pairs.jsonl.
"""
import json, os, sys, random, collections
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "outputs/analyses/glm_cluster_peer_tuned_openai.json"
CANON = "outputs/analyses/canon_all_real_forms.jsonl"
EMB = "outputs/analyses/glm_cluster_cache/emb_peer-review__openai__text-embedding-3-small.npz"
EX = "outputs/analyses/structural_metrics/clusters_peer-review.json"
OUT = "outputs/analyses/arbiter_pairs.jsonl"
BANDS = [(0.60, 1.01, 200), (0.45, 0.60, 150), (0.00, 0.45, 150)]
SEED = 123

d = json.load(open(RUN))
keys = d["keys"]
kset = set(keys)
key2text = {}
for line in open(CANON):
    o = json.loads(line)
    if o.get("task") == "peer-review" and o.get("key") in kset and o.get("canonical"):
        key2text[o["key"]] = o["canonical"]

nz = np.load(EMB, allow_pickle=True)
emb_keys = list(nz["keys"])
E = np.asarray(nz["emb"])
ki = {k: i for i, k in enumerate(emb_keys)}
order = [ki[k] for k in keys]
E = E[order]
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
sims = E @ E.T
ex = json.load(open(EX))
n = len(keys)
rng = random.Random(SEED)
all_pairs = [(i, j, float(sims[i, j])) for i in range(n) for j in range(i + 1, n)]
picked = []
for lo, hi, cnt in BANDS:
    cand = [p for p in all_pairs if lo <= p[2] < hi]
    rng.shuffle(cand)
    picked.extend(cand[:cnt])
    print(f"  band [{lo:.2f},{hi:.2f}): {len(cand)} available, sampled {min(cnt,len(cand))}")
rng.shuffle(picked)

with open(OUT, "w") as f:
    for pid, (i, j, s) in enumerate(picked):
        ka, kb = keys[i], keys[j]
        f.write(json.dumps({"pid": pid, "key_a": ka, "key_b": kb,
                            "text_a": key2text.get(ka, ""), "text_b": key2text.get(kb, ""),
                            "sim": round(s, 3),
                            "same_ex_cluster": ex.get(ka) == ex.get(kb)}) + "\n")
ncross = sum(1 for i, j, _ in picked if ex.get(keys[i]) != ex.get(keys[j]))
print(f"wrote {len(picked)} pairs -> {OUT}  ({ncross} cross-existing-cluster / undermerge-zone)")
