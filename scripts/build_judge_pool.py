"""Build a stratified pool of rubric PAIRS for the LLM same/different judge.

For each (bucket, task) we sample pairs across cosine-similarity bands (dense
near the 0.85-0.97 decision zone, plus easy negatives). High-cosine pairs come
from each rubric's nearest neighbours; mid/low from random pairs. The pool is
judged once and reused at every tau in the FP/FN sweep.

Input:  canon_all_real_forms.jsonl + nemotron embeddings emb_nemo_<bucket>_<task>.npy
Output: judge_pool.jsonl
        {bucket, task, idx_a, idx_b, key_a, key_b, canonical_a, canonical_b, cos}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
EMB = ROOT / "notebooks" / "_explore_cache" / "bge"
FORMS = OUT / "canon_all_real_forms.jsonl"

BANDS = [(0.97, 1.001), (0.94, 0.97), (0.91, 0.94), (0.88, 0.91),
         (0.85, 0.88), (0.78, 0.85), (0.55, 0.78)]
PER_BAND_PER_BUCKET = 230   # target pairs per band per bucket (pooled over tasks)
RNG = np.random.default_rng(13)


def load_forms():
    by_bt = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        by_bt[(r["bucket"], r["task"])].append(r)
    for v in by_bt.values():
        v.sort(key=lambda r: r["idx"])
    return by_bt


def task_pairs(emb):
    """Candidate (i,j,cos): each point's top-4 NN + a block of random pairs."""
    n = len(emb)
    if n < 3:
        return []
    out = {}
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1.0)
    k = min(4, n - 1)
    nn = np.argpartition(-sims, k, axis=1)[:, :k]
    for i in range(n):
        for j in nn[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a != b:
                out[(a, b)] = float(emb[a] @ emb[b])
    n_rand = min(6 * n, 40000)
    ri = RNG.integers(0, n, size=n_rand)
    rj = RNG.integers(0, n, size=n_rand)
    for a, b in zip(ri, rj):
        if a == b:
            continue
        a, b = (int(a), int(b)) if a < b else (int(b), int(a))
        out[(a, b)] = float(emb[a] @ emb[b])
    return [(a, b, c) for (a, b), c in out.items()]


def main():
    by_bt = load_forms()
    # bucket -> band-index -> list of (task, a, b, cos)
    pool = defaultdict(lambda: defaultdict(list))
    for (bucket, task), rows in by_bt.items():
        p = EMB / f"emb_bge_{bucket}_{task}.npy"
        if not p.exists():
            continue
        emb = np.load(p).astype(np.float64)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        for a, b, c in task_pairs(emb):
            for bi, (lo, hi) in enumerate(BANDS):
                if lo <= c < hi:
                    pool[bucket][bi].append((task, a, b, c))
                    break

    picked = []
    for bucket in sorted(pool):
        for bi, (lo, hi) in enumerate(BANDS):
            cand = pool[bucket][bi]
            if not cand:
                continue
            take = min(PER_BAND_PER_BUCKET, len(cand))
            idx = RNG.choice(len(cand), size=take, replace=False)
            for k in idx:
                task, a, b, c = cand[k]
                rows = by_bt[(bucket, task)]
                picked.append({
                    "bucket": bucket, "task": task, "idx_a": a, "idx_b": b,
                    "key_a": rows[a]["key"], "key_b": rows[b]["key"],
                    "canonical_a": rows[a]["canonical"],
                    "canonical_b": rows[b]["canonical"], "cos": round(c, 4)})

    RNG.shuffle(picked)
    with (OUT / "judge_pool.jsonl").open("w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    by_b = defaultdict(int)
    for r in picked:
        by_b[r["bucket"]] += 1
    print(f"wrote {len(picked)} judge pairs  ({dict(by_b)}) -> {OUT/'judge_pool.jsonl'}")


if __name__ == "__main__":
    main()
