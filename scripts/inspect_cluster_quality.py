"""Clustering-quality check for one task: are the singleton clusters actually
distinct concepts, or did the dedup under-merge them?

Method (clustering-free, uses only the medoid embeddings):
  - enumerate every general-bucket cluster, in the SAME order as
    embedding_diversity.py:load_cluster_texts (so the cached .npy aligns)
  - for each cluster, find its nearest neighbor by cosine similarity
  - report the nearest-neighbor-similarity distribution for SINGLETON clusters
  - print the highest-similarity singleton pairs for manual eyeballing
  - also dump exact-medoid-name collision groups

Usage: python scripts/inspect_cluster_quality.py creative-writing
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
HIER = ROOT / "outputs" / "hierarchy"
EMB_CACHE = ROOT / "notebooks" / "_explore_cache"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (s or "").lower())).strip()


def load_clusters(task: str) -> list[dict]:
    """Same enumeration order as embedding_diversity.load_cluster_texts."""
    d = json.loads((HIER / f"{task}_general_r1_refined.json").read_text())
    out = []
    for par in d.get("parented_trees", []):
        for ch in par.get("children", []):
            out.append({
                "name": ch.get("medoid_name", "") or "",
                "desc": ch.get("medoid_description", "") or "",
                "size": len(ch.get("rubrics", [])),
            })
    return out


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "creative-writing"
    clusters = load_clusters(task)
    emb = np.load(EMB_CACHE / f"emb_rubric_cluster_{task}.npy").astype(np.float32)
    assert len(clusters) == len(emb), f"misalign: {len(clusters)} clusters vs {len(emb)} embeddings"
    n = len(clusters)
    singleton = np.array([c["size"] == 1 for c in clusters])
    print(f"task={task}: {n} clusters, {singleton.sum()} singletons ({singleton.mean()*100:.0f}%)")

    # nearest-neighbor cosine sim (embeddings already unit-normalized)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    nn_idx = sim.argmax(axis=1)
    nn_sim = sim.max(axis=1)

    # NN-sim distribution for singletons
    s_nn = nn_sim[singleton]
    print(f"\nSINGLETON nearest-neighbor cosine sim:")
    for thr in [0.95, 0.90, 0.85, 0.80, 0.75]:
        frac = (s_nn >= thr).mean()
        print(f"  NN sim >= {thr}: {(s_nn>=thr).sum():>5} ({frac*100:.0f}% of singletons)")
    print(f"  median NN sim: {np.median(s_nn):.3f}")

    # highest-sim singleton-singleton pairs for eyeballing
    print(f"\n=== TOP 25 highest-similarity SINGLETON pairs (eyeball: same idea?) ===")
    pairs = []
    seen = set()
    order = np.argsort(-nn_sim)
    for i in order:
        if not singleton[i]:
            continue
        j = nn_idx[i]
        if not singleton[j]:
            continue
        key = tuple(sorted((int(i), int(j))))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((nn_sim[i], int(i), int(j)))
        if len(pairs) >= 25:
            break
    for s, i, j in pairs:
        ci, cj = clusters[i], clusters[j]
        print(f"\n  cos={s:.3f}")
        print(f"    A: {ci['name'][:70]}")
        print(f"       {ci['desc'][:140]}")
        print(f"    B: {cj['name'][:70]}")
        print(f"       {cj['desc'][:140]}")

    # exact-name collision groups
    by_name = defaultdict(list)
    for idx, c in enumerate(clusters):
        by_name[norm(c["name"])].append(idx)
    groups = sorted([(k, v) for k, v in by_name.items() if k and len(v) > 1],
                    key=lambda kv: -len(kv[1]))
    print(f"\n=== EXACT medoid-name collision groups (top 6) ===")
    for name, idxs in groups[:6]:
        print(f"\n  '{name}'  x{len(idxs)} separate clusters:")
        for idx in idxs:
            print(f"    [sz {clusters[idx]['size']}] {clusters[idx]['desc'][:130]}")


if __name__ == "__main__":
    main()
