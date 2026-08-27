"""Measure how badly the pipeline under-merges, using leaf-NAME embeddings.

The pipeline's own clustering ran on a LoRA-BGE space + HDBSCAN(eps=0) + an LLM
dedup pass with limited-recall candidate generation. The earlier nearest-
neighbour audit embedded the LLM-elaborated medoid DESCRIPTIONS, which diverge
between same-concept clusters and hid the problem.

Here we re-cluster the short, verbatim leaf NAMES (re-embedded with bge-large on
sk3) and ask: how many "concept families" did the pipeline split across
multiple clusters?

For a family found by leaf-name agglomerative clustering, if its leaves come
from >=2 distinct pipeline cluster_ids, the pipeline under-merged them.
  excess clusters in a family = (#distinct pipeline cluster_ids) - 1
  total excess / pipeline cluster count = the over-fragmentation rate.

Threshold tau IS the "definition of similarity" knob -- we sweep it; the chosen
operating point must be manually validated by reading sampled families.

Usage:
  python scripts/leaf_name_clusters.py --sweep
  python scripts/leaf_name_clusters.py --task creative-writing --tau 0.85 --show 30
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
EMB = ROOT / "notebooks" / "_explore_cache" / "bge"
LEAF_JSONL = OUT / "_sk3_leaf_input.jsonl"

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]


def load_task(task):
    rows = []
    for line in LEAF_JSONL.open():
        r = json.loads(line)
        if r["task"] == task:
            rows.append(r)
    rows.sort(key=lambda r: r["idx"])
    emb = np.load(EMB / f"emb_bge_leafname_{task}.npy").astype(np.float32)
    assert len(emb) == len(rows), f"{task}: {len(emb)} emb vs {len(rows)} rows"
    return rows, emb


def cluster(emb, tau, linkage="average"):
    """Agglomerative clustering at cosine distance threshold 1-tau."""
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage=linkage,
        distance_threshold=1.0 - tau)
    return model.fit_predict(emb)


def family_stats(rows, labels):
    """Per family: leaves, distinct pipeline cluster_ids -> excess merges."""
    fam = defaultdict(list)
    for r, lab in zip(rows, labels):
        fam[lab].append(r)
    n_pipe_clusters = len({r["cluster_id"] for r in rows})
    excess = 0
    leaves_in_undermerge = 0
    multi_fams = 0
    for lab, members in fam.items():
        cids = {m["cluster_id"] for m in members}
        if len(cids) >= 2:
            excess += len(cids) - 1
            leaves_in_undermerge += len(members)
            multi_fams += 1
    return {
        "n_leaves": len(rows),
        "n_pipe_clusters": n_pipe_clusters,
        "n_families": len(fam),
        "undermerge_families": multi_fams,
        "excess_clusters": excess,
        "leaves_in_undermerge": leaves_in_undermerge,
        "overfrag_pct": excess / max(1, n_pipe_clusters) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--task", default=None)
    ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        print("Over-fragmentation: % of pipeline clusters that are redundant")
        print("(excess = pipeline clusters beyond 1 per leaf-name family)\n")
        hdr = f"{'task':<24}" + "".join(f"{f'tau={t}':>12}" for t in
                                        (0.78, 0.83, 0.88, 0.93))
        print(hdr)
        for task in TASKS:
            rows, emb = load_task(task)
            cells = []
            for tau in (0.78, 0.83, 0.88, 0.93):
                s = family_stats(rows, cluster(emb, tau))
                cells.append(f"{s['overfrag_pct']:>10.0f}%")
            print(f"{task:<24}" + "".join(f"{c:>12}" for c in cells))
        return

    task = args.task or "creative-writing"
    rows, emb = load_task(task)
    labels = cluster(emb, args.tau)
    s = family_stats(rows, labels)
    print(f"== {task}  tau={args.tau} ==")
    for k, v in s.items():
        print(f"  {k:<22} {v:.0f}" if isinstance(v, float) else f"  {k:<22} {v}")

    if args.show:
        fam = defaultdict(list)
        for r, lab in zip(rows, labels):
            fam[lab].append(r)
        # families that span the most pipeline clusters
        ranked = sorted(fam.values(),
                        key=lambda m: -len({x["cluster_id"] for x in m}))
        print(f"\n-- top {args.show} under-merged families "
              f"(leaf-name family -> N pipeline clusters) --")
        for members in ranked[:args.show]:
            cids = {m["cluster_id"] for m in members}
            if len(cids) < 2:
                continue
            names = Counter(m["name"] for m in members)
            print(f"\n  FAMILY: {len(members)} leaves across {len(cids)} "
                  f"pipeline clusters")
            for nm, c in names.most_common(10):
                print(f"     [{c}x] {nm[:90]}")


if __name__ == "__main__":
    main()
