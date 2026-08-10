"""Re-cluster the canonical leaf forms at several agglomeration thresholds.

Complete-linkage agglomerative clustering (cosine) on bge embeddings of the
canonicalized leaf forms, per task. Reports how much the corpus compresses at
each tau, and prints the merged families at a strict tau for manual inspection.

tau = cosine-similarity merge threshold; complete linkage => every pair within
a cluster is within tau (conservative, avoids chaining / false merges).

Usage:
  python scripts/cluster_canon.py                       # sweep + families
  python scripts/cluster_canon.py --show-tau 0.97 --show-tasks creative-writing,code-review
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
EMB = ROOT / "notebooks" / "_explore_cache" / "bge"
FORMS = OUT / "canon_real_forms.jsonl"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]
TAUS = [0.985, 0.97, 0.955, 0.94, 0.92, 0.90]


def load_task(task):
    rows = [json.loads(l) for l in FORMS.open() if json.loads(l)["task"] == task]
    rows.sort(key=lambda r: r["idx"])
    emb = np.load(EMB / f"emb_bge_canon_{task}.npy").astype(np.float32)
    assert len(emb) == len(rows), f"{task}: {len(emb)} emb vs {len(rows)} rows"
    return rows, emb


def cluster(emb, tau):
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    m = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                linkage="complete", distance_threshold=1.0 - tau)
    return m.fit_predict(emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-tau", type=float, default=0.97)
    ap.add_argument("--show-tasks", default="creative-writing,code-review,peer-review")
    ap.add_argument("--n-families", type=int, default=12)
    args = ap.parse_args()

    data = {t: load_task(t) for t in TASKS}

    print("=" * 88)
    print("COMPRESSION by tau  —  n_clusters (compression%, = 1 - clusters/leaves)")
    print("=" * 88)
    hdr = f"{'task':<24}{'leaves':>8}" + "".join(f"{t:>13}" for t in TAUS)
    print(hdr)
    nc_grid = {}  # (task, tau) -> n_clusters, computed once
    for t in TASKS:
        emb = data[t][1]
        for tau in TAUS:
            nc_grid[(t, tau)] = len(set(cluster(emb, tau)))

    for t in TASKS:
        n = len(data[t][0])
        cells = [f"{nc_grid[(t,tau)]}({(1-nc_grid[(t,tau)]/n)*100:.0f}%)"
                 for tau in TAUS]
        print(f"{t:<24}{n:>8}" + "".join(f"{c:>13}" for c in cells))

    print("-" * len(hdr))
    tot_n = sum(len(data[t][0]) for t in TASKS)
    cells = [f"{sum(nc_grid[(t,tau)] for t in TASKS)}"
             f"({(1-sum(nc_grid[(t,tau)] for t in TASKS)/tot_n)*100:.0f}%)"
             for tau in TAUS]
    print(f"{'TOTAL':<24}{tot_n:>8}" + "".join(f"{c:>13}" for c in cells))

    # manual-inspection families at the strict tau
    tau = args.show_tau
    for task in args.show_tasks.split(","):
        rows, emb = data[task]
        lab = cluster(emb, tau)
        fam = defaultdict(list)
        for r, c in zip(rows, lab):
            fam[c].append(r)
        big = sorted([m for m in fam.values() if len(m) >= 2],
                     key=lambda m: -len(m))
        print(f"\n{'='*88}\n{task}: top {args.n_families} merged families at tau={tau} "
              f"({len(big)} multi-member families, "
              f"{sum(len(m) for m in big)} leaves merged)\n{'='*88}")
        for members in big[:args.n_families]:
            print(f"\n  FAMILY ({len(members)} leaves):")
            for m in members[:9]:
                print(f"    - {m['canonical'][:100]}")
            if len(members) > 9:
                print(f"    ... +{len(members)-9} more")


if __name__ == "__main__":
    main()
