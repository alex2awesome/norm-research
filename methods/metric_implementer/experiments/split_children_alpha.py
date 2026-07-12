"""Quick test (2026-06-26): does SPLITTING a metric's children into 2 behavioral clusters lower α?

Hypothesis: α measures a metric's internal behavioral coherence. A HIGH-α metric is incoherent — a
composite of distinct behavioral modes (children don't collapse). Splitting its children along those
modes (k-means on the σ signatures) should isolate coherent sub-metrics, each with LOWER α (more
internal collapse). The trivial counter-case: splitting an already-coherent blob → two coherent halves
→ α→0 (uninteresting); so the real signal is on HIGH-α (incoherent) parents.

CPU-only; runs on the per-metric checkpoints (…_metric{gi}_sigs.npz) the metric sweep wrote. Reports,
per metric: α on ALL children (parent) vs α on each of 2 behavioral child-clusters, + D/N (collapse
ratio) and whether the split lowered α.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .alpha_probe import collide, heaps_alpha, rarefaction, spectrum, _terminal_alpha


def alpha_of(sigs: np.ndarray, tau: float = 0.02):
    """α (terminal Heaps) + D/N on a group of signatures (the children of one (sub)metric)."""
    sigs = np.asarray(sigs, float)
    n = len(sigs)
    if n < 4:
        return float("nan"), 0, n
    labels = collide(sigs, tau)
    f, N, D = (lambda s: (s["f"], s["N"], s["D"]))(spectrum(labels, ["x"] * n))
    ms, S = rarefaction(f, N)
    return float(_terminal_alpha(ms, heaps_alpha(ms, S))), int(D), int(N)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="split_children_alpha", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="dir with …_metric{gi}_sigs.npz checkpoints")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=2, help="number of behavioral child-clusters")
    a = ap.parse_args(argv)
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise SystemExit(f"need sklearn: {e}")

    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "*_metric*_sigs.npz")))
    print(f"{len(ckpts)} checkpoints; τ={a.tau}, k={a.k}\n")
    print(f"{'metric':52s} {'α_parent':>8} {'D/N_p':>9} | " +
          "  ".join(f"α_cl{i+1:>2} D/N" for i in range(a.k)) + " | split_lowered")
    n_lowered = 0
    for ck in ckpts:
        z = np.load(ck, allow_pickle=True)
        tags = list(z["tags"])
        sigs = np.asarray(z["sigs"], float)
        name = str(z["name"]) if "name" in z else os.path.basename(ck)
        child_mask = np.array([t == "children" for t in tags])
        child_sigs = sigs[child_mask]
        if len(child_sigs) < 8:                       # too few children to split meaningfully
            continue
        a_par, d_par, n_par = alpha_of(child_sigs, a.tau)
        km = KMeans(n_clusters=a.k, n_init=10, random_state=0).fit(child_sigs)
        cell, lowered = [], False
        for c in range(a.k):
            sub = child_sigs[km.labels_ == c]
            ac, dc, nc = alpha_of(sub, a.tau)
            cell.append((ac, dc, nc))
            if np.isfinite(ac) and np.isfinite(a_par) and ac < a_par - 0.02:
                lowered = True
        if lowered:
            n_lowered += 1
        cells = "  ".join(f"{ac:5.2f} {dc:>3}/{nc:<3}" for ac, dc, nc in cell)
        print(f"{name[:52]:52s} {a_par:8.3f} {d_par:>3}/{n_par:<3}  | {cells} | "
              f"{'YES' if lowered else 'no'}")
    print(f"\nSplit lowered α on ≥1 cluster in {n_lowered}/{len(ckpts)} metrics.")


if __name__ == "__main__":
    main()
