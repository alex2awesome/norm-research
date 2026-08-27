"""Cross-check the dispersion result against a second, independent embedding
model (BAAI/bge-large-en-v1.5, re-embedded on sk3).

If bge + pca_top30 reproduces (a) the expanded dynamic range and (b) the
per-task ORDERING found with text-embedding-3-small + pca_top30, the ordering
is a real per-task signal rather than an OpenAI-embedding artifact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
CACHE = ROOT / "notebooks" / "_explore_cache"

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def mean_pairwise_cossim(emb):
    n = len(emb)
    if n < 2:
        return 1.0
    c = emb.mean(axis=0)
    return float((n * float(c @ c) - 1.0) / (n - 1))


def per_task_dispersion(loader, pca_k=None):
    """loader(task)->emb. If pca_k set, project pooled onto top-k PCs first."""
    mats, labels = [], []
    for t in TASKS:
        a = loader(t).astype(np.float64)
        mats.append(a)
        labels += [t] * len(a)
    X = np.vstack(mats)
    labels = np.array(labels)
    if pca_k:
        Xc = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        X = (U * S)[:, :pca_k]
    Z = unit(X)
    return {t: 1.0 - mean_pairwise_cossim(Z[labels == t]) for t in TASKS}


def main():
    openai = lambda t: np.load(CACHE / f"emb_rubric_cluster_{t}.npy")
    bge = lambda t: np.load(CACHE / "bge" / f"emb_bge_rubric_cluster_{t}.npy")

    rows = []
    for name, loader in [("openai", openai), ("bge", bge)]:
        for tag, k in [("baseline", None), ("pca_top30", 30)]:
            d = per_task_dispersion(loader, k)
            v = np.array([d[t] for t in TASKS])
            rows.append({"emb": name, "transform": tag, "disp": d,
                         "cv": v.std(ddof=1) / v.mean()})

    print("=" * 72)
    print("DYNAMIC RANGE (CV of within-task dispersion) by embedding x transform")
    print("=" * 72)
    for r in rows:
        print(f"  {r['emb']:<8} {r['transform']:<12} CV = {r['cv']:.3f}")

    # ordering agreement on pca_top30
    o = next(r for r in rows if r["emb"] == "openai" and r["transform"] == "pca_top30")
    b = next(r for r in rows if r["emb"] == "bge" and r["transform"] == "pca_top30")
    ov = [o["disp"][t] for t in TASKS]
    bv = [b["disp"][t] for t in TASKS]
    rho, p = spearmanr(ov, bv)

    # baseline ordering agreement, for contrast
    ob = next(r for r in rows if r["emb"] == "openai" and r["transform"] == "baseline")
    bb = next(r for r in rows if r["emb"] == "bge" and r["transform"] == "baseline")
    rho0, _ = spearmanr([ob["disp"][t] for t in TASKS],
                        [bb["disp"][t] for t in TASKS])

    print("\n" + "=" * 72)
    print("PER-TASK ORDERING: openai vs bge agreement (Spearman rho)")
    print("=" * 72)
    print(f"  baseline   rho = {rho0:+.3f}   (raw cosine - expected noisy)")
    print(f"  pca_top30  rho = {rho:+.3f}   p = {p:.4f}")
    print()
    print(f"  {'task':<26} {'openai':>9} {'rank':>5}   {'bge':>9} {'rank':>5}")
    orank = {t: i for i, t in enumerate(sorted(TASKS, key=lambda t: o['disp'][t]))}
    brank = {t: i for i, t in enumerate(sorted(TASKS, key=lambda t: b['disp'][t]))}
    for t in sorted(TASKS, key=lambda t: o["disp"][t]):
        print(f"  {t:<26} {o['disp'][t]:>9.4f} {orank[t]:>5}   "
              f"{b['disp'][t]:>9.4f} {brank[t]:>5}")

    # ---- metric/subtask dispersion RATIO (the downstream use), pca_top30 ----
    print("\n" + "=" * 72)
    print("METRIC/SUBTASK DISPERSION RATIO  (rubric disp / subtask disp, pca_top30)")
    print("  >1 = rubric space more spread than the task's own subtask space")
    print("=" * 72)
    ratios = {}
    for name, pre in [("openai", "emb"), ("bge", "bge/emb_bge")]:
        rub = per_task_dispersion(
            lambda t, p=pre: np.load(CACHE / f"{p}_rubric_cluster_{t}.npy"), 30)
        sub = per_task_dispersion(
            lambda t, p=pre: np.load(CACHE / f"{p}_subtask_desc_{t}.npy"), 30)
        ratios[name] = {t: rub[t] / sub[t] for t in TASKS}
    rr, _ = spearmanr([ratios["openai"][t] for t in TASKS],
                      [ratios["bge"][t] for t in TASKS])
    print(f"  openai-vs-bge ratio ordering: Spearman rho = {rr:+.3f}\n")
    print(f"  {'task':<26} {'openai':>9} {'bge':>9}")
    for t in sorted(TASKS, key=lambda t: -ratios["openai"][t]):
        print(f"  {t:<26} {ratios['openai'][t]:>9.3f} {ratios['bge'][t]:>9.3f}")


if __name__ == "__main__":
    main()
