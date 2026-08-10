"""Expand the dynamic range of the per-task rubric-cluster dispersion measure.

PROBLEM. With raw text-embedding-3-small vectors, within-task dispersion
(1 - mean pairwise cos sim) sits at 0.72-0.76 across all 11 tasks (std ~0.015).
That near-constant value is mostly an embedding-anisotropy artifact: every vector
lives in a narrow common cone, so cosine similarities are compressed and barely
discriminate tasks. We cannot use it as a per-task discriminator.

FIX TO TRY. Anisotropy removal. The cone is caused by a few dominant directions
shared by every embedding. Subtract them and the remaining geometry spreads out.
Standard recipes:
  - center            : subtract the pooled mean (kills the rank-1 common comp.)
  - all-but-the-top k  : Mu & Viswanath 2018 - subtract top-k principal comps.
  - whiten            : Su et al. 2021 - center + decorrelate + unit-variance.
  - whiten_topk       : whiten but keep only the top-k informative directions.
  - pca_topk          : project onto top-k PCs (denoise, no variance rescaling).

Every transform is FIT on the pooled rubric-cluster embeddings (task-agnostic),
then applied to each task. After the transform we re-unit-normalise and report
mean-pairwise-cosine dispersion via the closed form.

HEADLINE METRIC. CV = std/mean of the 11 within-task dispersion values. Higher
CV = more dynamic range = a more usable per-task discriminator. We also keep the
cross-task (inter-task pair) dispersion: a sane measure must have
cross-task > mean within-task, otherwise the transform has destroyed all signal.

Outputs:
  outputs/analyses/dispersion_transforms.parquet   (per task x transform)
  printed summary table ranked by CV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
EMB_CACHE = ROOT / "notebooks" / "_explore_cache"
OUT = ROOT / "outputs" / "analyses"

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]
RNG = np.random.default_rng(14)


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def mean_pairwise_cossim(emb: np.ndarray) -> float:
    """Closed form for unit vectors: (N*||centroid||^2 - 1) / (N - 1)."""
    n = len(emb)
    if n < 2:
        return 1.0
    c = emb.mean(axis=0)
    return float((n * float(c @ c) - 1.0) / (n - 1))


def load_pool(level: str):
    """Stack all tasks' cached embeddings; return (X, task_labels)."""
    mats, labels = [], []
    for t in TASKS:
        p = EMB_CACHE / f"emb_{level}_{t}.npy"
        if not p.exists():
            continue
        a = np.load(p).astype(np.float64)
        mats.append(a)
        labels += [t] * len(a)
    return np.vstack(mats), np.array(labels)


# ---- transforms (each fit on pooled X, returns fitted-fn) -------------------

def fit_transforms(X: np.ndarray) -> dict:
    """Return {name: transformed_pooled_matrix}. SVD fit once on pooled X."""
    mu = X.mean(axis=0)
    Xc = X - mu
    # economy SVD of centred pool: Xc = U S Vt ; PCs are rows of Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = U * S                       # (N, d) projection onto PCs
    n = len(X)
    out = {"baseline": X.copy(), "center": Xc.copy()}

    for k in (1, 2, 3, 5, 10, 20):
        # all-but-the-top-k: drop the k dominant directions
        rec = comps[:, :k] @ Vt[:k]
        out[f"abtt{k}"] = Xc - rec

    # full PCA whitening: divide every PC by its std (s/sqrt(n-1))
    scale = S / np.sqrt(max(n - 1, 1))
    scale[scale < 1e-9] = 1e-9
    out["whiten"] = comps / scale

    for k in (50, 100, 300):
        out[f"whiten_top{k}"] = comps[:, :k] / scale[:k]

    for k in (30, 100, 300):
        out[f"pca_top{k}"] = comps[:, :k].copy()

    return out


def summarise(name: str, Xt: np.ndarray, labels: np.ndarray) -> dict:
    """Per-task within dispersion + cross-task dispersion, on unit-normed Xt."""
    Z = unit(Xt)
    within = {}
    for t in TASKS:
        m = labels == t
        if m.sum() < 3:
            continue
        within[t] = 1.0 - mean_pairwise_cossim(Z[m])
    vals = np.array(list(within.values()))

    # cross-task: sample pairs, keep different-task ones
    n = len(Z)
    i = RNG.integers(0, n, size=400_000)
    j = RNG.integers(0, n, size=400_000)
    diff = labels[i] != labels[j]
    cross = 1.0 - float(np.mean(np.sum(Z[i[diff]] * Z[j[diff]], axis=1)))

    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    return {
        "transform": name,
        "within_mean": mean,
        "within_std": std,
        "within_cv": std / mean if mean else 0.0,
        "within_min": float(vals.min()),
        "within_max": float(vals.max()),
        "within_range": float(vals.max() - vals.min()),
        "cross_task": cross,
        "gap": cross - mean,
        "per_task": within,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, labels = load_pool("rubric_cluster")
    print(f"pooled rubric clusters: {X.shape}  ({len(set(labels))} tasks)\n")

    transforms = fit_transforms(X)
    results = [summarise(name, Xt, labels) for name, Xt in transforms.items()]

    # rank by dynamic range (CV), but only among transforms that keep
    # cross-task > within-task (a sane discriminator)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "per_task"}
                       for r in results])
    df = df.sort_values("within_cv", ascending=False).reset_index(drop=True)

    print("=" * 92)
    print("DYNAMIC RANGE of within-task dispersion, by anisotropy-removal transform")
    print("  within_cv = std/mean across 11 tasks  (higher = more discriminating)")
    print("  gap > 0 required: cross-task pairs must be MORE dispersed than within")
    print("=" * 92)
    hdr = f"{'transform':<14} {'w_mean':>8} {'w_std':>8} {'CV':>7} {'range':>8} {'cross':>8} {'gap':>8}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        flag = "" if r["gap"] > 0 else "  <- gap<=0 (signal destroyed)"
        print(f"{r['transform']:<14} {r['within_mean']:>8.4f} {r['within_std']:>8.4f} "
              f"{r['within_cv']:>7.3f} {r['within_range']:>8.4f} {r['cross_task']:>8.4f} "
              f"{r['gap']:>8.4f}{flag}")

    # per-task detail for baseline + the best sane transform
    sane = df[df.gap > 0]
    best = sane.iloc[0]["transform"] if len(sane) else df.iloc[0]["transform"]
    print("\n" + "=" * 92)
    print(f"PER-TASK dispersion: baseline vs best sane transform ({best})")
    print("=" * 92)
    bymap = {r["transform"]: r["per_task"] for r in results}
    base, bestp = bymap["baseline"], bymap[best]
    print(f"{'task':<26} {'baseline':>10} {'rank':>5}   {best:>14} {'rank':>5}")
    base_rank = {t: i for i, t in enumerate(sorted(base, key=base.get))}
    best_rank = {t: i for i, t in enumerate(sorted(bestp, key=bestp.get))}
    for t in sorted(bestp, key=bestp.get):
        print(f"{t:<26} {base[t]:>10.4f} {base_rank[t]:>5}   "
              f"{bestp[t]:>14.4f} {best_rank[t]:>5}")

    rows = []
    for r in results:
        for t, v in r["per_task"].items():
            rows.append({"transform": r["transform"], "task": t, "dispersion": v})
    pd.DataFrame(rows).to_parquet(OUT / "dispersion_transforms.parquet")
    print(f"\nwrote {OUT/'dispersion_transforms.parquet'}")


if __name__ == "__main__":
    main()
