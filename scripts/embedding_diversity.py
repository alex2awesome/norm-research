"""Clustering-free diversity metrics for rubric evaluation spaces.

For each task, embeds (a) its rubric-cluster medoids and (b) its subtask
descriptions, then computes three threshold-free / clustering-free measures:

  1. DISPERSION       — mean pairwise cosine distance (1 - mean pairwise cos sim).
                        Closed-form, N-independent. High = diffuse evaluation space.
  2. EFFECTIVE DIM    — participation ratio of the embedding covariance spectrum:
                        PR = (Σλ)² / Σλ². How many independent directions the
                        task's rubric space spans. Reported at full N and at a
                        common N=1000 subsample (PR has mild N-dependence).
  3. PAIRWISE-SIM     — distribution of cosine similarities over sampled pairs;
                        percentiles saved + a sample array for histogram plots.

The subtask-description metrics are the task-breadth control: compare rubric
dispersion to subtask dispersion per task.

Outputs:
  outputs/analyses/embedding_diversity_per_task.parquet
  outputs/analyses/pairwise_sim_samples.npz   (per task/level sampled sims)
  notebooks/_explore_cache/emb_*.npy          (cached embeddings)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
HIER = ROOT / "outputs" / "hierarchy"
OUT = ROOT / "outputs" / "analyses"
EMB_CACHE = ROOT / "notebooks" / "_explore_cache"
PAGES = EMB_CACHE / "pages.parquet"
KEY_PATH = Path("/Users/spangher/.openai-salt-lab-key.txt")

EMBED_MODEL = "text-embedding-3-small"
TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]
RNG = np.random.default_rng(14)
SUBSAMPLE_N = 1000        # common N for fair effective-dim comparison
N_PAIRS = 100_000         # sampled pairs for the similarity distribution


def embed_texts(client: OpenAI, texts: list[str], batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] if t else " " for t in texts[i:i + batch]]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        out.extend([d.embedding for d in resp.data])
    arr = np.array(out, dtype=np.float64)
    # unit-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def load_cluster_texts() -> dict[str, list[str]]:
    out = {}
    for task in TASKS:
        p = HIER / f"{task}_general_r1_refined.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        texts = []
        for par in d.get("parented_trees", []):
            for ch in par.get("children", []):
                nm = ch.get("medoid_name", "") or ""
                ds = ch.get("medoid_description", "") or ""
                texts.append(f"{nm}: {ds}".strip())
        out[task] = texts
    return out


def load_subtask_texts() -> dict[str, list[str]]:
    df = pd.read_parquet(PAGES)
    df = df[~df.is_error]
    out = {}
    for task in TASKS:
        sub = df[df.task == task]
        descs = [d for d in sub.subtask_description.fillna("").tolist() if d.strip()]
        out[task] = descs
    return out


def mean_pairwise_cossim(emb: np.ndarray) -> float:
    """Closed form for unit vectors: (N*||centroid||^2 - 1) / (N - 1)."""
    n = len(emb)
    if n < 2:
        return 1.0
    centroid = emb.mean(axis=0)
    return float((n * float(centroid @ centroid) - 1.0) / (n - 1))


def participation_ratio(emb: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ² of the centered covariance spectrum."""
    n = len(emb)
    if n < 3:
        return float(n)
    x = emb - emb.mean(axis=0)
    # singular values of centered X; eigenvalues of covariance ∝ s²
    s = np.linalg.svd(x, compute_uv=False)
    lam = s ** 2
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))


def pr_subsampled(emb: np.ndarray, n: int, reps: int = 5) -> float | None:
    if len(emb) < n:
        return None
    vals = []
    for _ in range(reps):
        idx = RNG.choice(len(emb), size=n, replace=False)
        vals.append(participation_ratio(emb[idx]))
    return float(np.mean(vals))


def sample_pairwise_sims(emb: np.ndarray, n_pairs: int) -> np.ndarray:
    n = len(emb)
    if n < 2:
        return np.array([])
    i = RNG.integers(0, n, size=n_pairs)
    j = RNG.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    return np.sum(emb[i] * emb[j], axis=1)


def analyze(level: str, texts_by_task: dict[str, list[str]], client: OpenAI):
    rows = []
    sims = {}
    for task in TASKS:
        texts = texts_by_task.get(task, [])
        if len(texts) < 3:
            continue
        cache = EMB_CACHE / f"emb_{level}_{task}.npy"
        if cache.exists():
            emb = np.load(cache)
        else:
            print(f"  embedding {level}/{task} ({len(texts)} texts)...")
            emb = embed_texts(client, texts)
            np.save(cache, emb)
        cossim = mean_pairwise_cossim(emb)
        ps = sample_pairwise_sims(emb, N_PAIRS)
        sims[f"{level}__{task}"] = ps
        rows.append({
            "level": level,
            "task": task,
            "n": len(emb),
            "mean_pairwise_cossim": cossim,
            "dispersion": 1.0 - cossim,
            "participation_ratio_full": participation_ratio(emb),
            "participation_ratio_n1000": pr_subsampled(emb, SUBSAMPLE_N),
            "sim_p10": float(np.percentile(ps, 10)) if len(ps) else None,
            "sim_p50": float(np.percentile(ps, 50)) if len(ps) else None,
            "sim_p90": float(np.percentile(ps, 90)) if len(ps) else None,
        })
    return rows, sims


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=KEY_PATH.read_text().strip())
    t0 = time.perf_counter()

    cluster_texts = load_cluster_texts()
    subtask_texts = load_subtask_texts()

    rows, sims = [], {}
    r1, s1 = analyze("rubric_cluster", cluster_texts, client)
    r2, s2 = analyze("subtask_desc", subtask_texts, client)
    rows = r1 + r2
    sims = {**s1, **s2}

    df = pd.DataFrame(rows)
    out_table = OUT / "embedding_diversity_per_task.parquet"
    df.to_parquet(out_table)
    np.savez_compressed(OUT / "pairwise_sim_samples.npz", **sims)

    print(f"\ndone in {time.perf_counter()-t0:.0f}s")
    print(f"wrote {out_table}")
    print()
    show = df[df.level == "rubric_cluster"].sort_values("dispersion", ascending=False)
    print("RUBRIC-CLUSTER diversity (sorted by dispersion):")
    print(show[["task", "n", "dispersion", "participation_ratio_n1000", "sim_p50", "sim_p90"]].to_string(index=False))
    print()
    show2 = df[df.level == "subtask_desc"].sort_values("dispersion", ascending=False)
    print("SUBTASK-DESCRIPTION diversity (the task-breadth control):")
    print(show2[["task", "n", "dispersion", "participation_ratio_n1000", "sim_p50", "sim_p90"]].to_string(index=False))


if __name__ == "__main__":
    main()
