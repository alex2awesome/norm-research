"""The depth dial — controlling the generality of what we discover (spec §6).

A tree node is already a stratified subpopulation, so the *pooling level of the contrast set*
sets the depth of the feature we get back:

- **narrow / leaf-level:** draw WRONG/RIGHT from a single deep gap node -> the LLM can only
  find a within-stratum (conditional) feature.
- **broad / root-level:** detect the diffuse signature first — the same residual theme
  recurring across many sibling/cousin gap nodes — by clustering the per-gap-node proposed
  descriptions by embedding similarity. When a cluster spans many nodes, pool WRONG/RIGHT
  across them and propose once. Pooling de-stratifies -> a population-wide contrast -> the
  feature splits near the root on reinsertion.
"""

from __future__ import annotations

from typing import List

import numpy as np


def embed(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed short descriptions. Uses sentence-transformers if available, else a hashing
    bag-of-words fallback so clustering still works offline."""
    if not texts:
        return np.zeros((0, 1))
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb)
    except Exception:
        return _hash_embed(texts)


def _hash_embed(texts: List[str], dim: int = 256) -> np.ndarray:
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms > 0, norms, 1.0)


def cluster_by_cosine(emb: np.ndarray, threshold: float) -> List[List[int]]:
    """Greedy agglomerative clustering by cosine similarity (vectors assumed normalized)."""
    n = len(emb)
    if n == 0:
        return []
    sim = emb @ emb.T
    unassigned = set(range(n))
    clusters: List[List[int]] = []
    while unassigned:
        seed = unassigned.pop()
        members = [seed]
        for j in list(unassigned):
            if sim[seed, j] >= threshold:
                members.append(j)
        unassigned -= set(members)
        clusters.append(sorted(members))
    return clusters


def find_poolable_clusters(descriptions: List[str], cfg) -> List[List[int]]:
    """Return clusters of gap-node indices whose proposed descriptions recur (>= min nodes)."""
    # Early-out before any embedding: a cluster can never reach min_nodes if there are fewer
    # descriptions than that, so skip the (network-bound) sentence-transformers call entirely.
    # Keeps the offline tests from triggering a HuggingFace download for MiniLM.
    if len(descriptions) < getattr(cfg, "pool_cluster_min_nodes", 3):
        return []
    emb = embed(descriptions, cfg.embedding_model)
    clusters = cluster_by_cosine(emb, cfg.pool_cosine_threshold)
    return [c for c in clusters if len(c) >= cfg.pool_cluster_min_nodes]
