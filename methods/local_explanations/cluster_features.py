"""Clustering steps 3-5: K-means on fine-tuned embeddings, optimal K, label clusters.

Following github.com/alex2awesome/schema-generation (steps 4-5).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import LocalExplanationConfig
from .parsing import parse_json_dict
from .prompts import render_cluster_naming_prompt

logger = logging.getLogger(__name__)


@dataclass
class FeatureCluster:
    """A cluster of raw features that describe the same concept."""
    cluster_id: int
    name: str                   # LLM-generated label
    description: str            # LLM-generated description
    member_features: List[str]
    member_weights: List[float]
    centroid: np.ndarray
    aggregate_weight: float = 0.0   # raw sum (kept for analysis)
    ranking_score: float = 0.0      # shrunk mean used for top-K selection

    def __post_init__(self):
        self.aggregate_weight = sum(self.member_weights)

    def compute_ranking_score(self, shrinkage_alpha: Optional[float]) -> float:
        """Compute and cache the ranking score used for top-K selection.

        shrinkage_alpha=None   → raw sum (legacy behavior)
        shrinkage_alpha=0      → plain mean
        shrinkage_alpha>0      → shrunk mean: sum / (count + alpha)
                                 small clusters get pulled toward 0
        """
        count = len(self.member_weights)
        if shrinkage_alpha is None:
            self.ranking_score = self.aggregate_weight
        elif count == 0:
            self.ranking_score = 0.0
        else:
            self.ranking_score = self.aggregate_weight / (count + float(shrinkage_alpha))
        return self.ranking_score


def embed_with_model(
    features: List[str],
    model_path: str,
    use_tail_only: bool = True,
) -> np.ndarray:
    """Embed features using a (possibly fine-tuned) sentence-transformer.

    If use_tail_only=True, embed only the em-dash tail (evaluative concept)
    of each feature, falling back to the full text if no em-dash present.
    This concentrates the embedding space on conceptual judgments rather than
    on per-paper specifics, matching what the similarity labeler saw.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path)
    if use_tail_only:
        to_embed = [
            f.split(" — ", 1)[1].strip() if " — " in f else f.strip()
            for f in features
        ]
    else:
        to_embed = features
    embeddings = model.encode(to_embed, show_progress_bar=True, batch_size=256)
    return np.array(embeddings)


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: List[int],
    seed: int = 42,
) -> Tuple[int, Dict[int, float]]:
    """Run K-means for multiple K values and select by silhouette score.

    Returns: (best_k, {k: silhouette_score}).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores = {}
    for k in k_range:
        if k >= len(embeddings):
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
        scores[k] = score
        logger.info("K=%d: silhouette=%.4f", k, score)

    if not scores:
        logger.warning("No valid K found — defaulting to min(30, n_features)")
        return min(30, len(embeddings)), scores

    best_k = max(scores, key=scores.get)
    logger.info("Optimal K=%d (silhouette=%.4f)", best_k, scores[best_k])
    return best_k, scores


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run K-means and return (assignments, centroids)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    assignments = km.fit_predict(embeddings)
    return assignments, km.cluster_centers_


def run_umap_hdbscan(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_components: int,
    min_dist: float,
    min_cluster_size: int,
    min_samples: Optional[int],
    seed: int = 42,
    return_clusterer: bool = False,
    y: Optional[np.ndarray] = None,
    target_weight: float = 0.0,
    cluster_selection_epsilon: float = 0.0,
):
    """Run UMAP → HDBSCAN. Returns (assignments, centroids_in_embedding_space).

    If return_clusterer=True, also returns the fitted HDBSCAN object (for soft
    membership vectors via all_points_membership_vectors) and the UMAP embedding.

    Noise points (HDBSCAN label -1) are dropped from the assignment array by
    leaving them labeled -1; downstream code must filter. Centroids are the
    mean of member points in the ORIGINAL embedding space (not UMAP space) so
    any downstream cosine-sim lookup still works.
    """
    import umap
    import hdbscan

    logger.info(
        "UMAP: n_neighbors=%d n_components=%d min_dist=%.2f on %d×%d",
        n_neighbors, n_components, min_dist,
        embeddings.shape[0], embeddings.shape[1],
    )
    # random_state=None to allow UMAP internal parallelism (warns + forces n_jobs=1
    # when seeded). Non-deterministic UMAP output is acceptable; HDBSCAN below is
    # deterministic given its UMAP input.
    umap_kwargs: Dict[str, Any] = dict(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=None,
        low_memory=True,
        verbose=True,
    )
    if y is not None and target_weight > 0.0:
        umap_kwargs["target_metric"] = "l2"
        umap_kwargs["target_weight"] = target_weight
        logger.info("Supervised UMAP: target_weight=%.2f", target_weight)
    reducer = umap.UMAP(**umap_kwargs)
    if y is not None and target_weight > 0.0:
        umap_emb = reducer.fit_transform(embeddings, y=y)
    else:
        umap_emb = reducer.fit_transform(embeddings)

    logger.info(
        "HDBSCAN: min_cluster_size=%d min_samples=%s on %d×%d",
        min_cluster_size, min_samples, umap_emb.shape[0], umap_emb.shape[1],
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        cluster_selection_epsilon=cluster_selection_epsilon,
        core_dist_n_jobs=-1,
        prediction_data=True,   # needed for all_points_membership_vectors
    )
    assignments = clusterer.fit_predict(umap_emb)

    unique = set(assignments.tolist())
    n_noise = int((assignments == -1).sum())
    n_clusters = len([c for c in unique if c != -1])
    logger.info(
        "HDBSCAN found %d clusters + %d noise points (%.1f%% noise)",
        n_clusters, n_noise, 100 * n_noise / len(assignments),
    )

    # Centroids in original embedding space for non-noise clusters
    centroids = np.zeros((n_clusters, embeddings.shape[1]), dtype=embeddings.dtype)
    sorted_ids = sorted(c for c in unique if c != -1)
    # Remap cluster ids to [0, n_clusters) so downstream code indexes centroids by id
    remap = {old: new for new, old in enumerate(sorted_ids)}
    new_assignments = np.full_like(assignments, -1)
    for old, new in remap.items():
        mask = assignments == old
        new_assignments[mask] = new
        centroids[new] = embeddings[mask].mean(axis=0)
    if return_clusterer:
        return new_assignments, centroids, clusterer, umap_emb, remap
    return new_assignments, centroids


def label_clusters_with_llm(
    features: List[str],
    weights: List[float],
    assignments: np.ndarray,
    n_clusters: int,
    generate_fn,  # callable(prompt: str) -> str
    samples_per_cluster: int = 10,
) -> Dict[int, Dict[str, str]]:
    """Label each K-means cluster by sampling members and asking LLM.

    Returns: dict mapping cluster_id -> {"name": ..., "description": ...}.
    """
    import random
    rng = random.Random(42)
    labels = {}

    for c_idx in range(n_clusters):
        member_indices = np.where(assignments == c_idx)[0]
        if len(member_indices) == 0:
            continue

        # Sample members (prefer higher-weighted ones)
        member_weights = [abs(weights[i]) for i in member_indices]
        total_w = sum(member_weights) + 1e-8
        probs = [w / total_w for w in member_weights]

        n_sample = min(samples_per_cluster, len(member_indices))
        sampled = rng.choices(
            list(member_indices), weights=probs, k=n_sample,
        )
        sampled_features = [features[i] for i in sampled]

        prompt = render_cluster_naming_prompt(sampled_features)
        response = generate_fn(prompt)
        parsed = parse_json_dict(response)

        labels[c_idx] = {
            "name": parsed.get("name", f"cluster_{c_idx}"),
            "description": parsed.get("description", ""),
        }

        if (c_idx + 1) % 20 == 0:
            logger.info("Labeled %d/%d clusters", c_idx + 1, n_clusters)

    logger.info("Labeled %d clusters", len(labels))
    return labels


def cluster_and_label(
    features: List[str],
    weights: List[float],
    model_path: str,
    config: LocalExplanationConfig,
    generate_fn,  # callable(prompt: str) -> str
    membership_vectors_path: Optional[str] = None,
) -> List[FeatureCluster]:
    """Full steps 3-5: embed → optimal K → cluster → label.

    If membership_vectors_path is given and clustering_method=hdbscan, also
    saves the soft-membership matrix (N, n_clusters) over ALL features (noise
    included) to that path via np.savez_compressed.

    Returns: list of FeatureCluster objects.
    """
    logger.info("Clustering %d features with model %s", len(features), model_path)

    # Embed with fine-tuned model
    embeddings = embed_with_model(features, model_path)

    if getattr(config, "clustering_method", "kmeans") == "hdbscan":
        want_mv = membership_vectors_path is not None
        if want_mv:
            assignments, centroids, clusterer, umap_emb, _remap = run_umap_hdbscan(
                embeddings,
                n_neighbors=config.umap_n_neighbors,
                n_components=config.umap_n_components,
                min_dist=config.umap_min_dist,
                min_cluster_size=config.hdbscan_min_cluster_size,
                min_samples=config.hdbscan_min_samples,
                seed=config.random_seed,
                return_clusterer=True,
            )
        else:
            assignments, centroids = run_umap_hdbscan(
                embeddings,
                n_neighbors=config.umap_n_neighbors,
                n_components=config.umap_n_components,
                min_dist=config.umap_min_dist,
                min_cluster_size=config.hdbscan_min_cluster_size,
                min_samples=config.hdbscan_min_samples,
                seed=config.random_seed,
            )
        n_clusters = centroids.shape[0]
        if n_clusters == 0:
            logger.warning("HDBSCAN found no clusters — aborting")
            return []

        if want_mv:
            try:
                import hdbscan as _hdb
                logger.info("Computing soft membership vectors over %d features × %d clusters",
                            assignments.shape[0], n_clusters)
                mv = _hdb.all_points_membership_vectors(clusterer)
                import numpy as _np
                _np.savez_compressed(
                    membership_vectors_path,
                    membership=mv,
                    assignments=assignments,
                )
                logger.info("Saved soft-membership matrix to %s (shape=%s)",
                            membership_vectors_path, mv.shape)
            except Exception as exc:
                logger.warning("Could not compute/save membership vectors: %s", exc)
    else:
        best_k, k_scores = find_optimal_k(
            embeddings, config.optimal_k_range, seed=config.random_seed,
        )
        # Force user-specified K when valid (previously this only logged)
        if (config.n_canonical_features
                and config.n_canonical_features in config.optimal_k_range
                and best_k != config.n_canonical_features):
            logger.info(
                "User override: K=%d (silhouette=%.4f vs optimal K=%d at %.4f)",
                config.n_canonical_features,
                k_scores.get(config.n_canonical_features, 0),
                best_k, k_scores.get(best_k, 0),
            )
            best_k = config.n_canonical_features
        assignments, centroids = run_kmeans(embeddings, best_k, seed=config.random_seed)
        n_clusters = best_k

    # Label clusters (skip the -1 noise cluster, which has no centroid)
    cluster_labels = label_clusters_with_llm(
        features, weights, assignments, n_clusters, generate_fn,
        samples_per_cluster=config.cluster_label_samples,
    )

    # Build FeatureCluster objects (noise points have assignment -1 and are dropped)
    clusters = []
    for c_idx in range(n_clusters):
        mask = assignments == c_idx
        member_indices = np.where(mask)[0]
        if len(member_indices) == 0:
            continue

        member_feats = [features[i] for i in member_indices]
        member_weights = [weights[i] for i in member_indices]
        label_info = cluster_labels.get(c_idx, {"name": f"cluster_{c_idx}", "description": ""})

        clusters.append(FeatureCluster(
            cluster_id=c_idx,
            name=label_info["name"],
            description=label_info["description"],
            member_features=member_feats,
            member_weights=member_weights,
            centroid=centroids[c_idx],
        ))

    # Compute ranking score (shrunk mean by default) and sort
    for c in clusters:
        c.compute_ranking_score(config.shrinkage_alpha)
    clusters.sort(key=lambda c: c.ranking_score, reverse=True)
    logger.info(
        "Built %d clusters. shrinkage_alpha=%s. Top-3 by ranking_score: %s",
        len(clusters),
        config.shrinkage_alpha,
        [
            (c.name, f"score={c.ranking_score:.3f}, sum={c.aggregate_weight:.2f}, n={len(c.member_weights)}")
            for c in clusters[:3]
        ],
    )
    return clusters
