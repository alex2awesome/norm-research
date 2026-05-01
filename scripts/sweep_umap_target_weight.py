#!/usr/bin/env python
"""Sweep UMAP target_weight for supervised clustering + analyze duplicate clusters.

For each target_weight in a grid, runs UMAP + HDBSCAN on cached embeddings +
per-feature P(y=1), computes per-cluster label-purity stats, names the top-K
clusters via gpt-5-mini, and dumps an artifact to disk.

After the sweep, identifies candidate "duplicate clusters": pairs with high
centroid cosine similarity but large |mean_p_positive| difference — the
pathology we're worried about.

Expensive pieces are cached:
  clustering/feature_embeddings.npy — BGE-large+LoRA embeddings over 429K features
  clustering/feature_label_props.npy — P(y=1|feature) aligned to feature list
  clustering/features.txt — feature text per row (for cross-file alignment)
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "methods" / "autometrics"))

from methods.autometrics.autometrics.task_descriptions import get_task_description
from methods.local_explanations.cluster_features import (
    FeatureCluster,
    embed_with_model,
    label_clusters_with_llm,
    run_umap_hdbscan,
)
from methods.local_explanations.config import (
    TASK_METADATA_REGISTRY,
    LocalExplanationConfig,
)
from methods.local_explanations.explainer import LocalExplainer
from methods.local_explanations.feature_weights import compute_star_weights
from scripts.run_local_explanations import (
    DATASET_CONFIGS,
    create_generate_fn,
    load_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_feature_label_props(star_results, true_labels):
    """For each unique feature, return P(y=1 | feature was extracted).

    Aggregates across all doc occurrences in features_for ∪ features_against.
    """
    feature_doc_counts = defaultdict(lambda: [0, 0])  # feature -> [n_pos_docs, n_neg_docs]
    for doc_id, result in star_results.items():
        y = true_labels.get(doc_id)
        if y is None:
            continue
        idx = 0 if y == 1 else 1
        seen_in_doc = set()
        for f in list(result.features_for) + list(result.features_against):
            f = f.strip()
            if not f or f in seen_in_doc:
                continue
            seen_in_doc.add(f)
            feature_doc_counts[f][idx] += 1

    props = {}
    for f, (n_pos, n_neg) in feature_doc_counts.items():
        total = n_pos + n_neg
        props[f] = n_pos / total if total > 0 else 0.5
    return props


def compute_cluster_purity(
    assignments: np.ndarray,
    feature_label_props: np.ndarray,
    n_clusters: int,
) -> list:
    """Per-cluster label-purity statistics."""
    purity = []
    for c in range(n_clusters):
        mask = assignments == c
        if mask.sum() == 0:
            continue
        member_props = feature_label_props[mask]
        purity.append({
            "cluster_id": int(c),
            "n_members": int(mask.sum()),
            "mean_p_positive": float(member_props.mean()),
            "median_p_positive": float(np.median(member_props)),
            "std_p_positive": float(member_props.std()),
            "n_pos_pure": int((member_props >= 0.7).sum()),
            "n_neg_pure": int((member_props <= 0.3).sum()),
            "n_mixed": int(((member_props > 0.3) & (member_props < 0.7)).sum()),
            "mixed_fraction": float(((member_props > 0.3) & (member_props < 0.7)).sum() / mask.sum()),
        })
    return purity


def name_top_k_by_ranking(
    features, weights, assignments, centroids, purity, feature_label_props,
    top_k, generate_fn, shrinkage_alpha, samples_per_cluster,
):
    """Name only the top-K clusters by ranking score."""
    n_clusters = centroids.shape[0]
    stats = []
    for c in range(n_clusters):
        mask = assignments == c
        members_idx = np.where(mask)[0]
        if len(members_idx) == 0:
            continue
        member_weights = [weights[i] for i in members_idx]
        agg = float(sum(member_weights))
        count = len(member_weights)
        if shrinkage_alpha is None:
            rs = agg
        else:
            rs = agg / (count + float(shrinkage_alpha))
        stats.append({
            "cluster_id": c,
            "members_idx": members_idx,
            "member_weights": member_weights,
            "aggregate_weight": agg,
            "ranking_score": rs,
            "count": count,
        })
    stats.sort(key=lambda s: s["ranking_score"], reverse=True)
    top = stats[:top_k]
    top_ids = [s["cluster_id"] for s in top]

    logger.info("Naming top %d clusters by ranking_score", len(top))
    top_assignments = np.full_like(assignments, -1)
    remap = {}
    for new_id, c in enumerate(top_ids):
        mask = assignments == c
        top_assignments[mask] = new_id
        remap[new_id] = c
    names = label_clusters_with_llm(
        features, weights, top_assignments, len(top_ids), generate_fn,
        samples_per_cluster=samples_per_cluster,
    )

    purity_by_id = {p["cluster_id"]: p for p in purity}
    named_top = []
    for new_id, s in enumerate(top):
        orig_id = s["cluster_id"]
        label = names.get(new_id, {"name": f"cluster_{orig_id}", "description": ""})
        p = purity_by_id.get(orig_id, {})
        member_sample = [features[i] for i in s["members_idx"][:20]]
        named_top.append({
            "cluster_id": orig_id,
            "name": label["name"],
            "description": label["description"],
            "aggregate_weight": s["aggregate_weight"],
            "ranking_score": s["ranking_score"],
            "count": s["count"],
            "mean_p_positive": p.get("mean_p_positive"),
            "mixed_fraction": p.get("mixed_fraction"),
            "n_pos_pure": p.get("n_pos_pure"),
            "n_neg_pure": p.get("n_neg_pure"),
            "centroid": centroids[orig_id].tolist(),
            "member_features": member_sample,
        })
    return named_top


def find_duplicate_clusters(named, sim_threshold=0.90, p_delta_threshold=0.3):
    """Detect pairs of clusters with high centroid similarity but divergent labels."""
    if not named:
        return []
    cents = np.array([n["centroid"] for n in named])
    cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
    sims = cents @ cents.T
    dupes = []
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            p_i = named[i].get("mean_p_positive")
            p_j = named[j].get("mean_p_positive")
            if p_i is None or p_j is None:
                continue
            if sims[i, j] >= sim_threshold and abs(p_i - p_j) >= p_delta_threshold:
                dupes.append({
                    "a_name": named[i]["name"],
                    "b_name": named[j]["name"],
                    "centroid_sim": float(sims[i, j]),
                    "p_pos_a": p_i,
                    "p_pos_b": p_j,
                    "p_delta": abs(p_i - p_j),
                })
    return dupes


def main():
    parser = argparse.ArgumentParser(description="Sweep UMAP target_weight")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--proposer-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Only name + save top-K clusters per config (by ranking_score)")
    parser.add_argument("--target-weights", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=100)
    parser.add_argument("--agency", type=str, default=None)
    args = parser.parse_args()

    dcfg = dict(DATASET_CONFIGS[args.dataset])
    if args.agency:
        dcfg["split_dir"] = (
            REPO_ROOT / "datasets" / "notice-and-comment" / "agencies" / args.agency
        )
    task_description = get_task_description(dcfg["dataset_name"])
    task_metadata = TASK_METADATA_REGISTRY[dcfg["dataset_name"]]

    train_df, _e, _t = load_splits(dcfg["split_dir"])
    id_col = dcfg["id_column"]
    label_col = dcfg["label_column"]
    if id_col not in train_df.columns:
        train_df[id_col] = [f"train_{i}" for i in range(len(train_df))]

    output_dir = Path(args.output_dir)
    cluster_dir = output_dir / "clustering"
    sweep_dir = cluster_dir / f"sweep_umap_target_weight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Sweep artifacts → %s", sweep_dir)

    config = LocalExplanationConfig(
        approach="star_local",
        n_canonical_features=args.top_k,
        output_dir=args.output_dir,
        clustering_method="hdbscan",
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
    )

    # Reload cached extractions
    explainer = LocalExplainer(
        config, scoring_backend=None,
        task_description=task_description, task_metadata=task_metadata,
        cache_dir=str(output_dir / "explanation_cache"),
    )
    doc_ids = train_df[id_col].astype(str).tolist()
    texts = train_df[dcfg["text_column"]].tolist()
    star_results = explainer.explain_star_local_batch(texts=texts, doc_ids=doc_ids)

    true_labels = dict(zip(
        train_df[id_col].astype(str),
        train_df[label_col].astype(int),
    ))
    weighted_features = compute_star_weights(
        star_results, true_labels, task_metadata.positive_label, config,
    )

    # Dedup + aggregate weights
    feature_weights = defaultdict(float)
    for feat, w in weighted_features:
        feat = feat.strip()
        if feat:
            feature_weights[feat] += w
    features = list(feature_weights.keys())
    weights = [feature_weights[f] for f in features]
    logger.info("Unique features: %d", len(features))

    # Per-feature P(y=1)
    feature_props_dict = compute_feature_label_props(star_results, true_labels)
    feature_label_props = np.array([feature_props_dict.get(f, 0.5) for f in features])
    logger.info(
        "Feature label props: mean=%.3f, %d pure-pos (≥0.7), %d pure-neg (≤0.3), %d mixed",
        feature_label_props.mean(),
        int((feature_label_props >= 0.7).sum()),
        int((feature_label_props <= 0.3).sum()),
        int(((feature_label_props > 0.3) & (feature_label_props < 0.7)).sum()),
    )

    # Cache features + label props for offline analysis
    np.save(sweep_dir / "feature_label_props.npy", feature_label_props)
    with open(sweep_dir / "features.txt", "w") as f:
        for feat in features:
            f.write(feat.replace("\n", " ") + "\n")

    # Embed once (~7 min), cache
    emb_cache = cluster_dir / "feature_embeddings.npy"
    if emb_cache.exists():
        logger.info("Loading cached embeddings from %s", emb_cache)
        embeddings = np.load(emb_cache)
        if embeddings.shape[0] != len(features):
            logger.warning("Cached embeddings shape %s != %d features — recomputing",
                           embeddings.shape, len(features))
            embeddings = None
    else:
        embeddings = None
    if embeddings is None:
        model_dir = cluster_dir / "similarity_model"
        embeddings = embed_with_model(features, str(model_dir))
        np.save(emb_cache, embeddings)
        logger.info("Cached embeddings to %s (shape=%s)", emb_cache, embeddings.shape)

    generate_fn = create_generate_fn(args.proposer_model)

    summary = []
    for tw in args.target_weights:
        logger.info("=" * 60)
        logger.info("Sweep config: target_weight=%.2f", tw)
        logger.info("=" * 60)

        assignments, centroids = run_umap_hdbscan(
            embeddings,
            n_neighbors=config.umap_n_neighbors,
            n_components=config.umap_n_components,
            min_dist=config.umap_min_dist,
            min_cluster_size=config.hdbscan_min_cluster_size,
            min_samples=config.hdbscan_min_samples,
            seed=config.random_seed,
            y=feature_label_props,
            target_weight=tw,
        )
        n_clusters = centroids.shape[0]
        if n_clusters == 0:
            logger.warning("target_weight=%.2f produced 0 clusters — skipping", tw)
            continue

        purity = compute_cluster_purity(assignments, feature_label_props, n_clusters)
        # Save full purity (for all clusters, not just top-K)
        purity_path = sweep_dir / f"purity_tw{tw:.2f}.json"
        with open(purity_path, "w") as f:
            json.dump(purity, f, indent=2)

        # Name top-K and save
        named = name_top_k_by_ranking(
            features, weights, assignments, centroids, purity, feature_label_props,
            top_k=args.top_k, generate_fn=generate_fn,
            shrinkage_alpha=config.shrinkage_alpha,
            samples_per_cluster=config.cluster_label_samples,
        )
        named_path = sweep_dir / f"top_{args.top_k}_named_tw{tw:.2f}.json"
        with open(named_path, "w") as f:
            json.dump(named, f, indent=2)

        dupes = find_duplicate_clusters(named)
        dupes_path = sweep_dir / f"duplicates_tw{tw:.2f}.json"
        with open(dupes_path, "w") as f:
            json.dump(dupes, f, indent=2)

        # Aggregate mixed fraction over all clusters (unweighted + weighted by size)
        total_members = sum(p["n_members"] for p in purity)
        weighted_mixed = sum(p["n_mixed"] for p in purity) / max(total_members, 1)
        cluster_level_mixed = sum(1 for p in purity if p["mixed_fraction"] > 0.5) / max(len(purity), 1)

        # Top-K discriminative-power proxy: mean |mean_p - 0.5| over the top-K clusters
        # that actually get rubrics + scored downstream. Higher = more label signal per
        # scoring-budget slot.
        top_k_abs_p_dev = [abs(n["mean_p_positive"] - 0.5) for n in named
                           if n.get("mean_p_positive") is not None]
        mean_abs_p_dev_topk = (float(np.mean(top_k_abs_p_dev))
                               if top_k_abs_p_dev else 0.0)

        summary.append({
            "target_weight": tw,
            "n_clusters": n_clusters,
            "noise_fraction": float((assignments == -1).sum() / len(assignments)),
            "member_level_mixed_fraction": weighted_mixed,
            "cluster_level_mixed_fraction": cluster_level_mixed,
            "n_duplicate_pairs_topk": len(dupes),
            "mean_abs_p_dev_topk": mean_abs_p_dev_topk,
        })
        logger.info(
            "tw=%.2f: %d clusters, %.1f%% noise, %.1f%% members in mixed clusters, "
            "%d dup pairs in top-%d, mean|p-0.5|_topk=%.3f",
            tw, n_clusters,
            100 * (assignments == -1).sum() / len(assignments),
            100 * weighted_mixed, len(dupes), args.top_k,
            mean_abs_p_dev_topk,
        )

    # Cross-config summary
    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Sweep summary saved to %s", summary_path)
    for s in summary:
        logger.info(
            "  tw=%.2f: K=%-4d noise=%.1f%%  member-mixed=%.1f%%  cluster-mixed=%.1f%%  "
            "dup-pairs=%d  mean|p-0.5|_topk=%.3f",
            s["target_weight"], s["n_clusters"],
            100 * s["noise_fraction"],
            100 * s["member_level_mixed_fraction"],
            100 * s["cluster_level_mixed_fraction"],
            s["n_duplicate_pairs_topk"],
            s["mean_abs_p_dev_topk"],
        )


if __name__ == "__main__":
    main()
