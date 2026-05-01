#!/usr/bin/env python
"""Dedup-first canonical feature selection.

Replaces the naive "top-K by ranking score" selection (which over-selects
phrasing variants of popular concepts) with:
  1. Take top-N (default 200) by shrunk-mean ranking
  2. Name all N via gpt-5-mini
  3. LLM dedup → concept families
  4. Rank families by sum of member ranking scores
  5. Pick top-K (default 30) families as canonical features
  6. Use highest-ranked member as the family representative
  7. Generate rubrics for the K family reps

Reuses cached embeddings + similarity model. Produces canonical_features.json
ready for Step 4 binary scoring.
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
from methods.local_explanations.rubric_generation import (
    CanonicalFeature,
    _make_feature_id,
    generate_rubrics,
)
from scripts.run_local_explanations import (
    DATASET_CONFIGS,
    create_generate_fn,
    load_splits,
)
from scripts.sweep_umap_target_weight import (
    compute_cluster_purity,
    compute_feature_label_props,
)
from scripts.sweep_cluster_selection_epsilon import run_dedup_judge

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def name_clusters(features, weights, assignments, cluster_ids, generate_fn,
                  samples_per_cluster):
    """Name a specific subset of clusters (rather than range(n))."""
    remap_assignments = np.full_like(assignments, -1)
    remap = {}
    for new_id, c in enumerate(cluster_ids):
        mask = assignments == c
        remap_assignments[mask] = new_id
        remap[new_id] = c
    names = label_clusters_with_llm(
        features, weights, remap_assignments, len(cluster_ids), generate_fn,
        samples_per_cluster=samples_per_cluster,
    )
    return {remap[new_id]: info for new_id, info in names.items()}


def build_named_clusters(features, weights, assignments, centroids, top_n,
                         shrinkage_alpha, samples_per_cluster, generate_fn):
    """Return list of dicts with cluster_id, name, description, ranking_score,
    aggregate_weight, count, members, centroid."""
    n_clusters = centroids.shape[0]
    stats = []
    for c in range(n_clusters):
        mask = assignments == c
        member_idx = np.where(mask)[0]
        if len(member_idx) == 0:
            continue
        member_w = [weights[i] for i in member_idx]
        agg = float(sum(member_w))
        count = len(member_w)
        rs = agg / (count + float(shrinkage_alpha)) if shrinkage_alpha is not None else agg
        stats.append({
            "cluster_id": c,
            "members_idx": member_idx,
            "member_weights": member_w,
            "aggregate_weight": agg,
            "ranking_score": rs,
            "count": count,
        })
    stats.sort(key=lambda s: s["ranking_score"], reverse=True)
    top = stats[:top_n]
    top_ids = [s["cluster_id"] for s in top]
    logger.info("Top-%d clusters by ranking_score selected for naming", len(top_ids))

    names_by_id = name_clusters(
        features, weights, assignments, top_ids, generate_fn, samples_per_cluster,
    )

    out = []
    for s in top:
        cid = s["cluster_id"]
        info = names_by_id.get(cid, {"name": f"cluster_{cid}", "description": ""})
        out.append({
            "cluster_id": cid,
            "name": info["name"],
            "description": info["description"],
            "ranking_score": s["ranking_score"],
            "aggregate_weight": s["aggregate_weight"],
            "count": s["count"],
            "member_features": [features[i] for i in s["members_idx"][:20]],
            "centroid": centroids[cid].tolist(),
        })
    return out


def select_top_families(named, dedup_groups, top_k):
    """Pick top-K concept families by sum of member ranking_scores.
    Returns list of (family_name, family_description, list_of_member_named_dicts)."""
    families = []
    for g in dedup_groups:
        members = [named[i] for i in g if 0 <= i < len(named)]
        if not members:
            continue
        members.sort(key=lambda m: m["ranking_score"], reverse=True)
        rep = members[0]
        family_score = sum(m["ranking_score"] for m in members)
        families.append({
            "rep": rep,
            "members": members,
            "family_score": family_score,
            "size": len(members),
        })
    families.sort(key=lambda f: f["family_score"], reverse=True)
    return families[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Dedup-first canonical feature selection")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--proposer-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--top-n-candidates", type=int, default=200,
                        help="Name this many clusters before dedup (broader pool)")
    parser.add_argument("--top-k-families", type=int, default=30,
                        help="Final number of canonical features (concept families)")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=100)
    parser.add_argument("--agency", type=str, default=None)
    args = parser.parse_args()

    dcfg = dict(DATASET_CONFIGS[args.dataset])
    if args.agency:
        dcfg["split_dir"] = (REPO_ROOT / "datasets" / "notice-and-comment" / "agencies" / args.agency)
    task_description = get_task_description(dcfg["dataset_name"])
    task_metadata = TASK_METADATA_REGISTRY[dcfg["dataset_name"]]

    train_df, _e, _t = load_splits(dcfg["split_dir"])
    id_col = dcfg["id_column"]
    label_col = dcfg["label_column"]
    if id_col not in train_df.columns:
        train_df[id_col] = [f"train_{i}" for i in range(len(train_df))]

    output_dir = Path(args.output_dir)
    cluster_dir = output_dir / "clustering"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = cluster_dir / f"dedup_first_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Artifacts → %s", artifact_dir)

    config = LocalExplanationConfig(
        approach="star_local",
        n_canonical_features=args.top_k_families,
        output_dir=args.output_dir,
        clustering_method="hdbscan",
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        umap_target_weight=0.0,
        hdbscan_cluster_selection_epsilon=0.0,
    )

    explainer = LocalExplainer(
        config, scoring_backend=None,
        task_description=task_description, task_metadata=task_metadata,
        cache_dir=str(output_dir / "explanation_cache"),
    )
    doc_ids = train_df[id_col].astype(str).tolist()
    texts = train_df[dcfg["text_column"]].tolist()
    star_results = explainer.explain_star_local_batch(texts=texts, doc_ids=doc_ids)

    true_labels = dict(zip(
        train_df[id_col].astype(str), train_df[label_col].astype(int),
    ))
    weighted_features = compute_star_weights(
        star_results, true_labels, task_metadata.positive_label, config,
    )
    feature_weights = defaultdict(float)
    for feat, w in weighted_features:
        feat = feat.strip()
        if feat:
            feature_weights[feat] += w
    features = list(feature_weights.keys())
    weights = [feature_weights[f] for f in features]
    logger.info("Unique features: %d", len(features))

    feature_props_dict = compute_feature_label_props(star_results, true_labels)
    feature_label_props = np.array([feature_props_dict.get(f, 0.5) for f in features])

    emb_cache = cluster_dir / "feature_embeddings.npy"
    if emb_cache.exists() and np.load(emb_cache).shape[0] == len(features):
        embeddings = np.load(emb_cache)
        logger.info("Loaded cached embeddings %s", embeddings.shape)
    else:
        model_dir = cluster_dir / "similarity_model"
        embeddings = embed_with_model(features, str(model_dir))
        np.save(emb_cache, embeddings)

    # Cluster (tw=0, eps=0)
    assignments, centroids = run_umap_hdbscan(
        embeddings,
        n_neighbors=config.umap_n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        seed=config.random_seed,
        y=None,
        target_weight=0.0,
        cluster_selection_epsilon=0.0,
    )
    n_clusters = centroids.shape[0]
    logger.info("HDBSCAN: %d clusters, %d noise (%.1f%%)",
                n_clusters, int((assignments == -1).sum()),
                100 * (assignments == -1).sum() / len(assignments))

    purity = compute_cluster_purity(assignments, feature_label_props, n_clusters)
    purity_by_id = {p["cluster_id"]: p for p in purity}

    generate_fn = create_generate_fn(args.proposer_model)

    # Step 1: name top-N candidates
    n_to_name = min(args.top_n_candidates, n_clusters)
    logger.info("Naming top-%d candidate clusters via gpt-5-mini (~$%.1f, ~%d min)",
                n_to_name, n_to_name * 0.005, n_to_name // 6)
    named = build_named_clusters(
        features, weights, assignments, centroids, top_n=n_to_name,
        shrinkage_alpha=config.shrinkage_alpha,
        samples_per_cluster=config.cluster_label_samples,
        generate_fn=generate_fn,
    )
    # Attach purity
    for n in named:
        p = purity_by_id.get(n["cluster_id"], {})
        n["mean_p_positive"] = p.get("mean_p_positive")
        n["mixed_fraction"] = p.get("mixed_fraction")

    with open(artifact_dir / "named_top_n.json", "w") as f:
        json.dump(named, f, indent=2)
    logger.info("Saved %d named clusters", len(named))

    # Step 2: LLM dedup judge
    logger.info("Running LLM dedup judge over %d clusters...", len(named))
    groups = run_dedup_judge(named, task_metadata.text_type, generate_fn)
    logger.info("Dedup → %d concept families", len(groups))
    with open(artifact_dir / "dedup_groups.json", "w") as f:
        json.dump({
            "n_named": len(named),
            "n_families": len(groups),
            "groups": groups,
            "merged_groups_names": [
                [named[i]["name"] for i in g if 0 <= i < len(named)]
                for g in groups if len(g) >= 2
            ],
        }, f, indent=2)

    # Step 3: rank families, take top-K
    top_families = select_top_families(named, groups, args.top_k_families)
    logger.info("Top-%d families selected (from %d total)",
                len(top_families), len(groups))
    for i, fam in enumerate(top_families[:10]):
        logger.info("  Family %d: rep=%r (size=%d, family_score=%.3f)",
                    i + 1, fam["rep"]["name"], fam["size"], fam["family_score"])

    # Build CanonicalFeature objects from family reps; merge member features
    # so the rubric prompt sees the union.
    canonical_clusters = []
    for fam in top_families:
        rep = fam["rep"]
        merged_members = []
        merged_weight = 0.0
        for m in fam["members"]:
            merged_members.extend(m["member_features"])
            merged_weight += m["aggregate_weight"]
        # Make a stub object for rubric_generation.generate_rubrics
        # generate_rubrics expects FeatureCluster-like with name, description,
        # member_features, aggregate_weight. Use a SimpleNamespace.
        from types import SimpleNamespace
        canonical_clusters.append(SimpleNamespace(
            cluster_id=rep["cluster_id"],
            name=rep["name"],
            description=rep["description"],
            member_features=merged_members[:50],
            member_weights=[],
            aggregate_weight=merged_weight,
            ranking_score=fam["family_score"],
            centroid=np.array(rep["centroid"]),
        ))

    # Step 4: rubrics
    logger.info("Generating rubrics for top-%d families", len(canonical_clusters))
    canonical = generate_rubrics(
        canonical_clusters,
        text_type=task_metadata.text_type,
        generate_fn=generate_fn,
        top_k=len(canonical_clusters),
    )

    # Save canonical_features.json (and timestamped copy)
    def _serialize(cf_list):
        return [
            {
                "feature_id": cf.feature_id,
                "name": cf.name,
                "description": cf.description,
                "rubric": cf.rubric,
                "aggregate_weight": cf.aggregate_weight,
                "cluster_size": cf.cluster_size,
                "member_features": cf.member_features[:20],
            }
            for cf in canonical
        ]

    out_ts = artifact_dir / "canonical_features.json"
    with open(out_ts, "w") as f:
        json.dump(_serialize(canonical), f, indent=2)
    latest_path = cluster_dir / "canonical_features.json"
    with open(latest_path, "w") as f:
        json.dump(_serialize(canonical), f, indent=2)
    logger.info("Wrote %d canonical features → %s", len(canonical), out_ts)
    logger.info("Updated latest pointer: %s", latest_path)


if __name__ == "__main__":
    main()
