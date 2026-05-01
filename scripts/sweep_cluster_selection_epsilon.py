#!/usr/bin/env python
"""Sweep HDBSCAN cluster_selection_epsilon + LLM dedup judge for overlap analysis.

For each epsilon in a grid, runs UMAP + HDBSCAN with that epsilon, names the
top-K clusters, and asks gpt-5-mini to group clusters that refer to the same
underlying concept. Reports an `overlap_rate` per config:

    overlap_rate = 1 − (n_distinct_concept_groups / n_clusters_named)

Lower overlap_rate = cleaner taxonomy with less duplication.

Reuses cached feature embeddings + label props from the target_weight sweep
(if present at clustering/feature_embeddings.npy and either a prior sweep dir's
feature_label_props.npy, or recomputed from the explanation cache).

Idea: pick smallest epsilon where overlap_rate < threshold (say 0.10) AND
n_clusters >= target K — that gives low duplication without over-merging.
"""

import argparse
import json
import logging
import re
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
from methods.local_explanations.parsing import parse_json_dict
from scripts.run_local_explanations import (
    DATASET_CONFIGS,
    create_generate_fn,
    load_splits,
)
from scripts.sweep_umap_target_weight import (
    compute_feature_label_props,
    compute_cluster_purity,
    name_top_k_by_ranking,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEDUP_JUDGE_PROMPT = """Below are {n} short canonical features (name + 1-line description) \
extracted from {text_type}. Group any features that describe the \
SAME underlying concept — even if phrased differently, at different levels of specificity, \
or from opposite framings (e.g. "has strong evaluation" and "lacks evaluation" are the same \
concept family).

Return a JSON object with a single key "groups". Each group is an array of the feature indices \
(0-based, matching the numbered list below) that belong together. Singletons are one-element \
arrays. Every index must appear in exactly one group.

Features:
{features_list}

Return only valid JSON:
{{"groups": [[0, 5], [1], [2, 7, 12], ...]}}"""


def render_dedup_prompt(named_clusters, text_type):
    lines = []
    for i, c in enumerate(named_clusters):
        desc = c.get("description", "")
        lines.append(f"[{i}] {c['name']} — {desc[:200]}")
    return DEDUP_JUDGE_PROMPT.format(
        n=len(named_clusters),
        text_type=text_type,
        features_list="\n".join(lines),
    )


def run_dedup_judge(named_clusters, text_type, generate_fn):
    """Ask gpt-5-mini to group duplicate clusters. Returns list of groups."""
    if not named_clusters:
        return []
    prompt = render_dedup_prompt(named_clusters, text_type)
    response = generate_fn(prompt)
    parsed = parse_json_dict(response)
    groups = parsed.get("groups")
    if not groups or not isinstance(groups, list):
        logger.warning("Dedup judge returned unparseable response; treating each as singleton")
        return [[i] for i in range(len(named_clusters))]
    # Validate: all indices must appear exactly once
    seen = set()
    clean_groups = []
    for g in groups:
        if not isinstance(g, list):
            continue
        clean_g = [int(i) for i in g if isinstance(i, (int, float)) and 0 <= int(i) < len(named_clusters) and int(i) not in seen]
        if clean_g:
            clean_groups.append(clean_g)
            seen.update(clean_g)
    # Append any missing indices as singletons
    for i in range(len(named_clusters)):
        if i not in seen:
            clean_groups.append([i])
    return clean_groups


def main():
    parser = argparse.ArgumentParser(description="Sweep HDBSCAN cluster_selection_epsilon")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--proposer-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.0, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--target-weight", type=float, default=0.0,
                        help="Fix UMAP target_weight for this sweep (use best from target_weight sweep)")
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
    sweep_dir = cluster_dir / f"sweep_epsilon_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    # Reuse cached embeddings from the target_weight sweep if available
    emb_cache = cluster_dir / "feature_embeddings.npy"
    if emb_cache.exists():
        embeddings = np.load(emb_cache)
        if embeddings.shape[0] != len(features):
            logger.warning("Cached embeddings mismatched — recomputing")
            embeddings = None
        else:
            logger.info("Loaded cached embeddings %s", embeddings.shape)
    else:
        embeddings = None
    if embeddings is None:
        model_dir = cluster_dir / "similarity_model"
        embeddings = embed_with_model(features, str(model_dir))
        np.save(emb_cache, embeddings)
        logger.info("Cached embeddings to %s", emb_cache)

    generate_fn = create_generate_fn(args.proposer_model)

    summary = []
    for eps in args.epsilons:
        logger.info("=" * 60)
        logger.info("Sweep: cluster_selection_epsilon=%.2f, target_weight=%.2f",
                    eps, args.target_weight)
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
            target_weight=args.target_weight,
            cluster_selection_epsilon=eps,
        )
        n_clusters = centroids.shape[0]
        if n_clusters == 0:
            logger.warning("epsilon=%.2f produced 0 clusters — skipping", eps)
            continue

        purity = compute_cluster_purity(assignments, feature_label_props, n_clusters)
        with open(sweep_dir / f"purity_eps{eps:.2f}.json", "w") as f:
            json.dump(purity, f, indent=2)

        n_to_name = min(args.top_k, n_clusters)
        named = name_top_k_by_ranking(
            features, weights, assignments, centroids, purity, feature_label_props,
            top_k=n_to_name, generate_fn=generate_fn,
            shrinkage_alpha=config.shrinkage_alpha,
            samples_per_cluster=config.cluster_label_samples,
        )
        with open(sweep_dir / f"top_{n_to_name}_named_eps{eps:.2f}.json", "w") as f:
            json.dump(named, f, indent=2)

        # LLM dedup judge
        groups = run_dedup_judge(named, task_metadata.text_type, generate_fn)
        n_groups = len(groups)
        overlap_rate = 1.0 - (n_groups / max(n_to_name, 1))
        # Gather merged cluster names for human inspection
        merged_groups = []
        for g in groups:
            if len(g) >= 2:
                merged_groups.append([named[i]["name"] for i in g])
        with open(sweep_dir / f"dedup_judge_eps{eps:.2f}.json", "w") as f:
            json.dump({
                "n_clusters_named": n_to_name,
                "n_distinct_groups": n_groups,
                "overlap_rate": overlap_rate,
                "groups": groups,
                "merged_groups_names": merged_groups,
            }, f, indent=2)

        top_k_abs_p_dev = [abs(n["mean_p_positive"] - 0.5) for n in named
                           if n.get("mean_p_positive") is not None]
        mean_abs_p_dev_topk = (float(np.mean(top_k_abs_p_dev))
                               if top_k_abs_p_dev else 0.0)

        summary.append({
            "epsilon": eps,
            "target_weight": args.target_weight,
            "n_clusters": n_clusters,
            "noise_fraction": float((assignments == -1).sum() / len(assignments)),
            "n_clusters_named": n_to_name,
            "n_distinct_concept_groups": n_groups,
            "overlap_rate": overlap_rate,
            "n_merged_groups": len(merged_groups),
            "mean_abs_p_dev_topk": mean_abs_p_dev_topk,
        })
        logger.info(
            "eps=%.2f: K=%d, %.1f%% noise, judge: %d distinct groups out of %d named "
            "(overlap_rate=%.3f, %d merged groups)",
            eps, n_clusters,
            100 * (assignments == -1).sum() / len(assignments),
            n_groups, n_to_name, overlap_rate, len(merged_groups),
        )

    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Sweep summary → %s", summary_path)
    logger.info("Per-config results:")
    for s in summary:
        logger.info(
            "  eps=%.2f: K=%-4d noise=%.1f%%  named=%-3d  distinct_groups=%-3d  "
            "overlap=%.3f  merged=%d",
            s["epsilon"], s["n_clusters"],
            100 * s["noise_fraction"], s["n_clusters_named"],
            s["n_distinct_concept_groups"], s["overlap_rate"], s["n_merged_groups"],
        )


if __name__ == "__main__":
    main()
