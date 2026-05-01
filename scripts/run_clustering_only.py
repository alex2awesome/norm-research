#!/usr/bin/env python
"""Run only Steps 3-6 of the clustering pipeline on cached STaR-Local extractions.

Skips Step 1 (similarity data) and Step 2 (fine-tune) entirely — assumes the
embedder at clustering/similarity_model/ is already ready (e.g. LoRA-merged
BGE-large). Skips Step 4 binary scoring (needs Llama-70B) and Step 5 L1
regression. Produces canonical_features.json with rubrics.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "methods" / "autometrics"))

from methods.autometrics.autometrics.task_descriptions import get_task_description
from methods.local_explanations.cluster_features import cluster_and_label
from methods.local_explanations.config import (
    TASK_METADATA_REGISTRY,
    LocalExplanationConfig,
)
from methods.local_explanations.explainer import LocalExplainer
from methods.local_explanations.feature_weights import compute_star_weights
from methods.local_explanations.rubric_generation import generate_rubrics
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


def main():
    parser = argparse.ArgumentParser(description="Clustering-only driver (no VLLM)")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--proposer-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--n-canonical-features", type=int, default=30)
    parser.add_argument("--agency", type=str, default=None)
    parser.add_argument("--clustering-method", type=str, default="kmeans",
                        choices=["kmeans", "hdbscan"])
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=100)
    args = parser.parse_args()

    dcfg = dict(DATASET_CONFIGS[args.dataset])
    if args.agency:
        dcfg["split_dir"] = (
            REPO_ROOT / "datasets" / "notice-and-comment" / "agencies" / args.agency
        )

    task_description = get_task_description(dcfg["dataset_name"])
    task_metadata = TASK_METADATA_REGISTRY[dcfg["dataset_name"]]

    train_df, _eval_df, _test_df = load_splits(dcfg["split_dir"])
    id_col = dcfg["id_column"]
    label_col = dcfg["label_column"]
    if id_col not in train_df.columns:
        train_df[id_col] = [f"train_{i}" for i in range(len(train_df))]

    config = LocalExplanationConfig(
        approach="star_local",
        n_canonical_features=args.n_canonical_features,
        output_dir=args.output_dir,
        clustering_method=args.clustering_method,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
    )

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "explanation_cache"
    cluster_dir = output_dir / "clustering"
    model_dir = cluster_dir / "similarity_model"

    if not (cache_dir / "star_local_v1.jsonl").exists():
        raise SystemExit(f"No cached extractions at {cache_dir}/star_local_v1.jsonl")
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"No similarity_model at {model_dir} — fine-tune first")

    # ---- Reload cached extractions (no VLLM needed) ----
    explainer = LocalExplainer(
        config, scoring_backend=None,
        task_description=task_description, task_metadata=task_metadata,
        cache_dir=str(cache_dir),
    )
    doc_ids = train_df[id_col].astype(str).tolist()
    texts = train_df[dcfg["text_column"]].tolist()
    star_results = explainer.explain_star_local_batch(texts=texts, doc_ids=doc_ids)

    # ---- Weight features ----
    true_labels = dict(zip(
        train_df[id_col].astype(str),
        train_df[label_col].astype(int),
    ))
    weighted_features = compute_star_weights(
        star_results, true_labels, task_metadata.positive_label, config,
    )

    # Dedup + aggregate weights (matches build_canonical_vocabulary)
    feature_weights = defaultdict(float)
    for feat, w in weighted_features:
        feat = feat.strip()
        if feat:
            feature_weights[feat] += w
    features = list(feature_weights.keys())
    weights = [feature_weights[f] for f in features]
    logger.info("Clustering %d unique features (from %d occurrences)",
                len(features), len(weighted_features))

    # ---- Step 3-5: embed with LoRA-BGE, cluster, label all clusters ----
    generate_fn = create_generate_fn(args.proposer_model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.clustering_method}_{timestamp}"
    mv_path = None
    if args.clustering_method == "hdbscan":
        mv_path = str(Path(args.output_dir) / "clustering" / f"membership_vectors_{run_tag}.npz")
    clusters = cluster_and_label(
        features, weights, str(model_dir), config, generate_fn,
        membership_vectors_path=mv_path,
    )

    if not clusters:
        raise SystemExit("No clusters produced")

    # Dump ALL clusters (full 429+ from HDBSCAN, or full K for K-means) BEFORE
    # rubric generation so we have the artifact even if rubric step fails.
    all_clusters_path = cluster_dir / f"all_clusters_{run_tag}.json"
    with open(all_clusters_path, "w") as f:
        json.dump(
            [
                {
                    "cluster_id": c.cluster_id,
                    "name": c.name,
                    "description": c.description,
                    "aggregate_weight": c.aggregate_weight,
                    "ranking_score": c.ranking_score,
                    "cluster_size": len(c.member_features),
                    "member_features": c.member_features[:20],
                }
                for c in clusters
            ],
            f,
            indent=2,
        )
    logger.info("Saved %d named clusters to %s", len(clusters), all_clusters_path)

    # ---- Step 6: rubrics for top-K ----
    canonical = generate_rubrics(
        clusters,
        text_type=task_metadata.text_type,
        generate_fn=generate_fn,
        top_k=config.n_canonical_features,
    )

    def _serialize_canonical(cf_list):
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
            for cf in cf_list
        ]

    canon_ts_path = cluster_dir / f"canonical_features_{run_tag}.json"
    with open(canon_ts_path, "w") as f:
        json.dump(_serialize_canonical(canonical), f, indent=2)
    logger.info("Saved %d canonical features to %s", len(canonical), canon_ts_path)

    # Update unsuffixed canonical_features.json as the "latest" pointer so the
    # existing cache-reload logic in build_canonical_vocabulary still works.
    latest_path = cluster_dir / "canonical_features.json"
    with open(latest_path, "w") as f:
        json.dump(_serialize_canonical(canonical), f, indent=2)
    logger.info("Also wrote latest pointer: %s", latest_path)


if __name__ == "__main__":
    main()
