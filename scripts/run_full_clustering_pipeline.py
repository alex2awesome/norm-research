#!/usr/bin/env python
"""Run the full clustering pipeline (Steps 1-6) for a task with cached extractions.

Unlike run_clustering_only.py (which skips Steps 1-2), this runs:
  Step 1: Similarity data (embed + candidate pairs + gpt-5-mini labeling + triplets)
  Step 2: LoRA fine-tune BGE-large on triplets
  Steps 3-5: UMAP + HDBSCAN + cluster naming
  Step 6: Dedup-first canonical feature selection + rubric generation

No VLLM/Llama needed — uses gpt-5-mini for labeling/naming/dedup/rubrics and
BGE-large (small GPU footprint) for embeddings. Fits in ~5 GB GPU.
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
from methods.local_explanations.clustering import build_canonical_vocabulary
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


def main():
    parser = argparse.ArgumentParser(description="Full clustering pipeline (no VLLM)")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--proposer-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--n-canonical-features", type=int, default=30)
    parser.add_argument("--agency", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5",
                        help="Base embedder for candidate pair generation")
    parser.add_argument("--similarity-model-base", type=str, default="BAAI/bge-large-en-v1.5",
                        help="Base model for LoRA fine-tuning")
    parser.add_argument("--similarity-labeling-model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-similarity-pairs", type=int, default=10000)
    parser.add_argument("--similarity-labeling-concurrency", type=int, default=30)
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
        embedding_model=args.embedding_model,
        similarity_model_base=args.similarity_model_base,
        similarity_labeling_model=args.similarity_labeling_model,
        max_similarity_pairs=args.max_similarity_pairs,
        similarity_labeling_concurrency=args.similarity_labeling_concurrency,
    )

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "explanation_cache"

    if not (cache_dir / "star_local_v1.jsonl").exists():
        raise SystemExit(f"No cached extractions at {cache_dir}/star_local_v1.jsonl")

    # Reload cached extractions (no VLLM needed)
    explainer = LocalExplainer(
        config, scoring_backend=None,
        task_description=task_description, task_metadata=task_metadata,
        cache_dir=str(cache_dir),
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

    # Build per-feature polarity map
    feature_polarity = {}
    for _doc_id, _result in star_results.items():
        for _f in _result.features_for:
            _f = _f.strip()
            if not _f:
                continue
            feature_polarity[_f] = "both" if feature_polarity.get(_f) == "against" else "for"
        for _f in _result.features_against:
            _f = _f.strip()
            if not _f:
                continue
            feature_polarity[_f] = "both" if feature_polarity.get(_f) == "for" else "against"

    logger.info("Total weighted features: %d", len(weighted_features))

    generate_fn = create_generate_fn(args.proposer_model)

    # Run the FULL clustering pipeline (Steps 1-6)
    # build_canonical_vocabulary handles caching internally:
    #   - triplets.json → skip Step 1 if exists
    #   - similarity_model/config.json → skip Step 2 if exists
    #   - canonical_features.json → skip everything if exists
    canonical = build_canonical_vocabulary(
        weighted_features, config,
        text_type=task_metadata.text_type,
        generate_fn=generate_fn,
        output_dir=str(output_dir / "clustering"),
        feature_polarity=feature_polarity,
    )

    if not canonical:
        logger.error("No canonical features produced")
        return

    logger.info("Pipeline complete: %d canonical features → %s",
                len(canonical), output_dir / "clustering" / "canonical_features.json")


if __name__ == "__main__":
    main()
