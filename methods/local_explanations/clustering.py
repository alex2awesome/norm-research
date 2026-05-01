"""Orchestrate the full clustering pipeline (steps 1-6).

Takes raw weighted features → produces canonical features with binary rubrics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cluster_features import cluster_and_label
from .config import LocalExplanationConfig
from .rubric_generation import CanonicalFeature, generate_rubrics
from .similarity_data import create_similarity_data
from .train_similarity_model import train_similarity_model

logger = logging.getLogger(__name__)


def build_canonical_vocabulary(
    weighted_features: List[Tuple[str, float]],
    config: LocalExplanationConfig,
    text_type: str,
    generate_fn,  # callable(prompt: str) -> str  — for similarity labeling, naming, rubrics
    output_dir: str,
    feature_polarity: Optional[Dict[str, str]] = None,
) -> List[CanonicalFeature]:
    """Full pipeline: raw features → canonical vocabulary with rubrics.

    Steps:
    1. Create supervised similarity data (pairs → LLM label → triplets)
    2. Train domain-specific similarity model (fine-tune sentence-transformer)
    3-5. K-means on fine-tuned embeddings → optimal K → label clusters
    6. Generate binary rubrics for top-K clusters

    Args:
        weighted_features: flat list of (feature_text, weight) from either approach.
        config: configuration.
        text_type: e.g. "scientific paper abstract".
        generate_fn: LLM text generation callable for all LLM steps.
        output_dir: directory for artifacts (model, triplets, etc.).

    Returns: list of CanonicalFeature objects ready for binary scoring.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Early-exit cache: if rubrics were already generated in a prior run, reload
    canon_path = out_path / "canonical_features.json"
    if canon_path.exists():
        from .rubric_generation import CanonicalFeature, _make_feature_id
        logger.info("Loading cached canonical features from %s", canon_path)
        with open(canon_path) as f:
            cached = json.load(f)
        canonical = [
            CanonicalFeature(
                feature_id=item.get("feature_id") or _make_feature_id(item["name"]),
                name=item["name"],
                description=item["description"],
                rubric=item["rubric"],
                aggregate_weight=item.get("aggregate_weight", 0.0),
                cluster_size=item.get("cluster_size", 0),
                member_features=item.get("member_features", []),
            )
            for item in cached
        ]
        logger.info("Loaded %d cached canonical features", len(canonical))
        return canonical

    # Deduplicate features, aggregating weights
    feature_weights: Dict[str, float] = {}
    for feat, weight in weighted_features:
        feat = feat.strip()
        if feat:
            feature_weights[feat] = feature_weights.get(feat, 0.0) + weight

    features = list(feature_weights.keys())
    weights = [feature_weights[f] for f in features]

    logger.info(
        "Clustering pipeline: %d unique features (from %d occurrences)",
        len(features), len(weighted_features),
    )

    if len(features) < 10:
        logger.warning("Too few features (%d) for clustering pipeline", len(features))
        # Fall back: each feature becomes its own canonical feature
        return _fallback_no_clustering(features, weights, text_type, generate_fn)

    # --- Step 1: Create similarity data ---
    triplets_path = out_path / "triplets.json"
    labeled_cache_path = out_path / "labeled_pairs.jsonl"
    if triplets_path.exists():
        logger.info("Loading cached triplets from %s", triplets_path)
        with open(triplets_path) as f:
            triplets = json.load(f)
    else:
        # Align polarities with the dedup'd features list (for cross-polarity filter)
        polarities = None
        if feature_polarity is not None:
            polarities = [feature_polarity.get(f, "unknown") for f in features]
        triplets = create_similarity_data(
            features, config, generate_fn,
            cache_path=labeled_cache_path,
            polarities=polarities,
        )
        with open(triplets_path, "w") as f:
            json.dump(triplets, f)
        logger.info("Saved %d triplets to %s", len(triplets), triplets_path)

    # --- Step 2: Train similarity model ---
    # Fine-tuning mpnet-base in the same process as the VLLM extractor tends to
    # hit CUDA OOM because sentence-transformers' Trainer auto-detects CUDA and
    # tries to use it even when the base model is on CPU. If training fails,
    # fall back to off-the-shelf embeddings — a real quality loss for clustering,
    # but lets the rest of the pipeline complete. Re-run clustering offline
    # later (with Qwen not loaded) to get the fine-tuned model.
    model_dir = out_path / "similarity_model"
    if model_dir.exists() and (model_dir / "config.json").exists():
        model_path = str(model_dir)
        logger.info("Using cached similarity model at %s", model_path)
    elif triplets:
        try:
            model_path = train_similarity_model(triplets, config, str(out_path))
        except Exception as exc:
            logger.warning(
                "Similarity model fine-tuning failed (%s) — falling back to base embedder %s",
                exc, config.embedding_model,
            )
            model_path = config.embedding_model
    else:
        # No triplets — fall back to off-the-shelf embeddings
        logger.warning("No triplets available — using base embedding model")
        model_path = config.embedding_model

    # --- Steps 3-5: Cluster and label ---
    clusters = cluster_and_label(
        features, weights, model_path, config, generate_fn,
    )

    if not clusters:
        logger.warning("No clusters produced — returning empty vocabulary")
        return []

    # --- Step 6: Generate rubrics ---
    canonical = generate_rubrics(
        clusters,
        text_type=text_type,
        generate_fn=generate_fn,
        top_k=config.n_canonical_features,
    )

    # Save canonical features
    canon_path = out_path / "canonical_features.json"
    with open(canon_path, "w") as f:
        json.dump(
            [
                {
                    "feature_id": cf.feature_id,
                    "name": cf.name,
                    "description": cf.description,
                    "rubric": cf.rubric,
                    "aggregate_weight": cf.aggregate_weight,
                    "cluster_size": cf.cluster_size,
                    "member_features": cf.member_features[:20],  # truncate for readability
                }
                for cf in canonical
            ],
            f,
            indent=2,
        )
    logger.info("Saved %d canonical features to %s", len(canonical), canon_path)

    return canonical


def _fallback_no_clustering(
    features: List[str],
    weights: List[float],
    text_type: str,
    generate_fn,
) -> List[CanonicalFeature]:
    """When there are too few features, skip clustering and generate rubrics directly."""
    from .rubric_generation import CanonicalFeature, _make_feature_id

    logger.info("Fallback: creating canonical features without clustering (%d features)", len(features))
    canonical = []
    for feat, weight in zip(features, weights):
        canonical.append(CanonicalFeature(
            feature_id=_make_feature_id(feat),
            name=feat,
            description=feat,
            rubric={
                "yes": f"The text clearly exhibits: {feat}",
                "no": f"The text does not exhibit: {feat}",
            },
            aggregate_weight=weight,
            cluster_size=1,
            member_features=[feat],
        ))
    return canonical
