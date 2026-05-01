"""Clustering step 6: Generate binary YES/NO rubrics for canonical features."""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .cluster_features import FeatureCluster
from .parsing import parse_json_dict
from .prompts import render_rubric_prompt

logger = logging.getLogger(__name__)


@dataclass
class CanonicalFeature:
    """A canonical feature with binary rubric, ready for scoring."""
    feature_id: str
    name: str
    description: str
    rubric: Dict[str, str]      # {"yes": "...", "no": "..."}
    aggregate_weight: float
    cluster_size: int
    member_features: List[str]


def _make_feature_id(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:16]


def generate_rubrics(
    clusters: List[FeatureCluster],
    text_type: str,
    generate_fn,  # callable(prompt: str) -> str
    top_k: int = 30,
) -> List[CanonicalFeature]:
    """Generate binary rubrics for the top-K clusters.

    Args:
        clusters: FeatureClusters sorted by aggregate_weight (descending).
        text_type: e.g. "scientific paper abstract" for prompt context.
        generate_fn: LLM generation callable.
        top_k: number of canonical features to produce.

    Returns: list of CanonicalFeature objects with rubrics.
    """
    selected = clusters[:top_k]
    logger.info("Generating rubrics for top %d clusters", len(selected))

    canonical_features = []
    for i, cluster in enumerate(selected):
        # Sample member features for context
        members_sample = cluster.member_features[:10]

        prompt = render_rubric_prompt(
            feature_name=cluster.name,
            feature_description=cluster.description,
            member_features=members_sample,
            text_type=text_type,
        )
        response = generate_fn(prompt)
        parsed = parse_json_dict(response)

        rubric = {
            "yes": parsed.get("yes", f"The text clearly exhibits: {cluster.name}"),
            "no": parsed.get("no", f"The text does not exhibit: {cluster.name}"),
        }

        cf = CanonicalFeature(
            feature_id=_make_feature_id(cluster.name),
            name=cluster.name,
            description=cluster.description,
            rubric=rubric,
            aggregate_weight=cluster.aggregate_weight,
            cluster_size=len(cluster.member_features),
            member_features=cluster.member_features,
        )
        canonical_features.append(cf)

        if (i + 1) % 10 == 0:
            logger.info("Generated rubrics for %d/%d features", i + 1, len(selected))

    logger.info("Generated %d canonical features with rubrics", len(canonical_features))
    return canonical_features
