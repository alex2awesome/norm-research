"""Approach A: Prior estimation and TF-IDF calibration."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import LocalExplanationConfig, TaskMetadata
from .parsing import parse_feature_list_json
from .prompts import render_prior_prompt

logger = logging.getLogger(__name__)


@dataclass
class PriorCluster:
    centroid: np.ndarray
    member_features: List[str]
    frequency: float  # normalized: how often this cluster appeared across prior calls
    label: int        # which class (0 or 1) this cluster was associated with


@dataclass
class PriorDistribution:
    features: List[str]
    embeddings: np.ndarray
    clusters: List[PriorCluster]
    cluster_assignments: np.ndarray  # index into clusters for each feature


class PriorEstimator:
    """Estimate prior feature distribution p(z|y) and compute TF-IDF calibration."""

    def __init__(
        self,
        config: LocalExplanationConfig,
        task_description: str,
        task_metadata: TaskMetadata,
    ):
        self.config = config
        self.task_description = task_description
        self.meta = task_metadata
        self._embed_model = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.config.embedding_model)
        return self._embed_model

    def estimate_prior(
        self,
        generate_fn,  # callable(prompt: str) -> str
    ) -> PriorDistribution:
        """Make N API calls asking "what features cause acceptance/rejection?"

        Args:
            generate_fn: A callable that takes a prompt string and returns
                the LLM's response string. Typically wraps an API call
                (e.g., dspy.LM or openai client).

        Returns:
            PriorDistribution with clustered prior features.
        """
        all_features = []
        all_labels = []

        for label_val, label_word in self.meta.label_words.items():
            n_calls = self.config.n_prior_calls // 2
            for i in range(n_calls):
                prompt = render_prior_prompt(
                    task_description=self.task_description,
                    text_type=self.meta.text_type,
                    label_description=label_word,
                    n_features=self.config.features_per_prior_call,
                )
                response = generate_fn(prompt)
                features = parse_feature_list_json(response)
                logger.info(
                    "Prior call %d/%d (label=%s): extracted %d features",
                    i + 1, n_calls, label_word, len(features),
                )
                all_features.extend(features)
                all_labels.extend([label_val] * len(features))

        if not all_features:
            logger.warning("No prior features extracted!")
            return PriorDistribution(
                features=[], embeddings=np.array([]),
                clusters=[], cluster_assignments=np.array([]),
            )

        # Embed and cluster
        model = self._get_embed_model()
        embeddings = model.encode(all_features, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # Simple K-means for prior clustering
        from sklearn.cluster import KMeans

        n_clusters = min(
            max(len(all_features) // 3, 5),
            len(all_features),
        )
        km = KMeans(n_clusters=n_clusters, random_state=self.config.random_seed, n_init=10)
        assignments = km.fit_predict(embeddings)

        # Build cluster objects
        clusters = []
        for c_idx in range(n_clusters):
            mask = assignments == c_idx
            member_indices = np.where(mask)[0]
            if len(member_indices) == 0:
                continue
            members = [all_features[i] for i in member_indices]
            member_labels = [all_labels[i] for i in member_indices]
            # Majority label
            majority_label = max(set(member_labels), key=member_labels.count)
            clusters.append(PriorCluster(
                centroid=km.cluster_centers_[c_idx],
                member_features=members,
                frequency=len(members) / len(all_features),
                label=majority_label,
            ))

        logger.info(
            "Prior estimation: %d features → %d clusters",
            len(all_features), len(clusters),
        )

        return PriorDistribution(
            features=all_features,
            embeddings=embeddings,
            clusters=clusters,
            cluster_assignments=assignments,
        )

    def calibrate_features(
        self,
        example_features: Dict[str, List[str]],
        prior: PriorDistribution,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """For each example's features, compute prior-calibrated weights.

        Features matching common prior clusters get downweighted (TF-IDF style).
        Novel features (low similarity to any prior) get full weight.

        Returns: doc_id -> [(feature_text, calibrated_weight), ...]
        """
        if not prior.clusters:
            # No prior — all features get weight 1.0
            return {
                doc_id: [(f, 1.0) for f in features]
                for doc_id, features in example_features.items()
            }

        # Collect all unique features for batch embedding
        all_features = set()
        for features in example_features.values():
            all_features.update(features)
        all_features = list(all_features)

        if not all_features:
            return {doc_id: [] for doc_id in example_features}

        model = self._get_embed_model()
        feature_embeddings = model.encode(all_features, show_progress_bar=False)
        feature_embeddings = np.array(feature_embeddings)

        # Compute cosine similarity to each prior cluster centroid
        centroids = np.array([c.centroid for c in prior.clusters])
        # Normalize
        feat_norms = feature_embeddings / (np.linalg.norm(feature_embeddings, axis=1, keepdims=True) + 1e-8)
        cent_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        # Cosine similarity matrix: (n_features, n_clusters)
        sim_matrix = feat_norms @ cent_norms.T

        # For each feature, find nearest prior cluster and compute weight
        feature_weights = {}
        for i, feat_text in enumerate(all_features):
            max_sim = sim_matrix[i].max()
            nearest_cluster_idx = sim_matrix[i].argmax()

            if max_sim >= self.config.prior_similarity_threshold:
                # Feature matches a prior cluster — downweight by IDF
                prior_freq = prior.clusters[nearest_cluster_idx].frequency
                weight = 1.0 / (self.config.prior_idf_smooth + prior_freq)
            else:
                # Novel feature — full weight
                weight = 1.0

            feature_weights[feat_text] = weight

        # Build output
        result = {}
        for doc_id, features in example_features.items():
            result[doc_id] = [
                (f, feature_weights.get(f, 1.0)) for f in features
            ]
        return result
