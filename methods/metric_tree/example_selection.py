"""Embedding-based representative example selection via clustering.

Selects diverse, representative examples from exception and correct subsets
using sentence embeddings + K-means clustering. Falls back to random sampling
if sentence-transformers is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("metric_tree.example_selection")

_embedding_model = None
_embedding_model_name = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Load and cache a sentence-transformers model."""
    global _embedding_model, _embedding_model_name
    if _embedding_model is None or _embedding_model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", model_name)
            _embedding_model = SentenceTransformer(model_name)
            _embedding_model_name = model_name
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to random sampling")
            _embedding_model = None
    return _embedding_model


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """Encode texts into embeddings. Returns ndarray of shape (n, dim) or None."""
    model = get_embedding_model(model_name)
    if model is None:
        return None
    try:
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None


def cluster_and_select(
    df: pd.DataFrame,
    text_column: str,
    k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    seed: int = 42,
) -> pd.DataFrame:
    """Select k representative examples via K-means on embeddings.

    Picks the example nearest to each cluster centroid. Falls back to
    df.sample(k) if embeddings fail or df is too small.
    """
    if len(df) <= k:
        return df.copy()

    texts = df[text_column].astype(str).tolist()
    embeddings = embed_texts(texts, model_name)

    if embeddings is None:
        return df.sample(n=k, random_state=seed).copy()

    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(embeddings)

        # Pick the example closest to each centroid
        selected_indices = []
        for c in range(k):
            cluster_mask = km.labels_ == c
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue
            centroid = km.cluster_centers_[c]
            dists = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
            nearest = cluster_indices[np.argmin(dists)]
            selected_indices.append(nearest)

        if not selected_indices:
            return df.sample(n=k, random_state=seed).copy()

        return df.iloc[selected_indices].copy()

    except Exception as e:
        logger.warning("Clustering failed: %s, falling back to random", e)
        return df.sample(n=k, random_state=seed).copy()


def select_representative_examples(
    exception_df: pd.DataFrame,
    correct_df: pd.DataFrame,
    text_column: str,
    k_per_class: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select representative examples from both exception and correct subsets.

    Returns (exception_sample, correct_sample) DataFrames.
    """
    logger.info(
        "Selecting %d representative examples per class (exception=%d, correct=%d)",
        k_per_class, len(exception_df), len(correct_df),
    )

    exception_sample = cluster_and_select(
        exception_df, text_column, k=k_per_class, model_name=model_name, seed=seed,
    )
    correct_sample = cluster_and_select(
        correct_df, text_column, k=k_per_class, model_name=model_name, seed=seed,
    )

    logger.info(
        "Selected %d exception + %d correct examples",
        len(exception_sample), len(correct_sample),
    )
    return exception_sample, correct_sample
