"""Binary scoring bridge: scores examples on TreeMetric binary criteria via VLLM backend."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autometrics.iterative_refinement.label_cache import LabelCache

from .data_structures import TreeMetric
from .token_utils import compute_scoring_text_budget, truncate_to_tokens

logger = logging.getLogger("metric_tree.scoring")

# Default depth threshold for clustering vs. discriminative mode
DEFAULT_CLUSTERING_DEPTH = 2


def _binary_rubric_to_criteria_text(metrics: List[TreeMetric]) -> str:
    """Format binary metrics into criteria text for the VLLM prompt."""
    parts = []
    for m in metrics:
        rubric = m.rubric
        yes_desc = rubric.get("yes", rubric.get("Yes", "Meets the criterion"))
        no_desc = rubric.get("no", rubric.get("No", "Does not meet the criterion"))
        parts.append(
            f"{m.name}\n"
            f"  YES: {yes_desc}\n"
            f"  NO: {no_desc}"
        )
    return "\n\n".join(parts)


def _metric_id_from_rubric(rubric_text: str) -> str:
    """Generate a stable hash ID from rubric text."""
    return hashlib.sha256(rubric_text.encode()).hexdigest()[:16]


def score_binary_subset(
    df: pd.DataFrame,
    indices: np.ndarray,
    metrics: List[TreeMetric],
    label_cache: LabelCache,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    task_description: str,
    scoring_backend: Any,
    batch_size: int = 200,
    verbose: bool = False,
    stage: str = "",
    max_text_tokens: int = 0,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> pd.DataFrame:
    """Score a subset of examples on binary TreeMetrics.

    Uses the VLLM backend's score_binary_batch method.
    Returns a DataFrame with id, label, and one column per metric (0 or 1).

    Token truncation:
      - If max_model_len > 0 and tokenizer are provided, computes the exact
        per-document budget from the actual criteria text.
      - Otherwise falls back to max_text_tokens if > 0.
    """
    subset_df = df.iloc[indices].reset_index(drop=True) if len(indices) < len(df) else df.copy()

    if not metrics:
        result = subset_df[[id_column, label_column]].copy()
        return result

    # Dynamic budget: measure actual criteria to get exact text budget
    if max_model_len > 0 and tokenizer is not None:
        max_text_tokens = compute_scoring_text_budget(
            metrics, task_description, max_model_len, tokenizer,
        )

    # Truncate text column
    if max_text_tokens > 0 and text_column in subset_df.columns:
        subset_df = subset_df.copy()
        subset_df[text_column] = subset_df[text_column].apply(
            lambda t: truncate_to_tokens(str(t), max_text_tokens, tokenizer=tokenizer),
        )

    criteria_text = _binary_rubric_to_criteria_text(metrics)
    metric_names = [m.name for m in metrics]
    metric_ids = {m.name: m.metric_id for m in metrics}

    # Check cache for already-scored (metric_id, doc_id) pairs
    doc_ids = subset_df[id_column].astype(str).tolist()
    texts = subset_df[text_column].astype(str).tolist()

    # Initialize score storage
    score_matrix = {}  # metric_name -> list of scores (or None for uncached)
    for name in metric_names:
        score_matrix[name] = [None] * len(doc_ids)

    # Check cache using LabelCache._load_metric API
    need_scoring_mask = [False] * len(doc_ids)
    for m in metrics:
        cached_scores = label_cache._load_metric(m.metric_id)
        for i, doc_id in enumerate(doc_ids):
            if doc_id in cached_scores:
                score_matrix[m.name][i] = int(round(cached_scores[doc_id]))
            else:
                need_scoring_mask[i] = True

    # Batch-score uncached examples
    uncached_indices = [i for i, need in enumerate(need_scoring_mask) if need]
    if uncached_indices:
        if verbose:
            logger.info("[%s] Scoring %d/%d examples on %d binary metrics",
                        stage, len(uncached_indices), len(doc_ids), len(metrics))

        # Process in batches
        for batch_start in range(0, len(uncached_indices), batch_size):
            batch_idx = uncached_indices[batch_start:batch_start + batch_size]
            batch_texts = [texts[i] for i in batch_idx]

            results = scoring_backend.score_binary_batch(
                task_description=task_description,
                criteria_text=criteria_text,
                criterion_names=metric_names,
                texts=batch_texts,
            )

            # Store results and update cache
            for name in metric_names:
                new_ids = []
                new_scores = []
                for j, i in enumerate(batch_idx):
                    val = results[j].get(name, 0)
                    score_matrix[name][i] = val
                    new_ids.append(doc_ids[i])
                    new_scores.append(float(val))
                label_cache.set_scores(metric_ids[name], new_ids, new_scores)
    else:
        if verbose:
            logger.info("[%s] All %d examples cached for %d metrics",
                        stage, len(doc_ids), len(metrics))

    # Build result DataFrame
    result = subset_df[[id_column, label_column]].copy()
    for name in metric_names:
        # Replace any remaining None with 0
        result[name] = [v if v is not None else 0 for v in score_matrix[name]]

    # Preserve text column
    if text_column not in result.columns and text_column in subset_df.columns:
        result[text_column] = subset_df[text_column].values

    return result


def score_ternary_subset(
    df: pd.DataFrame,
    indices: np.ndarray,
    metrics: List[TreeMetric],
    label_cache: LabelCache,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    task_description: str,
    scoring_backend: Any,
    batch_size: int = 200,
    verbose: bool = False,
    stage: str = "",
    max_text_tokens: int = 0,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> pd.DataFrame:
    """Score a subset of examples on ternary TreeMetrics (YES=1, NO=0, NA=NaN).

    Like score_binary_subset but uses the ternary scoring prompt that allows NA.
    Returns a DataFrame with id, label, and one column per metric (1, 0, or NaN).
    """
    subset_df = df.iloc[indices].reset_index(drop=True) if len(indices) < len(df) else df.copy()

    if not metrics:
        result = subset_df[[id_column, label_column]].copy()
        return result

    # Dynamic budget
    if max_model_len > 0 and tokenizer is not None:
        max_text_tokens = compute_scoring_text_budget(
            metrics, task_description, max_model_len, tokenizer,
        )

    # Truncate text column
    if max_text_tokens > 0 and text_column in subset_df.columns:
        subset_df = subset_df.copy()
        subset_df[text_column] = subset_df[text_column].apply(
            lambda t: truncate_to_tokens(str(t), max_text_tokens, tokenizer=tokenizer),
        )

    criteria_text = _binary_rubric_to_criteria_text(metrics)
    metric_names = [m.name for m in metrics]
    metric_ids = {m.name: m.metric_id for m in metrics}

    doc_ids = subset_df[id_column].astype(str).tolist()
    texts = subset_df[text_column].astype(str).tolist()

    # Initialize score storage: None = not yet scored
    score_matrix = {}
    for name in metric_names:
        score_matrix[name] = [None] * len(doc_ids)

    # Check cache — cached scores of -1 mean NA (stored as -1 in cache, restored as NaN)
    need_scoring_mask = [False] * len(doc_ids)
    for m in metrics:
        cached_scores = label_cache._load_metric(m.metric_id)
        for i, doc_id in enumerate(doc_ids):
            if doc_id in cached_scores:
                cached_val = cached_scores[doc_id]
                if cached_val < 0:  # -1 sentinel = NA
                    score_matrix[m.name][i] = float("nan")
                else:
                    score_matrix[m.name][i] = int(round(cached_val))
            else:
                need_scoring_mask[i] = True

    # Batch-score uncached examples using ternary prompt
    uncached_indices = [i for i, need in enumerate(need_scoring_mask) if need]
    if uncached_indices:
        if verbose:
            logger.info("[%s] Ternary scoring %d/%d examples on %d metrics",
                        stage, len(uncached_indices), len(doc_ids), len(metrics))

        for batch_start in range(0, len(uncached_indices), batch_size):
            batch_idx = uncached_indices[batch_start:batch_start + batch_size]
            batch_texts = [texts[i] for i in batch_idx]

            results = scoring_backend.score_ternary_batch(
                task_description=task_description,
                criteria_text=criteria_text,
                criterion_names=metric_names,
                texts=batch_texts,
            )

            for name in metric_names:
                new_ids = []
                new_scores = []
                for j, i in enumerate(batch_idx):
                    val = results[j].get(name)  # 1, 0, or None (NA)
                    if val is None:
                        score_matrix[name][i] = float("nan")
                        new_ids.append(doc_ids[i])
                        new_scores.append(-1.0)  # sentinel for NA in cache
                    else:
                        score_matrix[name][i] = val
                        new_ids.append(doc_ids[i])
                        new_scores.append(float(val))
                label_cache.set_scores(metric_ids[name], new_ids, new_scores)
    else:
        if verbose:
            logger.info("[%s] All %d examples cached for %d ternary metrics",
                        stage, len(doc_ids), len(metrics))

    # Build result DataFrame
    result = subset_df[[id_column, label_column]].copy()
    for name in metric_names:
        result[name] = score_matrix[name]  # 1, 0, or NaN

    if text_column not in result.columns and text_column in subset_df.columns:
        result[text_column] = subset_df[text_column].values

    return result


def build_binary_feature_matrix(
    scored_df: pd.DataFrame,
    feature_names: List[str],
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract binary numpy arrays from a scored DataFrame.

    Returns (X, y) where X has columns matching feature_names with values 0/1.
    """
    missing = [f for f in feature_names if f not in scored_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = scored_df[feature_names].values.astype(np.float64)
    y = scored_df[label_column].values.astype(np.float64)

    # Replace NaN with 0
    nan_mask = np.isnan(X)
    if nan_mask.any():
        logger.warning("Found %d NaN scores, replacing with 0", nan_mask.sum())
        X[nan_mask] = 0.0

    # Ensure binary
    X = (X >= 0.5).astype(np.float64)

    return X, y


def build_ternary_feature_matrix(
    scored_df: pd.DataFrame,
    feature_names: List[str],
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ternary numpy arrays from a scored DataFrame.

    Returns (X, y) where X has columns matching feature_names with values 0, 1, or NaN.
    NaN indicates the feature is not applicable (NA) to that example.
    """
    missing = [f for f in feature_names if f not in scored_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = scored_df[feature_names].values.astype(np.float64)
    y = scored_df[label_column].values.astype(np.float64)

    # Binarize non-NaN values (preserve NaN)
    non_nan = ~np.isnan(X)
    X[non_nan] = (X[non_nan] >= 0.5).astype(np.float64)

    return X, y


def compute_na_rate(X: np.ndarray) -> np.ndarray:
    """Compute NA rate for each feature column.

    Parameters
    ----------
    X : ndarray of shape (n, K) with values 0, 1, or NaN

    Returns
    -------
    ndarray of shape (K,) with fraction of NaN per feature.
    """
    n = X.shape[0]
    if n == 0:
        return np.zeros(X.shape[1])
    return np.isnan(X).sum(axis=0) / n


def compute_mutual_information_ternary(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute MI between each feature and label, ignoring NaN entries.

    For each feature, MI is computed only over examples where the feature is
    not NaN. This gives the MI for the subpopulation where the feature applies.

    Parameters
    ----------
    X : ndarray of shape (n, K) with values 0, 1, or NaN
    y : ndarray of shape (n,) with values 0 or 1

    Returns
    -------
    ndarray of shape (K,) with MI values for each feature.
    """
    K = X.shape[1]
    mi_values = np.zeros(K)
    eps = 1e-10

    for k in range(K):
        mask = ~np.isnan(X[:, k])
        x = X[mask, k]
        y_k = y[mask]
        n_k = len(x)

        if n_k == 0:
            continue

        p_y1 = y_k.mean()
        p_y0 = 1 - p_y1
        p_x1 = x.mean()
        p_x0 = 1 - p_x1

        p_11 = ((x == 1) & (y_k == 1)).mean()
        p_10 = ((x == 1) & (y_k == 0)).mean()
        p_01 = ((x == 0) & (y_k == 1)).mean()
        p_00 = ((x == 0) & (y_k == 0)).mean()

        mi = 0.0
        for p_xy, p_x, p_y in [
            (p_11, p_x1, p_y1),
            (p_10, p_x1, p_y0),
            (p_01, p_x0, p_y1),
            (p_00, p_x0, p_y0),
        ]:
            if p_xy > eps and p_x > eps and p_y > eps:
                mi += p_xy * np.log(p_xy / (p_x * p_y))

        mi_values[k] = mi

    return mi_values


def rank_features_for_node(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    na_threshold: float = 0.05,
    k_min: int = 3,
    k_max: int = 6,
) -> List[int]:
    """Rank and select features for a tree node using NA-penalized MI.

    Only considers features with NA rate < na_threshold at this node.
    Ranks by MI * (1 - NA_rate). Returns between k_min and k_max feature indices.

    Parameters
    ----------
    X : ndarray of shape (n, M) with values 0, 1, or NaN
    y : ndarray of shape (n,) with values 0 or 1
    feature_names : list of M feature names (for logging)
    na_threshold : max NA rate to consider a feature applicable
    k_min : minimum features to select
    k_max : maximum features to select

    Returns
    -------
    List of column indices into X for selected features.
    """
    na_rates = compute_na_rate(X)
    mi_values = compute_mutual_information_ternary(X, y)

    # Score = MI * (1 - NA_rate), but only for features below NA threshold
    scores = np.zeros(X.shape[1])
    for k in range(X.shape[1]):
        if na_rates[k] < na_threshold:
            scores[k] = mi_values[k] * (1.0 - na_rates[k])
        else:
            scores[k] = -1.0  # excluded

    # Sort by score descending
    ranked = np.argsort(-scores)

    # Select features with positive scores, up to k_max
    selected = []
    for idx in ranked:
        if scores[idx] <= 0:
            break
        selected.append(int(idx))
        if len(selected) >= k_max:
            break

    if len(selected) < k_min:
        logger.info("Only %d features pass NA threshold (%.0f%%), needed %d",
                     len(selected), na_threshold * 100, k_min)

    for idx in selected:
        logger.info("  Selected feature %s: MI=%.4f, NA_rate=%.3f, score=%.4f",
                     feature_names[idx], mi_values[idx], na_rates[idx], scores[idx])

    return selected


def compute_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute mutual information between each binary feature and binary label.

    Parameters
    ----------
    X : ndarray of shape (n, K) with values 0 or 1
    y : ndarray of shape (n,) with values 0 or 1

    Returns
    -------
    ndarray of shape (K,) with MI values for each feature.
    """
    n = len(y)
    if n == 0:
        return np.zeros(X.shape[1])

    eps = 1e-10
    p_y1 = y.mean()
    p_y0 = 1 - p_y1

    mi_values = []
    for k in range(X.shape[1]):
        x = X[:, k]
        p_x1 = x.mean()
        p_x0 = 1 - p_x1

        # Joint probabilities
        p_11 = ((x == 1) & (y == 1)).mean()
        p_10 = ((x == 1) & (y == 0)).mean()
        p_01 = ((x == 0) & (y == 1)).mean()
        p_00 = ((x == 0) & (y == 0)).mean()

        mi = 0.0
        for p_xy, p_x, p_y in [
            (p_11, p_x1, p_y1),
            (p_10, p_x1, p_y0),
            (p_01, p_x0, p_y1),
            (p_00, p_x0, p_y0),
        ]:
            if p_xy > eps and p_x > eps and p_y > eps:
                mi += p_xy * np.log(p_xy / (p_x * p_y))

        mi_values.append(mi)

    return np.array(mi_values)


def compute_clustering_scores(
    X: np.ndarray,
) -> np.ndarray:
    """Score features for clustering quality: high entropy + low pairwise correlation.

    For clustering mode, we want features that:
    1. Split the data roughly evenly (high entropy, ideally ~50/50)
    2. Are not redundant with each other (low pairwise correlation)

    Parameters
    ----------
    X : ndarray of shape (n, K) with values 0 or 1

    Returns
    -------
    ndarray of shape (K,) with clustering quality scores (higher = better).
    """
    n, K = X.shape
    if n == 0 or K == 0:
        return np.zeros(K)

    eps = 1e-10

    # 1. Entropy of each feature (max = log(2) ≈ 0.693 at 50/50)
    entropies = np.zeros(K)
    for k in range(K):
        p = X[:, k].mean()
        if eps < p < 1 - eps:
            entropies[k] = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    # 2. Penalize features that are too skewed (>90% or <10% YES)
    #    Scale entropy by a penalty: 1.0 at 50/50, 0.0 at 0/100 or 100/0
    balance_scores = np.zeros(K)
    for k in range(K):
        p = X[:, k].mean()
        # Triangular penalty: peaks at 0.5, zero at 0 and 1
        balance_scores[k] = min(p, 1 - p) * 2  # 0 to 1, max at p=0.5

    # 3. Pairwise redundancy penalty: for each feature, penalize if it's
    #    highly correlated with any already-higher-scoring feature
    #    Use absolute Pearson correlation
    corr_penalty = np.ones(K)
    if K > 1:
        # Compute pairwise absolute correlations
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds = np.where(stds < eps, 1.0, stds)
        X_norm = (X - means) / stds
        corr_matrix = np.abs(X_norm.T @ X_norm / n)
        np.fill_diagonal(corr_matrix, 0.0)
        # Penalty = 1 - max correlation with any other feature
        max_corr = corr_matrix.max(axis=1)
        corr_penalty = 1.0 - 0.5 * max_corr  # light penalty

    scores = entropies * balance_scores * corr_penalty

    return scores


def select_clustering_features(
    X: np.ndarray,
    candidate_names: list,
    K: int,
) -> list:
    """Greedy selection of K features optimized for clustering.

    Selects features one at a time, each time picking the feature with
    highest entropy that is least correlated with already-selected features.

    Returns list of selected column indices.
    """
    n, n_candidates = X.shape
    if n == 0 or n_candidates == 0:
        return list(range(min(K, n_candidates)))

    eps = 1e-10
    selected = []
    remaining = list(range(n_candidates))

    for _ in range(min(K, n_candidates)):
        best_idx = None
        best_score = -1.0

        for idx in remaining:
            p = X[:, idx].mean()
            # Entropy
            if eps < p < 1 - eps:
                entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            else:
                entropy = 0.0
            # Balance bonus
            balance = min(p, 1 - p) * 2

            # Redundancy penalty: max abs correlation with already-selected
            redundancy = 0.0
            if selected:
                x_new = X[:, idx]
                for s_idx in selected:
                    x_sel = X[:, s_idx]
                    # Pearson correlation
                    cov = ((x_new - x_new.mean()) * (x_sel - x_sel.mean())).mean()
                    std_new = x_new.std()
                    std_sel = x_sel.std()
                    if std_new > eps and std_sel > eps:
                        corr = abs(cov / (std_new * std_sel))
                        redundancy = max(redundancy, corr)

            score = entropy * balance * (1.0 - 0.7 * redundancy)
            logger.debug("  Clustering candidate %s: entropy=%.3f balance=%.2f redundancy=%.2f score=%.4f",
                         candidate_names[idx], entropy, balance, redundancy, score)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
            logger.info("  Clustering selected [%d]: %s (score=%.4f)",
                        len(selected), candidate_names[best_idx], best_score)

    return selected
