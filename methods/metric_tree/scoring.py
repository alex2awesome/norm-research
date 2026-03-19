"""Scoring bridge: converts TreeMetric objects to autometrics MetricSpec and delegates scoring."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autometrics.iterative_refinement.runner import (
    MetricSpec,
    _build_feature_frame,
    _build_metric_from_parts,
    _metric_id_from_rubric,
    _normalize_rubric,
    _rubric_to_text,
)
from autometrics.iterative_refinement.label_cache import LabelCache

from .data_structures import TreeMetric
from .token_utils import compute_scoring_text_budget, truncate_to_tokens

logger = logging.getLogger("metric_tree.scoring")


def tree_metric_to_metric_spec(
    tm: TreeMetric,
    *,
    judge_llm: Any,
    task_description: str,
    existing_names: set,
    scoring_backend: Any = None,
) -> MetricSpec:
    """Convert a TreeMetric to an autometrics MetricSpec for scoring."""
    return _build_metric_from_parts(
        name=tm.name,
        rubric=tm.rubric,
        rubric_text=tm.rubric_text,
        judge_llm=judge_llm,
        task_description=task_description,
        has_references=False,
        existing_names=existing_names,
        metric_id=tm.metric_id,
        scoring_backend=scoring_backend,
    )


def score_subset(
    df: pd.DataFrame,
    indices: np.ndarray,
    metrics: List[TreeMetric],
    label_cache: LabelCache,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    judge_llm: Any,
    task_description: str,
    batch_size: int = 200,
    verbose: bool = False,
    stage: str = "",
    scoring_backend: Any = None,
    max_text_tokens: int = 0,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> pd.DataFrame:
    """Score a subset of examples on the given TreeMetrics.

    Converts TreeMetric → MetricSpec, then delegates to _build_feature_frame.
    Returns a DataFrame with id, label, and one column per metric.

    Token truncation:
      - If max_model_len > 0 and tokenizer are provided, computes the exact
        per-document budget from the *actual* rubric text of the metrics being
        scored, overriding any passed-in max_text_tokens.
      - Otherwise falls back to max_text_tokens if > 0.
    """
    subset_df = df.iloc[indices].reset_index(drop=True) if len(indices) < len(df) else df.copy()

    # Dynamic budget: measure actual rubrics to get exact text budget.
    if max_model_len > 0 and tokenizer is not None:
        max_text_tokens = compute_scoring_text_budget(
            metrics, task_description, max_model_len, tokenizer,
        )

    # Truncate text column in token space to fit VLLM context window.
    if max_text_tokens > 0 and text_column in subset_df.columns:
        subset_df[text_column] = subset_df[text_column].apply(
            lambda t: truncate_to_tokens(str(t), max_text_tokens, tokenizer=tokenizer),
        )

    existing_names: set = set()
    metric_specs = []
    for tm in metrics:
        spec = tree_metric_to_metric_spec(
            tm,
            judge_llm=judge_llm,
            task_description=task_description,
            existing_names=existing_names,
            scoring_backend=scoring_backend,
        )
        metric_specs.append(spec)

    scored_df = _build_feature_frame(
        df=subset_df,
        metrics=metric_specs,
        label_cache=label_cache,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        batch_size=batch_size,
        verbose=verbose,
        stage=stage,
        score_all_metrics_together=True,
        judge_llm=judge_llm,
        task_description=task_description,
        scoring_backend=scoring_backend,
    )

    # Preserve text column — needed downstream by contrastive pair generation.
    if text_column not in scored_df.columns and text_column in subset_df.columns:
        scored_df = scored_df.merge(
            subset_df[[id_column, text_column]].astype({id_column: str}),
            on=id_column,
            how="left",
        )

    return scored_df


def build_feature_matrix(
    scored_df: pd.DataFrame,
    feature_names: List[str],
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract numpy arrays from a scored DataFrame.

    Returns (X, y) where X has columns matching feature_names.
    """
    missing = [f for f in feature_names if f not in scored_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = scored_df[feature_names].values.astype(np.float64)
    y = scored_df[label_column].values.astype(np.float64)

    # Handle NaN scores (default to 0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        logger.warning("Found %d NaN scores, replacing with 0", nan_mask.sum())
        X[nan_mask] = 0.0

    return X, y


def add_interaction_features(
    X: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
    """Add pairwise product interaction features to X.

    Returns (X_augmented, augmented_feature_names, interaction_pairs).
    """
    n_features = X.shape[1]
    interaction_cols = []
    interaction_names = []
    interaction_pairs = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            product = X[:, i] * X[:, j]
            interaction_cols.append(product)
            name = f"{feature_names[i]}__x__{feature_names[j]}"
            interaction_names.append(name)
            interaction_pairs.append((feature_names[i], feature_names[j]))

    if interaction_cols:
        X_interactions = np.column_stack(interaction_cols)
        X_augmented = np.hstack([X, X_interactions])
        all_names = feature_names + interaction_names
    else:
        X_augmented = X
        all_names = list(feature_names)

    return X_augmented, all_names, interaction_pairs
