"""Score examples on canonical features using the existing VLLM binary scoring pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .rubric_generation import CanonicalFeature

logger = logging.getLogger(__name__)


def _canonical_to_tree_metric(cf: CanonicalFeature):
    """Convert a CanonicalFeature to a TreeMetric for binary scoring."""
    from methods.metric_tree.data_structures import TreeMetric

    rubric_text = f"{cf.name}\n  YES: {cf.rubric['yes']}\n  NO: {cf.rubric['no']}"
    return TreeMetric(
        metric_id=cf.feature_id,
        name=cf.name,
        rubric_text=rubric_text,
        rubric=cf.rubric,
        source_node_id="local_explanations",
        scale="binary",
    )


def score_examples_on_vocabulary(
    df: pd.DataFrame,
    canonical_features: List[CanonicalFeature],
    scoring_backend: Any,
    task_description: str,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    label_cache: Optional[Any] = None,
    label_cache_dir: Optional[str] = None,
    tokenizer: Any = None,
    max_model_len: int = 0,
    batch_size: int = 8192,
    stage: str = "",
) -> pd.DataFrame:
    """Score all examples on canonical binary features.

    Converts CanonicalFeature -> TreeMetric, then calls the metric tree's
    binary scoring infrastructure.

    Returns: DataFrame with columns [id_column, label_column, feature_name_1, ..., feature_name_K]
             where feature values are 0 or 1.
    """
    from methods.metric_tree.scoring import score_binary_subset
    from autometrics.iterative_refinement.label_cache import LabelCache

    tree_metrics = [_canonical_to_tree_metric(cf) for cf in canonical_features]

    if label_cache is None:
        cache_dir = label_cache_dir or "/tmp/local_explanations_score_cache"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        label_cache = LabelCache(cache_dir)

    indices = np.arange(len(df))

    logger.info(
        "Scoring %d examples on %d canonical features",
        len(df), len(canonical_features),
    )

    # score_binary_subset returns a DataFrame keyed by id+label with one
    # column per metric.name (0/1 values).
    scored_df = score_binary_subset(
        df=df,
        indices=indices,
        metrics=tree_metrics,
        label_cache=label_cache,
        scoring_backend=scoring_backend,
        task_description=task_description,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        batch_size=batch_size,
        verbose=True,
        stage=stage or "local_explanations",
    )

    # Rename columns to safe names (strip spaces, cap length) — matches what
    # downstream code expects for the L1 LR.
    rename = {}
    for cf in canonical_features:
        if cf.name in scored_df.columns:
            rename[cf.name] = cf.name.replace(" ", "_")[:60]
    if rename:
        scored_df = scored_df.rename(columns=rename)

    return scored_df
