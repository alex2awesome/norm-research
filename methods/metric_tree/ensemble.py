"""Metric Tree ensemble: build multiple trees with varying seeds/temperatures."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from autometrics.generator.ContrastiveRubricProposer import ContrastiveRubricProposer
from autometrics.iterative_refinement.label_cache import LabelCache

from .config import TreeConfig
from .data_structures import MetricTree
from .inference import predict_batch
from .tree_builder import build_metric_tree

logger = logging.getLogger("metric_tree.ensemble")


def build_metric_tree_ensemble(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TreeConfig,
    proposer: ContrastiveRubricProposer,
    task_description: str,
    *,
    n_trees: int = 3,
    id_column: str,
    text_column: str,
    label_column: str,
    judge_llm: Any,
    cache_dir: Optional[str] = None,
    scoring_backend: Any = None,
    tokenizer: Any = None,
    token_budgets: Optional[dict] = None,
) -> List[MetricTree]:
    """Build an ensemble of Metric Trees with varying seeds and temperatures.

    All trees share a single LabelCache directory for cross-tree caching.
    """
    if cache_dir is None:
        cache_dir = str(Path(config.output_dir) / "label_cache")

    trees = []
    for i in range(n_trees):
        tree_config = deepcopy(config)
        tree_config.random_seed = config.random_seed + i * 17
        tree_config.llm_temperature = config.llm_temperature + i * 0.1

        logger.info("Building tree %d/%d (seed=%d, temp=%.2f)...",
                    i + 1, n_trees, tree_config.random_seed, tree_config.llm_temperature)

        tree = build_metric_tree(
            train_df=train_df,
            eval_df=eval_df,
            config=tree_config,
            proposer=proposer,
            task_description=task_description,
            id_column=id_column,
            text_column=text_column,
            label_column=label_column,
            judge_llm=judge_llm,
            cache_dir=cache_dir,  # shared cache
            scoring_backend=scoring_backend,
            tokenizer=tokenizer,
            token_budgets=token_budgets,
        )
        trees.append(tree)
        logger.info("Tree %d: %d nodes, %d metrics",
                    i + 1, len(tree.all_nodes), len(tree.all_metrics))

    return trees


def ensemble_predict(
    trees: List[MetricTree],
    df: pd.DataFrame,
    label_cache: LabelCache,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    judge_llm: Any,
    task_description: str,
    batch_size: int = 200,
    scoring_backend: Any = None,
    verbose: bool = False,
    max_model_len: int = 0,
    tokenizer: Any = None,
) -> pd.DataFrame:
    """Aggregate predictions from multiple trees via majority vote.

    Returns DataFrame with columns: id, prediction, probability,
    agreement (fraction of trees that agree), individual predictions.
    """
    all_predictions = []
    all_probs = []

    for i, tree in enumerate(trees):
        logger.info("Predicting with tree %d/%d...", i + 1, len(trees))
        result = predict_batch(
            tree=tree,
            df=df,
            label_cache=label_cache,
            id_column=id_column,
            text_column=text_column,
            label_column=label_column,
            judge_llm=judge_llm,
            task_description=task_description,
            batch_size=batch_size,
            scoring_backend=scoring_backend,
            verbose=verbose,
            max_model_len=max_model_len,
            tokenizer=tokenizer,
        )
        all_predictions.append(result["prediction"].values)
        all_probs.append(result["probability"].values)

    # Stack: shape (n_trees, n_examples)
    pred_matrix = np.stack(all_predictions, axis=0)
    prob_matrix = np.stack(all_probs, axis=0)

    # Confidence-weighted majority vote: each tree's vote is weighted by its
    # confidence (distance from 0.5). A tree that says P(1)=0.95 contributes
    # more than one that says P(1)=0.51.
    confidence_weights = np.abs(prob_matrix - 0.5)  # shape (n_trees, n_examples)
    # Weighted vote toward class 1: sum(w_i * pred_i) / sum(w_i)
    weight_sum = confidence_weights.sum(axis=0)
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)  # avoid division by zero
    weighted_vote = (confidence_weights * pred_matrix).sum(axis=0) / weight_sum
    ensemble_preds = (weighted_vote >= 0.5).astype(int)
    ensemble_probs = prob_matrix.mean(axis=0)

    # Agreement: fraction of trees that agree with the ensemble prediction
    agreement = np.array([
        (pred_matrix[:, j] == ensemble_preds[j]).mean()
        for j in range(len(ensemble_preds))
    ])

    result_df = pd.DataFrame({
        id_column: df[id_column].values,
        "prediction": ensemble_preds,
        "probability": ensemble_probs,
        "agreement": agreement,
    })

    if verbose:
        logger.info("Ensemble agreement: mean=%.3f, min=%.3f, max=%.3f",
                    agreement.mean(), agreement.min(), agreement.max())

    return result_df
