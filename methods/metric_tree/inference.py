"""Prediction through built Metric Trees."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autometrics.iterative_refinement.label_cache import LabelCache

from .data_structures import MetricTree, MetricTreeNode
from .scoring import (
    add_interaction_features,
    build_feature_matrix,
    score_subset,
)

logger = logging.getLogger("metric_tree.inference")


def _predict_at_node(
    node: MetricTreeNode,
    scored_df: pd.DataFrame,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a node's classifier to pre-scored data.

    Returns (predictions, probabilities).
    """
    base_metric_names = [m.name for m in node.all_metrics]
    available = [n for n in base_metric_names if n in scored_df.columns]

    if not available:
        raise ValueError(f"No metric columns found in scored_df for node {node.node_id}")

    X, _ = build_feature_matrix(scored_df, available, label_column)

    # Add interactions if needed
    if any("__x__" in fn for fn in node.feature_names):
        X_aug, aug_names, _ = add_interaction_features(X, available)
        name_set = set(node.feature_names)
        col_mask = [i for i, n in enumerate(aug_names) if n in name_set]
        X_final = X_aug[:, col_mask]
    else:
        name_set = set(node.feature_names)
        col_mask = [i for i, n in enumerate(available) if n in name_set]
        X_final = X[:, col_mask]

    X_scaled = node.scaler.transform(X_final)
    probs = node.classifier.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return preds, probs


def predict_single(
    tree: MetricTree,
    scored_row: Dict[str, float],
    label_column: str = "label",
) -> Tuple[int, float, str]:
    """Predict for a single example that has already been scored on all metrics.

    Returns (prediction, probability, resolving_node_id).
    """
    node = tree.root
    df = pd.DataFrame([scored_row])
    df[label_column] = 0  # dummy label for matrix building

    while node is not None:
        preds, probs = _predict_at_node(node, df, label_column)
        pred, prob = int(preds[0]), float(probs[0])
        confidence = max(prob, 1 - prob)

        # Check if resolved at this node
        if not node.children or confidence >= node.confidence_threshold:
            return pred, prob, node.node_id

        # Route to appropriate child
        if node.router is not None:
            # Learned routing: check if router thinks parent is correct
            base_names = [m.name for m in node.all_metrics]
            X, _ = build_feature_matrix(df, base_names, label_column)
            X_scaled = node.scaler.transform(X)
            router_prob = node.router.predict_proba(X_scaled)[0, 1]
            if router_prob >= 0.5:
                # Router thinks parent is correct, resolve here
                return pred, prob, node.node_id

        # Route based on prediction direction
        if pred == 1 and "false_positive" in node.children:
            node = node.children["false_positive"]
        elif pred == 0 and "false_negative" in node.children:
            node = node.children["false_negative"]
        elif "misclassified" in node.children:
            node = node.children["misclassified"]
        else:
            # No matching child, resolve here
            return pred, prob, node.node_id

    return pred, prob, node.node_id


def predict_root_only(
    tree: MetricTree,
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
) -> np.ndarray:
    """Classify ALL examples using only the root node's classifier.

    Used for computing the articulability gap: root-only accuracy vs. full tree.
    Returns array of predictions (0 or 1).
    """
    root = tree.root
    all_metrics = list(root.all_metrics)

    scored_df = score_subset(
        df, np.arange(len(df)), all_metrics, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        judge_llm=judge_llm, task_description=task_description,
        batch_size=batch_size, verbose=verbose,
        stage="root_only_predict", scoring_backend=scoring_backend,
        tokenizer=tokenizer, max_model_len=max_model_len,
    )

    preds, _ = _predict_at_node(root, scored_df, label_column)
    return preds


def predict_batch(
    tree: MetricTree,
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
    """Efficient batch prediction through the tree.

    1. Score all examples on root metrics → classify
    2. Partition by confidence → resolved at root vs. routed to children
    3. Score routed examples on child's local metrics (parent scores cached)
    4. Repeat until all resolved or at leaf

    Returns DataFrame with columns: id, prediction, probability, resolving_node.
    """
    n = len(df)
    predictions = np.zeros(n, dtype=int)
    probabilities = np.zeros(n, dtype=float)
    resolving_nodes = [""] * n

    # Track which examples are still unresolved
    unresolved = np.ones(n, dtype=bool)
    indices = np.arange(n)

    # BFS through tree levels
    queue: List[Tuple[MetricTreeNode, np.ndarray]] = [(tree.root, indices)]

    while queue:
        next_queue: List[Tuple[MetricTreeNode, np.ndarray]] = []

        for node, node_indices in queue:
            if len(node_indices) == 0:
                continue

            node_df = df.iloc[node_indices].reset_index(drop=True)

            # Score on this node's metrics
            all_metrics = list(node.all_metrics)
            scored_df = score_subset(
                node_df, np.arange(len(node_df)), all_metrics, label_cache,
                id_column=id_column, text_column=text_column, label_column=label_column,
                judge_llm=judge_llm, task_description=task_description,
                batch_size=batch_size, verbose=verbose,
                stage=f"predict_{node.node_id}", scoring_backend=scoring_backend,
                tokenizer=tokenizer, max_model_len=max_model_len,
            )

            # Predict at this node
            preds, probs = _predict_at_node(node, scored_df, label_column)
            confidence = np.maximum(probs, 1 - probs)

            # Determine resolved vs. unresolved
            if not node.children:
                # Leaf node: everything resolves here
                resolved_local = np.ones(len(node_indices), dtype=bool)
            else:
                resolved_local = confidence >= node.confidence_threshold

                # Learned router override
                if node.router is not None:
                    base_names = [m.name for m in node.all_metrics]
                    avail = [n_name for n_name in base_names if n_name in scored_df.columns]
                    X, _ = build_feature_matrix(scored_df, avail, label_column)
                    X_scaled = node.scaler.transform(X)
                    router_correct_probs = node.router.predict_proba(X_scaled)[:, 1]
                    # Override: router-confident examples are also resolved
                    resolved_local = resolved_local | (router_correct_probs >= 0.5)

            # Record resolved predictions
            resolved_global = node_indices[resolved_local]
            predictions[resolved_global] = preds[resolved_local]
            probabilities[resolved_global] = probs[resolved_local]
            for idx in resolved_global:
                resolving_nodes[idx] = node.node_id
            unresolved[resolved_global] = False

            # Route unresolved to children
            unresolved_local = ~resolved_local
            if unresolved_local.any() and node.children:
                unresolved_indices = node_indices[unresolved_local]
                unresolved_preds = preds[unresolved_local]

                for child_type, child_node in node.children.items():
                    if child_type == "false_positive":
                        child_mask = unresolved_preds == 1
                    elif child_type == "false_negative":
                        child_mask = unresolved_preds == 0
                    elif child_type == "misclassified":
                        child_mask = np.ones(len(unresolved_indices), dtype=bool)
                    else:
                        continue

                    child_indices = unresolved_indices[child_mask]
                    if len(child_indices) > 0:
                        next_queue.append((child_node, child_indices))

        queue = next_queue

    # Any remaining unresolved get root prediction
    still_unresolved = unresolved
    if still_unresolved.any():
        logger.warning("%d examples unresolved after tree traversal, using root prediction", still_unresolved.sum())
        # These should have been scored at some point, use last available
        for idx in indices[still_unresolved]:
            predictions[idx] = 0
            probabilities[idx] = 0.5
            resolving_nodes[idx] = "unresolved"

    result = pd.DataFrame({
        id_column: df[id_column].values,
        "prediction": predictions,
        "probability": probabilities,
        "resolving_node": resolving_nodes,
    })

    if verbose:
        node_counts = result["resolving_node"].value_counts()
        logger.info("Resolution distribution:\n%s", node_counts.to_string())

    return result
