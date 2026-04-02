"""Prediction through Partitioned Metric Trees — deterministic routing + base-rate leaves.

When use_router is enabled, per-node text classifiers gate which examples
continue deeper vs. stop at the current node's base-rate prediction.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autometrics.iterative_refinement.label_cache import LabelCache

from .data_structures import MetricTree, PartitionTreeNode
from .partition import assign_to_partitions, _format_key
from .router import predict_router
from .scoring import build_binary_feature_matrix, score_binary_subset

logger = logging.getLogger("metric_tree.inference")


def _predict_at_node(
    node: PartitionTreeNode,
    n: int,
    all_scores: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using the node's leaf regression model if available, else base rate.

    If a leaf regression model was fitted and all_scores are provided,
    use logistic regression on accumulated binary features.
    Otherwise, fall back to base rate prediction.

    Returns (predictions, probabilities).
    """
    if node.leaf_model is not None and all_scores is not None and all_scores.shape[0] == n:
        try:
            useful_cols = node.leaf_model._useful_cols
            X = all_scores[:, useful_cols]
            probs = node.leaf_model.predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)
            return preds, probs
        except Exception as e:
            logger.warning("Leaf model prediction failed at %s: %s, falling back to base rate",
                          node.node_id, e)

    preds = np.ones(n, dtype=int) if node.base_rate >= 0.5 else np.zeros(n, dtype=int)
    probs = np.full(n, node.base_rate)
    return preds, probs


def _resolve_at_node(
    node: PartitionTreeNode,
    global_idx: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    resolving_nodes: list,
    all_scores: np.ndarray = None,
) -> None:
    """Predict at a node and store results for the given global indices."""
    preds, probs = _predict_at_node(node, len(global_idx), all_scores=all_scores)
    predictions[global_idx] = preds
    probabilities[global_idx] = probs
    node_id = node.node_id
    for idx in global_idx:
        resolving_nodes[idx] = node_id


def _apply_router_gate(
    node: PartitionTreeNode,
    child: PartitionTreeNode,
    actual_global_idx: np.ndarray,
    df: pd.DataFrame,
    text_column: str,
    router_threshold: float,
    embedding_model_name: str,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    resolving_nodes: list,
    queue: deque,
    accum_scores: np.ndarray = None,
) -> None:
    """Apply router gating: examples above threshold continue, others stop."""
    texts = df.iloc[actual_global_idx][text_column].astype(str).tolist()
    minority_probs = predict_router(
        node.router, texts,
        embedding_model_name=embedding_model_name,
    )

    continue_mask = minority_probs > router_threshold
    continue_idx = actual_global_idx[continue_mask]
    stop_idx = actual_global_idx[~continue_mask]

    if len(continue_idx) > 0:
        if accum_scores is not None:
            queue.append((child, continue_idx, accum_scores[continue_mask]))
        else:
            queue.append((child, continue_idx, np.empty((len(continue_idx), 0))))
    if len(stop_idx) > 0:
        # Stopped examples get the child's base-rate prediction
        # (they landed in this partition but the router says they're majority-class)
        resolve_node = child
        stop_scores = accum_scores[~continue_mask] if accum_scores is not None else None
        _resolve_at_node(resolve_node, stop_idx, predictions, probabilities, resolving_nodes,
                        all_scores=stop_scores)

    logger.debug(
        "Router at %s: %d/%d continue (threshold=%.2f)",
        node.node_id, len(continue_idx), len(actual_global_idx), router_threshold,
    )


def predict_batch(
    tree: MetricTree,
    df: pd.DataFrame,
    label_cache: LabelCache,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    task_description: str,
    scoring_backend: Any,
    batch_size: int = 200,
    verbose: bool = False,
    max_model_len: int = 0,
    tokenizer: Any = None,
    use_router: Optional[bool] = None,
    router_threshold: Optional[float] = None,
    # Legacy params (ignored, kept for API compatibility)
    judge_llm: Any = None,
) -> pd.DataFrame:
    """Batch prediction through the partition tree.

    1. Score all examples on root's K metrics -> binary
    2. Assign to partitions
    3. BFS: for each (node, partition_key, indices):
       - Look up child = node.children.get(partition_key)
       - If child exists and not leaf: optionally apply router gate, then
         score on child's metrics, assign to sub-partitions, enqueue
       - If child is leaf or missing: predict via base rate
    4. Return DataFrame with id, prediction, probability, resolving_node

    When use_router=True, per-node text classifiers filter which examples
    continue deeper. Examples where p(minority | text) <= threshold stop
    at the current node's base-rate prediction.
    """
    # Resolve router config
    cfg = tree.config
    if use_router is None:
        use_router = getattr(cfg, "use_router", False) if cfg else False
    if router_threshold is None:
        router_threshold = getattr(cfg, "router_threshold", 0.5) if cfg else 0.5
    embedding_model_name = getattr(cfg, "embedding_model", "all-MiniLM-L6-v2") if cfg else "all-MiniLM-L6-v2"

    if use_router:
        logger.info("Router-gated inference enabled (threshold=%.2f)", router_threshold)

    n = len(df)
    predictions = np.zeros(n, dtype=int)
    probabilities = np.full(n, 0.5)
    resolving_nodes = [""] * n

    root = tree.root
    if root is None:
        return pd.DataFrame({
            id_column: df[id_column].values,
            "prediction": predictions,
            "probability": probabilities,
            "resolving_node": resolving_nodes,
        })

    # Score all examples on root's metrics
    logger.info("Scoring %d examples on root metrics (%d)...", n, len(root.local_metrics))
    root_scored = score_binary_subset(
        df, np.arange(n), root.local_metrics, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        task_description=task_description, scoring_backend=scoring_backend,
        batch_size=batch_size, verbose=verbose,
        stage="predict_root", tokenizer=tokenizer, max_model_len=max_model_len,
    )

    root_metric_names = [m.name for m in root.local_metrics]
    X_root, _ = build_binary_feature_matrix(root_scored, root_metric_names, label_column)

    # BFS queue: (node, global_indices, accumulated_scores)
    # accumulated_scores tracks all binary features scored so far for each example
    queue: deque[Tuple[PartitionTreeNode, np.ndarray, np.ndarray]] = deque()

    # Assign to root partitions
    root_partitions = assign_to_partitions(X_root)

    for p_key, local_idx in root_partitions.items():
        child = root.children.get(p_key)
        accum_scores = X_root[local_idx]  # root-level scores for these examples
        if child is not None and not child.is_leaf and child.local_metrics:
            # Router gate at root level
            if use_router and root.router is not None:
                _apply_router_gate(
                    root, child, local_idx, df, text_column,
                    router_threshold, embedding_model_name,
                    predictions, probabilities, resolving_nodes, queue,
                )
            else:
                queue.append((child, local_idx, accum_scores))
        else:
            resolve_node = child if child is not None else root
            _resolve_at_node(resolve_node, local_idx, predictions, probabilities, resolving_nodes,
                           all_scores=accum_scores)

    # BFS through deeper levels
    while queue:
        node, global_idx, parent_accum_scores = queue.popleft()

        if len(global_idx) == 0:
            continue

        # Score on this node's local metrics
        node_df = df.iloc[global_idx].reset_index(drop=True)
        node_scored = score_binary_subset(
            node_df, np.arange(len(node_df)), node.local_metrics, label_cache,
            id_column=id_column, text_column=text_column, label_column=label_column,
            task_description=task_description, scoring_backend=scoring_backend,
            batch_size=batch_size, verbose=verbose,
            stage=f"predict_{node.node_id}", tokenizer=tokenizer, max_model_len=max_model_len,
        )

        local_names = [m.name for m in node.local_metrics]
        X_local, _ = build_binary_feature_matrix(node_scored, local_names, label_column)

        # Accumulate: concatenate parent's accumulated scores with this node's local scores
        node_accum_scores = np.hstack([parent_accum_scores, X_local])

        # Assign to partitions based on THIS node's local scores
        node_partitions = assign_to_partitions(X_local)

        for p_key, local_idx in node_partitions.items():
            actual_global_idx = global_idx[local_idx]
            child_accum = node_accum_scores[local_idx]
            child = node.children.get(p_key)

            if child is not None and not child.is_leaf and child.local_metrics:
                # Router gate: filter which examples continue deeper
                if use_router and node.router is not None:
                    _apply_router_gate(
                        node, child, actual_global_idx, df, text_column,
                        router_threshold, embedding_model_name,
                        predictions, probabilities, resolving_nodes, queue,
                    )
                else:
                    queue.append((child, actual_global_idx, child_accum))
            else:
                resolve_node = child if child is not None else node
                _resolve_at_node(
                    resolve_node, actual_global_idx,
                    predictions, probabilities, resolving_nodes,
                    all_scores=child_accum,
                )

    # Handle any unresolved (shouldn't happen with fallback, but just in case)
    unresolved = np.array([rn == "" for rn in resolving_nodes])
    if unresolved.any():
        logger.warning("%d examples unresolved, using root prediction", unresolved.sum())
        preds, probs = _predict_at_node(root, int(unresolved.sum()))
        predictions[unresolved] = preds
        probabilities[unresolved] = probs
        for idx in np.where(unresolved)[0]:
            resolving_nodes[idx] = "root_fallback"

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


def predict_root_only(
    tree: MetricTree,
    df: pd.DataFrame,
    label_cache: LabelCache,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    task_description: str,
    scoring_backend: Any,
    batch_size: int = 200,
    verbose: bool = False,
    max_model_len: int = 0,
    tokenizer: Any = None,
    judge_llm: Any = None,
) -> np.ndarray:
    """Classify ALL examples using only the root node's base rate per partition.

    Used for computing the articulability gap. No router gating here —
    this is the baseline that uses only root-level features.
    Returns array of predictions (0 or 1).
    """
    root = tree.root
    if root is None:
        return np.zeros(len(df), dtype=int)

    scored_df = score_binary_subset(
        df, np.arange(len(df)), root.local_metrics, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        task_description=task_description, scoring_backend=scoring_backend,
        batch_size=batch_size, verbose=verbose,
        stage="root_only_predict", tokenizer=tokenizer, max_model_len=max_model_len,
    )

    metric_names = [m.name for m in root.local_metrics]
    X, _ = build_binary_feature_matrix(scored_df, metric_names, label_column)

    # Per-partition base rate prediction
    n = len(df)
    predictions = np.zeros(n, dtype=int)
    partitions = assign_to_partitions(X)
    for p_key, local_idx in partitions.items():
        child = root.children.get(p_key)
        resolve_node = child if child is not None else root
        preds, _ = _predict_at_node(resolve_node, len(local_idx))
        predictions[local_idx] = preds

    return predictions
