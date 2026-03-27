"""Iterative tree restructuring over a global ternary score matrix.

Pipeline:
1. Build initial greedy tree (existing tree_builder)
2. Score ALL metrics on ALL examples (ternary: YES/NO/NA)
3. Deduplicate metrics (embedding similarity + LLM consolidation)
4. Rebuild tree over full score matrix using NA-aware greedy algorithm
5. Gap-fill: generate new metrics for partitions lacking good features
6. Repeat 2-5, saving each iteration and evaluating on held-out data
7. Stop when eval accuracy plateaus
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from autometrics.iterative_refinement.label_cache import LabelCache
from autometrics.iterative_refinement.runner import _coerce_binary_labels

from .config import TreeConfig
from .data_structures import MetricTree, PartitionTreeNode, TreeMetric
from .partition import assign_to_partitions, count_contrastive_pairs, _format_key
from .scoring import (
    score_ternary_subset,
    build_ternary_feature_matrix,
    compute_mutual_information_ternary,
    compute_na_rate,
    rank_features_for_node,
    score_binary_subset,
    build_binary_feature_matrix,
)
from .inference import predict_batch
from .serialization import save_tree

logger = logging.getLogger("metric_tree.restructure")


# ── Tree Rebuilding ──────────────────────────────────────────────────────────


def _compute_base_rate(labels: np.ndarray) -> float:
    """Compute base rate (fraction positive) from binary label array."""
    n = len(labels)
    if n == 0:
        return 0.5
    return float((labels == 1).sum()) / n


def rebuild_tree_from_scores(
    score_matrix: np.ndarray,
    labels: np.ndarray,
    all_metrics: List[TreeMetric],
    config: TreeConfig,
    task_description: str = "",
) -> MetricTree:
    """Build a tree from a pre-scored (examples × metrics) ternary matrix.

    Parameters
    ----------
    score_matrix : ndarray of shape (N, M) with values 0, 1, or NaN
    labels : ndarray of shape (N,) with values 0 or 1
    all_metrics : list of M TreeMetric objects (column order matches score_matrix)
    config : TreeConfig
    task_description : str

    Returns
    -------
    MetricTree with optimal structure over the given scores.
    """
    N, M = score_matrix.shape
    metric_names = [m.name for m in all_metrics]

    all_nodes: Dict[str, PartitionTreeNode] = {}
    all_metrics_dict: Dict[str, TreeMetric] = {m.metric_id: m for m in all_metrics}

    def _build_node(
        example_idx: np.ndarray,
        available_features: List[int],  # indices into score_matrix columns
        depth: int,
        parent_id: Optional[str],
        partition_key: Tuple[int, ...],
        node_id_prefix: str,
    ) -> PartitionTreeNode:
        """Recursively build a node of the restructured tree."""
        n = len(example_idx)
        node_labels = labels[example_idx]
        base_rate = _compute_base_rate(node_labels)
        n_pos = int((node_labels == 1).sum())
        n_neg = n - n_pos

        node_id = node_id_prefix if node_id_prefix else "root"

        # Base cases
        is_leaf = False
        if depth >= config.max_depth:
            is_leaf = True
        elif n < config.min_partition_size:
            is_leaf = True
        elif count_contrastive_pairs(node_labels) < config.min_contrastive_pairs:
            is_leaf = True

        # Check minority fraction pruning
        if not is_leaf and config.min_minority_fraction > 0:
            minority_frac = min(base_rate, 1 - base_rate)
            if minority_frac < config.min_minority_fraction:
                is_leaf = True

        # Select features for this node
        selected_indices = []
        if not is_leaf and available_features:
            X_avail = score_matrix[np.ix_(example_idx, available_features)]
            selected_indices = rank_features_for_node(
                X_avail, node_labels,
                feature_names=[metric_names[i] for i in available_features],
                na_threshold=config.restructure_na_threshold,
                k_min=config.restructure_k_min,
                k_max=config.restructure_k_max,
            )
            # Map back to global feature indices
            selected_global = [available_features[i] for i in selected_indices]

            if len(selected_indices) < config.restructure_k_min:
                is_leaf = True
                selected_global = []
        else:
            selected_global = []

        # Build node
        local_metrics = [all_metrics[i] for i in selected_global]
        # all_metrics for this node = parent's all_metrics + this node's local_metrics
        # (we reconstruct this from the tree structure)

        node = PartitionTreeNode(
            node_id=node_id,
            depth=depth,
            parent_id=parent_id,
            partition_key=partition_key,
            local_metrics=local_metrics,
            all_metrics=local_metrics,  # will be filled in later
            point_indices=example_idx,
            local_scores=score_matrix[np.ix_(example_idx, selected_global)] if selected_global else np.empty((n, 0)),
            all_scores=np.empty((n, 0)),  # filled in later
            base_rate=base_rate,
            n_positive=n_pos,
            n_negative=n_neg,
            is_leaf=is_leaf,
        )
        all_nodes[node_id] = node

        if is_leaf:
            return node

        # Partition on selected features (NaN → impute to mode for partitioning)
        X_local = score_matrix[np.ix_(example_idx, selected_global)].copy()
        # Impute NaN to column mode (should be very rare given NA threshold)
        for col in range(X_local.shape[1]):
            nan_mask = np.isnan(X_local[:, col])
            if nan_mask.any():
                non_nan = X_local[~nan_mask, col]
                mode = 1.0 if non_nan.mean() >= 0.5 else 0.0
                X_local[nan_mask, col] = mode

        X_local_binary = (X_local >= 0.5).astype(int)
        partitions = assign_to_partitions(X_local_binary)

        # Remaining features for children
        remaining = [f for f in available_features if f not in selected_global]

        # Build children
        for p_key, local_idx in partitions.items():
            child_example_idx = example_idx[local_idx]

            if len(child_example_idx) < config.min_partition_size:
                # Too small — make a leaf child
                child_id = f"{node_id}_p{''.join(str(v) for v in p_key)}"
                child_labels = labels[child_example_idx]
                child_br = _compute_base_rate(child_labels)
                child = PartitionTreeNode(
                    node_id=child_id,
                    depth=depth + 1,
                    parent_id=node_id,
                    partition_key=p_key,
                    local_metrics=[],
                    all_metrics=local_metrics,
                    point_indices=child_example_idx,
                    base_rate=child_br,
                    n_positive=int((child_labels == 1).sum()),
                    n_negative=len(child_labels) - int((child_labels == 1).sum()),
                    is_leaf=True,
                )
                all_nodes[child_id] = child
                node.children[p_key] = child
            else:
                child_id = f"{node_id}_p{''.join(str(v) for v in p_key)}"
                child = _build_node(
                    child_example_idx,
                    remaining,
                    depth + 1,
                    parent_id=node_id,
                    partition_key=p_key,
                    node_id_prefix=child_id,
                )
                node.children[p_key] = child

        return node

    # Start recursive build
    all_example_idx = np.arange(N)
    all_feature_idx = list(range(M))
    root = _build_node(
        all_example_idx, all_feature_idx,
        depth=0, parent_id=None, partition_key=(),
        node_id_prefix="root",
    )

    tree = MetricTree(
        root=root,
        config=config,
        all_nodes=all_nodes,
        all_metrics=all_metrics_dict,
        task_description=task_description,
    )

    logger.info("Rebuilt tree: %d nodes, %d metrics, max_depth=%d",
                len(all_nodes), len(all_metrics), config.max_depth)

    return tree


# ── Metric Deduplication ─────────────────────────────────────────────────────


def dedup_metrics_by_embedding(
    metrics: List[TreeMetric],
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[TreeMetric]:
    """Remove duplicate metrics using embedding cosine similarity.

    Within each cluster of similar metrics, keeps the first one (preserving
    order, which typically means higher-MI metrics survive if pre-sorted).

    Parameters
    ----------
    metrics : list of TreeMetric
    threshold : cosine similarity threshold for dedup
    model_name : sentence-transformer model for embeddings

    Returns
    -------
    Deduplicated list of TreeMetric.
    """
    if len(metrics) <= 1:
        return metrics

    from .example_selection import embed_texts

    # Build text representations: name + rubric
    texts = []
    for m in metrics:
        rubric_str = f"YES: {m.rubric.get('yes', '')} NO: {m.rubric.get('no', '')}"
        texts.append(f"{m.name}: {rubric_str}")

    embeddings = embed_texts(texts, model_name=model_name)
    if embeddings is None:
        logger.warning("Embedding failed, skipping dedup")
        return metrics

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normed = embeddings / norms

    # Greedy dedup: iterate in order, skip if too similar to any kept metric
    kept = []
    kept_embeddings = []
    for i, m in enumerate(metrics):
        if kept_embeddings:
            sims = normed[i] @ np.array(kept_embeddings).T
            if sims.max() > threshold:
                dup_idx = int(sims.argmax())
                logger.info("Dedup: %s too similar to %s (sim=%.3f), removing",
                            m.name, kept[dup_idx].name, sims.max())
                continue
        kept.append(m)
        kept_embeddings.append(normed[i])

    logger.info("Embedding dedup: %d → %d metrics (threshold=%.2f)",
                len(metrics), len(kept), threshold)
    return kept


def dedup_metrics_by_llm(
    metrics: List[TreeMetric],
    scoring_backend: Any,
    task_description: str = "",
) -> List[TreeMetric]:
    """Consolidate duplicate metrics using LLM-based dedup.

    Sends metrics in batches to the LLM to identify groups measuring
    the same concept, then keeps one representative per group.

    Parameters
    ----------
    metrics : list of TreeMetric (already embedding-deduped)
    scoring_backend : VLLM backend with generate_text method
    task_description : for context

    Returns
    -------
    Deduplicated list of TreeMetric.
    """
    if len(metrics) <= 1:
        return metrics

    # Format metrics for the LLM
    metric_descriptions = []
    for i, m in enumerate(metrics):
        rubric_str = f"YES: {m.rubric.get('yes', '')} NO: {m.rubric.get('no', '')}"
        metric_descriptions.append(f"{i+1}. {m.name}: {rubric_str}")

    prompt = (
        f"Task: {task_description}\n\n"
        f"Below are {len(metrics)} evaluation criteria. Some may measure the same "
        f"underlying concept. Group any duplicates together.\n\n"
        + "\n\n".join(metric_descriptions) +
        "\n\nFor each group of duplicates, output the NUMBER of the best representative "
        "(the one with the clearest, most specific rubric). Output as JSON:\n"
        "{\"keep\": [list of numbers to keep], \"remove\": [list of numbers to remove]}\n"
        "If all metrics are distinct, keep all of them."
    )

    try:
        raw = scoring_backend.generate_text(prompt, max_tokens=512)
        import json, re
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            result = json.loads(raw[start:end+1])
            keep_indices = [int(x) - 1 for x in result.get("keep", [])]
            # Validate indices
            keep_indices = [i for i in keep_indices if 0 <= i < len(metrics)]
            if keep_indices:
                kept = [metrics[i] for i in keep_indices]
                removed = len(metrics) - len(kept)
                if removed > 0:
                    logger.info("LLM dedup: %d → %d metrics (%d removed)",
                                len(metrics), len(kept), removed)
                return kept
    except Exception as e:
        logger.warning("LLM dedup failed: %s, keeping all metrics", e)

    return metrics


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_tree_on_split(
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
) -> Dict[str, float]:
    """Evaluate a tree on a data split, returning accuracy, AUC, etc."""
    predictions_df = predict_batch(
        tree=tree,
        df=df,
        label_cache=label_cache,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        task_description=task_description,
        scoring_backend=scoring_backend,
        batch_size=batch_size,
        verbose=verbose,
        max_model_len=max_model_len,
        tokenizer=tokenizer,
    )

    true_labels = _coerce_binary_labels(df, label_column)[label_column].values
    preds = predictions_df["prediction"].values
    probs = predictions_df["probability"].values

    accuracy = float((preds == true_labels).mean())

    try:
        auc = float(roc_auc_score(true_labels, probs))
    except ValueError:
        auc = 0.5

    n_nodes = len(tree.all_nodes)
    n_leaves = sum(1 for n in tree.all_nodes.values() if n.is_leaf)
    n_metrics = len(tree.all_metrics)

    results = {
        "accuracy": accuracy,
        "auc": auc,
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "n_metrics": n_metrics,
    }

    # Depth distribution
    if "resolving_node" in predictions_df.columns:
        depth_counts = {}
        for _, row in predictions_df.iterrows():
            node_id = row["resolving_node"]
            if node_id in tree.all_nodes:
                d = tree.all_nodes[node_id].depth
                depth_counts[d] = depth_counts.get(d, 0) + 1
        results["depth_distribution"] = depth_counts

    return results


# ── Global Ternary Scoring ───────────────────────────────────────────────────


def score_all_metrics_globally(
    df: pd.DataFrame,
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
    max_model_len: int = 0,
    tokenizer: Any = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Score ALL metrics on ALL examples using ternary (YES/NO/NA) scoring.

    Returns
    -------
    score_matrix : ndarray of shape (N, M) with values 0, 1, or NaN
    labels : ndarray of shape (N,) with values 0 or 1
    metric_names : list of M metric names (column order)
    """
    scored_df = score_ternary_subset(
        df, np.arange(len(df)), metrics, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        task_description=task_description, scoring_backend=scoring_backend,
        batch_size=batch_size, verbose=verbose,
        stage="global_ternary", tokenizer=tokenizer, max_model_len=max_model_len,
    )

    metric_names = [m.name for m in metrics]
    X, y = build_ternary_feature_matrix(scored_df, metric_names, label_column)

    return X, y, metric_names


# ── Collect All Metrics From Tree ────────────────────────────────────────────


def collect_all_metrics_from_tree(tree: MetricTree) -> List[TreeMetric]:
    """Extract all unique TreeMetric objects from a tree, in order of appearance."""
    seen = set()
    metrics = []
    for node in tree.all_nodes.values():
        for m in node.local_metrics:
            if m.metric_id not in seen:
                seen.add(m.metric_id)
                metrics.append(m)
    return metrics


# ── Main Restructuring Loop ─────────────────────────────────────────────────


def restructure_tree(
    initial_tree: MetricTree,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_cache: LabelCache,
    config: TreeConfig,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    task_description: str,
    scoring_backend: Any,
    proposer: Any = None,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> Tuple[MetricTree, List[Dict[str, Any]]]:
    """Run the iterative restructuring pipeline.

    Parameters
    ----------
    initial_tree : MetricTree from the greedy builder
    train_df, eval_df, test_df : data splits
    label_cache : for caching scores
    config : TreeConfig with restructuring params
    scoring_backend : VLLM backend
    proposer : metric proposer for gap-filling (optional)
    tokenizer, max_model_len : for token budgeting

    Returns
    -------
    (best_tree, iteration_results) where iteration_results is a list of
    dicts with eval metrics per iteration.
    """
    output_dir = Path(config.output_dir)
    n_iterations = config.restructure_iterations

    if n_iterations <= 0:
        logger.info("Restructuring disabled (restructure_iterations=0)")
        return initial_tree, []

    # Collect all metrics from initial tree
    all_metrics = collect_all_metrics_from_tree(initial_tree)
    logger.info("Initial tree has %d unique metrics", len(all_metrics))

    # Evaluate initial tree
    logger.info("Evaluating initial tree on eval set...")
    eval_results = evaluate_tree_on_split(
        initial_tree, eval_df, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        task_description=task_description, scoring_backend=scoring_backend,
        batch_size=config.label_batch_size, verbose=config.verbose,
        max_model_len=max_model_len, tokenizer=tokenizer,
    )
    logger.info("Initial tree eval: accuracy=%.4f, AUC=%.4f",
                eval_results["accuracy"], eval_results["auc"])

    # Also evaluate on test set
    logger.info("Evaluating initial tree on test set...")
    test_results = evaluate_tree_on_split(
        initial_tree, test_df, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        task_description=task_description, scoring_backend=scoring_backend,
        batch_size=config.label_batch_size, verbose=config.verbose,
        max_model_len=max_model_len, tokenizer=tokenizer,
    )
    logger.info("Initial tree test: accuracy=%.4f, AUC=%.4f",
                test_results["accuracy"], test_results["auc"])

    iteration_results = [{
        "iteration": 0,
        "stage": "initial",
        "n_metrics": len(all_metrics),
        "eval_accuracy": eval_results["accuracy"],
        "eval_auc": eval_results["auc"],
        "test_accuracy": test_results["accuracy"],
        "test_auc": test_results["auc"],
        "n_nodes": eval_results["n_nodes"],
        "n_leaves": eval_results["n_leaves"],
    }]

    # Save initial tree
    save_tree(initial_tree, str(output_dir / "iteration_0"))
    _save_iteration_results(iteration_results, output_dir)

    best_tree = initial_tree
    best_eval_auc = eval_results["auc"]

    for iteration in range(1, n_iterations + 1):
        logger.info("\n=== Restructuring Iteration %d/%d ===", iteration, n_iterations)

        # Step 1: Deduplicate metrics
        logger.info("Deduplicating %d metrics...", len(all_metrics))
        deduped = dedup_metrics_by_embedding(
            all_metrics, threshold=config.dedup_embedding_threshold,
            model_name=config.embedding_model,
        )
        deduped = dedup_metrics_by_llm(deduped, scoring_backend, task_description)
        logger.info("After dedup: %d metrics", len(deduped))

        # Step 2: Score all metrics on all training examples (ternary)
        logger.info("Global ternary scoring: %d metrics × %d examples...",
                     len(deduped), len(train_df))
        score_matrix, train_labels, metric_names = score_all_metrics_globally(
            train_df, deduped, label_cache,
            id_column=id_column, text_column=text_column, label_column=label_column,
            task_description=task_description, scoring_backend=scoring_backend,
            batch_size=config.label_batch_size, verbose=config.verbose,
            max_model_len=max_model_len, tokenizer=tokenizer,
        )

        # Log NA rates
        na_rates = compute_na_rate(score_matrix)
        for i, name in enumerate(metric_names):
            logger.info("  %s: NA_rate=%.3f", name, na_rates[i])

        # Step 3: Rebuild tree from score matrix
        logger.info("Rebuilding tree from score matrix...")
        new_tree = rebuild_tree_from_scores(
            score_matrix, train_labels, deduped, config, task_description,
        )

        # Step 4: Evaluate on eval set
        logger.info("Evaluating restructured tree on eval set...")
        eval_results = evaluate_tree_on_split(
            new_tree, eval_df, label_cache,
            id_column=id_column, text_column=text_column, label_column=label_column,
            task_description=task_description, scoring_backend=scoring_backend,
            batch_size=config.label_batch_size, verbose=config.verbose,
            max_model_len=max_model_len, tokenizer=tokenizer,
        )
        logger.info("Iteration %d eval: accuracy=%.4f, AUC=%.4f",
                     iteration, eval_results["accuracy"], eval_results["auc"])

        # Evaluate on test set too
        logger.info("Evaluating restructured tree on test set...")
        test_results = evaluate_tree_on_split(
            new_tree, test_df, label_cache,
            id_column=id_column, text_column=text_column, label_column=label_column,
            task_description=task_description, scoring_backend=scoring_backend,
            batch_size=config.label_batch_size, verbose=config.verbose,
            max_model_len=max_model_len, tokenizer=tokenizer,
        )
        logger.info("Iteration %d test: accuracy=%.4f, AUC=%.4f",
                     iteration, test_results["accuracy"], test_results["auc"])

        iter_result = {
            "iteration": iteration,
            "stage": "restructured",
            "n_metrics": len(deduped),
            "eval_accuracy": eval_results["accuracy"],
            "eval_auc": eval_results["auc"],
            "test_accuracy": test_results["accuracy"],
            "test_auc": test_results["auc"],
            "n_nodes": eval_results["n_nodes"],
            "n_leaves": eval_results["n_leaves"],
        }
        iteration_results.append(iter_result)

        # Save this iteration's tree
        save_tree(new_tree, str(output_dir / f"iteration_{iteration}"))
        _save_iteration_results(iteration_results, output_dir)

        # Track best
        if eval_results["auc"] > best_eval_auc:
            best_eval_auc = eval_results["auc"]
            best_tree = new_tree
            logger.info("New best tree (eval AUC=%.4f)", best_eval_auc)

        # Step 5: Gap-filling — generate new metrics for weak partitions
        if proposer is not None:
            new_metrics = _gap_fill(
                new_tree, train_df, train_labels, score_matrix, deduped,
                config, proposer, scoring_backend, label_cache,
                id_column=id_column, text_column=text_column,
                label_column=label_column, task_description=task_description,
                tokenizer=tokenizer, max_model_len=max_model_len,
            )
            if new_metrics:
                logger.info("Gap-filling added %d new metrics", len(new_metrics))
                all_metrics = deduped + new_metrics
            else:
                all_metrics = deduped
                logger.info("No gap-filling metrics generated")
        else:
            all_metrics = deduped

        # Check for convergence
        if iteration >= 2:
            prev_auc = iteration_results[-2]["eval_auc"]
            curr_auc = eval_results["auc"]
            if curr_auc <= prev_auc + 0.005:
                logger.info("Eval AUC plateaued (%.4f → %.4f), stopping", prev_auc, curr_auc)
                break

    logger.info("\nRestructuring complete. Best eval AUC=%.4f across %d iterations",
                best_eval_auc, len(iteration_results))

    return best_tree, iteration_results


# ── Gap-Filling ──────────────────────────────────────────────────────────────


def _gap_fill(
    tree: MetricTree,
    train_df: pd.DataFrame,
    train_labels: np.ndarray,
    score_matrix: np.ndarray,
    existing_metrics: List[TreeMetric],
    config: TreeConfig,
    proposer: Any,
    scoring_backend: Any,
    label_cache: LabelCache,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    task_description: str,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> List[TreeMetric]:
    """Generate new metrics for leaf partitions with poor base rates.

    Identifies leaf nodes where:
    - The partition is large enough to generate from
    - The base rate is between 0.3 and 0.7 (neither class dominates)
    - There are enough contrastive pairs

    For each such partition, proposes new discriminative metrics.
    """
    new_metrics = []

    for node_id, node in tree.all_nodes.items():
        if not node.is_leaf:
            continue
        if len(node.point_indices) < config.min_partition_size * 2:
            continue

        node_labels = train_labels[node.point_indices]
        n_contrastive = count_contrastive_pairs(node_labels)
        if n_contrastive < config.min_contrastive_pairs:
            continue

        base_rate = _compute_base_rate(node_labels)
        if base_rate < 0.2 or base_rate > 0.8:
            continue  # too pure to benefit from more features

        logger.info("Gap-filling for leaf %s: n=%d, base_rate=%.3f, contrastive=%d",
                     node_id, len(node.point_indices), base_rate, n_contrastive)

        # Get texts for this partition
        partition_df = train_df.iloc[node.point_indices]
        pos_df = partition_df[partition_df[label_column] == 1]
        neg_df = partition_df[partition_df[label_column] == 0]

        try:
            proposed = proposer.propose(
                task_description=task_description,
                parent=node,
                partition_key=node.partition_key,
                positive_df=pos_df,
                negative_df=neg_df,
                id_column=id_column,
                text_column=text_column,
                label_column=label_column,
                num_metrics=config.n_rubrics_to_propose,
                scoring_backend=scoring_backend,
                contrastive_pairs_k=config.contrastive_pairs_k,
                population_size=len(node.point_indices),
                positive_rate=base_rate,
            )
            if proposed:
                # Convert proposed dicts to TreeMetric objects
                import hashlib
                for p in proposed:
                    rubric = p.get("rubric", {})
                    rubric_text = str(rubric)
                    metric_id = hashlib.sha256(rubric_text.encode()).hexdigest()[:16]
                    # Check if this metric_id already exists
                    if not any(m.metric_id == metric_id for m in existing_metrics):
                        tm = TreeMetric(
                            metric_id=metric_id,
                            name=p["name"],
                            rubric_text=rubric_text,
                            rubric=rubric,
                            source_node_id=node_id,
                            scale=p.get("scale", "binary"),
                        )
                        new_metrics.append(tm)
                logger.info("  Proposed %d new metrics for %s", len(proposed), node_id)
        except Exception as e:
            logger.warning("Gap-fill proposal failed for %s: %s", node_id, e)

    return new_metrics


# ── Utilities ────────────────────────────────────────────────────────────────


def _save_iteration_results(results: List[Dict], output_dir: Path) -> None:
    """Save iteration results as JSON for tracking progress."""
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "restructuring_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
