"""Core Metric Tree builder: root node, exception nodes, recursive growth."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from autometrics.generator.ContrastiveRubricProposer import (
    ContrastiveRubricProposer,
    _sanitize_metric_name,
)
from autometrics.iterative_refinement.runner import (
    _coerce_binary_labels,
    _format_examples,
    _metric_id_from_rubric,
    _normalize_rubric,
    _rubric_to_text,
    _ensure_unique_name,
    _truncate_text,
)
from autometrics.iterative_refinement.label_cache import LabelCache

from .config import TreeConfig
from .data_structures import MetricTree, MetricTreeNode, TreeMetric
from .example_selection import select_representative_examples
from .proposer import ExceptionMetricProposer
from .routing import build_learned_router, tune_threshold
from .scoring import (
    add_interaction_features,
    build_feature_matrix,
    score_subset,
)
from .token_utils import (
    compute_generation_example_budget,
    compute_scoring_text_budget,
    truncate_to_tokens,
)

logger = logging.getLogger("metric_tree.builder")


def _create_tree_metrics(
    raw_metrics: List[Dict[str, Any]],
    source_node_id: str,
    existing_metric_ids: set,
    existing_names: set,
) -> List[TreeMetric]:
    """Convert raw metric dicts from the proposer into TreeMetric objects."""
    tree_metrics = []
    for m in raw_metrics:
        name = _sanitize_metric_name(m.get("name", "Metric"))
        scale = m.get("scale", "ordinal")
        rubric = _normalize_rubric(m.get("rubric", {}), scale)
        rubric_text = _rubric_to_text(rubric)
        metric_id = _metric_id_from_rubric(rubric_text)

        # Skip duplicates
        if metric_id in existing_metric_ids:
            logger.info("Skipping duplicate metric %s (%s)", name, metric_id[:8])
            continue

        name = _ensure_unique_name(name, existing_names, metric_id)
        existing_names.add(name)
        existing_metric_ids.add(metric_id)

        tree_metrics.append(TreeMetric(
            metric_id=metric_id,
            name=name,
            rubric_text=rubric_text,
            rubric=rubric,
            source_node_id=source_node_id,
            scale=scale,
        ))
    return tree_metrics


def _fit_l1_selector(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: TreeConfig,
) -> Tuple[List[str], np.ndarray]:
    """Fit L1-penalized logistic regression for metric selection.

    Returns (selected_feature_names, coefficients_for_selected).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegressionCV(
        penalty="l1",
        solver="liblinear",
        Cs=10,
        cv=min(config.cv_folds, max(2, int(y.sum()), int((1 - y).sum()))),
        class_weight=config.class_weight,
        max_iter=1000,
        random_state=config.random_seed,
    )
    clf.fit(X_scaled, y)

    coefs = clf.coef_.ravel()
    nonzero_mask = np.abs(coefs) > 1e-6
    selected = [feature_names[i] for i in range(len(feature_names)) if nonzero_mask[i]]
    selected_coefs = coefs[nonzero_mask]

    logger.info(
        "L1 selection: %d/%d features selected (C=%.4f)",
        len(selected), len(feature_names), clf.C_[0] if hasattr(clf, "C_") else 0,
    )
    return selected, selected_coefs


def _fit_final_classifier(
    X: np.ndarray,
    y: np.ndarray,
    config: TreeConfig,
) -> Tuple[Any, StandardScaler]:
    """Fit L2-regularized logistic regression on selected features.

    Returns (classifier, scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegressionCV(
        penalty="l2",
        solver="lbfgs",
        Cs=10,
        cv=min(config.cv_folds, max(2, int(y.sum()), int((1 - y).sum()))),
        class_weight=config.class_weight,
        max_iter=1000,
        random_state=config.random_seed,
    )
    clf.fit(X_scaled, y)

    return clf, scaler


def _sample_examples(
    df: pd.DataFrame,
    label_column: str,
    n_per_class: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample positive and negative examples for metric generation."""
    rng = random.Random(seed)
    pos_df = df[df[label_column] == 1]
    neg_df = df[df[label_column] == 0]

    pos_sample = pos_df.sample(n=min(n_per_class, len(pos_df)), random_state=seed)
    neg_sample = neg_df.sample(n=min(n_per_class, len(neg_df)), random_state=seed)

    return pos_sample, neg_sample


def build_root_node(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TreeConfig,
    label_cache: LabelCache,
    proposer: ContrastiveRubricProposer,
    task_description: str,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    judge_llm: Any,
    scoring_backend: Any = None,
    tokenizer: Any = None,
    token_budgets: Optional[Dict] = None,
) -> Tuple[MetricTreeNode, Dict[str, TreeMetric]]:
    """Build the root node of the Metric Tree.

    Steps:
    1. Sample positive/negative examples
    2. Propose candidate metrics via ContrastiveRubricProposer
    3. Score train + eval on candidates
    4. L1 selection on eval set
    5. Optionally add interaction features and re-select
    6. Fit final L2 classifier on selected features
    7. Predict on train → correct_mask
    8. Tune confidence threshold on eval
    """
    node_id = "root"
    budgets = token_budgets or {}
    max_model_len = budgets.get("max_model_len", 0)
    logger.info("Building root node (max_model_len=%d)...", max_model_len)

    # 1. Sample examples for metric generation
    pos_sample, neg_sample = _sample_examples(train_df, label_column, seed=config.random_seed)
    # Dynamic generation budget: current_metrics="" at root, measure exact budget
    if max_model_len > 0 and tokenizer is not None:
        n_total = len(pos_sample) + len(neg_sample)
        gen_example_tokens = compute_generation_example_budget(
            current_metrics_text="",
            task_description=task_description,
            max_model_len=max_model_len,
            n_total_examples=max(1, n_total),
            tokenizer=tokenizer,
        )
        pos_sample = pos_sample.copy()
        neg_sample = neg_sample.copy()
        pos_sample[text_column] = pos_sample[text_column].apply(
            lambda t: truncate_to_tokens(str(t), gen_example_tokens, tokenizer=tokenizer))
        neg_sample[text_column] = neg_sample[text_column].apply(
            lambda t: truncate_to_tokens(str(t), gen_example_tokens, tokenizer=tokenizer))
    pos_text = _format_examples(pos_sample, id_column, text_column, label_column)
    neg_text = _format_examples(neg_sample, id_column, text_column, label_column)

    # 2. Propose candidate metrics
    logger.info("Proposing %d metrics...", config.n_metrics_to_propose)
    raw_metrics = proposer.propose(
        task_description=task_description,
        positive_examples=pos_text,
        negative_examples=neg_text,
        current_metrics="",
        contrastive_pairs="",
        num_metrics=config.n_metrics_to_propose,
        num_rubrics=config.n_rubrics_to_propose,
    )
    logger.info("Proposer returned %d metrics", len(raw_metrics))

    if not raw_metrics:
        raise RuntimeError("Proposer returned no metrics for root node")

    # 3. Create TreeMetric objects
    existing_metric_ids: set = set()
    existing_names: set = set()
    candidate_metrics = _create_tree_metrics(
        raw_metrics, node_id, existing_metric_ids, existing_names,
    )

    if not candidate_metrics:
        raise RuntimeError("No valid metrics after deduplication for root node")

    # 4. Score train and eval
    train_indices = np.arange(len(train_df))
    eval_indices = np.arange(len(eval_df))

    logger.info("Scoring %d train examples on %d metrics...", len(train_df), len(candidate_metrics))
    train_scored = score_subset(
        train_df, train_indices, candidate_metrics, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        judge_llm=judge_llm, task_description=task_description,
        batch_size=config.label_batch_size, verbose=config.verbose,
        stage="root_train", scoring_backend=scoring_backend,
        tokenizer=tokenizer, max_model_len=max_model_len,
    )

    logger.info("Scoring %d eval examples...", len(eval_df))
    eval_scored = score_subset(
        eval_df, eval_indices, candidate_metrics, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        judge_llm=judge_llm, task_description=task_description,
        batch_size=config.label_batch_size, verbose=config.verbose,
        stage="root_eval", scoring_backend=scoring_backend,
        tokenizer=tokenizer, max_model_len=max_model_len,
    )

    # 5. L1 feature selection on eval
    candidate_names = [m.name for m in candidate_metrics]
    X_eval, y_eval = build_feature_matrix(eval_scored, candidate_names, label_column)
    selected_names, _ = _fit_l1_selector(X_eval, y_eval, candidate_names, config)

    if not selected_names:
        logger.warning("L1 selected no features, keeping all candidates")
        selected_names = candidate_names

    # Filter metrics to selected
    selected_metrics = [m for m in candidate_metrics if m.name in set(selected_names)]

    # 6. Optionally add interactions (at root, depth=0)
    feature_names = [m.name for m in selected_metrics]
    interaction_pairs = []

    X_train, y_train = build_feature_matrix(train_scored, feature_names, label_column)

    if config.use_interactions and len(feature_names) >= 2:
        logger.info("Adding interaction features...")
        X_train_aug, aug_names, interaction_pairs = add_interaction_features(X_train, feature_names)
        X_eval_aug, _, _ = add_interaction_features(
            build_feature_matrix(eval_scored, feature_names, label_column)[0],
            feature_names,
        )
        y_eval_for_interactions = build_feature_matrix(eval_scored, feature_names, label_column)[1]

        # Re-run L1 selection with interactions
        selected_aug_names, _ = _fit_l1_selector(
            X_eval_aug, y_eval_for_interactions, aug_names, config,
        )

        if selected_aug_names:
            # Use augmented features
            feature_names = selected_aug_names
            # Rebuild train matrix with selected augmented features
            aug_name_set = set(selected_aug_names)
            col_mask = [i for i, n in enumerate(aug_names) if n in aug_name_set]
            X_train = X_train_aug[:, col_mask]

            # Determine which interaction pairs survived
            interaction_pairs = [
                ip for ip, n in zip(
                    [(None, None)] * len([m.name for m in selected_metrics])  # base features have no pair
                    + interaction_pairs,
                    aug_names,
                )
                if n in aug_name_set and ip != (None, None)
            ]

    # 7. Fit final L2 classifier
    logger.info("Fitting final classifier on %d features...", len(feature_names))
    classifier, scaler = _fit_final_classifier(X_train, y_train, config)

    # Predict on train set
    X_train_scaled = scaler.transform(X_train)
    train_probs = classifier.predict_proba(X_train_scaled)[:, 1]
    train_preds = (train_probs >= 0.5).astype(int)
    correct_mask = (train_preds == y_train)
    train_acc = correct_mask.mean()
    logger.info("Root train accuracy: %.3f", train_acc)

    # 8. Tune threshold on eval
    # Rebuild eval features for selected feature set
    eval_feature_names_base = [m.name for m in selected_metrics]
    X_eval_base, y_eval_base = build_feature_matrix(eval_scored, eval_feature_names_base, label_column)

    if config.use_interactions and interaction_pairs:
        X_eval_aug, eval_aug_names, _ = add_interaction_features(X_eval_base, eval_feature_names_base)
        aug_name_set = set(feature_names)
        col_mask = [i for i, n in enumerate(eval_aug_names) if n in aug_name_set]
        X_eval_final = X_eval_aug[:, col_mask]
    else:
        X_eval_final = X_eval_base

    X_eval_scaled = scaler.transform(X_eval_final)
    eval_probs = classifier.predict_proba(X_eval_scaled)[:, 1]
    eval_preds = (eval_probs >= 0.5).astype(int)
    eval_acc = (eval_preds == y_eval_base).mean()
    logger.info("Root eval accuracy: %.3f", eval_acc)

    confidence_threshold, _, _ = tune_threshold(eval_probs, y_eval_base)

    # Build node
    node = MetricTreeNode(
        node_id=node_id,
        depth=0,
        parent_id=None,
        local_metrics=selected_metrics,
        all_metrics=selected_metrics,
        classifier=classifier,
        scaler=scaler,
        feature_names=feature_names,
        point_indices=np.arange(len(train_df)),
        predictions=train_preds,
        probabilities=train_probs,
        correct_mask=correct_mask,
        confidence_threshold=confidence_threshold,
        interaction_pairs=interaction_pairs,
        train_accuracy=float(train_acc),
        eval_accuracy=float(eval_acc),
    )

    # Optionally build learned router
    if config.use_learned_router:
        node.router = build_learned_router(X_train_scaled, correct_mask, config)

    # Collect all metrics
    all_metrics_dict = {m.metric_id: m for m in selected_metrics}

    return node, all_metrics_dict


def _extract_parent_coefficients(parent: MetricTreeNode) -> str:
    """Format parent's logistic regression coefficients sorted by absolute magnitude."""
    if parent.classifier is None or not hasattr(parent.classifier, 'coef_'):
        return ""

    coefs = parent.classifier.coef_.ravel()
    names = parent.feature_names

    if len(coefs) != len(names):
        return ""

    # Sort by absolute magnitude (most important first)
    pairs = sorted(zip(names, coefs), key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for name, coef in pairs:
        direction = "+" if coef > 0 else "-"
        lines.append(f"  {direction} {name}: {coef:.3f}")

    return "\n".join(lines)


def build_exception_node(
    parent: MetricTreeNode,
    error_type: str,
    error_indices: np.ndarray,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TreeConfig,
    label_cache: LabelCache,
    exception_proposer: ExceptionMetricProposer,
    task_description: str,
    tree: MetricTree,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    judge_llm: Any,
    scoring_backend: Any = None,
    tokenizer: Any = None,
    token_budgets: Optional[Dict] = None,
) -> Optional[MetricTreeNode]:
    """Build a child node for exceptions (misclassified points).

    Critical design: child trains on the FULL prediction bucket from parent,
    not just misclassified points.
    - false_positive child: train on all parent-predicted-positive (TP + FP)
    - false_negative child: train on all parent-predicted-negative (TN + FN)
    """
    node_id = f"{parent.node_id}_{error_type}"
    depth = parent.depth + 1
    budgets = token_budgets or {}
    max_model_len = budgets.get("max_model_len", 0)
    logger.info("Building exception node %s (depth=%d, %d errors)...", node_id, depth, len(error_indices))

    # Determine the full prediction bucket
    if error_type == "false_positive":
        # All points parent predicted positive
        bucket_mask = parent.predictions == 1
    elif error_type == "false_negative":
        # All points parent predicted negative
        bucket_mask = parent.predictions == 0
    else:
        # "misclassified" fallback: use all parent's points
        bucket_mask = np.ones(len(parent.predictions), dtype=bool)

    bucket_indices = parent.point_indices[bucket_mask]
    bucket_correct = parent.correct_mask[bucket_mask]

    if len(bucket_indices) < config.min_subset_size:
        logger.info("Bucket too small (%d < %d), skipping", len(bucket_indices), config.min_subset_size)
        return None

    # Partition into exception vs correct within this bucket
    exception_mask = ~bucket_correct
    correct_in_bucket_mask = bucket_correct

    exception_df = train_df.iloc[bucket_indices[exception_mask]].reset_index(drop=True)
    correct_in_bucket_df = train_df.iloc[bucket_indices[correct_in_bucket_mask]].reset_index(drop=True)

    if len(exception_df) == 0:
        logger.info("No exceptions in bucket for %s, skipping", node_id)
        return None

    logger.info("Proposing exception metrics for %s (%d exceptions, %d correct)...",
                node_id, len(exception_df), len(correct_in_bucket_df))

    # Step 1: Select representative examples via embedding-based clustering
    exception_sample, correct_sample = select_representative_examples(
        exception_df, correct_df=correct_in_bucket_df,
        text_column=text_column,
        k_per_class=config.exception_examples_per_class,
        model_name=config.embedding_model,
        seed=config.random_seed,
    )

    # Step 2: Score samples on parent metrics (mostly cached)
    sample_combined = pd.concat([exception_sample, correct_sample], ignore_index=True)
    sample_indices = np.arange(len(sample_combined))
    sample_scored = None
    if parent.all_metrics:
        try:
            sample_scored = score_subset(
                sample_combined, sample_indices, parent.all_metrics, label_cache,
                id_column=id_column, text_column=text_column, label_column=label_column,
                judge_llm=judge_llm, task_description=task_description,
                batch_size=config.label_batch_size, verbose=config.verbose,
                stage=f"{node_id}_sample", scoring_backend=scoring_backend,
                tokenizer=tokenizer, max_model_len=max_model_len,
            )
            logger.info("Scored %d sample examples on %d parent metrics",
                        len(sample_scored), len(parent.all_metrics))
        except Exception as e:
            logger.warning("Failed to score sample on parent metrics: %s", e)

    # Step 3: Compute token budget with correct n_total_examples (~10, not 47K)
    gen_example_tokens = 0
    if max_model_len > 0 and tokenizer is not None:
        from .proposer import _format_parent_context
        parent_coefficients = _extract_parent_coefficients(parent)
        parent_context_text = _format_parent_context(parent, error_type, coefficients=parent_coefficients)
        exception_task = (
            f"{task_description}\n\n"
            f"=== EXCEPTION ANALYSIS MODE ===\n"
            f"The parent classifier has already applied general rules but misclassifies "
            f"some examples ({error_type})."
        )
        n_total = len(exception_sample) + len(correct_sample)
        gen_example_tokens = compute_generation_example_budget(
            current_metrics_text=parent_context_text,
            task_description=exception_task,
            max_model_len=max_model_len,
            n_total_examples=max(1, n_total),
            tokenizer=tokenizer,
        )
    else:
        parent_coefficients = _extract_parent_coefficients(parent)

    # Step 4: Call proposer with enriched args
    raw_metrics = exception_proposer.propose(
        task_description=task_description,
        parent=parent,
        error_type=error_type,
        exception_df=exception_sample,
        correct_df=correct_sample,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        num_metrics=config.n_metrics_to_propose,
        num_rubrics=config.n_rubrics_to_propose,
        max_example_tokens=gen_example_tokens,
        tokenizer=tokenizer,
        sample_scored=sample_scored,
        parent_coefficients=parent_coefficients,
        scoring_backend=scoring_backend,
        enable_error_analysis=config.enable_error_analysis,
        contrastive_pairs_k=config.contrastive_pairs_k,
    )

    if not raw_metrics:
        logger.warning("No exception metrics proposed for %s", node_id)
        return None

    # Create TreeMetric objects
    existing_metric_ids = set(tree.all_metrics.keys())
    existing_names = {m.name for m in tree.all_metrics.values()}
    new_metrics = _create_tree_metrics(raw_metrics, node_id, existing_metric_ids, existing_names)

    if not new_metrics:
        logger.warning("No new metrics after dedup for %s", node_id)
        return None

    # Score bucket examples on NEW metrics only (parent scores already cached)
    bucket_df = train_df.iloc[bucket_indices].reset_index(drop=True)
    bucket_local_indices = np.arange(len(bucket_df))

    logger.info("Scoring %d bucket examples on %d new metrics...", len(bucket_df), len(new_metrics))
    new_scored = score_subset(
        bucket_df, bucket_local_indices, new_metrics, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        judge_llm=judge_llm, task_description=task_description,
        batch_size=config.label_batch_size, verbose=config.verbose,
        stage=f"{node_id}_new", scoring_backend=scoring_backend,
        tokenizer=tokenizer, max_model_len=max_model_len,
    )

    # Also score on parent's metrics (will mostly come from cache)
    parent_scored = score_subset(
        bucket_df, bucket_local_indices, parent.all_metrics, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        judge_llm=judge_llm, task_description=task_description,
        batch_size=config.label_batch_size, verbose=config.verbose,
        stage=f"{node_id}_parent", scoring_backend=scoring_backend,
        tokenizer=tokenizer, max_model_len=max_model_len,
    )

    # Combine features: parent metrics + new metrics
    all_node_metrics = list(parent.all_metrics) + new_metrics
    all_feature_names = [m.name for m in all_node_metrics]

    # Build combined scored DataFrame
    combined_scored = parent_scored.copy()
    for m in new_metrics:
        if m.name in new_scored.columns:
            combined_scored[m.name] = new_scored[m.name].values

    # Build child's label: 1 = exception (misclassified), 0 = correct
    # Actually, for classification we want to use the ORIGINAL labels
    # The child learns to classify within the bucket using original labels
    X_all, y_all = build_feature_matrix(combined_scored, all_feature_names, label_column)

    # L1 selection
    selected_names, _ = _fit_l1_selector(X_all, y_all, all_feature_names, config)

    # Check if any NEW exception metrics survived L1 selection.
    # Spec: "no new metrics selected — meaning the LLM couldn't find useful exception
    # criteria" → don't create the child node. Without new metrics the child just
    # repeats the parent's classifier on a subset.
    new_metric_names = {m.name for m in new_metrics}
    new_survived = [n for n in (selected_names or []) if n in new_metric_names]

    if not new_survived:
        logger.info(
            "L1 selected no NEW exception metrics for %s — parent becomes forced "
            "leaf (inarticulate residual). %d parent metrics survived but no new ones.",
            node_id, len(selected_names or []),
        )
        return None

    # Interactions (only if depth <= interaction_max_depth)
    feature_names = selected_names
    interaction_pairs = []

    selected_col_mask = [i for i, n in enumerate(all_feature_names) if n in set(selected_names)]
    X_selected = X_all[:, selected_col_mask]

    if config.use_interactions and depth <= config.interaction_max_depth and len(feature_names) >= 2:
        X_aug, aug_names, interaction_pairs = add_interaction_features(X_selected, feature_names)
        selected_aug, _ = _fit_l1_selector(X_aug, y_all, aug_names, config)
        if selected_aug:
            feature_names = selected_aug
            aug_name_set = set(selected_aug)
            col_mask = [i for i, n in enumerate(aug_names) if n in aug_name_set]
            X_selected = X_aug[:, col_mask]

    # Fit final classifier
    classifier, scaler_node = _fit_final_classifier(X_selected, y_all, config)

    # Predict on bucket
    X_scaled = scaler_node.transform(X_selected)
    probs = classifier.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    correct_mask = (preds == y_all)
    train_acc = correct_mask.mean()
    logger.info("Node %s train accuracy: %.3f", node_id, train_acc)

    # Eval accuracy (score eval on same metrics, predict)
    eval_acc = _evaluate_node_on_eval(
        eval_df, all_node_metrics, feature_names, classifier, scaler_node,
        label_cache, config, id_column, text_column, label_column,
        judge_llm, task_description, scoring_backend, node_id,
        interaction_pairs, tokenizer=tokenizer, max_model_len=max_model_len,
    )

    # Tune threshold
    confidence_threshold, _, _ = tune_threshold(probs, y_all)

    # Build selected metrics list
    selected_name_set = set()
    for fn in feature_names:
        # Strip interaction suffix
        if "__x__" in fn:
            parts = fn.split("__x__")
            selected_name_set.update(parts)
        else:
            selected_name_set.add(fn)

    local_selected = [m for m in new_metrics if m.name in selected_name_set]
    all_selected = [m for m in all_node_metrics if m.name in selected_name_set]

    node = MetricTreeNode(
        node_id=node_id,
        depth=depth,
        parent_id=parent.node_id,
        local_metrics=local_selected,
        all_metrics=all_selected,
        classifier=classifier,
        scaler=scaler_node,
        feature_names=feature_names,
        point_indices=bucket_indices,
        predictions=preds,
        probabilities=probs,
        correct_mask=correct_mask,
        confidence_threshold=confidence_threshold,
        interaction_pairs=interaction_pairs,
        train_accuracy=float(train_acc),
        eval_accuracy=float(eval_acc),
    )

    if config.use_learned_router:
        node.router = build_learned_router(X_scaled, correct_mask, config)

    # Register new metrics in tree
    for m in new_metrics:
        tree.all_metrics[m.metric_id] = m

    return node


def _evaluate_node_on_eval(
    eval_df, metrics, feature_names, classifier, scaler,
    label_cache, config, id_column, text_column, label_column,
    judge_llm, task_description, scoring_backend, node_id,
    interaction_pairs, tokenizer=None, max_model_len=0,
):
    """Score eval set and compute accuracy for a node."""
    try:
        eval_indices = np.arange(len(eval_df))
        eval_scored = score_subset(
            eval_df, eval_indices, metrics, label_cache,
            id_column=id_column, text_column=text_column, label_column=label_column,
            judge_llm=judge_llm, task_description=task_description,
            batch_size=config.label_batch_size, verbose=False,
            stage=f"{node_id}_eval", scoring_backend=scoring_backend,
            tokenizer=tokenizer, max_model_len=max_model_len,
        )

        base_names = [m.name for m in metrics]
        X_eval, y_eval = build_feature_matrix(eval_scored, base_names, label_column)

        # Rebuild feature matrix with selected features
        if any("__x__" in fn for fn in feature_names):
            X_eval_aug, aug_names, _ = add_interaction_features(X_eval, base_names)
            name_set = set(feature_names)
            col_mask = [i for i, n in enumerate(aug_names) if n in name_set]
            X_eval_final = X_eval_aug[:, col_mask]
        else:
            name_set = set(feature_names)
            col_mask = [i for i, n in enumerate(base_names) if n in name_set]
            X_eval_final = X_eval[:, col_mask]

        X_eval_scaled = scaler.transform(X_eval_final)
        eval_preds = (classifier.predict_proba(X_eval_scaled)[:, 1] >= 0.5).astype(int)
        eval_acc = float((eval_preds == y_eval).mean())
        logger.info("Node %s eval accuracy: %.3f", node_id, eval_acc)
        return eval_acc
    except Exception as e:
        logger.warning("Failed to evaluate node %s on eval: %s", node_id, e)
        return 0.0


def grow_tree(
    node: MetricTreeNode,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TreeConfig,
    label_cache: LabelCache,
    exception_proposer: ExceptionMetricProposer,
    task_description: str,
    tree: MetricTree,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    judge_llm: Any,
    scoring_backend: Any = None,
    tokenizer: Any = None,
    token_budgets: Optional[Dict] = None,
) -> None:
    """Recursively grow the tree from a node by handling its misclassifications."""
    if node.depth >= config.max_depth:
        logger.info("Max depth %d reached at node %s", config.max_depth, node.node_id)
        return

    y_train = train_df.iloc[node.point_indices][label_column].values
    misclassified = ~node.correct_mask
    n_misclassified = misclassified.sum()
    logger.info("Node %s: %d/%d misclassified", node.node_id, n_misclassified, len(node.correct_mask))

    if n_misclassified < config.min_subset_size:
        logger.info("Too few misclassified (%d < %d), leaf node", n_misclassified, config.min_subset_size)
        return

    # Split misclassified into false positives and false negatives
    fp_mask = misclassified & (node.predictions == 1)  # predicted 1, true 0
    fn_mask = misclassified & (node.predictions == 0)  # predicted 0, true 1

    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    logger.info("Node %s: %d false positives, %d false negatives", node.node_id, n_fp, n_fn)

    built_any = False

    # Try to build separate FP and FN children
    for error_type, count in [("false_positive", n_fp), ("false_negative", n_fn)]:
        if count >= config.min_subset_size:
            error_indices = node.point_indices[fp_mask if error_type == "false_positive" else fn_mask]
            child = build_exception_node(
                parent=node,
                error_type=error_type,
                error_indices=error_indices,
                train_df=train_df,
                eval_df=eval_df,
                config=config,
                label_cache=label_cache,
                exception_proposer=exception_proposer,
                task_description=task_description,
                tree=tree,
                id_column=id_column,
                text_column=text_column,
                label_column=label_column,
                judge_llm=judge_llm,
                scoring_backend=scoring_backend,
                tokenizer=tokenizer,
                token_budgets=token_budgets,
            )
            if child is not None:
                node.children[error_type] = child
                tree.all_nodes[child.node_id] = child
                built_any = True

    # Fallback: single "misclassified" child if neither FP nor FN met min_subset_size alone
    if not built_any and n_misclassified >= config.min_subset_size:
        error_indices = node.point_indices[misclassified]
        child = build_exception_node(
            parent=node,
            error_type="misclassified",
            error_indices=error_indices,
            train_df=train_df,
            eval_df=eval_df,
            config=config,
            label_cache=label_cache,
            exception_proposer=exception_proposer,
            task_description=task_description,
            tree=tree,
            id_column=id_column,
            text_column=text_column,
            label_column=label_column,
            judge_llm=judge_llm,
            scoring_backend=scoring_backend,
            tokenizer=tokenizer,
            token_budgets=token_budgets,
        )
        if child is not None:
            node.children["misclassified"] = child
            tree.all_nodes[child.node_id] = child

    # Recurse on children
    for child_type, child_node in node.children.items():
        grow_tree(
            node=child_node,
            train_df=train_df,
            eval_df=eval_df,
            config=config,
            label_cache=label_cache,
            exception_proposer=exception_proposer,
            task_description=task_description,
            tree=tree,
            id_column=id_column,
            text_column=text_column,
            label_column=label_column,
            judge_llm=judge_llm,
            scoring_backend=scoring_backend,
            tokenizer=tokenizer,
            token_budgets=token_budgets,
        )


def build_metric_tree(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TreeConfig,
    proposer: ContrastiveRubricProposer,
    task_description: str,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    judge_llm: Any,
    cache_dir: Optional[str] = None,
    scoring_backend: Any = None,
    tokenizer: Any = None,
    token_budgets: Optional[Dict] = None,
) -> MetricTree:
    """Build a complete Metric Tree.

    Orchestrates: root construction → recursive exception node growth.
    """
    # Set up label cache
    if cache_dir is None:
        cache_dir = str(Path(config.output_dir) / "label_cache")
    label_cache = LabelCache(cache_dir)

    # Coerce labels to binary
    train_df = _coerce_binary_labels(train_df, label_column)
    eval_df = _coerce_binary_labels(eval_df, label_column)

    # Initialize tree
    tree = MetricTree(config=config, task_description=task_description)

    # Build root
    root_node, root_metrics = build_root_node(
        train_df=train_df,
        eval_df=eval_df,
        config=config,
        label_cache=label_cache,
        proposer=proposer,
        task_description=task_description,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        judge_llm=judge_llm,
        scoring_backend=scoring_backend,
        tokenizer=tokenizer,
        token_budgets=token_budgets,
    )

    tree.root = root_node
    tree.all_nodes[root_node.node_id] = root_node
    tree.all_metrics.update(root_metrics)

    # Build exception proposer
    exception_proposer = ExceptionMetricProposer(proposer)

    # Grow tree recursively
    grow_tree(
        node=root_node,
        train_df=train_df,
        eval_df=eval_df,
        config=config,
        label_cache=label_cache,
        exception_proposer=exception_proposer,
        task_description=task_description,
        tree=tree,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        judge_llm=judge_llm,
        scoring_backend=scoring_backend,
        tokenizer=tokenizer,
        token_budgets=token_budgets,
    )

    n_nodes = len(tree.all_nodes)
    n_metrics = len(tree.all_metrics)
    logger.info("Metric Tree complete: %d nodes, %d unique metrics", n_nodes, n_metrics)

    return tree
