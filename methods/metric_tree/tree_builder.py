"""Core Metric Tree builder: partition-based with binary features and base-rate leaves."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autometrics.generator.ContrastiveRubricProposer import (
    ContrastiveRubricProposer,
    _sanitize_metric_name,
)
from autometrics.iterative_refinement.runner import (
    _coerce_binary_labels,
    _format_examples,
    _normalize_rubric,
    _rubric_to_text,
    _ensure_unique_name,
)
from autometrics.iterative_refinement.label_cache import LabelCache

from .config import TreeConfig
from .data_structures import MetricTree, PartitionTreeNode, TreeMetric
from .example_selection import cluster_and_select, select_representative_examples
# mahalanobis removed — using base-rate prediction instead
from .partition import (
    assign_to_partitions,
    count_contrastive_pairs,
    format_partition_description,
    prune_partitions,
    _format_key,
)
from .proposer import PartitionMetricProposer
from .router import train_node_router
from .scoring import (
    DEFAULT_CLUSTERING_DEPTH,
    _binary_rubric_to_criteria_text,
    _metric_id_from_rubric,
    build_binary_feature_matrix,
    compute_mutual_information,
    score_binary_subset,
    select_clustering_features,
)
from .token_utils import (
    compute_generation_example_budget,
    truncate_to_tokens,
)

logger = logging.getLogger("metric_tree.builder")


def _normalize_binary_rubric(rubric_raw: dict) -> Dict[str, str]:
    """Normalize a rubric dict to have 'yes' and 'no' keys."""
    normalized = {}
    for k, v in rubric_raw.items():
        k_lower = str(k).strip().lower()
        if k_lower in ("yes", "1", "true"):
            normalized["yes"] = str(v)
        elif k_lower in ("no", "0", "false"):
            normalized["no"] = str(v)
    # Fallback if keys are ordinal (1-5) — take 4-5 as yes, 1-2 as no
    if "yes" not in normalized and "no" not in normalized:
        vals = list(rubric_raw.values())
        keys = list(rubric_raw.keys())
        if len(vals) >= 2:
            # Assume higher keys = yes, lower keys = no
            normalized["yes"] = str(vals[-1])
            normalized["no"] = str(vals[0])
        elif len(vals) == 1:
            normalized["yes"] = str(vals[0])
            normalized["no"] = "Does not meet the criterion."
    if "yes" not in normalized:
        normalized["yes"] = "Meets the criterion."
    if "no" not in normalized:
        normalized["no"] = "Does not meet the criterion."
    return normalized


def _create_binary_tree_metrics(
    raw_metrics: List[Dict[str, Any]],
    source_node_id: str,
    existing_metric_ids: set,
    existing_names: set,
) -> List[TreeMetric]:
    """Convert raw metric dicts from the proposer into binary TreeMetric objects."""
    tree_metrics = []
    for m in raw_metrics:
        name = _sanitize_metric_name(m.get("name", "Metric"))
        rubric_raw = m.get("rubric", {})
        rubric = _normalize_binary_rubric(rubric_raw)
        rubric_text = f"YES: {rubric['yes']}\nNO: {rubric['no']}"
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
            scale="binary",
        ))
    return tree_metrics


def _sample_examples(
    df: pd.DataFrame,
    label_column: str,
    n_per_class: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample positive and negative examples for metric generation."""
    pos_df = df[df[label_column] == 1]
    neg_df = df[df[label_column] == 0]

    pos_sample = pos_df.sample(n=min(n_per_class, len(pos_df)), random_state=seed)
    neg_sample = neg_df.sample(n=min(n_per_class, len(neg_df)), random_state=seed)

    return pos_sample, neg_sample


def _format_sibling_and_uncle_context(
    current_key: Tuple[int, ...],
    all_partitions: Dict[Tuple[int, ...], np.ndarray],
    parent_metric_names: List[str],
    parent: PartitionTreeNode,
    tree: MetricTree,
    train_df: pd.DataFrame,
    label_column: str,
) -> str:
    """Format 2-3 sentence contrastive summaries of sibling and uncle/aunt branches.

    Siblings: other partitions at the same level (same parent).
    Uncle/aunts: parent's siblings (same grandparent).
    """
    lines = []

    # --- Siblings: other partitions from the same parent ---
    sibling_descs = []
    for key, indices in all_partitions.items():
        if key == current_key:
            continue
        global_idx = parent.point_indices[indices]
        labels = train_df.iloc[global_idx][label_column].values.astype(float)
        acc_rate = float((labels == 1).mean()) if len(labels) > 0 else 0.5
        desc = format_partition_description(key, parent_metric_names)
        sibling_descs.append(
            f"  - [{desc}]: {len(indices):,} examples, "
            f"{acc_rate:.0%} accepted"
        )

    if sibling_descs:
        lines.append("SIBLING BRANCHES (other partitions from the same parent node):")
        # Show at most 4 siblings to keep it concise
        for s in sibling_descs[:4]:
            lines.append(s)
        if len(sibling_descs) > 4:
            lines.append(f"  ... and {len(sibling_descs) - 4} more sibling partitions")
        lines.append("")

    # --- Uncle/aunt: parent's siblings (nodes with the same grandparent) ---
    if parent.parent_id and tree.all_nodes:
        grandparent = tree.all_nodes.get(parent.parent_id)
        gp_metric_names = (
            [m.name for m in grandparent.local_metrics]
            if grandparent and grandparent.local_metrics else []
        )
        uncle_descs = []
        for nid, node in tree.all_nodes.items():
            if node.parent_id == parent.parent_id and nid != parent.node_id:
                n_examples = len(node.point_indices)
                acc_rate = node.n_positive / max(1, node.n_positive + node.n_negative)
                # partition_key is defined by grandparent's metrics
                local_desc = format_partition_description(
                    node.partition_key, gp_metric_names,
                ) if gp_metric_names and node.partition_key else ""
                uncle_descs.append(
                    f"  - {node.node_id} [{local_desc}]: {n_examples:,} examples, "
                    f"{acc_rate:.0%} accepted"
                )

        if uncle_descs:
            lines.append("PARENT'S SIBLINGS (uncle/aunt branches — same grandparent, different parent partition):")
            for u in uncle_descs[:3]:
                lines.append(u)
            if len(uncle_descs) > 3:
                lines.append(f"  ... and {len(uncle_descs) - 3} more")
            lines.append("")

    return "\n".join(lines)


def _compute_base_rate(labels: np.ndarray) -> float:
    """Compute base rate (fraction of positive examples)."""
    n = len(labels)
    if n == 0:
        return 0.5
    return float((labels == 1).sum()) / n


def _format_round_history(round_history: List[Dict]) -> str:
    """Format accumulated history of all prior rounds' failed features."""
    if not round_history:
        return ""

    lines = ["PRIOR ATTEMPTS (all produced features that were too lenient):"]
    for entry in round_history:
        lines.append(f"\n  Round {entry['round']}:")
        for feat in entry["skewed"]:
            lines.append(
                f"    - '{feat['name']}': P(YES) = {feat['p_yes']:.1%}"
                f"  (YES criterion: \"{feat['rubric_yes']}\")"
            )
    lines.append(
        "\nAll the above features were USELESS because nearly every paper scored YES. "
        "Do NOT repeat similar features or similarly lenient rubrics. "
        "Your new features must be MUCH more specific and demanding."
    )
    return "\n".join(lines)


def _format_yes_examples(
    scored_df: pd.DataFrame,
    skewed: List[Tuple],
    id_column: str,
    text_column: str,
    label_column: str,
    max_chars: int = 250,
) -> str:
    """Pick papers that scored YES on the most skewed feature — both accepted and rejected.

    Shows the model concrete evidence that its rubric was too easy.
    """
    if not skewed:
        return ""

    # Pick the most skewed feature
    worst_idx, worst_metric, worst_p = max(skewed, key=lambda x: abs(x[2] - 0.5))
    if worst_metric.name not in scored_df.columns:
        return ""

    yes_papers = scored_df[scored_df[worst_metric.name] == 1]
    if len(yes_papers) == 0:
        return ""

    rejected_yes = yes_papers[yes_papers[label_column] == 0].head(2)
    accepted_yes = yes_papers[yes_papers[label_column] == 1].head(1)

    lines = [
        f"WHY YOUR RUBRIC WAS TOO EASY — example papers that ALL scored YES on "
        f"'{worst_metric.name}' (P(YES)={worst_p:.1%}):",
        f"  Your YES criterion was: \"{worst_metric.rubric.get('yes', '?')[:200]}\"",
        "",
    ]
    for _, row in rejected_yes.iterrows():
        text = str(row[text_column])[:max_chars].replace("\n", " ")
        lines.append(f"  REJECTED paper (but scored YES!): {text}...")
    for _, row in accepted_yes.iterrows():
        text = str(row[text_column])[:max_chars].replace("\n", " ")
        lines.append(f"  ACCEPTED paper (scored YES): {text}...")
    lines.append(
        f"\n  Both accepted AND rejected papers easily pass your '{worst_metric.name}' "
        f"criterion. The rubric is too broad — nearly any paper qualifies as YES."
    )

    return "\n".join(lines)


def _propose_and_refine(
    proposer: PartitionMetricProposer,
    config: TreeConfig,
    label_cache: LabelCache,
    sample_df: pd.DataFrame,
    pos_sample: pd.DataFrame,
    neg_sample: pd.DataFrame,
    task_description: str,
    node_id: str,
    *,
    parent: Any = None,
    partition_key: Tuple[int, ...] = (),
    id_column: str,
    text_column: str,
    label_column: str,
    scoring_backend: Any,
    gen_example_tokens: int = 0,
    tokenizer: Any = None,
    max_model_len: int = 0,
    population_size: int = 0,
    positive_rate: float = 0.5,
    sample_scored: Any = None,
    sibling_context: str = "",
    exception_context: str = "",
) -> List[TreeMetric]:
    """Propose metrics, score a sample, refine skewed ones, return final candidates.

    Iterates up to config.max_refinement_rounds times:
    1. Propose candidate features
    2. Score a sample (config.refinement_sample_size)
    3. Check balance: features with P(YES) outside [min_balance, 1-min_balance] are skewed
    4. Re-propose replacements for skewed features
    """
    K_propose = config.n_rubrics_to_propose
    min_bal = config.min_feature_balance
    existing_metric_ids: set = set()
    existing_names: set = set()

    # Initial proposal
    logger.info("Proposing %d binary metrics (round 1)...", K_propose)
    raw_metrics = proposer.propose(
        task_description=task_description,
        parent=parent,
        partition_key=partition_key,
        positive_df=pos_sample,
        negative_df=neg_sample,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        num_metrics=K_propose,
        max_example_tokens=gen_example_tokens,
        tokenizer=tokenizer,
        scoring_backend=scoring_backend,
        contrastive_pairs_k=config.contrastive_pairs_k,
        population_size=population_size,
        positive_rate=positive_rate,
        sample_scored=sample_scored,
        sibling_context=sibling_context,
        exception_context=exception_context,
    )
    logger.info("Proposer returned %d metrics", len(raw_metrics))

    if not raw_metrics:
        return []

    candidate_metrics = _create_binary_tree_metrics(
        raw_metrics, node_id, existing_metric_ids, existing_names,
    )
    if not candidate_metrics:
        return []

    # Refinement loop: score different samples each round, accumulate history,
    # show the model its own failures with concrete examples.
    sample_size = min(config.refinement_sample_size, len(sample_df))
    round_history: List[Dict] = []  # cumulative history of all failed rounds

    for round_num in range(config.max_refinement_rounds):
        # Draw a DIFFERENT sample each round for robustness
        rng = np.random.RandomState(config.random_seed + round_num)
        round_indices = rng.choice(len(sample_df), size=sample_size, replace=False)
        round_sample = sample_df.iloc[round_indices].reset_index(drop=True)

        # Score sample on current candidates
        logger.info("Refinement round %d/%d: scoring %d-example sample on %d candidates...",
                     round_num + 1, config.max_refinement_rounds,
                     len(round_sample), len(candidate_metrics))
        sample_scored_df = score_binary_subset(
            round_sample, np.arange(len(round_sample)), candidate_metrics, label_cache,
            id_column=id_column, text_column=text_column, label_column=label_column,
            task_description=task_description, scoring_backend=scoring_backend,
            batch_size=config.label_batch_size, verbose=False,
            stage=f"{node_id}_refine_r{round_num+1}",
            tokenizer=tokenizer, max_model_len=max_model_len,
        )

        # Check balance
        cand_names = [m.name for m in candidate_metrics]
        X_sample, _ = build_binary_feature_matrix(sample_scored_df, cand_names, label_column)

        skewed = []
        good = []
        for i, m in enumerate(candidate_metrics):
            p_yes = float(X_sample[:, i].mean())
            logger.info("  %s: P(YES)=%.3f", m.name, p_yes)
            if p_yes < min_bal or p_yes > (1 - min_bal):
                skewed.append((i, m, p_yes))
            else:
                good.append((i, m, p_yes))

        if not skewed:
            logger.info("All %d features have acceptable balance — no refinement needed", len(candidate_metrics))
            break

        logger.info("%d/%d features are skewed (P(YES) outside [%.0f%%, %.0f%%]) — round %d/%d",
                     len(skewed), len(candidate_metrics), min_bal*100, (1-min_bal)*100,
                     round_num + 1, config.max_refinement_rounds)

        # Record this round's failures in history
        round_history.append({
            "round": round_num + 1,
            "skewed": [
                {
                    "name": m.name,
                    "p_yes": p,
                    "rubric_yes": m.rubric.get("yes", "?")[:150],
                }
                for _, m, p in skewed
            ],
        })

        # Gather concrete examples of papers that scored YES on skewed features
        example_papers = _format_yes_examples(
            sample_scored_df, skewed, id_column, text_column, label_column,
        )

        # Format cumulative prior rounds history
        prior_summary = _format_round_history(round_history)

        # Ask proposer to refine with full history + concrete examples
        skewed_info = [{"name": m.name, "p_yes": p, "rubric": m.rubric} for _, m, p in skewed]
        good_info = [{"name": m.name, "p_yes": p, "rubric": m.rubric} for _, m, p in good]

        replacement_raws = proposer.refine(
            task_description=task_description,
            skewed_features=skewed_info,
            good_features=good_info,
            num_replacements=len(skewed),
            parent=parent,
            partition_key=partition_key,
            positive_df=pos_sample,
            negative_df=neg_sample,
            id_column=id_column,
            text_column=text_column,
            label_column=label_column,
            max_example_tokens=gen_example_tokens,
            tokenizer=tokenizer,
            population_size=population_size,
            positive_rate=positive_rate,
            round_num=round_num + 1,
            prior_rounds_summary=prior_summary,
            example_papers_on_skewed=example_papers,
        )

        if not replacement_raws:
            logger.warning("Refinement returned no replacements (round %d), keeping current candidates",
                          round_num + 1)
            break

        replacement_metrics = _create_binary_tree_metrics(
            replacement_raws, node_id, existing_metric_ids, existing_names,
        )
        logger.info("Refinement round %d produced %d replacement metrics",
                     round_num + 1, len(replacement_metrics))

        # Replace skewed metrics with new ones
        new_candidates = [m for _, m, _ in good]
        new_candidates.extend(replacement_metrics)
        candidate_metrics = new_candidates

        if not candidate_metrics:
            logger.warning("No candidates remaining after refinement round %d", round_num + 1)
            break

    if round_history:
        logger.info("Refinement used %d rounds total", len(round_history))
    logger.info("Final candidate pool: %d metrics", len(candidate_metrics))
    return candidate_metrics


def build_root_node(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TreeConfig,
    label_cache: LabelCache,
    proposer: PartitionMetricProposer,
    task_description: str,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    scoring_backend: Any,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> Tuple[PartitionTreeNode, Dict[str, TreeMetric]]:
    """Build the root node of the Partition Metric Tree.

    Steps:
    1. Sample positive/negative examples
    2. Propose K_propose binary metrics with refinement loop on sample
    3. Score full train + eval on final candidates
    4. Select top K by clustering quality or MI
    5. Compute base rate
    """
    node_id = "root"
    K = config.n_binary_metrics_per_level
    logger.info("Building root node (proposing %d, selecting %d)...", config.n_rubrics_to_propose, K)

    # 1. Sample examples for metric generation
    pos_sample, neg_sample = _sample_examples(
        train_df, label_column, n_per_class=config.exception_examples_per_class,
        seed=config.random_seed,
    )

    # Population stats for prompt
    combined_df = pd.concat([train_df, eval_df], ignore_index=True)
    population_size = len(combined_df)
    positive_rate = float((combined_df[label_column].astype(float) == 1).mean())

    # Token budget for examples
    gen_example_tokens = 0
    if max_model_len > 0 and tokenizer is not None:
        n_total = len(pos_sample) + len(neg_sample)
        gen_example_tokens = compute_generation_example_budget(
            current_metrics_text="",
            task_description=task_description,
            max_model_len=max_model_len,
            n_total_examples=max(1, n_total),
            tokenizer=tokenizer,
        )

    # 2. Propose + refine on sample
    candidate_metrics = _propose_and_refine(
        proposer=proposer,
        config=config,
        label_cache=label_cache,
        sample_df=combined_df,
        pos_sample=pos_sample,
        neg_sample=neg_sample,
        task_description=task_description,
        node_id=node_id,
        parent=None,
        partition_key=(),
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        scoring_backend=scoring_backend,
        gen_example_tokens=gen_example_tokens,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        population_size=population_size,
        positive_rate=positive_rate,
    )

    if not candidate_metrics:
        raise RuntimeError("No valid metrics after proposal + refinement for root node")

    # 3. Score full train+eval on final candidates
    all_indices = np.arange(len(combined_df))
    logger.info("Scoring %d examples on %d final candidate metrics...", len(combined_df), len(candidate_metrics))
    scored_df = score_binary_subset(
        combined_df, all_indices, candidate_metrics, label_cache,
        id_column=id_column, text_column=text_column, label_column=label_column,
        task_description=task_description, scoring_backend=scoring_backend,
        batch_size=config.label_batch_size, verbose=config.verbose,
        stage="root_score", tokenizer=tokenizer, max_model_len=max_model_len,
    )

    # 4. Select top K features
    candidate_names = [m.name for m in candidate_metrics]
    X_all, y_all = build_binary_feature_matrix(scored_df, candidate_names, label_column)

    # Root is always depth 0 → use clustering selection if clustering_depth > 0
    is_clustering = 0 < config.clustering_depth
    if is_clustering:
        logger.info("Using CLUSTERING selection (high-entropy, low-redundancy)")
        for i, name in enumerate(candidate_names):
            p = X_all[:, i].mean()
            logger.info("  %s: P(YES)=%.3f (balance=%.2f)", name, p, min(p, 1-p)*2)
        top_k_indices = select_clustering_features(X_all, candidate_names, K)
    else:
        mi_scores = compute_mutual_information(X_all, y_all)
        for name, mi in zip(candidate_names, mi_scores):
            logger.info("  MI(%s) = %.4f", name, mi)
        top_k_indices = list(np.argsort(mi_scores)[::-1][:K])

    selected_metrics = [candidate_metrics[i] for i in top_k_indices]
    selected_names = [m.name for m in selected_metrics]
    logger.info("Selected %d metrics: %s", len(selected_metrics), selected_names)

    # Get binary scores for selected metrics only
    X_selected, y_selected = build_binary_feature_matrix(scored_df, selected_names, label_column)

    # Split back into train/eval
    n_train = len(train_df)
    X_train = X_selected[:n_train]
    y_train = y_selected[:n_train]
    X_eval = X_selected[n_train:]
    y_eval = y_selected[n_train:]

    # 6. Compute base rate
    base_rate = _compute_base_rate(y_train)
    logger.info("Root base rate: %.4f (%d pos, %d neg)",
                base_rate, int((y_train == 1).sum()), int((y_train == 0).sum()))

    # Train router if enabled
    root_router = None
    if config.use_router and n_train >= config.router_min_examples:
        logger.info("Training root router...")
        train_texts = train_df[text_column].astype(str).tolist()
        root_router = train_node_router(
            texts=train_texts,
            labels=y_train,
            base_rate=base_rate,
            embedding_model_name=config.embedding_model,
            n_epochs=config.router_n_epochs,
            batch_size=config.router_batch_size,
            learning_rate=config.router_learning_rate,
            hidden_dim=config.router_hidden_dim,
            dropout=config.router_dropout,
            seed=config.random_seed,
            min_examples=config.router_min_examples,
        )

    # Build node
    node = PartitionTreeNode(
        node_id=node_id,
        depth=0,
        parent_id=None,
        partition_key=(),
        local_metrics=selected_metrics,
        all_metrics=list(selected_metrics),
        point_indices=np.arange(n_train),
        local_scores=X_train,
        all_scores=X_train,
        base_rate=base_rate,
        n_positive=int((y_train == 1).sum()),
        n_negative=int((y_train == 0).sum()),
        router=root_router,
        router_minority_is_positive=root_router.minority_is_positive if root_router else None,
    )

    all_metrics_dict = {m.metric_id: m for m in selected_metrics}
    return node, all_metrics_dict


def build_partition_children(
    parent: PartitionTreeNode,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TreeConfig,
    label_cache: LabelCache,
    proposer: PartitionMetricProposer,
    task_description: str,
    tree: MetricTree,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    scoring_backend: Any,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> None:
    """Build child partitions for a parent node.

    1. Assign parent's examples to 2^K partitions
    2. Prune small partitions
    3. For each surviving partition:
       a. Check contrastive pairs → leaf if insufficient
       b. Propose K new binary metrics
       c. Score, select by MI, compute base rate
    """
    K = config.n_binary_metrics_per_level
    K_propose = config.n_rubrics_to_propose

    # Assign to partitions based on parent's local_scores
    partitions = assign_to_partitions(parent.local_scores)
    metric_names = [m.name for m in parent.local_metrics]

    logger.info(
        "Node %s: %d partitions from %d examples (K=%d)",
        parent.node_id, len(partitions), len(parent.point_indices), K,
    )
    for key, indices in partitions.items():
        logger.info("  Partition %s: %d examples", _format_key(key), len(indices))

    # Prune
    partitions = prune_partitions(partitions, config.min_partition_size)
    if not partitions:
        logger.info("No surviving partitions for %s", parent.node_id)
        parent.is_leaf = True
        return

    # Process each partition
    for partition_key, local_indices in partitions.items():
        # local_indices are indices into parent.point_indices
        global_indices = parent.point_indices[local_indices]
        partition_df = train_df.iloc[global_indices].reset_index(drop=True)
        labels = train_df.iloc[global_indices][label_column].values.astype(float)

        n_pairs = count_contrastive_pairs(labels)
        key_str = _format_key(partition_key)
        child_node_id = f"{parent.node_id}_p{key_str}"

        logger.info(
            "Partition %s (%s): %d examples, %d contrastive pairs",
            key_str, format_partition_description(partition_key, metric_names),
            len(global_indices), n_pairs,
        )

        # Accumulate parent's all_scores for this partition
        parent_all_scores = parent.all_scores[local_indices]

        # Check minority fraction for --errors-only pruning
        minority_frac = min(
            float((labels == 1).sum()), float((labels == 0).sum())
        ) / max(len(labels), 1)

        if n_pairs < config.min_contrastive_pairs:
            logger.info("  Too few pairs (%d < %d) → leaf", n_pairs, config.min_contrastive_pairs)
            child = PartitionTreeNode(
                node_id=child_node_id,
                depth=parent.depth + 1,
                parent_id=parent.node_id,
                partition_key=partition_key,
                local_metrics=[],
                all_metrics=list(parent.all_metrics),
                point_indices=global_indices,
                local_scores=np.empty((len(global_indices), 0)),
                all_scores=parent_all_scores,
                base_rate=_compute_base_rate(labels),
                n_positive=int((labels == 1).sum()),
                n_negative=int((labels == 0).sum()),
                is_leaf=True,
            )
            parent.children[partition_key] = child
            tree.all_nodes[child_node_id] = child
            continue

        if config.min_minority_fraction > 0 and minority_frac < config.min_minority_fraction:
            logger.info("  Minority fraction %.1f%% < %.1f%% → leaf (--errors-only pruning)",
                       minority_frac * 100, config.min_minority_fraction * 100)
            child = PartitionTreeNode(
                node_id=child_node_id,
                depth=parent.depth + 1,
                parent_id=parent.node_id,
                partition_key=partition_key,
                local_metrics=[],
                all_metrics=list(parent.all_metrics),
                point_indices=global_indices,
                local_scores=np.empty((len(global_indices), 0)),
                all_scores=parent_all_scores,
                base_rate=_compute_base_rate(labels),
                n_positive=int((labels == 1).sum()),
                n_negative=int((labels == 0).sum()),
                is_leaf=True,
            )
            parent.children[partition_key] = child
            tree.all_nodes[child_node_id] = child
            continue

        # Select representative examples for proposer, emphasizing "exceptions"
        # (minority-class examples that the base-rate prediction gets wrong)
        pos_df = partition_df[partition_df[label_column] == 1]
        neg_df = partition_df[partition_df[label_column] == 0]

        if len(pos_df) == 0 or len(neg_df) == 0:
            # Degenerate: single-class partition → leaf
            child = PartitionTreeNode(
                node_id=child_node_id,
                depth=parent.depth + 1,
                parent_id=parent.node_id,
                partition_key=partition_key,
                local_metrics=[],
                all_metrics=list(parent.all_metrics),
                point_indices=global_indices,
                local_scores=np.empty((len(global_indices), 0)),
                all_scores=parent_all_scores,
                base_rate=_compute_base_rate(labels),
                n_positive=int((labels == 1).sum()),
                n_negative=int((labels == 0).sum()),
                is_leaf=True,
            )
            parent.children[partition_key] = child
            tree.all_nodes[child_node_id] = child
            continue

        # Determine majority/minority class: show MORE minority examples (exceptions)
        base_rate_here = _compute_base_rate(labels)
        majority_is_positive = base_rate_here >= 0.5
        k_base = config.exception_examples_per_class

        if majority_is_positive:
            # Rejected papers are exceptions — show more of them
            k_minority = min(k_base * 2, len(neg_df))
            k_majority = k_base
            exception_class = "rejected"
        else:
            # Accepted papers are exceptions — show more of them
            k_minority = min(k_base * 2, len(pos_df))
            k_majority = k_base
            exception_class = "accepted"

        pos_sample = cluster_and_select(
            pos_df, text_column,
            k=k_minority if not majority_is_positive else k_majority,
            model_name=config.embedding_model, seed=config.random_seed,
        )
        neg_sample = cluster_and_select(
            neg_df, text_column,
            k=k_minority if majority_is_positive else k_majority,
            model_name=config.embedding_model, seed=config.random_seed,
        )

        # Build exception context for the proposer
        n_minority = int((labels == 0).sum()) if majority_is_positive else int((labels == 1).sum())
        exception_context = (
            f"EXCEPTIONS: This partition predicts '{('accepted' if majority_is_positive else 'rejected')}' "
            f"for all {len(labels):,} examples (base rate = {base_rate_here:.0%} accepted). "
            f"But {n_minority:,} examples ({n_minority/len(labels):.0%}) are {exception_class} — "
            f"the partition gets them WRONG. We show extra {exception_class} examples below "
            f"because they represent the prediction errors your features should help explain."
        )

        # Token budget
        gen_example_tokens = 0
        if max_model_len > 0 and tokenizer is not None:
            partition_context = format_partition_description(partition_key, metric_names)
            current_metrics_text = "\n".join([
                f"{m.name}: YES={m.rubric.get('yes', '')[:100]}"
                for m in parent.all_metrics
            ])
            n_total = len(pos_sample) + len(neg_sample)
            gen_example_tokens = compute_generation_example_budget(
                current_metrics_text=current_metrics_text,
                task_description=task_description,
                max_model_len=max_model_len,
                n_total_examples=max(1, n_total),
                tokenizer=tokenizer,
            )

        # Score sample on parent metrics (for contrastive pairs in proposer)
        sample_combined = pd.concat([pos_sample, neg_sample], ignore_index=True)
        sample_scored = None
        if parent.all_metrics:
            try:
                sample_scored = score_binary_subset(
                    sample_combined, np.arange(len(sample_combined)),
                    parent.all_metrics, label_cache,
                    id_column=id_column, text_column=text_column, label_column=label_column,
                    task_description=task_description, scoring_backend=scoring_backend,
                    batch_size=config.label_batch_size, verbose=False,
                    stage=f"{child_node_id}_sample",
                    tokenizer=tokenizer, max_model_len=max_model_len,
                )
            except Exception as e:
                logger.warning("Failed to score sample: %s", e)

        # Build sibling + uncle/aunt contrastive context
        sibling_ctx = _format_sibling_and_uncle_context(
            current_key=partition_key,
            all_partitions=partitions,
            parent_metric_names=metric_names,
            parent=parent,
            tree=tree,
            train_df=train_df,
            label_column=label_column,
        )

        # Propose + refine on sample
        partition_pop_size = len(partition_df)
        partition_pos_rate = float((labels == 1).mean()) if len(labels) > 0 else 0.5

        logger.info("  Proposing %d binary metrics for partition %s (with refinement)...",
                     config.n_rubrics_to_propose, key_str)
        new_metrics = _propose_and_refine(
            proposer=proposer,
            config=config,
            label_cache=label_cache,
            sample_df=partition_df,
            pos_sample=pos_sample,
            neg_sample=neg_sample,
            task_description=task_description,
            node_id=child_node_id,
            parent=parent,
            partition_key=partition_key,
            id_column=id_column,
            text_column=text_column,
            label_column=label_column,
            scoring_backend=scoring_backend,
            gen_example_tokens=gen_example_tokens,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            population_size=partition_pop_size,
            positive_rate=partition_pos_rate,
            sample_scored=sample_scored,
            sibling_context=sibling_ctx,
            exception_context=exception_context,
        )

        if not new_metrics:
            logger.warning("  No metrics after proposal+refinement for %s → leaf", key_str)
            child = PartitionTreeNode(
                node_id=child_node_id,
                depth=parent.depth + 1,
                parent_id=parent.node_id,
                partition_key=partition_key,
                local_metrics=[],
                all_metrics=list(parent.all_metrics),
                point_indices=global_indices,
                local_scores=np.empty((len(global_indices), 0)),
                all_scores=parent_all_scores,
                base_rate=_compute_base_rate(labels),
                n_positive=int((labels == 1).sum()),
                n_negative=int((labels == 0).sum()),
                is_leaf=True,
            )
            parent.children[partition_key] = child
            tree.all_nodes[child_node_id] = child
            continue

        # Score partition on new metrics
        logger.info("  Scoring %d examples on %d new metrics...", len(partition_df), len(new_metrics))
        new_scored = score_binary_subset(
            partition_df, np.arange(len(partition_df)), new_metrics, label_cache,
            id_column=id_column, text_column=text_column, label_column=label_column,
            task_description=task_description, scoring_backend=scoring_backend,
            batch_size=config.label_batch_size, verbose=config.verbose,
            stage=f"{child_node_id}_new", tokenizer=tokenizer, max_model_len=max_model_len,
        )

        # Select top K — clustering vs. discriminative based on child depth
        new_names = [m.name for m in new_metrics]
        X_new, y_new = build_binary_feature_matrix(new_scored, new_names, label_column)
        child_depth = parent.depth + 1
        is_clustering = child_depth < config.clustering_depth

        if is_clustering:
            logger.info("  Using CLUSTERING selection (depth=%d)", child_depth)
            for i, name in enumerate(new_names):
                p = X_new[:, i].mean()
                logger.info("    %s: P(YES)=%.3f (balance=%.2f)", name, p, min(p, 1-p)*2)
            top_k_idx = select_clustering_features(X_new, new_names, K)
        else:
            logger.info("  Using DISCRIMINATIVE selection (MI, depth=%d)", child_depth)
            mi_scores = compute_mutual_information(X_new, y_new)
            for name, mi in zip(new_names, mi_scores):
                logger.info("    MI(%s) = %.4f", name, mi)
            top_k_idx = list(np.argsort(mi_scores)[::-1][:K])

        selected_new = [new_metrics[i] for i in top_k_idx]
        selected_new_names = [m.name for m in selected_new]
        logger.info("  Selected: %s", selected_new_names)

        # Build accumulated scores: parent_all_scores + new local scores
        X_local, _ = build_binary_feature_matrix(new_scored, selected_new_names, label_column)
        X_all_child = np.hstack([parent_all_scores, X_local])
        all_child_metrics = list(parent.all_metrics) + selected_new

        # Compute base rate
        br = _compute_base_rate(labels)
        logger.info("  Child %s: base_rate=%.3f (%d pos, %d neg)",
                    key_str, br, int((labels == 1).sum()), int((labels == 0).sum()))

        # Train router if enabled
        child_router = None
        if config.use_router and len(global_indices) >= config.router_min_examples:
            partition_texts = train_df.iloc[global_indices][text_column].astype(str).tolist()
            child_router = train_node_router(
                texts=partition_texts,
                labels=labels,
                base_rate=br,
                embedding_model_name=config.embedding_model,
                n_epochs=config.router_n_epochs,
                batch_size=config.router_batch_size,
                learning_rate=config.router_learning_rate,
                hidden_dim=config.router_hidden_dim,
                dropout=config.router_dropout,
                seed=config.random_seed,
                min_examples=config.router_min_examples,
            )

        child = PartitionTreeNode(
            node_id=child_node_id,
            depth=parent.depth + 1,
            parent_id=parent.node_id,
            partition_key=partition_key,
            local_metrics=selected_new,
            all_metrics=all_child_metrics,
            point_indices=global_indices,
            local_scores=X_local,
            all_scores=X_all_child,
            base_rate=br,
            n_positive=int((labels == 1).sum()),
            n_negative=int((labels == 0).sum()),
            router=child_router,
            router_minority_is_positive=child_router.minority_is_positive if child_router else None,
        )

        parent.children[partition_key] = child
        tree.all_nodes[child_node_id] = child

        # Register new metrics
        for m in selected_new:
            tree.all_metrics[m.metric_id] = m


def grow_partition_tree(
    node: PartitionTreeNode,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TreeConfig,
    label_cache: LabelCache,
    proposer: PartitionMetricProposer,
    task_description: str,
    tree: MetricTree,
    *,
    id_column: str,
    text_column: str,
    label_column: str,
    scoring_backend: Any,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> None:
    """Recursively grow the tree from a node by building partition children."""
    if node.depth >= config.max_depth:
        logger.info("Max depth %d reached at node %s", config.max_depth, node.node_id)
        node.is_leaf = True
        return

    if node.is_leaf:
        return

    if not node.local_metrics:
        logger.info("Node %s has no local metrics, marking as leaf", node.node_id)
        node.is_leaf = True
        return

    build_partition_children(
        parent=node,
        train_df=train_df,
        eval_df=eval_df,
        config=config,
        label_cache=label_cache,
        proposer=proposer,
        task_description=task_description,
        tree=tree,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        scoring_backend=scoring_backend,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
    )

    # Recurse on children
    for child in node.children.values():
        if not child.is_leaf:
            grow_partition_tree(
                node=child,
                train_df=train_df,
                eval_df=eval_df,
                config=config,
                label_cache=label_cache,
                proposer=proposer,
                task_description=task_description,
                tree=tree,
                id_column=id_column,
                text_column=text_column,
                label_column=label_column,
                scoring_backend=scoring_backend,
                tokenizer=tokenizer,
                max_model_len=max_model_len,
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
    cache_dir: Optional[str] = None,
    scoring_backend: Any = None,
    tokenizer: Any = None,
    max_model_len: int = 0,
) -> MetricTree:
    """Build a complete Partitioned Metric Tree.

    Orchestrates: root construction → recursive partition growth.
    """
    if scoring_backend is None:
        raise ValueError("scoring_backend is required for binary scoring")

    # Set up label cache
    if cache_dir is None:
        cache_dir = str(Path(config.output_dir) / "label_cache")
    label_cache = LabelCache(cache_dir)

    # Coerce labels to binary
    train_df = _coerce_binary_labels(train_df, label_column)
    eval_df = _coerce_binary_labels(eval_df, label_column)

    # Initialize tree
    tree = MetricTree(config=config, task_description=task_description)

    # Build partition proposer with clustering depth from config
    partition_proposer = PartitionMetricProposer(
        proposer, clustering_depth=config.clustering_depth,
    )

    # Build root
    root_node, root_metrics = build_root_node(
        train_df=train_df,
        eval_df=eval_df,
        config=config,
        label_cache=label_cache,
        proposer=partition_proposer,
        task_description=task_description,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        scoring_backend=scoring_backend,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
    )

    tree.root = root_node
    tree.all_nodes[root_node.node_id] = root_node
    tree.all_metrics.update(root_metrics)

    # Grow tree recursively
    grow_partition_tree(
        node=root_node,
        train_df=train_df,
        eval_df=eval_df,
        config=config,
        label_cache=label_cache,
        proposer=partition_proposer,
        task_description=task_description,
        tree=tree,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        scoring_backend=scoring_backend,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
    )

    n_nodes = len(tree.all_nodes)
    n_metrics = len(tree.all_metrics)
    n_leaves = sum(1 for n in tree.all_nodes.values() if n.is_leaf or not n.children)
    logger.info("Metric Tree complete: %d nodes (%d leaves), %d unique metrics", n_nodes, n_leaves, n_metrics)

    return tree
