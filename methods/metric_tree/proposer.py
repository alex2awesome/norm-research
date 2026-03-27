"""PartitionMetricProposer: proposes binary metrics for partition tree nodes."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from autometrics.generator.ContrastiveRubricProposer import (
    ContrastiveRubricProposer,
    _parse_metrics_json,
    _sanitize_metric_name,
)
from autometrics.iterative_refinement.matching import exact_match
from autometrics.iterative_refinement.runner import (
    _format_examples,
    _truncate_text,
)

from .data_structures import PartitionTreeNode, TreeMetric
from .partition import format_partition_description
from .token_utils import truncate_to_tokens

logger = logging.getLogger("metric_tree.proposer")


def _build_ancestor_chain(
    node: Optional[PartitionTreeNode],
    partition_key: Tuple[int, ...] = (),
) -> List[Tuple[str, List[TreeMetric], Tuple[int, ...]]]:
    """Walk from root to current node, collecting (node_id, local_metrics, partition_key) at each level.

    Returns list of (description_string, local_metrics, key) from root downward,
    ending with the current (pending) partition_key under `node`.
    """
    chain: List[Tuple[str, List[TreeMetric], Tuple[int, ...]]] = []
    if node is None:
        return chain

    # Build path from current node up to root
    reverse_chain: List[Tuple[PartitionTreeNode, Tuple[int, ...]]] = []
    cur = node
    cur_key = partition_key
    while cur is not None:
        reverse_chain.append((cur, cur_key))
        cur_key = cur.partition_key
        # Walk up: find parent by parent_id (not stored as reference, so we stop)
        cur = None  # We don't have parent references, just the current node's info

    # We only have the current node + its partition key. But all_metrics accumulates
    # the full path, so we can reconstruct the chain from all_metrics.
    return reverse_chain[::-1]


def _format_partition_context(
    node: Optional[PartitionTreeNode],
    partition_key: Tuple[int, ...] = (),
) -> str:
    """Format parent/ancestor partition context for the proposer prompt.

    Provides a full description of the path from root to the current partition,
    so the LLM understands exactly what subpopulation it's working with.
    """
    if node is None:
        return ""

    lines = []

    # Build the full partition path description
    if node.all_metrics:
        # Describe what this partition IS — the full chain of decisions
        desc = format_partition_description(partition_key, [m.name for m in node.local_metrics])
        lines.append(f"CURRENT PARTITION: {desc}")
        lines.append("")

        # Show the full accumulated context — ALL features from root to here
        lines.append("FULL PATH from root to this partition (all features that define this subgroup):")
        for m in node.all_metrics:
            lines.append(f"  - {m.name}: YES means \"{m.rubric.get('yes', '?')[:120]}\"")
        lines.append("")

        # Summarize in plain language what kind of examples are in this partition
        yes_features = []
        no_features = []
        # node.local_metrics correspond to partition_key
        for val, m in zip(partition_key, node.local_metrics):
            if val == 1:
                yes_features.append(m.name.replace('_', ' '))
            else:
                no_features.append(m.name.replace('_', ' '))
        # Also include parent's partition key info from all_metrics minus local_metrics
        parent_metrics = node.all_metrics[:len(node.all_metrics) - len(node.local_metrics)]
        if parent_metrics and node.partition_key:
            for val, m in zip(node.partition_key, parent_metrics[-len(node.partition_key):]):
                if val == 1:
                    yes_features.append(m.name.replace('_', ' '))
                else:
                    no_features.append(m.name.replace('_', ' '))

        summary_parts = []
        if yes_features:
            summary_parts.append(f"characterized by: {', '.join(yes_features)}")
        if no_features:
            summary_parts.append(f"NOT characterized by: {', '.join(no_features)}")
        if summary_parts:
            lines.append(f"In plain language, these are examples {'; '.join(summary_parts)}.")

    return "\n".join(lines)


def _format_examples_with_binary_scores(
    df: pd.DataFrame,
    scored_df: Optional[pd.DataFrame],
    metrics: List[TreeMetric],
    id_column: str,
    text_column: str,
    label_column: str,
) -> str:
    """Format examples showing binary metric scores alongside text."""
    lines: List[str] = []
    metric_names = [m.name for m in metrics]

    for _, row in df.iterrows():
        doc_id = row[id_column]
        label = row[label_column]
        text = _truncate_text(str(row[text_column]))

        scores_str = ""
        if scored_df is not None and len(metric_names) > 0:
            score_row = scored_df[scored_df[id_column] == doc_id]
            if len(score_row) > 0:
                score_row = score_row.iloc[0]
                score_parts = []
                for mn in metric_names:
                    if mn in score_row.index:
                        val = score_row[mn]
                        if pd.notna(val):
                            score_parts.append(f"{mn}={'YES' if int(val) == 1 else 'NO'}")
                if score_parts:
                    scores_str = f" scores={{{', '.join(score_parts)}}}"

        lines.append(f"[id={doc_id} label={label}{scores_str}]\n{text}")

    return "\n\n".join(lines).strip()


def _generate_contrastive_pairs(
    sample_scored: Optional[pd.DataFrame],
    metrics: List[TreeMetric],
    id_column: str,
    label_column: str,
    text_column: str,
    k_pairs: int = 3,
) -> str:
    """Generate contrastive pairs from scored sample using exact_match.

    Finds pairs with similar binary scores but opposite labels.
    """
    if sample_scored is None or len(metrics) == 0:
        return ""

    feature_columns = [m.name for m in metrics]
    available = [c for c in feature_columns if c in sample_scored.columns]
    if not available:
        return ""

    try:
        pairs = exact_match(
            df=sample_scored,
            id_column=id_column,
            label_column=label_column,
            feature_columns=available,
            positive_label=1,
            k_pairs=k_pairs,
            seen_pairs=set(),
            max_feature_dist=1.0,
        )
    except Exception as e:
        logger.warning("Contrastive pair generation failed: %s", e)
        return ""

    if not pairs:
        return ""

    lines = []
    for pos_id, neg_id, dist in pairs:
        pos_row = sample_scored[sample_scored[id_column].astype(str) == str(pos_id)]
        neg_row = sample_scored[sample_scored[id_column].astype(str) == str(neg_id)]

        if len(pos_row) == 0 or len(neg_row) == 0:
            continue

        pos_row = pos_row.iloc[0]
        neg_row = neg_row.iloc[0]

        pos_text = _truncate_text(str(pos_row[text_column]), max_chars=300)
        neg_text = _truncate_text(str(neg_row[text_column]), max_chars=300)

        # Don't show per-pair feature scores — all items in this partition
        # share the same accumulated feature values (that defines the partition).
        lines.append(
            f"PAIR:\n"
            f"  ACCEPTED [id={pos_id}]:\n    {pos_text}\n"
            f"  REJECTED [id={neg_id}]:\n    {neg_text}"
        )

    result = "\n\n".join(lines)
    logger.info("Generated %d contrastive pairs", len(lines))
    return result


class PartitionMetricProposer:
    """Proposes binary metrics for partition tree nodes.

    Wraps ContrastiveRubricProposer with binary-forcing prompt injection
    and partition context.

    Two distinct generation modes:
    - **Clustering mode** (root / early depth): propose descriptive features that
      characterize *what kind of example* this is — axes along which the data
      naturally varies, roughly independent of the label.  The goal is to create
      meaningful subpopulations, not to predict the label directly.
    - **Discriminative mode** (deeper / within a partition): propose features
      that explain *why, within this specific subpopulation*, some examples are
      positive and others negative.  These are the partition-specific error
      explanations.
    """

    # Depth threshold: at depth < clustering_depth, use clustering mode.
    # At depth >= clustering_depth, switch to discriminative mode.
    clustering_depth: int = 2

    def __init__(
        self,
        base_proposer: ContrastiveRubricProposer,
        clustering_depth: int = 2,
    ):
        self.base_proposer = base_proposer
        self.clustering_depth = clustering_depth

    def propose(
        self,
        *,
        task_description: str,
        parent: Optional[PartitionTreeNode] = None,
        partition_key: Tuple[int, ...] = (),
        positive_df: pd.DataFrame,
        negative_df: pd.DataFrame,
        id_column: str,
        text_column: str,
        label_column: str,
        num_metrics: int = 5,
        max_example_tokens: int = 0,
        tokenizer: Any = None,
        sample_scored: Optional[pd.DataFrame] = None,
        scoring_backend: Any = None,
        contrastive_pairs_k: int = 3,
        population_size: int = 0,
        positive_rate: float = 0.5,
        sibling_context: str = "",
        exception_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Propose binary metrics for a tree node.

        At shallow depths (< clustering_depth), proposes descriptive clustering
        features.  At deeper depths, proposes discriminative features specific
        to the partition.
        """
        # Determine current depth
        current_depth = 0 if parent is None else parent.depth + 1
        is_clustering = current_depth < self.clustering_depth

        # Build partition context
        partition_context = ""
        accumulated_metrics: List[TreeMetric] = []
        if parent is not None:
            partition_context = _format_partition_context(parent, partition_key)
            accumulated_metrics = parent.all_metrics

        # Truncate texts
        if max_example_tokens > 0:
            positive_df = positive_df.copy()
            negative_df = negative_df.copy()
            positive_df[text_column] = positive_df[text_column].apply(
                lambda t: truncate_to_tokens(str(t), max_example_tokens, tokenizer=tokenizer),
            )
            negative_df[text_column] = negative_df[text_column].apply(
                lambda t: truncate_to_tokens(str(t), max_example_tokens, tokenizer=tokenizer),
            )

        # Format examples — in discriminative mode, skip per-example scores
        # since all examples in the partition share the same accumulated scores
        # (that's what defines the partition). Scores are summarized once in the prompt.
        if is_clustering and sample_scored is not None and accumulated_metrics:
            pos_examples = _format_examples_with_binary_scores(
                positive_df, sample_scored, accumulated_metrics,
                id_column, text_column, label_column,
            )
            neg_examples = _format_examples_with_binary_scores(
                negative_df, sample_scored, accumulated_metrics,
                id_column, text_column, label_column,
            )
        else:
            pos_examples = _format_examples(
                positive_df, id_column, text_column, label_column,
            )
            neg_examples = _format_examples(
                negative_df, id_column, text_column, label_column,
            )

        # Generate contrastive pairs (only useful in discriminative mode)
        contrastive_pairs = ""
        if not is_clustering:
            contrastive_pairs = _generate_contrastive_pairs(
                sample_scored, accumulated_metrics,
                id_column, label_column, text_column,
                k_pairs=contrastive_pairs_k,
            )

        # Build the task description based on mode
        if is_clustering:
            binary_task = self._build_clustering_prompt(
                task_description, num_metrics, partition_context,
                accumulated_metrics,
                population_size=population_size,
                positive_rate=positive_rate,
            )
        else:
            binary_task = self._build_discriminative_prompt(
                task_description, num_metrics, partition_context,
                accumulated_metrics,
                population_size=population_size,
                positive_rate=positive_rate,
                sibling_context=sibling_context,
                exception_context=exception_context,
            )

        # Current metrics context
        current_metrics_text = ""
        if accumulated_metrics:
            lines = ["Existing binary metrics:"]
            for m in accumulated_metrics:
                lines.append(f"  - {m.name}: YES={m.rubric.get('yes', '?')[:150]}")
            current_metrics_text = "\n".join(lines)

        logger.info("Proposing %d metrics in %s mode (depth=%d)",
                     num_metrics, "CLUSTERING" if is_clustering else "DISCRIMINATIVE",
                     current_depth)

        return self.base_proposer.propose(
            task_description=binary_task,
            positive_examples=pos_examples,
            negative_examples=neg_examples,
            current_metrics=current_metrics_text,
            contrastive_pairs=contrastive_pairs,
            num_metrics=num_metrics,
            num_rubrics=0,  # no ordinal rubrics
        )

    def _build_clustering_prompt(
        self,
        task_description: str,
        num_metrics: int,
        partition_context: str,
        accumulated_metrics: List[TreeMetric],
        population_size: int = 0,
        positive_rate: float = 0.5,
    ) -> str:
        """Build prompt for clustering mode: descriptive features that partition the data."""
        pop_info = ""
        if population_size > 0:
            pop_info = (
                f"\n\nPOPULATION: {population_size:,} examples "
                f"({positive_rate:.0%} positive, {1-positive_rate:.0%} negative). "
                f"Your features will be applied to ALL of them. A useful feature should "
                f"be YES for roughly 20-80% of examples — if nearly all examples would "
                f"get the same answer, the feature is not specific enough.\n"
            )

        prompt = (
            f"{task_description}{pop_info}\n\n"
            f"=== CLUSTERING FEATURES (Descriptive, NOT Predictive) ===\n\n"
            f"Your goal is to propose {num_metrics} binary (yes/no) features that DESCRIBE "
            f"what kind of example this is — NOT whether it is positive or negative.\n\n"
            f"These features should CLUSTER the data along meaningful, naturally-varying "
            f"axes. Think of them as typological descriptors: what *category* or *type* "
            f"of example is this?\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"  - Features must be roughly INDEPENDENT of the label. A good clustering "
            f"feature has a mix of positive and negative examples on BOTH sides "
            f"(YES and NO). If 90%+ of examples would get the same answer, the "
            f"feature is too generic.\n"
            f"  - Features should describe STRUCTURAL or TYPOLOGICAL properties: "
            f"the kind of approach, the domain, the methodology type, the scope, "
            f"the framing — NOT quality judgments.\n"
            f"  - Avoid features that are essentially 'is this good?' in disguise. "
            f"Do NOT propose features like 'Clear Contribution', 'Sound Methodology', "
            f"'Novel Approach', 'Well-Supported Claims', 'Significant Impact' — "
            f"these are quality assessments that almost all examples will score YES on.\n"
            f"  - Instead, propose features that DIVIDE the space roughly evenly "
            f"(ideally 30-70% YES) along dimensions like:\n"
            f"    * Empirical study vs. theoretical/analytical work\n"
            f"    * Proposes a new method/model vs. applies existing methods to new problem\n"
            f"    * Single-domain focused vs. cross-domain or interdisciplinary\n"
            f"    * Primarily quantitative results vs. primarily qualitative analysis\n"
            f"    * Addresses a well-established problem vs. defines a new problem\n"
            f"    * Large-scale evaluation vs. small-scale/proof-of-concept\n"
            f"  These are EXAMPLES of the right kind of axis. Propose features "
            f"specific to the domain described in the task.\n\n"
            f"For each criterion, provide:\n"
            f"  - A clear, specific name\n"
            f"  - A detailed description of what constitutes YES\n"
            f"  - A detailed description of what constitutes NO\n\n"
            f"The rubric for each metric MUST use keys 'yes' and 'no' with "
            f"detailed multi-sentence descriptions. The scale must be 'binary'."
        )

        if partition_context:
            prompt += (
                f"\n\n=== CURRENT PARTITION ===\n"
                f"{partition_context}\n\n"
                f"Propose new CLUSTERING features that further subdivide this "
                f"subpopulation along descriptive axes. Do NOT re-propose "
                f"features that overlap with the existing ones listed above. "
                f"Think about what meaningful sub-types exist WITHIN this group."
            )

        return prompt

    def _build_discriminative_prompt(
        self,
        task_description: str,
        num_metrics: int,
        partition_context: str,
        accumulated_metrics: List[TreeMetric],
        population_size: int = 0,
        positive_rate: float = 0.5,
        sibling_context: str = "",
        exception_context: str = "",
    ) -> str:
        """Build prompt for discriminative mode: features that explain label differences."""
        rejection_rate = 1.0 - positive_rate

        # --- Base rate with contextualization of WHY we show it ---
        pop_info = ""
        if population_size > 0:
            pop_info = (
                f"\n\nPOPULATION: {population_size:,} examples in this partition.\n"
                f"ACCEPTANCE RATE: {positive_rate:.0%} accepted, {rejection_rate:.0%} rejected.\n"
                f"Why this matters: even after filtering through all our existing features "
                f"(see below), {rejection_rate:.0%} of papers in this group are STILL "
                f"rejected. Our existing features capture what these papers have in common, "
                f"but they completely miss whatever causes {rejection_rate:.0%} to fail. "
                f"Your job is to find THAT missing signal.\n"
            )

        # --- Partition context: what defines this subpopulation ---
        partition_summary = ""
        if partition_context:
            partition_summary = (
                f"\n\n=== WHAT THIS PARTITION IS ===\n"
                f"{partition_context}\n\n"
                f"Given that ALL papers in this group share the above characteristics, "
                f"your features must be SPECIFIC to this type of paper. Generic quality "
                f"criteria (like 'Clear Contribution' or 'Rigorous Methodology') are "
                f"useless — they score YES for 95%+ of papers and fail to discriminate.\n"
            )

        # --- Exception context: minority-class examples the partition gets wrong ---
        exception_section = ""
        if exception_context:
            exception_section = (
                f"\n\n=== PREDICTION ERRORS (\"Exceptions\") ===\n"
                f"{exception_context}\n"
                f"Focus on what makes these exception examples DIFFERENT from the majority. "
                f"Your features should capture properties that distinguish exceptions from "
                f"correctly-predicted examples within this partition.\n"
            )

        # --- Shared feature summary (one-time, not repeated per pair) ---
        existing_feature_summary = ""
        if accumulated_metrics:
            feat_lines = []
            for m in accumulated_metrics:
                feat_lines.append(
                    f"  - {m.name.replace('_', ' ')}: "
                    f"YES = \"{m.rubric.get('yes', '?')[:80]}\""
                )
            existing_feature_summary = (
                f"EXISTING FEATURES (all scored identically for accepted AND rejected "
                f"papers in this partition — these FAILED to discriminate):\n"
                f"{chr(10).join(feat_lines)}\n\n"
                f"Every paper below — whether accepted or rejected — scored the same "
                f"on ALL of the above features. Yet some were accepted and others "
                f"rejected. The above features are blind to whatever is driving "
                f"the decision.\n\n"
            )

        # --- Sibling / uncle-aunt contrastive summaries ---
        sibling_section = ""
        if sibling_context:
            sibling_section = (
                f"\n\n=== OTHER BRANCHES (for contrast) ===\n"
                f"{sibling_context}\n"
                f"Use this context to understand what makes YOUR partition distinctive "
                f"and what kinds of features might matter specifically here.\n"
            )

        # --- Main discriminative prompt ---
        prompt = (
            f"{task_description}{pop_info}{partition_summary}"
            f"{exception_section}{sibling_section}\n"
            f"=== DISCRIMINATIVE FEATURES: What did our prior features miss? ===\n\n"
            f"{existing_feature_summary}"
            f"Your goal: propose {num_metrics} binary (yes/no) features that capture "
            f"SPECIFIC, CONCRETE differences between accepted and rejected papers of "
            f"this type — things our existing features missed.\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"  1. DEMAND SPECIFICITY in your rubrics. Write YES criteria that are "
            f"HARD TO SATISFY — only 20-80% of papers should qualify. If you write "
            f"'the paper has a clear methodology', virtually every paper will score YES. "
            f"Instead write something like 'the paper includes ablation studies or "
            f"sensitivity analysis comparing at least 3 variants of the approach'.\n"
            f"  2. THINK ABOUT FAILURE MODES. What specific weaknesses cause papers "
            f"like these to be rejected? Examples of useful features:\n"
            f"    - 'Compares against strong recent baselines (not just trivial ones)'\n"
            f"    - 'Provides error analysis or failure case discussion'\n"
            f"    - 'Reports confidence intervals or statistical significance tests'\n"
            f"    - 'Addresses limitations of the proposed approach explicitly'\n"
            f"    - 'Validates on multiple datasets or across different conditions'\n"
            f"  3. BE SPECIFIC TO THIS SUBPOPULATION. Think about what distinguishes "
            f"accepted from rejected papers SPECIFICALLY among papers of this type.\n"
            f"  4. DO NOT propose generic quality signals like 'Novel Contribution', "
            f"'Sound Methodology', 'Clear Research Question', 'Significant Impact', "
            f"'Well-Supported Claims'. These are always YES and waste a feature slot.\n\n"
            f"For each criterion, provide:\n"
            f"  - A clear, specific name\n"
            f"  - A YES description that is DEMANDING — most papers should NOT trivially satisfy it\n"
            f"  - A NO description that captures a specific, concrete deficiency\n\n"
            f"The rubric for each metric MUST use keys 'yes' and 'no' with "
            f"detailed multi-sentence descriptions. The scale must be 'binary'."
        )

        if partition_context and not partition_summary:
            prompt += (
                f"\n\n=== PARTITION CONTEXT ===\n"
                f"{partition_context}\n\n"
                f"Do NOT re-propose metrics that overlap with existing ones listed above."
            )

        return prompt

    def refine(
        self,
        *,
        task_description: str,
        skewed_features: List[Dict[str, float]],
        good_features: List[Dict[str, Any]],
        num_replacements: int,
        parent: Optional[PartitionTreeNode] = None,
        partition_key: Tuple[int, ...] = (),
        positive_df: pd.DataFrame,
        negative_df: pd.DataFrame,
        id_column: str,
        text_column: str,
        label_column: str,
        max_example_tokens: int = 0,
        tokenizer: Any = None,
        population_size: int = 0,
        positive_rate: float = 0.5,
        round_num: int = 1,
        prior_rounds_summary: str = "",
        example_papers_on_skewed: str = "",
    ) -> List[Dict[str, Any]]:
        """Re-propose features to replace skewed ones, with cumulative history.

        Shows the model all prior failed attempts plus concrete examples of
        papers that scored YES on skewed features, so it understands WHY
        its rubrics were too lenient.
        """
        current_depth = 0 if parent is None else parent.depth + 1
        is_clustering = current_depth < self.clustering_depth

        # Build feedback about what went wrong THIS round
        skewed_report = []
        for f in skewed_features:
            skewed_report.append(
                f"  - {f['name']}: P(YES) = {f['p_yes']:.1%} — TOO SKEWED"
                f"  (YES: \"{f['rubric'].get('yes', '?')[:100]}\")"
            )
        good_report = []
        for f in good_features:
            good_report.append(
                f"  - {f['name']}: P(YES) = {f['p_yes']:.1%} — acceptable"
            )

        # Truncate texts
        if max_example_tokens > 0:
            positive_df = positive_df.copy()
            negative_df = negative_df.copy()
            positive_df[text_column] = positive_df[text_column].apply(
                lambda t: truncate_to_tokens(str(t), max_example_tokens, tokenizer=tokenizer),
            )
            negative_df[text_column] = negative_df[text_column].apply(
                lambda t: truncate_to_tokens(str(t), max_example_tokens, tokenizer=tokenizer),
            )

        pos_examples = _format_examples(
            positive_df, id_column, text_column, label_column,
        )
        neg_examples = _format_examples(
            negative_df, id_column, text_column, label_column,
        )

        # Build refinement prompt
        refinement_task = (
            f"{task_description}\n\n"
            f"=== FEATURE REFINEMENT (Round {round_num + 1}) ===\n\n"
        )

        if population_size > 0:
            refinement_task += (
                f"POPULATION: {population_size:,} examples "
                f"({positive_rate:.0%} positive, {1-positive_rate:.0%} negative).\n\n"
            )

        # Show cumulative history of ALL prior failed rounds
        if prior_rounds_summary:
            refinement_task += f"{prior_rounds_summary}\n\n"

        # Show concrete example papers that scored YES on skewed features
        if example_papers_on_skewed:
            refinement_task += f"{example_papers_on_skewed}\n\n"

        # Current round's skewed features
        refinement_task += (
            f"LATEST SKEWED features (from round {round_num}, REPLACE THESE):\n"
            f"{chr(10).join(skewed_report)}\n\n"
        )

        if good_report:
            refinement_task += (
                f"GOOD features (keep, do not re-propose):\n"
                f"{chr(10).join(good_report)}\n\n"
            )

        if is_clustering:
            refinement_task += (
                f"Propose {num_replacements} REPLACEMENT binary features.\n"
                f"These must CLUSTER the data — describe what TYPE of example this is, "
                f"not whether it's good or bad.\n"
                f"CRITICAL: Each feature should have P(YES) between 20-80%. "
                f"If a feature would be YES for nearly all examples, it is too generic. "
                f"If it would be YES for almost none, it is too narrow.\n"
                f"Be MORE SPECIFIC than the skewed features above. "
                f"Think about concrete, observable differences in content, methodology, "
                f"or framing that would split the population roughly in half.\n\n"
            )
        else:
            refinement_task += (
                f"Propose {num_replacements} REPLACEMENT binary features.\n"
                f"You have now seen {round_num} round(s) of features that were ALL "
                f"too lenient — nearly every paper scored YES. The problem is your "
                f"YES criteria are too broad and generic.\n\n"
                f"To break this pattern, your replacement features MUST:\n"
                f"  1. Have NARROW, DEMANDING YES criteria. Think: 'Would a mediocre "
                f"paper in this field satisfy this criterion?' If yes, the criterion "
                f"is too easy.\n"
                f"  2. Focus on SPECIFIC, TESTABLE properties — not vague quality:\n"
                f"    GOOD: 'Includes ablation study removing each component'\n"
                f"    GOOD: 'Compares against >=3 baselines published in last 2 years'\n"
                f"    GOOD: 'Reports confidence intervals or significance tests'\n"
                f"    BAD:  'Sound methodology' (too vague, every paper scores YES)\n"
                f"    BAD:  'Clear contribution' (too vague, every paper scores YES)\n"
                f"  3. Target SPECIFIC FAILURE MODES: what concrete deficiency would "
                f"cause a reviewer to reject a paper of this type?\n\n"
            )

        refinement_task += (
            f"For each criterion, provide:\n"
            f"  - A clear, specific name\n"
            f"  - A YES description that is DEMANDING — a mediocre paper should NOT qualify\n"
            f"  - A NO description that captures a specific, concrete deficiency\n\n"
            f"The rubric for each metric MUST use keys 'yes' and 'no' with "
            f"detailed multi-sentence descriptions. The scale must be 'binary'."
        )

        # Include partition context if available
        accumulated_metrics: List[TreeMetric] = []
        if parent is not None:
            partition_context = _format_partition_context(parent, partition_key)
            accumulated_metrics = parent.all_metrics
            if partition_context:
                refinement_task += f"\n\n=== PARTITION CONTEXT ===\n{partition_context}"

        current_metrics_text = ""
        if accumulated_metrics or good_features:
            lines = ["Existing/kept features (do NOT re-propose):"]
            for m in accumulated_metrics:
                lines.append(f"  - {m.name}")
            for f in good_features:
                lines.append(f"  - {f['name']}")
            current_metrics_text = "\n".join(lines)

        logger.info("Refinement round %d: replacing %d skewed features, keeping %d good ones",
                     round_num + 1, len(skewed_features), len(good_features))

        return self.base_proposer.propose(
            task_description=refinement_task,
            positive_examples=pos_examples,
            negative_examples=neg_examples,
            current_metrics=current_metrics_text,
            contrastive_pairs="",
            num_metrics=num_replacements,
            num_rubrics=0,
        )
