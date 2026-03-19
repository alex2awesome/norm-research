"""ExceptionMetricProposer: wraps ContrastiveRubricProposer for child nodes."""

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
    _normalize_rubric,
    _rubric_to_text,
    _truncate_text,
)

from .data_structures import MetricTreeNode, TreeMetric
from .token_utils import truncate_to_tokens

logger = logging.getLogger("metric_tree.proposer")


def _format_parent_context(
    parent: MetricTreeNode,
    error_type: str,
    coefficients: str = "",
) -> str:
    """Format parent node metrics and context for the exception prompt.

    Optionally includes parent logistic regression coefficients so the LLM
    knows which metrics matter most.
    """
    lines = [f"Parent node ({parent.node_id}) uses the following metrics:"]
    for tm in parent.all_metrics:
        lines.append(f"  - {tm.name}: {_truncate_text(tm.rubric_text, max_chars=300)}")
    lines.append(f"\nError type: {error_type}")
    lines.append(f"Parent train accuracy: {parent.train_accuracy:.3f}")
    if coefficients:
        lines.append(f"\nParent metric importance (logistic regression coefficients):\n{coefficients}")
    return "\n".join(lines)


def _format_examples_with_scores(
    df: pd.DataFrame,
    scored_df: Optional[pd.DataFrame],
    parent_metrics: List[TreeMetric],
    id_column: str,
    text_column: str,
    label_column: str,
) -> str:
    """Format examples showing parent metric scores alongside text.

    Output per example:
        [id=123 label=1 parent_scores={Clarity: 4, Depth: 2}]
        <text>
    """
    lines: List[str] = []
    metric_names = [m.name for m in parent_metrics]

    for _, row in df.iterrows():
        doc_id = row[id_column]
        label = row[label_column]
        text = _truncate_text(str(row[text_column]))

        # Look up parent scores for this example if scored_df is available
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
                            score_parts.append(f"{mn}: {int(val)}")
                if score_parts:
                    scores_str = f" parent_scores={{{', '.join(score_parts)}}}"

        lines.append(f"[id={doc_id} label={label}{scores_str}]\n{text}")

    return "\n\n".join(lines).strip()


def _generate_contrastive_pairs(
    sample_scored: Optional[pd.DataFrame],
    parent_metrics: List[TreeMetric],
    id_column: str,
    label_column: str,
    text_column: str,
    k_pairs: int = 3,
) -> str:
    """Generate contrastive pairs from scored sample using exact_match.

    Finds pairs with similar parent scores but opposite labels (exception vs correct).
    Returns formatted string for the contrastive_pairs field.
    """
    if sample_scored is None or len(parent_metrics) == 0:
        return ""

    feature_columns = [m.name for m in parent_metrics]
    # Check that feature columns exist in the scored df
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

        pos_scores = {c: int(pos_row[c]) for c in available if pd.notna(pos_row[c])}
        neg_scores = {c: int(neg_row[c]) for c in available if pd.notna(neg_row[c])}

        pos_text = _truncate_text(str(pos_row[text_column]), max_chars=300)
        neg_text = _truncate_text(str(neg_row[text_column]), max_chars=300)

        lines.append(
            f"PAIR (distance={dist:.1f}):\n"
            f"  Exception [id={pos_id}] scores={pos_scores}:\n    {pos_text}\n"
            f"  Correct   [id={neg_id}] scores={neg_scores}:\n    {neg_text}"
        )

    result = "\n\n".join(lines)
    logger.info("Generated %d contrastive pairs", len(lines))
    return result


def _analyze_error_patterns(
    scoring_backend: Any,
    task_description: str,
    exception_text: str,
    correct_text: str,
    parent_context: str,
    error_type: str,
) -> str:
    """Run an LLM call to analyze confusion patterns before proposing metrics.

    Returns analysis text to inject into the enriched task description.
    """
    if scoring_backend is None:
        logger.info("No scoring_backend available, skipping error pattern analysis")
        return ""

    prompt = (
        f"You are analyzing classification errors to help propose better evaluation metrics.\n\n"
        f"Task: {task_description}\n\n"
        f"{parent_context}\n\n"
        f"Below are examples the parent classifier got WRONG ({error_type}):\n"
        f"{exception_text}\n\n"
        f"Below are examples the parent classifier got RIGHT from the same prediction bucket:\n"
        f"{correct_text}\n\n"
        f"Analyze the patterns:\n"
        f"1. What distinguishes the misclassified examples from the correctly classified ones?\n"
        f"2. What subtle features does the parent model seem to miss?\n"
        f"3. Are there specific content patterns, structural differences, or edge cases?\n\n"
        f"Be specific and concrete. Focus on differences that could be captured by new evaluation metrics."
    )

    try:
        analysis = scoring_backend.generate_text(prompt, max_tokens=1024)
        logger.info("Error pattern analysis complete (%d chars)", len(analysis))
        return analysis
    except Exception as e:
        logger.warning("Error pattern analysis failed: %s", e)
        return ""


class ExceptionMetricProposer:
    """Proposes metrics for exception (child) nodes by wrapping ContrastiveRubricProposer.

    Encodes exception context into the existing generate_metrics fields:
    - task_description: appends EXCEPTION ANALYSIS MODE framing + error analysis
    - current_metrics: lists parent metrics with coefficients
    - positive/negative_examples: representative examples with parent scores
    - contrastive_pairs: matched pairs with similar parent scores but opposite labels
    """

    def __init__(
        self,
        base_proposer: ContrastiveRubricProposer,
    ):
        self.base_proposer = base_proposer

    def propose(
        self,
        *,
        task_description: str,
        parent: MetricTreeNode,
        error_type: str,
        exception_df: Any,  # pd.DataFrame - the misclassified points
        correct_df: Any,  # pd.DataFrame - the correctly classified from same bucket
        id_column: str,
        text_column: str,
        label_column: str,
        num_metrics: int = 5,
        num_rubrics: int = 5,
        max_example_tokens: int = 0,
        tokenizer: Any = None,
        sample_scored: Optional[pd.DataFrame] = None,
        parent_coefficients: str = "",
        scoring_backend: Any = None,
        enable_error_analysis: bool = True,
        contrastive_pairs_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Propose metrics that distinguish exceptions from correct predictions.

        For a false_positive child:
          - exception_df: false positives (parent predicted positive, true label negative)
          - correct_df: true positives (parent predicted positive, true label positive)

        For a false_negative child:
          - exception_df: false negatives (parent predicted negative, true label positive)
          - correct_df: true negatives (parent predicted negative, true label negative)

        New optional kwargs:
          - sample_scored: scored DataFrame with parent metric scores for the sample
          - parent_coefficients: formatted string of parent logistic regression coefficients
          - scoring_backend: VLLM backend for error pattern analysis LLM call
          - enable_error_analysis: whether to run error pattern analysis
          - contrastive_pairs_k: number of contrastive pairs to generate
        """
        # Build current metrics context from parent (now with coefficients)
        parent_context = _format_parent_context(parent, error_type, coefficients=parent_coefficients)

        # Truncate texts in token space to fit context window
        if max_example_tokens > 0:
            exception_df = exception_df.copy()
            correct_df = correct_df.copy()
            exception_df[text_column] = exception_df[text_column].apply(
                lambda t: truncate_to_tokens(str(t), max_example_tokens, tokenizer=tokenizer),
            )
            correct_df[text_column] = correct_df[text_column].apply(
                lambda t: truncate_to_tokens(str(t), max_example_tokens, tokenizer=tokenizer),
            )

        # Format examples WITH parent scores if scored_df available
        if sample_scored is not None:
            exception_examples = _format_examples_with_scores(
                exception_df, sample_scored, parent.all_metrics,
                id_column, text_column, label_column,
            )
            correct_examples = _format_examples_with_scores(
                correct_df, sample_scored, parent.all_metrics,
                id_column, text_column, label_column,
            )
        else:
            exception_examples = _format_examples(
                exception_df, id_column, text_column, label_column,
            )
            correct_examples = _format_examples(
                correct_df, id_column, text_column, label_column,
            )

        # Generate contrastive pairs from scored sample
        contrastive_pairs = _generate_contrastive_pairs(
            sample_scored, parent.all_metrics,
            id_column, label_column, text_column,
            k_pairs=contrastive_pairs_k,
        )

        # Run error pattern analysis (extra LLM call)
        error_analysis = ""
        if enable_error_analysis and scoring_backend is not None:
            error_analysis = _analyze_error_patterns(
                scoring_backend, task_description,
                exception_examples, correct_examples,
                parent_context, error_type,
            )

        # Build exception-aware task description with analysis injected
        exception_task = (
            f"{task_description}\n\n"
            f"=== EXCEPTION ANALYSIS MODE ===\n"
            f"The parent classifier has already applied general rules but misclassifies "
            f"some examples ({error_type}). Your goal is to propose NEW metrics that "
            f"capture the subtle distinctions the parent model MISSED. Focus on edge cases "
            f"and exceptions rather than broad patterns. The parent metrics are listed in "
            f"'current_metrics' — do NOT re-propose similar metrics."
        )

        if error_analysis:
            exception_task += (
                f"\n\n=== ERROR PATTERN ANALYSIS ===\n"
                f"An analysis of the misclassified examples found these patterns:\n"
                f"{error_analysis}\n\n"
                f"Use these insights to inform your metric proposals. Each metric should "
                f"target a specific confusion pattern identified above."
            )

        return self.base_proposer.propose(
            task_description=exception_task,
            positive_examples=exception_examples,
            negative_examples=correct_examples,
            current_metrics=parent_context,
            contrastive_pairs=contrastive_pairs,
            num_metrics=num_metrics,
            num_rubrics=num_rubrics,
        )
