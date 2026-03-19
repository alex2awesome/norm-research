"""Token-aware text truncation using the HuggingFace Llama tokenizer."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("metric_tree.token_utils")

_tokenizer = None


def get_tokenizer(model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
    """Load and cache the HuggingFace tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        logger.info("Loading tokenizer: %s", model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


def count_tokens(text: str, tokenizer=None) -> int:
    """Count exact tokens in text."""
    tok = tokenizer or get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def truncate_to_tokens(text: str, max_tokens: int, tokenizer=None) -> str:
    """Truncate text to exactly max_tokens tokens, decoding back to a clean string."""
    tok = tokenizer or get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    truncated_ids = ids[:max_tokens]
    return tok.decode(truncated_ids, skip_special_tokens=True)


def compute_scoring_text_budget(
    metrics,
    task_description: str,
    max_model_len: int,
    tokenizer=None,
) -> int:
    """Compute exact per-document token budget for scoring prompts.

    Measures the actual rubric text (from the metrics to be scored) rather
    than estimating.  This is called at the point of use so the rubric size
    is known exactly.

    The scoring prompt uses the single-text signature (text appears once):
        template_overhead + task_description + rubrics + text + output_tokens
    """
    tok = tokenizer or get_tokenizer()

    TEMPLATE_OVERHEAD = 250   # signature docstring + field labels + chat template + CoT rationale
    OUTPUT_TOKENS = 1024      # SamplingParams(max_tokens=1024)
    SAFETY = 100

    task_tokens = count_tokens(task_description, tok) if task_description else 0

    # Build rubrics_text exactly as runner.py line 341 does
    rubrics_text = "\n\n".join([f"{m.name}\n{m.rubric_text}" for m in metrics])
    rubric_tokens = count_tokens(rubrics_text, tok) if rubrics_text else 0

    fixed = TEMPLATE_OVERHEAD + task_tokens + rubric_tokens + OUTPUT_TOKENS + SAFETY
    # Single-text signature: text appears once (no input/output duplication)
    text_budget = max_model_len - fixed
    # Cap at 2000 tokens: metrics are proposed from ~690-token snippets, so
    # scoring on much longer text gives diminishing returns while hurting throughput.
    MAX_SCORING_TEXT_TOKENS = 2000
    text_budget = min(text_budget, MAX_SCORING_TEXT_TOKENS)

    logger.info(
        "Scoring budget (model_len=%d): template=%d + task=%d + rubrics=%d + output=%d + safety=%d "
        "= %d fixed → %d tok/doc (capped at %d)",
        max_model_len, TEMPLATE_OVERHEAD, task_tokens, rubric_tokens,
        OUTPUT_TOKENS, SAFETY, fixed, max(64, text_budget), MAX_SCORING_TEXT_TOKENS,
    )
    return max(64, text_budget)


def compute_generation_example_budget(
    current_metrics_text: str,
    task_description: str,
    max_model_len: int,
    n_total_examples: int = 10,
    tokenizer=None,
) -> int:
    """Compute exact per-example token budget for generation prompts.

    Measures the actual current_metrics text rather than estimating.
    Called at the point of use so the parent-context size is known exactly.

    The generation prompt layout is:
        template_overhead + task_description + positive_examples + negative_examples
        + current_metrics + misc + output_tokens
    """
    tok = tokenizer or get_tokenizer()

    TEMPLATE_OVERHEAD = 400   # longer signature docstring + field descs + chat template + CoT
    OUTPUT_TOKENS = 8192      # SamplingParams(max_tokens=8192)
    MISC = 20                 # contrastive_pairs + num_metrics + num_rubrics fields
    SAFETY = 100
    EXAMPLE_FORMAT_OVERHEAD = 15  # [id=X label=Y]\n per example

    task_tokens = count_tokens(task_description, tok) if task_description else 0
    metrics_tokens = count_tokens(current_metrics_text, tok) if current_metrics_text else 0

    fixed = TEMPLATE_OVERHEAD + task_tokens + metrics_tokens + MISC + OUTPUT_TOKENS + SAFETY
    total_for_examples = max_model_len - fixed
    per_example = (total_for_examples // max(1, n_total_examples)) - EXAMPLE_FORMAT_OVERHEAD

    logger.info(
        "Generation budget (model_len=%d): template=%d + task=%d + metrics=%d + output=%d "
        "+ misc=%d + safety=%d = %d fixed → %d tok/example (%d examples)",
        max_model_len, TEMPLATE_OVERHEAD, task_tokens, metrics_tokens,
        OUTPUT_TOKENS, MISC, SAFETY, fixed, max(500, per_example), n_total_examples,
    )
    return max(500, per_example)
