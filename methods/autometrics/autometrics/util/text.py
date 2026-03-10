from typing import List

import litellm

# Match train_reward_model.py default max_length
DEFAULT_PROMPT_MAX_TOKENS = 1024


def count_tokens_with_litellm(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using litellm.token_counter with fallback."""
    try:
        return litellm.token_counter(model=model_name, text=text)
    except Exception:
        # Fallback: rough estimation (1.3 tokens per word on average)
        return int(len(text.split()) * 1.3)


def truncate_text_to_token_limit(text: str, max_tokens: int, model_name: str) -> str:
    """Truncate a single text to a token limit using litellm token_counter."""
    if text is None:
        return ""
    text = str(text)
    try:
        token_count = count_tokens_with_litellm(text, model_name)
    except Exception:
        token_count = int(len(text.split()) * 1.3)
    if token_count <= max_tokens:
        return text

    words = text.split()
    approx_len = max(1, int(max_tokens / 1.3))
    if len(words) > approx_len:
        words = words[:approx_len]
    truncated = " ".join(words)

    # Refine if still too long
    for _ in range(10):
        try:
            token_count = count_tokens_with_litellm(truncated, model_name)
        except Exception:
            token_count = int(len(truncated.split()) * 1.3)
        if token_count <= max_tokens:
            return truncated + "..."
        if len(words) <= 10:
            return truncated + "..."
        words = words[: int(len(words) * 0.9)]
        truncated = " ".join(words)
    return truncated + "..."


def truncate_examples_to_token_limit(
    examples: List[str],
    model_name: str,
    max_tokens: int = DEFAULT_PROMPT_MAX_TOKENS,
) -> List[str]:
    """Apply token-based truncation to each example."""
    return [truncate_text_to_token_limit(ex, max_tokens, model_name) for ex in examples]
