from __future__ import annotations

import pytest
import json

from methods.codability.lexicon_distill.train_gemma4_similarity_lora import (
    label_token_ids,
    read_rows,
)


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        # Treat a final digit as one dedicated next token.
        if text and text[-1] in "012":
            return {"input_ids": [ord(char) for char in text[:-1]] + [1000 + int(text[-1])]}
        return {"input_ids": [ord(char) for char in text]}


def test_label_tokens_are_unique_stable_next_tokens() -> None:
    assert label_token_ids(FakeTokenizer(), "prompt:") == (1000, 1001, 1002)


class BadBoundaryTokenizer(FakeTokenizer):
    def __call__(self, text: str, add_special_tokens: bool = False):
        if text and text[-1] in "012":
            return {"input_ids": [999]}
        return super().__call__(text, add_special_tokens=add_special_tokens)


def test_label_boundary_drift_fails_closed() -> None:
    with pytest.raises(ValueError):
        label_token_ids(BadBoundaryTokenizer(), "prompt:")


def test_auxiliary_scope_uses_auxiliary_distribution(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "level": "R1", "task": "humor", "split": "train",
                "target_probs": [0.0, 0.0, 1.0], "example_weight": 1.0,
                "family_distributions": {"sonnet": [0.0, 0.0, 1.0], "opus": [1.0, 0.0, 0.0]},
            }
        ) + "\n",
        encoding="utf-8",
    )
    rows = read_rows(path, level="R1", task=None, primary_only=False, auxiliary_only=True)
    assert rows[0]["target_probs"] == [1.0, 0.0, 0.0]
    assert rows[0]["example_weight"] == 0.25
