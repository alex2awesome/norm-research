"""Configuration for the verification library discovery algorithm."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class VerificationLibraryConfig:
    """Controls the generate-refactor-evaluate loop."""

    # Algorithm
    approach: str = "approach2"  # "v1", "approach1", "approach2"
    batch_size: int = 20  # generate programs in batches of this size
    refactor_size: int = 3  # refactor library after every N new programs
    max_examples: int = 200  # total training examples to process
    programs_per_example: int = 1

    # LLM — generation (bulk, cheap)
    generation_model: str = "qwen/qwen3-coder-next"
    generation_base_url: str = "https://openrouter.ai/api/v1"
    generation_concurrency: int = 30
    max_tokens_program: int = 2048
    temperature_program: float = 0.7

    # LLM — refactoring (few calls, quality matters)
    # env-overridable pointer: `export SONNET_MODEL=claude-sonnet-5` to switch (default unchanged).
    refactoring_model: str = field(
        default_factory=lambda: os.environ.get("SONNET_MODEL", "claude-sonnet-4-20250514"))
    refactoring_base_url: str = ""  # empty = Anthropic native API
    refactoring_concurrency: int = 5
    max_tokens_refactor: int = 8192
    temperature_refactor: float = 0.3

    # Sandbox
    sandbox_timeout: float = 5.0  # seconds per program execution
    max_program_length: int = 5000  # max chars in generated code

    # Evaluation
    test_set_size: int = 500
    combination_method: str = "logistic"  # "vote", "mean", "logistic"
    min_programs_for_logistic: int = 5

    # Library management
    library_similarity_threshold: int = 300  # switch to similarity routing above this many lines
    library_similarity_top_k: int = 8  # show top-k similar functions to refactorer
    embedding_model: str = "jinaai/jina-code-embeddings-0.5b"  # local: jina 0.5b, server: nomic-ai/nomic-embed-code

    # Convergence
    convergence_window: int = 3
    convergence_edit_threshold: float = 0.05
    convergence_accuracy_threshold: float = 0.005

    # Output
    output_dir: str = "outputs/verification_library"
    random_seed: int = 42


@dataclass
class DatasetConfig:
    """Per-dataset metadata."""

    name: str
    text_type: str
    positive_label: str
    negative_label: str
    text_column: str = "text"
    label_column: str = "judgement"
    id_column: str = "id"


DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "notice-and-comment": DatasetConfig(
        name="NoticeAndComment",
        text_type="public comment on a proposed federal regulation",
        positive_label="substantive (responded to by agency)",
        negative_label="non-substantive (not responded to)",
    ),
    "peer-review": DatasetConfig(
        name="PeerReviewAcceptance",
        text_type="scientific paper submitted to an academic venue",
        positive_label="accepted",
        negative_label="rejected",
    ),
    "code-review": DatasetConfig(
        name="CodeReviewAcceptance",
        text_type="pull request with code changes and review comments",
        positive_label="merged",
        negative_label="not merged",
        label_column="label",
    ),
    "press-release": DatasetConfig(
        name="PressReleaseModeling",
        text_type="press release",
        positive_label="picked up by news media",
        negative_label="not picked up by news media",
        label_column="label",
    ),
    "creative-writing": DatasetConfig(
        name="CreativeWriting",
        text_type="creative story written in response to a writing prompt",
        positive_label="highly upvoted",
        negative_label="not highly upvoted",
    ),
    "humor": DatasetConfig(
        name="Humor",
        text_type="Reddit joke post",
        positive_label="funny (highly upvoted)",
        negative_label="not funny (low score)",
    ),
}
