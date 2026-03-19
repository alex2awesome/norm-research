"""TreeConfig dataclass for Metric Tree hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TreeConfig:
    """Configuration for building a Metric Tree."""

    # Tree structure
    max_depth: int = 3
    min_subset_size: int = 20

    # Metric generation
    n_metrics_to_propose: int = 5
    n_rubrics_to_propose: int = 5

    # Regularization
    sparsity_penalty: str = "l1"
    cv_folds: int = 5
    class_weight: Optional[str] = "balanced"

    # Interaction terms
    use_interactions: bool = True
    interaction_max_depth: int = 0  # only at root by default

    # LLM settings
    llm_temperature: float = 0.7

    # Reproducibility
    random_seed: int = 42

    # Routing
    confidence_threshold: float = 0.7
    use_learned_router: bool = False

    # Scoring
    label_batch_size: int = 200
    max_text_tokens: int = 512

    # Data
    eval_fraction: float = 0.4

    # Exception proposer
    exception_examples_per_class: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    enable_error_analysis: bool = True
    contrastive_pairs_k: int = 3

    # Output
    output_dir: str = "outputs/metric_tree"
    verbose: bool = True
