"""TreeConfig dataclass for Metric Tree hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TreeConfig:
    """Configuration for building a partitioned binary Metric Tree."""

    # Tree structure
    max_depth: int = 3
    n_binary_metrics_per_level: int = 3     # K: binary metrics per node → 2^K partitions
    n_rubrics_to_propose: int = 5           # propose more, select K by MI
    min_partition_size: int = 20
    min_contrastive_pairs: int = 3          # stop branching if fewer pairs
    clustering_depth: int = 2               # depths < this use clustering (descriptive) features;
                                            # depths >= this use discriminative (predictive) features

    # Metric generation
    contrastive_pairs_k: int = 5            # pairs for proposer context
    exception_examples_per_class: int = 25
    refinement_sample_size: int = 1000      # sample size per refinement round (different sample each round)
    max_refinement_rounds: int = 5          # max rounds of propose→score→refine with cumulative history
    min_feature_balance: float = 0.15       # features with P(YES) outside [min, 1-min] are "skewed"
    proposer_max_retries: int = 3           # retry proposal with fresh examples on total failure

    # Error-based pruning: only extend partitions with enough minority-class examples
    min_minority_fraction: float = 0.0    # 0 = extend all (default); e.g. 0.15 = prune pure partitions

    # LLM settings
    llm_temperature: float = 0.7

    # Scoring
    label_batch_size: int = 8192
    max_text_tokens: int = 512

    # Data
    eval_fraction: float = 0.4
    embedding_model: str = "all-MiniLM-L6-v2"

    # Reproducibility
    random_seed: int = 42

    # Router: per-node text classifier for selective deepening
    use_router: bool = False                      # enable per-node text classifier routing
    router_threshold: float = 0.5                 # p(minority) > threshold → continue deeper
    router_n_epochs: int = 20
    router_learning_rate: float = 1e-3
    router_hidden_dim: int = 128
    router_dropout: float = 0.1
    router_batch_size: int = 64
    router_min_examples: int = 40                 # skip router if partition has fewer examples

    # Restructuring: iterative tree rebuilding over global score matrix
    restructure_iterations: int = 3           # max restructuring iterations (0 = no restructuring)
    restructure_na_threshold: float = 0.05    # max NA rate to consider a feature applicable at a node
    restructure_k_min: int = 3                # min features per node after restructuring
    restructure_k_max: int = 6                # max features per node after restructuring
    restructure_min_mi: float = 0.001         # min MI to include a feature
    dedup_embedding_threshold: float = 0.85   # cosine similarity threshold for embedding-based dedup

    # Output
    output_dir: str = "outputs/metric_tree"
    verbose: bool = True
