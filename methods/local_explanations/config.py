from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LocalExplanationConfig:
    """Configuration for both local explanation approaches."""

    # Approach selection
    approach: str = "star_local"  # "rationalization" or "star_local"

    # --- Approach A: Prior estimation ---
    n_prior_calls: int = 15
    features_per_prior_call: int = 10
    prior_similarity_threshold: float = 0.7  # cosine sim above this = matches prior
    prior_idf_smooth: float = 1.0

    # --- Approach A: Per-example rationalization ---
    features_per_example: int = 7

    # --- Approach B: STaR weights ---
    # Matrix interpretation:
    #   a feature is rewarded when its listed side matches the true label,
    #   penalized when it contradicts it. Magnitude reflects confidence:
    #   ±1.0 when model COMMITTED (winning side), ±0.3/+0.5 when it merely LISTED (losing side).
    weight_correct_winning: float = 1.0    # model championed, was right
    weight_correct_losing: float = -0.3    # listed on wrong side, model correctly discounted
    weight_incorrect_winning: float = -1.0 # model championed, was wrong (misleading)
    weight_incorrect_losing: float = 0.5   # listed on right side, model wrongly discounted

    # --- Clustering pipeline ---
    n_canonical_features: int = 30           # target K for final vocabulary
    n_initial_clusters: int = 200            # K-means initial over-clustering
    cluster_dedup_threshold: float = 0.85    # cosine sim for embedding dedup
    similarity_pair_low: float = 0.3         # min cosine sim for candidate pairs
    similarity_pair_high: float = 0.9        # max cosine sim for candidate pairs
    max_similarity_pairs: int = 10000        # max pairs for similarity labeling
    similarity_labeling_model: str = "gpt-5-mini"  # reasoning model — slower per call, far better label quality on conceptual similarity
    similarity_labeling_concurrency: int = 30  # async in-flight requests via OpenAI AsyncClient
    triplet_max_positives: int = 10
    triplet_max_negatives_per_positive: int = 10
    similarity_model_base: str = "microsoft/mpnet-base"
    similarity_model_epochs: int = 3
    similarity_model_batch_size: int = 64
    cluster_label_samples: int = 10          # members to sample when labeling a cluster
    optimal_k_range: list = field(default_factory=lambda: [10, 20, 30, 50, 75, 100])

    # --- Clustering method ---
    # "kmeans"  → silhouette sweep over optimal_k_range (legacy)
    # "hdbscan" → UMAP dim-reduce + HDBSCAN density clustering (auto K)
    clustering_method: str = "kmeans"
    umap_n_neighbors: int = 30
    umap_n_components: int = 20
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 100
    hdbscan_min_samples: Optional[int] = None  # defaults to min_cluster_size
    # Supervised UMAP: 0.0 = unsupervised (default); >0 pulls features with
    # similar per-feature P(y=1) toward each other before HDBSCAN.
    umap_target_weight: float = 0.0
    # HDBSCAN cluster_selection_epsilon: prevents splitting clusters whose
    # children are within this distance in UMAP space. Higher = fewer
    # duplicate-ish leaf clusters. 0.0 = no constraint (default).
    hdbscan_cluster_selection_epsilon: float = 0.0

    # --- Cluster ranking / small-sample mitigation ---
    # Shrinkage when ranking clusters for top-K selection:
    #   ranking_score = sum(member_weights) / (count + shrinkage_alpha)
    # shrinkage_alpha=0     → plain mean weight (mitigation #1)
    # shrinkage_alpha>0     → shrunk mean (mitigation #3), small clusters pulled toward 0
    # shrinkage_alpha=None  → legacy raw sum (old behavior)
    shrinkage_alpha: Optional[float] = 5.0

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Scoring & prediction ---
    label_batch_size: int = 8192
    max_text_tokens: int = 512
    eval_fraction: float = 0.4

    # --- VLLM generation ---
    explanation_max_tokens: int = 1024
    explanation_temperature: float = 0.0

    # --- Output ---
    output_dir: str = "outputs/local_explanations"
    verbose: bool = True
    random_seed: int = 42


@dataclass
class TaskMetadata:
    """Per-dataset metadata for prompt rendering."""
    text_type: str                   # e.g. "scientific paper abstract"
    positive_label: str              # e.g. "accepted"
    negative_label: str              # e.g. "rejected"
    label_words: Dict[int, str] = field(default_factory=dict)  # {1: "accepted", 0: "rejected"}


TASK_METADATA_REGISTRY: Dict[str, TaskMetadata] = {
    "PeerReviewAcceptance": TaskMetadata(
        text_type="scientific paper abstract",
        positive_label="accepted",
        negative_label="rejected",
        label_words={1: "accepted", 0: "rejected"},
    ),
    "PressReleaseModeling": TaskMetadata(
        text_type="press release",
        positive_label="newsworthy",
        negative_label="not newsworthy",
        label_words={1: "newsworthy", 0: "not newsworthy"},
    ),
    "CodeReviewAcceptance": TaskMetadata(
        text_type="GitHub pull request",
        positive_label="accepted",
        negative_label="rejected",
        label_words={1: "accepted", 0: "rejected"},
    ),
    "MathAnswerQuality": TaskMetadata(
        text_type="mathematics question-answer pair from Math StackExchange",
        positive_label="preferred",
        negative_label="non-preferred",
        label_words={1: "preferred", 0: "non-preferred"},
    ),
    "NoticeAndComment": TaskMetadata(
        text_type="public comment on a proposed regulation",
        positive_label="substantive",
        negative_label="non-substantive",
        label_words={1: "substantive", 0: "non-substantive"},
    ),
}
