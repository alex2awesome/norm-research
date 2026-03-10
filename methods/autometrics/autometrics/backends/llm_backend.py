"""Protocol and data classes for pluggable LLM scoring backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """Standardized response from a single-metric LLM scoring call."""

    score: float
    reasoning: str = ""
    raw_text: str = ""


@dataclass
class MultiMetricResponse:
    """Response from multi-metric scoring (one example, many metrics)."""

    scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    raw_text: str = ""


@runtime_checkable
class ScoringBackend(Protocol):
    """Protocol that all scoring backends must satisfy.

    Three core operations for scoring, plus two optional operations for
    metric generation and deduplication (needed when running fully offline
    with a single VLLM process).
    """

    def score_single_metric_batch(
        self,
        task_description: str,
        axis: str,
        inputs: List[str],
        outputs: List[str],
        references: Optional[List[Optional[str]]] = None,
    ) -> List[LLMResponse]:
        """Score a batch of (input, output) pairs on a single metric axis."""
        ...

    def score_multi_metric_batch(
        self,
        task_description: str,
        rubrics_text: str,
        metric_names: List[str],
        inputs: List[str],
        outputs: List[str],
    ) -> List[MultiMetricResponse]:
        """Score a batch of (input, output) pairs on multiple metrics at once."""
        ...

    def generate_metrics(
        self,
        task_description: str,
        positive_examples: str,
        negative_examples: str,
        current_metrics: str,
        contrastive_pairs: str,
        num_metrics: int,
        num_rubrics: int,
    ) -> str:
        """Propose new metrics given contrastive examples.

        Returns raw JSON string of proposed metrics (same format as
        ``ContrastiveRubricProposer``).
        """
        ...

    def check_dedup(
        self,
        existing_metrics: str,
        candidate_metric: str,
    ) -> str:
        """Check if a candidate metric is distinct from existing metrics.

        Returns verdict string: ``"distinct"`` or ``"duplicate: <name>"``.
        """
        ...

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """Generic text generation from a user-message prompt.

        Used for auxiliary tasks like metric-card generation where
        structured DSPy signatures are not required.
        """
        ...
