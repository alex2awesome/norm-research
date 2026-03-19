"""Core data structures for the Metric Tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TreeMetric:
    """A single metric in the tree, with rubric and provenance."""

    metric_id: str
    name: str
    rubric_text: str
    rubric: Dict[str, str]
    source_node_id: str
    scale: str = "ordinal"


@dataclass
class MetricTreeNode:
    """A node in the Metric Tree hierarchy."""

    node_id: str
    depth: int
    parent_id: Optional[str] = None

    # Metrics: local = proposed at this node; all = local + inherited
    local_metrics: List[TreeMetric] = field(default_factory=list)
    all_metrics: List[TreeMetric] = field(default_factory=list)

    # Fitted classifier (sklearn LogisticRegression)
    classifier: Any = None
    scaler: Any = None  # StandardScaler
    feature_names: List[str] = field(default_factory=list)

    # Data partition
    point_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    predictions: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    correct_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Routing
    confidence_threshold: float = 0.7
    router: Any = None  # CalibratedClassifierCV for learned routing

    # Interaction terms
    interaction_pairs: List[Tuple[str, str]] = field(default_factory=list)

    # Children: keyed by error type ("false_positive", "false_negative", "misclassified")
    children: Dict[str, "MetricTreeNode"] = field(default_factory=dict)

    # Performance
    train_accuracy: float = 0.0
    eval_accuracy: float = 0.0


@dataclass
class MetricTree:
    """Complete Metric Tree with all nodes and metrics."""

    root: Optional[MetricTreeNode] = None
    config: Any = None  # TreeConfig
    all_nodes: Dict[str, MetricTreeNode] = field(default_factory=dict)
    all_metrics: Dict[str, TreeMetric] = field(default_factory=dict)
    task_description: str = ""
