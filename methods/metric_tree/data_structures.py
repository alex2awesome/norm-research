"""Core data structures for the Partitioned Metric Tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TreeMetric:
    """A single binary metric in the tree, with rubric and provenance."""

    metric_id: str
    name: str
    rubric_text: str
    rubric: Dict[str, str]       # {"yes": "...", "no": "..."}
    source_node_id: str
    scale: str = "binary"


@dataclass
class PartitionTreeNode:
    """A node in the Partitioned Metric Tree hierarchy."""

    node_id: str
    depth: int
    parent_id: Optional[str] = None

    # The binary combination from the parent's metrics that defines this partition
    partition_key: Tuple[int, ...] = ()

    # Metrics: local = K binary metrics proposed at this level
    local_metrics: List[TreeMetric] = field(default_factory=list)
    # Accumulated from root to here (full binary feature vector)
    all_metrics: List[TreeMetric] = field(default_factory=list)

    # Data partition (indices into training data)
    point_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # Scores: binary 0/1 arrays
    local_scores: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))   # (n_points, K)
    all_scores: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))     # (n_points, total_metrics)

    # Children: keyed by partition tuple e.g. (0, 1, 1) for K=3
    children: Dict[Tuple[int, ...], "PartitionTreeNode"] = field(default_factory=dict)

    # Prediction: simple base-rate within the partition
    base_rate: float = 0.5
    n_positive: int = 0
    n_negative: int = 0

    is_leaf: bool = False

    # Leaf regression: logistic regression on accumulated binary features
    leaf_model: Any = None  # fitted sklearn LogisticRegression (or None)

    # Router: per-node text classifier for selective deepening (optional)
    router: Any = None                              # trained NodeRouter instance
    router_minority_is_positive: Optional[bool] = None  # whether minority class = label 1


@dataclass
class MetricTree:
    """Complete Partitioned Metric Tree with all nodes and metrics."""

    root: Optional[PartitionTreeNode] = None
    config: Any = None  # TreeConfig
    all_nodes: Dict[str, PartitionTreeNode] = field(default_factory=dict)
    all_metrics: Dict[str, TreeMetric] = field(default_factory=dict)
    task_description: str = ""
