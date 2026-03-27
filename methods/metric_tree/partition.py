"""Partition assignment and pruning for binary-feature combinatorial branching."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("metric_tree.partition")


def assign_to_partitions(
    binary_scores: np.ndarray,
) -> Dict[Tuple[int, ...], np.ndarray]:
    """Assign examples to 2^K partitions based on binary score vectors.

    Parameters
    ----------
    binary_scores : ndarray of shape (n, K) with values 0 or 1

    Returns
    -------
    Dict mapping partition key (tuple of ints) to array of row indices.
    """
    n = binary_scores.shape[0]
    partitions: Dict[Tuple[int, ...], list] = {}
    for i in range(n):
        key = tuple(int(v) for v in binary_scores[i])
        partitions.setdefault(key, []).append(i)

    return {k: np.array(v, dtype=int) for k, v in partitions.items()}


def prune_partitions(
    partitions: Dict[Tuple[int, ...], np.ndarray],
    min_size: int,
) -> Dict[Tuple[int, ...], np.ndarray]:
    """Remove partitions smaller than min_size.

    Returns the surviving partitions.
    """
    pruned = {}
    for key, indices in partitions.items():
        if len(indices) >= min_size:
            pruned[key] = indices
        else:
            logger.info(
                "Pruning partition %s (size=%d < min=%d)",
                _format_key(key), len(indices), min_size,
            )
    return pruned


def count_contrastive_pairs(labels: np.ndarray) -> int:
    """Count the number of contrastive pairs available.

    A contrastive pair requires one positive and one negative example.
    Returns min(n_positive, n_negative).
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return min(n_pos, n_neg)


def format_partition_description(
    partition_key: Tuple[int, ...],
    metric_names: List[str],
) -> str:
    """Format a human-readable description of a partition.

    Example: "Has_Novel_Method = YES, Uses_Empirical_Data = NO, ..."
    """
    parts = []
    for val, name in zip(partition_key, metric_names):
        label = "YES" if val == 1 else "NO"
        parts.append(f"{name} = {label}")
    return ", ".join(parts)


def _format_key(key: Tuple[int, ...]) -> str:
    """Compact string representation: (1, 0, 1) -> '101'."""
    return "".join(str(v) for v in key)
