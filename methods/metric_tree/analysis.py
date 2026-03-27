"""Tree analysis and articulability gap measurement."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_structures import MetricTree, PartitionTreeNode
from .partition import format_partition_description, _format_key

logger = logging.getLogger("metric_tree.analysis")


def analyze_tree_complexity(tree: MetricTree) -> pd.DataFrame:
    """Compute per-node statistics for the tree.

    Returns a DataFrame with columns: node_id, depth, parent_id,
    n_local_metrics, n_all_metrics, n_points, partition_key,
    n_positive, n_negative, base_rate, n_children, is_leaf.
    """
    rows = []
    for node_id, node in tree.all_nodes.items():
        rows.append({
            "node_id": node.node_id,
            "depth": node.depth,
            "parent_id": node.parent_id or "",
            "partition_key": _format_key(node.partition_key) if node.partition_key else "",
            "n_local_metrics": len(node.local_metrics),
            "n_all_metrics": len(node.all_metrics),
            "n_points": len(node.point_indices),
            "n_positive": node.n_positive,
            "n_negative": node.n_negative,
            "base_rate": node.base_rate,
            "n_children": len(node.children),
            "is_leaf": node.is_leaf or len(node.children) == 0,
        })
    return pd.DataFrame(rows)


def measure_depth_distribution(
    tree: MetricTree,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute resolution depth histogram from prediction results."""
    node_depths = {nid: node.depth for nid, node in tree.all_nodes.items()}

    depths = []
    for node_id in predictions_df["resolving_node"]:
        if node_id in node_depths:
            depths.append(node_depths[node_id])
        else:
            depths.append(-1)

    depth_series = pd.Series(depths, name="depth")
    counts = depth_series.value_counts().sort_index()

    result = pd.DataFrame({
        "depth": counts.index,
        "count": counts.values,
        "fraction": counts.values / len(predictions_df),
    })

    return result


def export_tree_summary(tree: MetricTree, path: str) -> None:
    """Export a human-readable tree description to a text file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("PARTITIONED METRIC TREE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Task: {tree.task_description[:200]}")
    lines.append(f"Total nodes: {len(tree.all_nodes)}")
    lines.append(f"Total unique metrics: {len(tree.all_metrics)}")
    n_leaves = sum(1 for n in tree.all_nodes.values() if n.is_leaf or not n.children)
    lines.append(f"Leaf nodes: {n_leaves}")
    lines.append("")

    def _format_node(node: PartitionTreeNode, indent: int = 0) -> List[str]:
        prefix = "  " * indent
        node_lines = []

        # Header with partition key
        pk_str = ""
        if node.partition_key:
            metric_names = []
            parent = tree.all_nodes.get(node.parent_id or "")
            if parent and parent.local_metrics:
                metric_names = [m.name for m in parent.local_metrics]
            pk_str = f" [{format_partition_description(node.partition_key, metric_names)}]"

        node_lines.append(f"{prefix}[{node.node_id}] depth={node.depth}{pk_str}")
        node_lines.append(
            f"{prefix}  base_rate={node.base_rate:.3f}"
        )
        node_lines.append(
            f"{prefix}  Points: {len(node.point_indices)} "
            f"(+{node.n_positive} / -{node.n_negative})"
        )

        # Local metrics
        if node.local_metrics:
            node_lines.append(f"{prefix}  Binary metrics ({len(node.local_metrics)}):")
            for m in node.local_metrics:
                node_lines.append(f"{prefix}    - {m.name}")
                yes_desc = m.rubric.get("yes", "")[:100]
                no_desc = m.rubric.get("no", "")[:100]
                node_lines.append(f"{prefix}      YES: {yes_desc}")
                node_lines.append(f"{prefix}      NO:  {no_desc}")

        if node.is_leaf or not node.children:
            node_lines.append(f"{prefix}  [LEAF — base-rate prediction]")

        # Children
        for pk, child in sorted(node.children.items(), key=lambda x: _format_key(x[0])):
            node_lines.append(f"{prefix}  -> Partition {_format_key(pk)}:")
            node_lines.extend(_format_node(child, indent + 2))

        return node_lines

    if tree.root:
        lines.extend(_format_node(tree.root))

    lines.append("")
    lines.append("=" * 80)
    lines.append("ALL METRICS")
    lines.append("=" * 80)
    for metric_id, m in sorted(tree.all_metrics.items()):
        lines.append(f"\n[{metric_id[:8]}] {m.name} (source: {m.source_node_id})")
        lines.append(f"  YES: {m.rubric.get('yes', '')[:200]}")
        lines.append(f"  NO:  {m.rubric.get('no', '')[:200]}")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Tree summary exported to %s (%d chars)", path, len(text))


def compute_articulability_gap(
    tree: MetricTree,
    predictions_df: pd.DataFrame,
    true_labels: np.ndarray,
    root_only_predictions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Measure the articulability gap: how much AUC improves with depth."""
    from sklearn.metrics import roc_auc_score

    tree_preds = predictions_df["prediction"].values
    tree_probs = predictions_df["probability"].values
    tree_accuracy = float((tree_preds == true_labels).mean())

    try:
        tree_auc = float(roc_auc_score(true_labels, tree_probs))
    except ValueError:
        tree_auc = 0.5

    if root_only_predictions is not None:
        root_accuracy = float((root_only_predictions == true_labels).mean())
    else:
        root_accuracy = tree.root.base_rate if tree.root else 0.0

    # Per-depth accuracy
    node_depths = {nid: node.depth for nid, node in tree.all_nodes.items()}
    per_depth: Dict[int, Dict[str, int]] = {}
    for node_id in predictions_df["resolving_node"].unique():
        depth = node_depths.get(node_id, -1)
        mask = (predictions_df["resolving_node"] == node_id).values
        if mask.any():
            if depth not in per_depth:
                per_depth[depth] = {"correct": 0, "total": 0}
            per_depth[depth]["correct"] += int((tree_preds[mask] == true_labels[mask]).sum())
            per_depth[depth]["total"] += int(mask.sum())

    per_depth_accuracy = {
        d: stats["correct"] / stats["total"]
        for d, stats in sorted(per_depth.items())
        if stats["total"] > 0
    }

    return {
        "root_accuracy": root_accuracy,
        "tree_accuracy": tree_accuracy,
        "tree_auc": tree_auc,
        "articulability_gap": tree_accuracy - root_accuracy,
        "per_depth_accuracy": per_depth_accuracy,
        "n_examples": len(true_labels),
    }
