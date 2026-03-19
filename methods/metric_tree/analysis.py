"""Tree analysis and articulability gap measurement."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_structures import MetricTree, MetricTreeNode

logger = logging.getLogger("metric_tree.analysis")


def analyze_tree_complexity(tree: MetricTree) -> pd.DataFrame:
    """Compute per-node statistics for the tree.

    Returns a DataFrame with columns: node_id, depth, parent_id,
    n_local_metrics, n_all_metrics, n_features, n_points, train_accuracy,
    eval_accuracy, n_children, confidence_threshold.
    """
    rows = []
    for node_id, node in tree.all_nodes.items():
        rows.append({
            "node_id": node.node_id,
            "depth": node.depth,
            "parent_id": node.parent_id or "",
            "n_local_metrics": len(node.local_metrics),
            "n_all_metrics": len(node.all_metrics),
            "n_features": len(node.feature_names),
            "n_points": len(node.point_indices),
            "n_correct": int(node.correct_mask.sum()) if len(node.correct_mask) > 0 else 0,
            "train_accuracy": node.train_accuracy,
            "eval_accuracy": node.eval_accuracy,
            "n_children": len(node.children),
            "confidence_threshold": node.confidence_threshold,
            "n_interactions": len(node.interaction_pairs),
        })
    return pd.DataFrame(rows)


def measure_depth_distribution(
    tree: MetricTree,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute resolution depth histogram from prediction results.

    Expects predictions_df to have a 'resolving_node' column (from predict_batch).
    Returns DataFrame with columns: depth, count, fraction.
    """
    node_depths = {nid: node.depth for nid, node in tree.all_nodes.items()}

    depths = []
    for node_id in predictions_df["resolving_node"]:
        if node_id in node_depths:
            depths.append(node_depths[node_id])
        else:
            depths.append(-1)  # unresolved

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
    lines.append("METRIC TREE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Task: {tree.task_description[:200]}")
    lines.append(f"Total nodes: {len(tree.all_nodes)}")
    lines.append(f"Total unique metrics: {len(tree.all_metrics)}")
    lines.append("")

    def _format_node(node: MetricTreeNode, indent: int = 0) -> List[str]:
        prefix = "  " * indent
        node_lines = []
        node_lines.append(f"{prefix}[{node.node_id}] depth={node.depth}")
        node_lines.append(f"{prefix}  Train acc: {node.train_accuracy:.3f} | Eval acc: {node.eval_accuracy:.3f}")
        node_lines.append(f"{prefix}  Points: {len(node.point_indices)} | Confidence threshold: {node.confidence_threshold:.3f}")
        node_lines.append(f"{prefix}  Features ({len(node.feature_names)}):")

        for fn in node.feature_names:
            node_lines.append(f"{prefix}    - {fn}")

        if node.local_metrics:
            node_lines.append(f"{prefix}  Local metrics ({len(node.local_metrics)}):")
            for m in node.local_metrics:
                node_lines.append(f"{prefix}    [{m.metric_id[:8]}] {m.name} (scale={m.scale})")
                # Show abbreviated rubric
                rubric_preview = m.rubric_text[:150].replace("\n", " ")
                node_lines.append(f"{prefix}      Rubric: {rubric_preview}...")

        if node.interaction_pairs:
            node_lines.append(f"{prefix}  Interactions ({len(node.interaction_pairs)}):")
            for a, b in node.interaction_pairs:
                node_lines.append(f"{prefix}    {a} x {b}")

        if node.classifier is not None:
            try:
                coefs = node.classifier.coef_.ravel()
                for i, (fn, c) in enumerate(zip(node.feature_names, coefs)):
                    if abs(c) > 1e-4:
                        node_lines.append(f"{prefix}  Coef {fn}: {c:.4f}")
            except Exception:
                pass

        for child_type, child_node in node.children.items():
            node_lines.append(f"{prefix}  -> Child [{child_type}]:")
            node_lines.extend(_format_node(child_node, indent + 2))

        return node_lines

    if tree.root:
        lines.extend(_format_node(tree.root))

    lines.append("")
    lines.append("=" * 80)
    lines.append("ALL METRICS")
    lines.append("=" * 80)
    for metric_id, m in sorted(tree.all_metrics.items()):
        lines.append(f"\n[{metric_id[:8]}] {m.name} (source: {m.source_node_id}, scale={m.scale})")
        lines.append(f"  Rubric:\n    {m.rubric_text[:500]}")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Tree summary exported to %s (%d chars)", path, len(text))


def compute_articulability_gap(
    tree: MetricTree,
    predictions_df: pd.DataFrame,
    true_labels: np.ndarray,
    root_only_predictions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Measure the articulability gap: how much accuracy improves with depth.

    The spec defines three accuracy numbers:
    1. Root-only accuracy: classify ALL test points using root's classifier only.
    2. Full tree accuracy: the complete rule-plus-exception hierarchy.
    3. Implicit model accuracy: external (not computed here).

    Parameters
    ----------
    root_only_predictions : optional array of predictions from root classifier
        applied to ALL test points. If not provided, we approximate using
        predictions from examples that resolved at root depth.

    Returns dict with:
    - root_accuracy: accuracy using only root node on ALL test points
    - tree_accuracy: accuracy using full tree
    - articulability_gap: tree_accuracy - root_accuracy
    - per_depth_accuracy: accuracy at each depth level
    """
    tree_preds = predictions_df["prediction"].values
    tree_accuracy = float((tree_preds == true_labels).mean())

    # Root-only accuracy: root classifier applied to ALL test points
    if root_only_predictions is not None:
        root_accuracy = float((root_only_predictions == true_labels).mean())
    else:
        # Approximate: use tree predictions for root-resolved, but note this
        # underestimates root-only accuracy since deeper-resolved examples
        # may have been classified differently by the root alone.
        # For proper measurement, caller should pass root_only_predictions.
        root_accuracy = tree.root.eval_accuracy if tree.root else 0.0
        logger.warning(
            "root_only_predictions not provided; using root eval_accuracy (%.3f) "
            "as approximation. For accurate gap measurement, score all test points "
            "through root classifier only and pass as root_only_predictions.",
            root_accuracy,
        )

    # Per-depth accuracy: how accurate are predictions at each resolution depth
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
        "articulability_gap": tree_accuracy - root_accuracy,
        "per_depth_accuracy": per_depth_accuracy,
        "n_examples": len(true_labels),
    }
