"""Visualization utilities for Partitioned Metric Tree analysis."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .partition import format_partition_description, _format_key


def _collect_nodes_bfs(tree) -> List[Any]:
    """Collect all nodes in BFS order from a MetricTree."""
    from collections import deque
    if tree.root is None:
        return []
    queue = deque([tree.root])
    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for key in sorted(node.children.keys()):
            queue.append(node.children[key])
    return ordered


def plot_tree_structure(
    tree,
    *,
    ax=None,
    figsize: Tuple[float, float] = (18, 12),
    show_metrics: bool = True,
    max_metric_name_len: int = 25,
    node_width: float = 3.5,
    node_height_base: float = 1.2,
    level_gap: float = 3.5,
    sibling_gap: float = 4.5,
):
    """Render the Partition Metric Tree as a hierarchical node diagram."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if tree.root is None:
        print("Empty tree (no root node).")
        return None, None

    nodes = _collect_nodes_bfs(tree)
    if not nodes:
        print("No nodes in tree.")
        return None, None

    depth_groups: Dict[int, list] = {}
    for node in nodes:
        depth_groups.setdefault(node.depth, []).append(node)

    positions: Dict[str, Tuple[float, float]] = {}
    for depth, group in depth_groups.items():
        n = len(group)
        for i, node in enumerate(group):
            x = (i - (n - 1) / 2) * sibling_gap
            y = -depth * level_gap
            positions[node.node_id] = (x, y)

    node_heights = {}
    for node in nodes:
        n_lines = 3
        if show_metrics and node.local_metrics:
            n_lines += min(len(node.local_metrics), 4) + 1
        node_heights[node.node_id] = node_height_base + 0.2 * min(n_lines, 8)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    depth_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

    # Draw edges
    for node in nodes:
        if node.parent_id and node.parent_id in positions:
            px, py = positions[node.parent_id]
            cx, cy = positions[node.node_id]
            parent_h = node_heights.get(node.parent_id, node_height_base)
            child_h = node_heights.get(node.node_id, node_height_base)

            ax.annotate(
                "",
                xy=(cx, cy + child_h / 2),
                xytext=(px, py - parent_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#6B7280", lw=1.5),
            )

            # Edge label: partition key
            if node.partition_key:
                edge_label = _format_key(node.partition_key)
                mid_x = (px + cx) / 2
                mid_y = (py - parent_h / 2 + cy + child_h / 2) / 2
                ax.text(
                    mid_x + 0.1, mid_y, edge_label,
                    fontsize=8, color="#6B7280", fontweight="bold",
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#6B7280", alpha=0.8),
                )

    # Draw nodes
    for node in nodes:
        x, y = positions[node.node_id]
        h = node_heights[node.node_id]
        w = node_width
        color = depth_colors[node.depth % len(depth_colors)]

        is_leaf = node.is_leaf or not node.children

        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color if not is_leaf else "#FEE2E2",
            edgecolor="#1F2937",
            alpha=0.15,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        border = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor="none",
            edgecolor=color if not is_leaf else "#EF4444",
            linewidth=2,
        )
        ax.add_patch(border)

        text_lines = []
        text_lines.append(f"{node.node_id}")
        n_pts = len(node.point_indices)
        text_lines.append(f"BR:{node.base_rate:.3f} n={n_pts} +{node.n_positive}/-{node.n_negative}")

        if is_leaf:
            text_lines.append("[LEAF]")

        if show_metrics and node.local_metrics:
            text_lines.append("--- binary metrics ---")
            for m in node.local_metrics[:3]:
                text_lines.append(f"  {m.name[:max_metric_name_len]}")
            if len(node.local_metrics) > 3:
                text_lines.append(f"  +{len(node.local_metrics) - 3} more")

        full_text = "\n".join(text_lines)
        ax.text(
            x, y, full_text,
            fontsize=6.5, fontfamily="monospace",
            ha="center", va="center",
            color="#1F2937",
        )

    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    margin = 2.5
    ax.set_xlim(min(all_x) - margin - node_width, max(all_x) + margin + node_width)
    ax.set_ylim(min(all_y) - margin - 2, max(all_y) + margin + 2)

    n_leaves = sum(1 for n in nodes if n.is_leaf or not n.children)
    ax.set_title(
        f"Partition Metric Tree ({len(nodes)} nodes, {n_leaves} leaves, {len(tree.all_metrics)} metrics)",
        fontsize=13, fontweight="bold", pad=15,
    )

    return fig, ax


def plot_metrics_by_depth(tree, *, ax=None, figsize=(12, 5)):
    """Bar chart showing number of binary metrics added at each depth level."""
    import matplotlib.pyplot as plt

    if tree.root is None:
        print("Empty tree.")
        return None, None

    nodes = _collect_nodes_bfs(tree)
    depth_data = {}
    for node in nodes:
        d = node.depth
        if d not in depth_data:
            depth_data[d] = {"n_local": 0, "n_all": 0, "names": []}
        depth_data[d]["n_local"] += len(node.local_metrics)
        depth_data[d]["n_all"] += len(node.all_metrics)
        depth_data[d]["names"].extend([m.name for m in node.local_metrics])

    depths = sorted(depth_data.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bar_width = 0.35
    x = np.arange(len(depths))

    ax.bar(
        x - bar_width / 2,
        [depth_data[d]["n_local"] for d in depths],
        bar_width,
        label="New (local) metrics",
        color="#3B82F6",
        alpha=0.8,
    )
    ax.bar(
        x + bar_width / 2,
        [depth_data[d]["n_all"] for d in depths],
        bar_width,
        label="Total (inherited + local)",
        color="#10B981",
        alpha=0.8,
    )

    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Number of Binary Metrics")
    ax.set_title("Binary Metrics Added at Each Tree Depth")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Depth {d}" for d in depths])
    ax.legend()

    fig.tight_layout()
    return fig, ax


def compute_complexity_by_depth(tree) -> pd.DataFrame:
    """Compute metric complexity statistics at each tree depth."""
    if tree.root is None:
        return pd.DataFrame()

    nodes = _collect_nodes_bfs(tree)
    depth_stats: Dict[int, dict] = {}

    for node in nodes:
        d = node.depth
        if d not in depth_stats:
            depth_stats[d] = {
                "n_nodes": 0, "n_local_metrics": 0, "n_all_metrics": 0,
                "rubric_lengths": [], "base_rates": [],
                "n_positive": [], "n_negative": [],
            }

        stats = depth_stats[d]
        stats["n_nodes"] += 1
        stats["n_local_metrics"] += len(node.local_metrics)
        stats["n_all_metrics"] += len(node.all_metrics)
        stats["base_rates"].append(node.base_rate)
        stats["n_positive"].append(node.n_positive)
        stats["n_negative"].append(node.n_negative)

        for m in node.local_metrics:
            stats["rubric_lengths"].append(len(m.rubric_text or ""))

    rows = []
    for d in sorted(depth_stats.keys()):
        s = depth_stats[d]
        rows.append({
            "depth": d,
            "n_nodes": s["n_nodes"],
            "n_local_metrics": s["n_local_metrics"],
            "n_all_metrics": s["n_all_metrics"],
            "mean_rubric_len": np.mean(s["rubric_lengths"]) if s["rubric_lengths"] else 0,
            "mean_base_rate": np.mean(s["base_rates"]),
            "total_points": sum(s["n_positive"]) + sum(s["n_negative"]),
        })

    return pd.DataFrame(rows)


def format_tree_text(tree, indent: int = 2) -> str:
    """Return a text representation of the tree for display."""
    if tree.root is None:
        return "(empty tree)"

    lines = []

    def _fmt(node, level=0):
        prefix = " " * (indent * level)
        n_pts = len(node.point_indices)
        pk_desc = ""
        if node.partition_key:
            parent = tree.all_nodes.get(node.parent_id or "")
            if parent and parent.local_metrics:
                pk_desc = f" [{format_partition_description(node.partition_key, [m.name for m in parent.local_metrics])}]"
            else:
                pk_desc = f" [{_format_key(node.partition_key)}]"

        lines.append(
            f"{prefix}[{node.node_id}] depth={node.depth}{pk_desc}  "
            f"BR={node.base_rate:.3f}  n={n_pts} (+{node.n_positive}/-{node.n_negative})"
        )

        if node.local_metrics:
            lines.append(f"{prefix}  Binary metrics ({len(node.local_metrics)}):")
            for m in node.local_metrics:
                lines.append(f"{prefix}    - {m.name}")

        if node.is_leaf or not node.children:
            lines.append(f"{prefix}  [LEAF — base-rate]")

        for pk, child in sorted(node.children.items(), key=lambda x: _format_key(x[0])):
            lines.append(f"{prefix}  -> partition {_format_key(pk)}")
            _fmt(child, level + 1)

    _fmt(tree.root)
    return "\n".join(lines)


def compute_rubric_complexity_metrics(tree) -> pd.DataFrame:
    """Compute per-metric complexity indicators."""
    if tree.root is None:
        return pd.DataFrame()

    nodes = _collect_nodes_bfs(tree)
    node_depths = {n.node_id: n.depth for n in nodes}

    rows = []
    seen = set()
    for node in nodes:
        for m in node.local_metrics:
            if m.metric_id in seen:
                continue
            seen.add(m.metric_id)

            yes_text = m.rubric.get("yes", "")
            no_text = m.rubric.get("no", "")
            total_len = len(yes_text) + len(no_text)

            rows.append({
                "metric_id": m.metric_id,
                "name": m.name,
                "source_node_id": m.source_node_id,
                "depth": node_depths.get(m.source_node_id, -1),
                "yes_desc_len": len(yes_text),
                "no_desc_len": len(no_text),
                "total_rubric_len": total_len,
                "name_len": len(m.name),
            })

    return pd.DataFrame(rows)
