"""Visualization utilities for Metric Tree analysis.

Provides functions for:
- Rendering the tree structure as a graphical diagram
- Plotting metrics added at each depth level
- Analyzing metric/rubric complexity across tree depth
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
        for child_type in sorted(node.children.keys()):
            queue.append(node.children[child_type])
    return ordered


def plot_tree_structure(
    tree,
    *,
    ax=None,
    figsize: Tuple[float, float] = (16, 10),
    show_metrics: bool = True,
    show_coefficients: bool = True,
    max_metric_name_len: int = 25,
    node_width: float = 3.0,
    node_height_base: float = 1.2,
    level_gap: float = 3.5,
    sibling_gap: float = 4.0,
):
    """Render the Metric Tree as a hierarchical node diagram.

    Each node box shows:
    - Node ID, depth, accuracy (train/eval)
    - Local metrics added at this node (with coefficients if available)
    - Number of data points and error type

    Parameters
    ----------
    tree : MetricTree
    ax : matplotlib Axes, optional
    figsize : figure size if ax is None
    show_metrics : show metric names inside node boxes
    show_coefficients : show classifier coefficients next to metrics
    max_metric_name_len : truncate metric names beyond this length
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if tree.root is None:
        print("Empty tree (no root node).")
        return None, None

    nodes = _collect_nodes_bfs(tree)
    if not nodes:
        print("No nodes in tree.")
        return None, None

    # Assign positions: BFS by depth, spread siblings horizontally
    depth_groups: Dict[int, list] = {}
    for node in nodes:
        depth_groups.setdefault(node.depth, []).append(node)

    max_depth = max(depth_groups.keys())
    positions: Dict[str, Tuple[float, float]] = {}
    for depth, group in depth_groups.items():
        n = len(group)
        for i, node in enumerate(group):
            x = (i - (n - 1) / 2) * sibling_gap
            y = -depth * level_gap
            positions[node.node_id] = (x, y)

    # Compute node box heights based on content
    node_heights = {}
    for node in nodes:
        n_lines = 2  # header + accuracy line
        if show_metrics and node.local_metrics:
            n_lines += min(len(node.local_metrics), 6) + 1  # +1 for "Metrics:" label
        if n_lines > 8:
            n_lines = 8
        node_heights[node.node_id] = node_height_base + 0.2 * n_lines

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Color palette by depth
    depth_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

    # Draw edges first
    for node in nodes:
        if node.parent_id and node.parent_id in positions:
            px, py = positions[node.parent_id]
            cx, cy = positions[node.node_id]
            parent_h = node_heights.get(node.parent_id, node_height_base)
            child_h = node_heights.get(node.node_id, node_height_base)

            # Find edge label (error type)
            edge_label = ""
            for et, child in tree.all_nodes.get(node.parent_id, node).children.items() if hasattr(tree.all_nodes.get(node.parent_id), 'children') else []:
                pass
            # Get error type from node_id
            parts = node.node_id.rsplit("_", 1)
            if len(parts) > 1 and parts[-1] in ("positive", "negative"):
                edge_label = f"false_{parts[-1]}"
            elif len(parts) > 1 and parts[-1] == "misclassified":
                edge_label = "misclassified"
            # Check for two-word suffix like false_positive
            if "false_positive" in node.node_id:
                edge_label = "false_positive"
            elif "false_negative" in node.node_id:
                edge_label = "false_negative"
            elif "misclassified" in node.node_id:
                edge_label = "misclassified"

            ax.annotate(
                "",
                xy=(cx, cy + child_h / 2),
                xytext=(px, py - parent_h / 2),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#6B7280",
                    lw=1.5,
                    connectionstyle="arc3,rad=0.0",
                ),
            )
            if edge_label:
                mid_x = (px + cx) / 2
                mid_y = (py - parent_h / 2 + cy + child_h / 2) / 2
                ax.text(
                    mid_x + 0.1, mid_y, edge_label.replace("_", " "),
                    fontsize=7, color="#6B7280", style="italic",
                    ha="left", va="center",
                )

    # Draw nodes
    for node in nodes:
        x, y = positions[node.node_id]
        h = node_heights[node.node_id]
        w = node_width
        color = depth_colors[node.depth % len(depth_colors)]

        # Node box
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="#1F2937",
            alpha=0.15,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # Border
        border = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor="none",
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(border)

        # Text content
        text_lines = []
        text_lines.append(f"{node.node_id}")
        n_pts = len(node.point_indices) if hasattr(node, 'point_indices') else 0
        text_lines.append(
            f"acc: {node.train_accuracy:.3f} / {node.eval_accuracy:.3f}  |  n={n_pts}"
        )

        if show_metrics and node.local_metrics:
            text_lines.append("--- metrics ---")
            for m in node.local_metrics[:5]:
                name = m.name[:max_metric_name_len]
                if show_coefficients and node.classifier is not None:
                    try:
                        coefs = node.classifier.coef_.ravel()
                        idx = node.feature_names.index(m.name) if m.name in node.feature_names else -1
                        if 0 <= idx < len(coefs):
                            name += f" ({coefs[idx]:+.3f})"
                    except (ValueError, IndexError, AttributeError):
                        pass
                text_lines.append(f"  {name}")
            if len(node.local_metrics) > 5:
                text_lines.append(f"  ... +{len(node.local_metrics) - 5} more")

        # Render text
        full_text = "\n".join(text_lines)
        ax.text(
            x, y, full_text,
            fontsize=7, fontfamily="monospace",
            ha="center", va="center",
            color="#1F2937",
        )

    # Auto-scale
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    margin = 2.5
    ax.set_xlim(min(all_x) - margin - node_width, max(all_x) + margin + node_width)
    ax.set_ylim(min(all_y) - margin - 2, max(all_y) + margin + 2)

    ax.set_title(
        f"Metric Tree Structure ({len(nodes)} nodes, {len(tree.all_metrics)} metrics)",
        fontsize=13, fontweight="bold", pad=15,
    )

    return fig, ax


def plot_metrics_by_depth(tree, *, ax=None, figsize=(12, 5)):
    """Bar chart showing number of local (new) metrics added at each depth level.

    Also annotates with metric names.
    """
    import matplotlib.pyplot as plt

    if tree.root is None:
        print("Empty tree.")
        return None, None

    nodes = _collect_nodes_bfs(tree)
    depth_data = {}
    for node in nodes:
        d = node.depth
        if d not in depth_data:
            depth_data[d] = {"n_local": 0, "n_all": 0, "names": [], "node_ids": []}
        depth_data[d]["n_local"] += len(node.local_metrics)
        depth_data[d]["n_all"] += len(node.all_metrics)
        depth_data[d]["names"].extend([m.name for m in node.local_metrics])
        depth_data[d]["node_ids"].append(node.node_id)

    depths = sorted(depth_data.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bar_width = 0.35
    x = np.arange(len(depths))

    bars_local = ax.bar(
        x - bar_width / 2,
        [depth_data[d]["n_local"] for d in depths],
        bar_width,
        label="New (local) metrics",
        color="#3B82F6",
        alpha=0.8,
    )
    bars_all = ax.bar(
        x + bar_width / 2,
        [depth_data[d]["n_all"] for d in depths],
        bar_width,
        label="Total (inherited + local)",
        color="#10B981",
        alpha=0.8,
    )

    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Number of Metrics")
    ax.set_title("Metrics Added at Each Tree Depth")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Depth {d}" for d in depths])
    ax.legend()

    # Annotate with metric names
    for i, d in enumerate(depths):
        names = depth_data[d]["names"]
        if names:
            label = "\n".join(textwrap.shorten(n, 30) for n in names[:5])
            if len(names) > 5:
                label += f"\n+{len(names) - 5} more"
            ax.annotate(
                label,
                xy=(x[i] - bar_width / 2, depth_data[d]["n_local"]),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=6,
                ha="center",
                va="bottom",
            )

    fig.tight_layout()
    return fig, ax


def compute_complexity_by_depth(tree) -> pd.DataFrame:
    """Compute metric/rubric complexity statistics at each tree depth.

    Returns a DataFrame with columns:
    - depth, n_nodes, n_local_metrics, n_all_metrics
    - mean_rubric_len_chars, median_rubric_len_chars, total_rubric_len_chars
    - mean_rubric_len_words, mean_rubric_n_levels
    - mean_metric_name_len
    - mean_train_acc, mean_eval_acc
    """
    if tree.root is None:
        return pd.DataFrame()

    nodes = _collect_nodes_bfs(tree)
    depth_stats: Dict[int, dict] = {}

    for node in nodes:
        d = node.depth
        if d not in depth_stats:
            depth_stats[d] = {
                "n_nodes": 0,
                "n_local_metrics": 0,
                "n_all_metrics": 0,
                "rubric_lengths_chars": [],
                "rubric_lengths_words": [],
                "rubric_n_levels": [],
                "metric_name_lengths": [],
                "train_accs": [],
                "eval_accs": [],
                "n_features": [],
                "n_interactions": [],
            }

        stats = depth_stats[d]
        stats["n_nodes"] += 1
        stats["n_local_metrics"] += len(node.local_metrics)
        stats["n_all_metrics"] += len(node.all_metrics)
        stats["train_accs"].append(node.train_accuracy)
        stats["eval_accs"].append(node.eval_accuracy)
        stats["n_features"].append(len(node.feature_names))
        stats["n_interactions"].append(len(node.interaction_pairs))

        for m in node.local_metrics:
            rubric_text = m.rubric_text or ""
            stats["rubric_lengths_chars"].append(len(rubric_text))
            stats["rubric_lengths_words"].append(len(rubric_text.split()))
            # Count distinct levels in rubric (e.g. lines starting with 1:, 2:, etc.)
            n_levels = sum(1 for line in rubric_text.split("\n") if line.strip()[:2].rstrip(":").isdigit())
            stats["rubric_n_levels"].append(n_levels)
            stats["metric_name_lengths"].append(len(m.name))

    rows = []
    for d in sorted(depth_stats.keys()):
        s = depth_stats[d]
        rl = s["rubric_lengths_chars"]
        wl = s["rubric_lengths_words"]
        rows.append({
            "depth": d,
            "n_nodes": s["n_nodes"],
            "n_local_metrics": s["n_local_metrics"],
            "n_all_metrics": s["n_all_metrics"],
            "mean_rubric_len_chars": np.mean(rl) if rl else 0,
            "median_rubric_len_chars": np.median(rl) if rl else 0,
            "total_rubric_len_chars": sum(rl),
            "mean_rubric_len_words": np.mean(wl) if wl else 0,
            "mean_rubric_n_levels": np.mean(s["rubric_n_levels"]) if s["rubric_n_levels"] else 0,
            "mean_metric_name_len": np.mean(s["metric_name_lengths"]) if s["metric_name_lengths"] else 0,
            "mean_train_acc": np.mean(s["train_accs"]),
            "mean_eval_acc": np.mean(s["eval_accs"]),
            "mean_n_features": np.mean(s["n_features"]),
            "mean_n_interactions": np.mean(s["n_interactions"]),
        })

    return pd.DataFrame(rows)


def plot_complexity_by_depth(tree, *, figsize=(16, 10)):
    """Multi-panel plot of metric/rubric complexity across tree depth.

    Panels:
    1. Mean rubric length (chars and words) by depth
    2. Total rubric length (prompt budget) by depth
    3. Number of rubric levels by depth
    4. Train/eval accuracy by depth
    5. Number of features (including interactions) by depth
    6. Distribution of rubric lengths (box plot per depth)
    """
    import matplotlib.pyplot as plt

    if tree.root is None:
        print("Empty tree.")
        return None

    complexity_df = compute_complexity_by_depth(tree)
    if complexity_df.empty:
        print("No complexity data.")
        return None

    nodes = _collect_nodes_bfs(tree)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Panel 1: Mean rubric length
    ax = axes[0, 0]
    ax.bar(complexity_df["depth"], complexity_df["mean_rubric_len_chars"], color="#3B82F6", alpha=0.8)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Mean Rubric Length (chars)")
    ax.set_title("Mean Rubric Length by Depth")
    for _, row in complexity_df.iterrows():
        ax.text(row["depth"], row["mean_rubric_len_chars"] + 5,
                f'{row["mean_rubric_len_words"]:.0f}w', ha="center", fontsize=8)

    # Panel 2: Total rubric length
    ax = axes[0, 1]
    ax.bar(complexity_df["depth"], complexity_df["total_rubric_len_chars"], color="#10B981", alpha=0.8)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Total Rubric Length (chars)")
    ax.set_title("Total Rubric Length (Prompt Budget) by Depth")

    # Panel 3: Number of rubric levels
    ax = axes[0, 2]
    ax.bar(complexity_df["depth"], complexity_df["mean_rubric_n_levels"], color="#F59E0B", alpha=0.8)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Mean # Rubric Levels")
    ax.set_title("Rubric Granularity by Depth")

    # Panel 4: Accuracy by depth
    ax = axes[1, 0]
    ax.plot(complexity_df["depth"], complexity_df["mean_train_acc"], "o-", label="Train", color="#3B82F6")
    ax.plot(complexity_df["depth"], complexity_df["mean_eval_acc"], "s--", label="Eval", color="#EF4444")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Depth")
    ax.legend()
    ax.set_ylim(0.4, 1.0)

    # Panel 5: Features & interactions
    ax = axes[1, 1]
    ax.bar(complexity_df["depth"], complexity_df["mean_n_features"], color="#8B5CF6", alpha=0.8, label="Features")
    ax.bar(complexity_df["depth"], complexity_df["mean_n_interactions"],
           bottom=complexity_df["mean_n_features"], color="#EC4899", alpha=0.5, label="Interactions")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Count")
    ax.set_title("Features & Interactions by Depth")
    ax.legend()

    # Panel 6: Box plot of rubric lengths per depth
    ax = axes[1, 2]
    depth_rubric_lengths: Dict[int, list] = {}
    for node in nodes:
        for m in node.local_metrics:
            depth_rubric_lengths.setdefault(node.depth, []).append(len(m.rubric_text or ""))
    if depth_rubric_lengths:
        depths_sorted = sorted(depth_rubric_lengths.keys())
        data = [depth_rubric_lengths[d] for d in depths_sorted]
        bp = ax.boxplot(data, labels=[f"D{d}" for d in depths_sorted], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3B82F6")
            patch.set_alpha(0.3)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Rubric Length (chars)")
    ax.set_title("Rubric Length Distribution by Depth")

    fig.suptitle("Metric & Rubric Complexity Across Tree Depth", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def format_tree_text(tree, indent: int = 2) -> str:
    """Return a text representation of the tree for display in notebooks."""
    if tree.root is None:
        return "(empty tree)"

    lines = []

    def _fmt(node, level=0):
        prefix = " " * (indent * level)
        # Header
        n_pts = len(node.point_indices) if hasattr(node, 'point_indices') else 0
        lines.append(
            f"{prefix}[{node.node_id}] depth={node.depth}  "
            f"train_acc={node.train_accuracy:.3f}  eval_acc={node.eval_accuracy:.3f}  "
            f"n_points={n_pts}  threshold={node.confidence_threshold:.3f}"
        )
        # Local metrics
        if node.local_metrics:
            lines.append(f"{prefix}  New metrics ({len(node.local_metrics)}):")
            for m in node.local_metrics:
                coef_str = ""
                if node.classifier is not None:
                    try:
                        idx = node.feature_names.index(m.name)
                        coef = node.classifier.coef_.ravel()[idx]
                        coef_str = f"  coef={coef:+.4f}"
                    except (ValueError, IndexError):
                        pass
                lines.append(f"{prefix}    - {m.name} ({m.scale}){coef_str}")
        # Inherited count
        n_inherited = len(node.all_metrics) - len(node.local_metrics)
        if n_inherited > 0:
            lines.append(f"{prefix}  Inherited metrics: {n_inherited}")
        # Features
        lines.append(f"{prefix}  Features: {node.feature_names}")
        if node.interaction_pairs:
            lines.append(f"{prefix}  Interactions: {len(node.interaction_pairs)}")
        # Children
        for et, child in sorted(node.children.items()):
            lines.append(f"{prefix}  -> [{et}]")
            _fmt(child, level + 1)

    _fmt(tree.root)
    return "\n".join(lines)


def compute_rubric_complexity_metrics(tree) -> pd.DataFrame:
    """Compute per-metric complexity indicators.

    Returns DataFrame with one row per metric:
    - metric_id, name, source_node_id, depth
    - rubric_len_chars, rubric_len_words, rubric_n_sentences
    - rubric_n_levels, rubric_avg_level_len
    - rubric_unique_words, rubric_lexical_diversity
    - name_len, name_n_words
    """
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

            text = m.rubric_text or ""
            words = text.split()
            # Sentence count (rough: split on . ! ?)
            sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

            # Level analysis
            levels = []
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped and stripped[0].isdigit() and ":" in stripped[:4]:
                    levels.append(stripped)

            level_lengths = [len(l) for l in levels]
            unique_words = set(w.lower() for w in words)

            rows.append({
                "metric_id": m.metric_id,
                "name": m.name,
                "source_node_id": m.source_node_id,
                "depth": node_depths.get(m.source_node_id, -1),
                "rubric_len_chars": len(text),
                "rubric_len_words": len(words),
                "rubric_n_sentences": len(sentences),
                "rubric_n_levels": len(levels),
                "rubric_avg_level_len": np.mean(level_lengths) if level_lengths else 0,
                "rubric_max_level_len": max(level_lengths) if level_lengths else 0,
                "rubric_unique_words": len(unique_words),
                "rubric_lexical_diversity": len(unique_words) / max(len(words), 1),
                "name_len": len(m.name),
                "name_n_words": len(m.name.replace("_", " ").split()),
            })

    return pd.DataFrame(rows)
