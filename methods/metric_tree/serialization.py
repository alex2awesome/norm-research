"""Save and load complete MetricTree objects, including per-node routers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import TreeConfig
from .data_structures import MetricTree, PartitionTreeNode, TreeMetric
from .router import NodeRouter, save_router, load_router

logger = logging.getLogger("metric_tree.serialization")


def _node_to_dict(node: PartitionTreeNode) -> dict:
    """Serialize a PartitionTreeNode to a JSON-safe dict (without large arrays)."""
    return {
        "node_id": node.node_id,
        "depth": node.depth,
        "parent_id": node.parent_id,
        "partition_key": list(node.partition_key) if node.partition_key else [],
        "local_metric_ids": [m.metric_id for m in node.local_metrics],
        "all_metric_ids": [m.metric_id for m in node.all_metrics],
        "n_points": len(node.point_indices),
        "base_rate": node.base_rate,
        "n_positive": node.n_positive,
        "n_negative": node.n_negative,
        "is_leaf": node.is_leaf,
        "children_keys": [list(k) for k in node.children.keys()],
        "has_router": node.router is not None,
        "router_minority_is_positive": node.router_minority_is_positive,
    }


def _metric_to_dict(m: TreeMetric) -> dict:
    """Serialize a TreeMetric to a JSON-safe dict."""
    return {
        "metric_id": m.metric_id,
        "name": m.name,
        "rubric_text": m.rubric_text,
        "rubric": m.rubric,
        "source_node_id": m.source_node_id,
        "scale": m.scale,
    }


def save_tree(tree: MetricTree, output_dir: str) -> None:
    """Save a complete MetricTree to disk.

    Creates:
      - tree_metadata.json: tree structure, metrics, config
      - routers/{node_id}.npz: per-node router weights (if any)
      - arrays/{node_id}_*.npy: score arrays for each node
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save routers
    router_dir = out / "routers"
    router_dir.mkdir(exist_ok=True)
    n_routers = 0

    for node_id, node in tree.all_nodes.items():
        if node.router is not None:
            router_path = str(router_dir / f"{node_id}.pt")
            save_router(node.router, router_path)
            n_routers += 1

    # Save score arrays
    array_dir = out / "arrays"
    array_dir.mkdir(exist_ok=True)

    for node_id, node in tree.all_nodes.items():
        if len(node.point_indices) > 0:
            np.save(str(array_dir / f"{node_id}_point_indices.npy"), node.point_indices)
        if node.local_scores.size > 0:
            np.save(str(array_dir / f"{node_id}_local_scores.npy"), node.local_scores)
        if node.all_scores.size > 0:
            np.save(str(array_dir / f"{node_id}_all_scores.npy"), node.all_scores)

    # Save metadata
    metadata = {
        "task_description": tree.task_description,
        "metrics": {mid: _metric_to_dict(m) for mid, m in tree.all_metrics.items()},
        "nodes": {nid: _node_to_dict(n) for nid, n in tree.all_nodes.items()},
        "root_id": tree.root.node_id if tree.root else None,
        "n_routers": n_routers,
    }

    # Save config if present
    if tree.config is not None:
        cfg = tree.config
        metadata["config"] = {
            k: getattr(cfg, k) for k in cfg.__dataclass_fields__
        }

    with open(out / "tree_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Tree saved to %s (%d nodes, %d metrics, %d routers)",
                output_dir, len(tree.all_nodes), len(tree.all_metrics), n_routers)


def load_tree(output_dir: str) -> MetricTree:
    """Load a complete MetricTree from disk, including routers.

    Returns a MetricTree with all nodes, metrics, and trained routers restored.
    """
    out = Path(output_dir)

    with open(out / "tree_metadata.json") as f:
        metadata = json.load(f)

    # Reconstruct config
    config = None
    if "config" in metadata:
        config = TreeConfig(**{
            k: v for k, v in metadata["config"].items()
            if k in TreeConfig.__dataclass_fields__
        })

    # Reconstruct metrics
    all_metrics = {}
    for mid, mdata in metadata["metrics"].items():
        all_metrics[mid] = TreeMetric(
            metric_id=mdata["metric_id"],
            name=mdata["name"],
            rubric_text=mdata["rubric_text"],
            rubric=mdata["rubric"],
            source_node_id=mdata["source_node_id"],
            scale=mdata.get("scale", "binary"),
        )

    # Reconstruct nodes (first pass: create nodes without children links)
    all_nodes = {}
    for nid, ndata in metadata["nodes"].items():
        # Load arrays
        array_dir = out / "arrays"
        point_indices = np.array([], dtype=int)
        local_scores = np.empty((0, 0))
        all_scores = np.empty((0, 0))

        pi_path = array_dir / f"{nid}_point_indices.npy"
        if pi_path.exists():
            point_indices = np.load(str(pi_path))
        ls_path = array_dir / f"{nid}_local_scores.npy"
        if ls_path.exists():
            local_scores = np.load(str(ls_path))
        as_path = array_dir / f"{nid}_all_scores.npy"
        if as_path.exists():
            all_scores = np.load(str(as_path))

        # Load router
        router = None
        router_path = out / "routers" / f"{nid}.pt"
        if ndata.get("has_router") and router_path.exists():
            router = load_router(str(router_path))

        node = PartitionTreeNode(
            node_id=ndata["node_id"],
            depth=ndata["depth"],
            parent_id=ndata.get("parent_id"),
            partition_key=tuple(ndata.get("partition_key", [])),
            local_metrics=[all_metrics[mid] for mid in ndata["local_metric_ids"] if mid in all_metrics],
            all_metrics=[all_metrics[mid] for mid in ndata["all_metric_ids"] if mid in all_metrics],
            point_indices=point_indices,
            local_scores=local_scores,
            all_scores=all_scores,
            base_rate=ndata["base_rate"],
            n_positive=ndata["n_positive"],
            n_negative=ndata["n_negative"],
            is_leaf=ndata["is_leaf"],
            router=router,
            router_minority_is_positive=ndata.get("router_minority_is_positive"),
        )
        all_nodes[nid] = node

    # Second pass: wire up children
    for nid, ndata in metadata["nodes"].items():
        node = all_nodes[nid]
        for child_key_list in ndata.get("children_keys", []):
            child_key = tuple(child_key_list)
            # Find the child node by matching partition_key and parent_id
            for cid, cnode in all_nodes.items():
                if cnode.parent_id == nid and tuple(cnode.partition_key) == child_key:
                    node.children[child_key] = cnode
                    break

    # Build tree
    tree = MetricTree(
        root=all_nodes.get(metadata.get("root_id")),
        config=config,
        all_nodes=all_nodes,
        all_metrics=all_metrics,
        task_description=metadata.get("task_description", ""),
    )

    n_routers = sum(1 for n in all_nodes.values() if n.router is not None)
    logger.info("Tree loaded from %s (%d nodes, %d metrics, %d routers)",
                output_dir, len(all_nodes), len(all_metrics), n_routers)

    return tree
