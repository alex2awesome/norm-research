"""Flag feature-gap nodes (spec §2.1).

A terminal node ``N`` is a *feature gap* iff, on held-out (test) items routed to it:

1. the within-node metric GLM predicts the label poorly (deviance/item above threshold, or
   AUC near chance), **and**
2. no available split fixes it -- which, because the tree already stopped there, holds for
   every terminal node by construction.

(1) alone at a non-terminal node is just an ordinary splittable node. (1)+(2) localizes the
articulability gap to a subpopulation. Whether the gap is signal or irreducible noise is left
to the LLM reinsertion test (``loop.py`` + ``guards.py``), not decided here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .mob.glmtree import GapTree, TreeNode, _binomial_deviance


@dataclass
class GapNode:
    node: TreeNode
    test_indices: np.ndarray        # rows (into the test arrays) routed to this node
    deviance_per_item: float
    auc: float                      # NaN if a single class is present
    n_test: int
    reasons: Tuple[str, ...]        # which criteria flagged it


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def evaluate_terminal_nodes(
    tree: GapTree, X_test: np.ndarray, y_test: np.ndarray,
    Z_test: Dict[str, Tuple[np.ndarray, str]],
) -> Dict[str, dict]:
    """Per terminal node: held-out deviance/item, AUC, and the routed test indices."""
    n = len(y_test)
    term_ids = tree.terminal_for(Z_test, n)
    by_node: Dict[str, List[int]] = {}
    for i, nid in enumerate(term_ids):
        by_node.setdefault(nid, []).append(i)

    nodes = {node.node_id: node for node in tree.terminal_nodes()}
    out: Dict[str, dict] = {}
    for nid, idxs in by_node.items():
        node = nodes.get(nid)
        if node is None:
            continue
        idx = np.array(idxs)
        p = tree.node_predict_proba(node, X_test[idx])
        y = y_test[idx]
        dev = _binomial_deviance(y, p)
        out[nid] = {
            "node": node, "test_indices": idx,
            "deviance_per_item": dev / max(len(idx), 1),
            "auc": _auc(y, p), "n_test": len(idx),
        }
    return out


def flag_gap_nodes(
    tree: GapTree, X_test: np.ndarray, y_test: np.ndarray,
    Z_test: Dict[str, Tuple[np.ndarray, str]], cfg,
) -> List[GapNode]:
    """Return terminal nodes whose held-out fit is poor (the feature gaps)."""
    stats = evaluate_terminal_nodes(tree, X_test, y_test, Z_test)
    gaps: List[GapNode] = []
    for nid, s in stats.items():
        if s["n_test"] < cfg.gap_min_test_items:
            continue
        reasons = []
        if s["deviance_per_item"] > cfg.gap_deviance_per_item:
            reasons.append("high_deviance")
        if np.isfinite(s["auc"]) and s["auc"] <= cfg.gap_auc_threshold:
            reasons.append("low_auc")
        if reasons:
            gaps.append(GapNode(
                node=s["node"], test_indices=s["test_indices"],
                deviance_per_item=s["deviance_per_item"], auc=s["auc"],
                n_test=s["n_test"], reasons=tuple(reasons),
            ))
    # worst gaps first
    gaps.sort(key=lambda g: g.deviance_per_item, reverse=True)
    return gaps
