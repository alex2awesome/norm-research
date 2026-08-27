"""Keep/drop guards — all evaluated on ``items_test`` (spec §7).

1. Residualized contrast — applied upstream in :mod:`contrast` (the strongest guard).
2. Redundancy — if the new feature is well predicted by the existing metric columns it is a
   recombination, not new information -> drop. The R^2 ceiling is the scorer's own
   reliability, so we normalize by it (a low R^2 driven by judge noise is not novelty).
3. Gap-closure — keep only if held-out deviance at the original gap node drops materially.
4. Measured importance — ``minimal_depth`` from the refit tree; never LLM confidence.
5. Reliability discount — a noisy extractor makes a true root feature look deeper; correct
   the apparent importance by the reliability estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mob.glmtree import GapTree, _binomial_deviance


# -- guard 2: redundancy ----------------------------------------------------------------

@dataclass
class RedundancyResult:
    r2: float
    normalized_r2: float            # r2 / reliability (judge-noise-adjusted)
    redundant: bool


def redundancy_check(
    new_levels: np.ndarray, existing_X: np.ndarray, applicable: np.ndarray,
    reliability: float, tau: float,
) -> RedundancyResult:
    """R^2 of predicting the new feature from existing metric columns (test set).

    ``existing_X`` should exclude the new feature's own column(s). Fit only on rows where the
    new feature is applicable.
    """
    from sklearn.linear_model import LinearRegression

    mask = applicable & np.isfinite(new_levels)
    if mask.sum() < 10 or existing_X.shape[1] == 0:
        return RedundancyResult(0.0, 0.0, False)
    y = new_levels[mask]
    Xe = existing_X[mask]
    if np.std(y) < 1e-9:
        return RedundancyResult(1.0, 1.0, True)   # constant feature carries no information
    r2 = float(LinearRegression().fit(Xe, y).score(Xe, y))
    r2 = max(0.0, r2)
    norm = r2 / max(reliability, 1e-3)
    return RedundancyResult(r2=r2, normalized_r2=norm, redundant=norm > tau)


# -- guard 3: gap-closure ---------------------------------------------------------------

def subset_deviance(
    tree: GapTree, X_test: np.ndarray, y_test: np.ndarray,
    Z_test: Dict[str, Tuple[np.ndarray, str]], subset_idx: np.ndarray,
) -> Tuple[float, List[str]]:
    """Held-out deviance/item for ``subset_idx`` under ``tree`` + their terminal node ids."""
    n = len(y_test)
    term_ids = tree.terminal_for(Z_test, n)
    nodes = {node.node_id: node for node in tree.terminal_nodes()}
    dev = 0.0
    out_ids: List[str] = []
    for i in subset_idx:
        nid = term_ids[i]
        out_ids.append(nid)
        node = nodes.get(nid)
        if node is None:
            continue
        p = tree.node_predict_proba(node, X_test[i:i + 1])
        dev += _binomial_deviance(y_test[i:i + 1], p)
    return dev / max(len(subset_idx), 1), out_ids


@dataclass
class GapClosureResult:
    old_deviance_per_item: float
    new_deviance_per_item: float
    drop_fraction: float
    deflagged_fraction: float       # of old gap items, fraction no longer in a flagged node
    closed: bool


def gap_closure_check(
    old_deviance_per_item: float, new_tree: GapTree,
    X_test_new: np.ndarray, y_test: np.ndarray,
    Z_test_new: Dict[str, Tuple[np.ndarray, str]],
    old_gap_test_idx: np.ndarray, new_flagged_ids: set, cfg,
) -> GapClosureResult:
    new_dev, new_ids = subset_deviance(new_tree, X_test_new, y_test, Z_test_new, old_gap_test_idx)
    drop_frac = ((old_deviance_per_item - new_dev) / old_deviance_per_item
                 if old_deviance_per_item > 0 else 0.0)
    deflagged = np.mean([nid not in new_flagged_ids for nid in new_ids]) if new_ids else 1.0
    closed = drop_frac >= cfg.min_deviance_drop_frac
    return GapClosureResult(
        old_deviance_per_item=old_deviance_per_item, new_deviance_per_item=new_dev,
        drop_fraction=float(drop_frac), deflagged_fraction=float(deflagged), closed=bool(closed),
    )


# -- guards 4 + 5: measured importance with reliability discount ------------------------

@dataclass
class ImportanceResult:
    minimal_depth: int
    raw_importance: float           # 1 / (1 + minimal_depth)
    reliability: float
    corrected_importance: float     # discounted by reliability (spec §5)


def measured_importance(tree: GapTree, metric_name: str, reliability: float) -> ImportanceResult:
    d = tree.minimal_depth(metric_name)
    raw = 1.0 / (1.0 + d)
    # Reliability DISCOUNT: a noisy extractor inflates apparent importance, so shrink it by the
    # scorer's test-retest reliability (a reliability of 0.5 halves the credited importance).
    # (Previously divided by reliability, which perversely credited noisier features more.)
    corrected = float(min(1.0, raw * max(reliability, 0.0)))
    return ImportanceResult(d, raw, reliability, corrected)
