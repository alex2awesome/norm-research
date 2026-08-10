"""GapTree: model-based recursive partitioning with a logistic node model.

Within-node model:  P(y=1 | node) = sigmoid(sum_m beta_m * level_m(item))   (metric LEVELS)
Partitioning:       split on the covariate z whose M-fluctuation test is most significant,
                    at the cutpoint maximizing the partitioned binomial log-likelihood.

A node is **terminal** when no covariate's (Bonferroni-adjusted, permutation) p-value clears
``alpha`` -- i.e. the metric->label relationship is stable and no available ``z`` repairs it.
Terminal nodes with poor held-out fit are the *feature gaps* (see ``gaps.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mfluctuation import (
    FluctResult,
    cov_inverse,
    fit_node_glm,
    score_contributions,
    test_node,
)

ZFrame = Dict[str, Tuple[np.ndarray, str]]  # var -> (values over ALL rows, "numeric"|"categorical")


# --------------------------------------------------------------------------------------

@dataclass
class Split:
    """A binary split rule. Left child if the predicate holds."""

    variable: str
    kind: str                                 # "numeric" | "categorical"
    threshold: Optional[float] = None         # numeric: left iff value <= threshold
    left_levels: Optional[frozenset] = None   # categorical: left iff value in left_levels

    def goes_left(self, value) -> bool:
        if self.kind == "numeric":
            return float(value) <= self.threshold
        return value in self.left_levels

    def describe(self) -> str:
        if self.kind == "numeric":
            return f"{self.variable} <= {self.threshold:.4g}"
        return f"{self.variable} in {sorted(self.left_levels)}"


@dataclass
class TreeNode:
    node_id: str
    depth: int
    indices: np.ndarray                       # rows (into the discover arrays) at this node
    parent_id: Optional[str] = None
    beta: Optional[np.ndarray] = None         # fitted GLM coefficients (intercept first)
    deviance: float = float("nan")            # discover-set binomial deviance
    n_pos: int = 0
    n_neg: int = 0
    base_rate: float = 0.5
    split: Optional[Split] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    is_terminal: bool = False
    fluct: List[FluctResult] = field(default_factory=list)   # test results at this node


# --------------------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _binomial_deviance(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-2.0 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _loglik_fit(X: np.ndarray, y: np.ndarray) -> float:
    """Fitted-GLM log-likelihood on (X, y); -inf if a child is unusable."""
    if len(y) == 0:
        return -np.inf
    _, p, _ = fit_node_glm(X, y)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --------------------------------------------------------------------------------------

class GapTree:
    """Gap-detecting classification tree (MOB with a logistic node model)."""

    def __init__(self, config):
        self.config = config
        self.root: Optional[TreeNode] = None
        self.feature_names: List[str] = []   # X columns (metric levels)
        self.z_names: List[str] = []
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._Z: Optional[ZFrame] = None
        self._node_counter = 0
        self.root_std_coef: Dict[str, float] = {}

    # -- fitting ----------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, Z: ZFrame, feature_names: List[str]) -> "GapTree":
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._Z = Z
        self.feature_names = list(feature_names)
        self.z_names = list(Z.keys())
        self._node_counter = 0
        rng = np.random.default_rng(self.config.random_seed)
        self.root = self._grow(np.arange(len(self._y)), depth=0, parent_id=None, rng=rng)
        self._compute_root_std_coef()
        return self

    def _new_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"

    def _grow(self, idx: np.ndarray, depth: int, parent_id: Optional[str],
              rng: np.random.Generator) -> TreeNode:
        cfg = self.config
        X_node, y_node = self._X[idx], self._y[idx]
        beta, p, X_design = fit_node_glm(X_node, y_node)
        node = TreeNode(
            node_id=self._new_id(), depth=depth, indices=idx, parent_id=parent_id,
            beta=beta, deviance=_binomial_deviance(y_node, p),
            n_pos=int(y_node.sum()), n_neg=int(len(y_node) - y_node.sum()),
            base_rate=float(y_node.mean()) if len(y_node) else 0.5,
        )

        # stop conditions
        too_small = len(idx) < 2 * cfg.min_node_size
        too_deep = cfg.max_depth is not None and depth >= cfg.max_depth
        pure = node.n_pos == 0 or node.n_neg == 0
        if too_small or too_deep or pure:
            node.is_terminal = True
            return node

        # instability test
        psi = score_contributions(X_design, y_node, p)
        Jinv = cov_inverse(psi)
        z_local = {name: (vals[idx], kind) for name, (vals, kind) in self._Z.items()}
        results = test_node(
            psi, z_local, trim=cfg.mfluct_trim, n_perm=cfg.n_permutations,
            bonferroni=cfg.bonferroni, rng=rng, Jinv=Jinv,
        )
        node.fluct = results
        best = results[0] if results else None
        if best is None or best.adj_pvalue >= cfg.alpha:
            node.is_terminal = True
            return node

        # exhaustive cutpoint search on the selected covariate
        split = self._best_split(idx, best)
        if split is None:
            node.is_terminal = True
            return node

        node.split = split
        zvals = self._Z[split.variable][0][idx]
        go_left = np.array([split.goes_left(v) for v in zvals])
        node.left = self._grow(idx[go_left], depth + 1, node.node_id, rng)
        node.right = self._grow(idx[~go_left], depth + 1, node.node_id, rng)
        return node

    def _best_split(self, idx: np.ndarray, res: FluctResult) -> Optional[Split]:
        """Find the cutpoint on ``res.variable`` maximizing partitioned log-likelihood."""
        cfg = self.config
        X_node, y_node = self._X[idx], self._y[idx]
        zvals = self._Z[res.variable][0][idx]
        best_ll, best_split = -np.inf, None

        if res.kind == "numeric":
            candidates = self._numeric_cutpoints(zvals.astype(float), cfg.n_cutpoint_candidates)
            for thr in candidates:
                left = zvals.astype(float) <= thr
                ll = self._partitioned_ll(X_node, y_node, left, cfg.min_node_size)
                if ll > best_ll:
                    best_ll, best_split = ll, Split(res.variable, "numeric", threshold=float(thr))
        else:
            for left_levels in self._categorical_partitions(zvals, y_node):
                left = np.array([v in left_levels for v in zvals])
                ll = self._partitioned_ll(X_node, y_node, left, cfg.min_node_size)
                if ll > best_ll:
                    best_ll = ll
                    best_split = Split(res.variable, "categorical", left_levels=frozenset(left_levels))
        return best_split

    @staticmethod
    def _numeric_cutpoints(vals: np.ndarray, cap: int) -> np.ndarray:
        uniq = np.unique(vals)
        if len(uniq) < 2:
            return np.array([])
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        if cap and len(mids) > cap:
            sel = np.linspace(0, len(mids) - 1, cap).round().astype(int)
            mids = mids[np.unique(sel)]
        return mids

    @staticmethod
    def _categorical_partitions(vals: np.ndarray, y: np.ndarray):
        """Yield candidate left-level sets for a binary categorical split.

        Small cardinality: exhaustive over non-trivial subsets. Larger: order levels by
        within-node positive rate and try threshold splits along that order (the standard
        MOB heuristic that avoids the 2^(C-1) blow-up).
        """
        levels = list(np.unique(vals))
        C = len(levels)
        if C < 2:
            return
        if C <= 6:
            seen = set()
            for r in range(1, C):
                for combo in combinations(levels, r):
                    key = frozenset(combo)
                    comp = frozenset(levels) - key
                    if key in seen or comp in seen:
                        continue
                    seen.add(key)
                    yield set(combo)
        else:
            rates = {lv: y[vals == lv].mean() if np.any(vals == lv) else 0.0 for lv in levels}
            ordered = sorted(levels, key=lambda lv: rates[lv])
            for cut in range(1, C):
                yield set(ordered[:cut])

    @staticmethod
    def _partitioned_ll(X: np.ndarray, y: np.ndarray, left: np.ndarray, min_node: int) -> float:
        nL, nR = int(left.sum()), int((~left).sum())
        if nL < min_node or nR < min_node:
            return -np.inf
        return _loglik_fit(X[left], y[left]) + _loglik_fit(X[~left], y[~left])

    def _compute_root_std_coef(self) -> None:
        if self.root is None or self.root.beta is None:
            return
        std = self._X.std(axis=0)
        coef = self.root.beta[1:]  # drop intercept
        self.root_std_coef = {
            name: float(coef[i] * std[i]) for i, name in enumerate(self.feature_names)
        }

    # -- inference ---------------------------------------------------------------------

    def terminal_nodes(self) -> List[TreeNode]:
        out: List[TreeNode] = []

        def walk(n: Optional[TreeNode]):
            if n is None:
                return
            if n.is_terminal:
                out.append(n)
            else:
                walk(n.left)
                walk(n.right)

        walk(self.root)
        return out

    def all_nodes(self) -> List[TreeNode]:
        out: List[TreeNode] = []

        def walk(n: Optional[TreeNode]):
            if n is None:
                return
            out.append(n)
            walk(n.left)
            walk(n.right)

        walk(self.root)
        return out

    def _route_row(self, zrow: Dict[str, object]) -> TreeNode:
        node = self.root
        while node is not None and not node.is_terminal and node.split is not None:
            node = node.left if node.split.goes_left(zrow[node.split.variable]) else node.right
        return node

    def terminal_for(self, Z_rows: ZFrame, n_rows: int) -> List[str]:
        """Terminal node id for each of ``n_rows`` items described by ``Z_rows``."""
        ids = []
        for i in range(n_rows):
            zrow = {name: vals[i] for name, (vals, _) in Z_rows.items()}
            node = self._route_row(zrow)
            ids.append(node.node_id if node is not None else None)
        return ids

    @staticmethod
    def node_predict_proba(node: TreeNode, X_rows: np.ndarray) -> np.ndarray:
        X_rows = np.atleast_2d(np.asarray(X_rows, dtype=float))
        X_design = np.column_stack([np.ones(len(X_rows)), X_rows])
        return _sigmoid(X_design @ node.beta)

    def feature_active_coverage(
        self, feature_name: str, X: np.ndarray, Z_rows: ZFrame, n_rows: int,
        threshold: float = 0.5,
    ) -> float:
        """Fraction of the population in leaves where ``feature_name`` is *active*.

        A feature is active in a leaf when its standardized within-leaf coefficient exceeds
        ``threshold`` (a real effect on the log-odds, per SD of the feature). This is the
        faithful measure of a feature's generality / coverage — independent of which gap node
        happened to surface it — so a conditional feature (active in one branch) reads as
        narrow and a broad feature (active across leaves) reads as general.
        """
        if feature_name not in self.feature_names:
            return float("nan")
        col = self.feature_names.index(feature_name)
        term_ids = self.terminal_for(Z_rows, n_rows)
        nodes = {nd.node_id: nd for nd in self.terminal_nodes()}
        by_node: Dict[str, List[int]] = {}
        for i, nid in enumerate(term_ids):
            by_node.setdefault(nid, []).append(i)
        covered = 0
        for nid, rows in by_node.items():
            node = nodes.get(nid)
            if node is None or node.beta is None:
                continue
            sd = float(np.std(X[np.array(rows), col]))
            std_coef = abs(node.beta[1 + col] * sd)
            if std_coef >= threshold:
                covered += len(rows)
        return covered / max(n_rows, 1)

    # -- measured generality -----------------------------------------------------------

    def minimal_depth(self, metric_name: str, root_coef_pctile: float = 0.5) -> int:
        """Shallowest depth at which the tree splits on ``metric_name``.

        Falls back to depth 0 when the metric is not used as a split but is a dominant
        root coefficient (>= the ``root_coef_pctile`` quantile of |std coef|); otherwise
        returns a deep sentinel (it is a narrow/marginal feature).
        """
        depths = [n.depth for n in self.all_nodes()
                  if n.split is not None and n.split.variable == metric_name]
        if depths:
            return min(depths)
        if self.root_std_coef:
            mags = np.abs(list(self.root_std_coef.values()))
            thr = np.quantile(mags, root_coef_pctile)
            if abs(self.root_std_coef.get(metric_name, 0.0)) >= thr and thr > 0:
                return 0
        max_d = max((n.depth for n in self.all_nodes()), default=0)
        return max_d + 1
