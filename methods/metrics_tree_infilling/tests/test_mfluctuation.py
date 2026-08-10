"""Unit tests for the MOB engine on synthetic data with *known* instability structure.

Fast (no R): these check that the M-fluctuation test detects a planted coefficient break
on the right covariate, ignores a null covariate, stops on stable data, and that GapTree
grows/splits accordingly. The extensive numeric comparison against R partykit lives in
``validate_against_partykit.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.mob.glmtree import GapTree
from methods.metrics_tree_infilling.mob.mfluctuation import (
    fit_node_glm,
    score_contributions,
)
from methods.metrics_tree_infilling.mob.mfluctuation import test_node as run_fluct_test


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _make_break_data(n=800, seed=0, beta=2.5, split=0.5):
    """Coefficient of x on y flips sign at z >= split. z_null is independent noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.uniform(size=n)
    z_null = rng.uniform(size=n)
    sign = np.where(z >= split, -1.0, 1.0)
    logit = sign * beta * x
    y = (rng.uniform(size=n) < _sigmoid(logit)).astype(float)
    return x, y, z, z_null


def _make_stable_data(n=800, seed=1, beta=2.5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.uniform(size=n)
    y = (rng.uniform(size=n) < _sigmoid(beta * x)).astype(float)
    return x, y, z


def _pooled_psi(x, y):
    X = x.reshape(-1, 1)
    _, p, X_design = fit_node_glm(X, y)
    return score_contributions(X_design, y, p)


def test_numeric_break_detected_and_null_ignored():
    x, y, z, z_null = _make_break_data()
    psi = _pooled_psi(x, y)
    rng = np.random.default_rng(123)
    res = run_fluct_test(
        psi,
        {"z_break": (z, "numeric"), "z_null": (z_null, "numeric")},
        n_perm=199, bonferroni=False, rng=rng,
    )
    by = {r.variable: r for r in res}
    assert by["z_break"].pvalue < 0.05, by["z_break"]
    assert by["z_null"].pvalue > 0.10, by["z_null"]
    # split-location hint should land near the planted break at 0.5
    assert abs(by["z_break"].split_value - 0.5) < 0.15, by["z_break"].split_value


def test_categorical_break_detected():
    x, y, z, _ = _make_break_data()
    group = (z >= 0.5).astype(int)
    psi = _pooled_psi(x, y)
    rng = np.random.default_rng(7)
    res = run_fluct_test(psi, {"grp": (group, "categorical")}, n_perm=199, bonferroni=False, rng=rng)
    assert res[0].variable == "grp"
    assert res[0].pvalue < 0.05, res[0]


def test_stable_data_not_flagged():
    x, y, z = _make_stable_data()
    psi = _pooled_psi(x, y)
    rng = np.random.default_rng(5)
    res = run_fluct_test(psi, {"z": (z, "numeric")}, n_perm=199, bonferroni=False, rng=rng)
    assert res[0].pvalue > 0.05, res[0]


def test_gaptree_splits_on_break_variable():
    x, y, z, z_null = _make_break_data()
    cfg = InfillConfig(n_permutations=199, min_node_size=40, max_depth=2, random_seed=0)
    tree = GapTree(cfg).fit(
        x.reshape(-1, 1), y,
        {"z_break": (z, "numeric"), "z_null": (z_null, "numeric")},
        feature_names=["x"],
    )
    assert tree.root.split is not None
    assert tree.root.split.variable == "z_break"
    assert abs(tree.root.split.threshold - 0.5) < 0.2


def test_gaptree_terminal_on_stable():
    x, y, z = _make_stable_data()
    cfg = InfillConfig(n_permutations=199, min_node_size=40, max_depth=3, random_seed=0)
    tree = GapTree(cfg).fit(
        x.reshape(-1, 1), y, {"z": (z, "numeric")}, feature_names=["x"],
    )
    assert tree.root.is_terminal
    assert tree.root.split is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
