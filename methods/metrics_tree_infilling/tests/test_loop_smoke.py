"""End-to-end smoke test of the infilling loop on a synthetic planted-feature corpus.

No live LLM: a fake judge scorer reads metric levels out of each item's text, and a fake
proposer returns the planted hidden feature. The corpus is built so that in a subpopulation
(m1 >= 0.5) the label is driven by a hidden attribute the known metrics cannot see -> the tree
should isolate that region as a gap node, the proposer's feature should close it, and the
guards should keep it.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.feature_gen import ProposedFeature
from methods.metrics_tree_infilling.io_metrics import (
    MetricSpec, ScoreMatrix, materialize,
)
from methods.metrics_tree_infilling.loop import run_infill


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _make_corpus(n=1400, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    m0 = rng.uniform(size=n)
    m1 = rng.uniform(size=n)
    m2 = rng.uniform(size=n)
    hidden = rng.integers(0, 2, size=n)            # the unarticulated feature
    region = m1 >= 0.5
    # outside region: label driven by m0 (a known metric); inside: driven by hidden only
    logit = np.where(region, 5.0 * (hidden - 0.5), 4.0 * (m0 - 0.5))
    y = (rng.uniform(size=n) < _sigmoid(logit)).astype(int)
    text = [f"m0={m0[i]:.4f} m1={m1[i]:.4f} m2={m2[i]:.4f} HIDDEN={hidden[i]} review body filler"
            for i in range(n)]
    return pd.DataFrame({"id": np.arange(n), "text": text, "judgement": y})


_BASE = [
    MetricSpec("m0", "m0", "metric m0", "judge", guidance="value of m0"),
    MetricSpec("m1", "m1", "metric m1", "judge", guidance="value of m1"),
    MetricSpec("m2", "m2", "metric m2", "judge", guidance="value of m2"),
]


def _fake_judge_scorer(metrics, texts):
    n, M = len(texts), len(metrics)
    levels = np.zeros((n, M))
    applicable = np.ones((n, M), dtype=bool)
    for j, m in enumerate(metrics):
        if m.name in ("m0", "m1", "m2"):
            pat = re.compile(rf"{m.name}=([0-9.]+)")
            levels[:, j] = [float(pat.search(t).group(1)) for t in texts]
        elif "widget" in m.name.lower() or "HIDDEN" in (m.guidance or ""):
            levels[:, j] = [1.0 if re.search(r"HIDDEN=1", t) else 0.0 for t in texts]
        else:
            applicable[:, j] = False
            levels[:, j] = np.nan
    return levels, applicable


def _fake_proposer(prompt):  # noqa: ARG001 — ignores prompt, returns the planted feature
    return ('{"name": "has_widget", "description": "presence of the hidden widget signal", '
            '"rubric": "Return 1 if HIDDEN=1 else 0"}')


def test_loop_discovers_and_keeps_hidden_feature():
    df = _make_corpus()
    cfg = InfillConfig(
        n_permutations=199, min_node_size=50, max_depth=3, random_seed=0,
        max_outer_rounds=3, reliability_sample_size=60, gap_deviance_per_item=1.25,
    )
    # honest split
    rng = np.random.default_rng(cfg.random_seed)
    perm = rng.permutation(len(df))
    cut = int(0.7 * len(df))
    df_d = df.iloc[perm[:cut]].reset_index(drop=True)
    df_t = df.iloc[perm[cut:]].reset_index(drop=True)

    metrics = list(_BASE)
    sm_d = materialize(metrics, df_d, cfg, _fake_judge_scorer)
    sm_t = materialize(metrics, df_t, cfg, _fake_judge_scorer)

    # there should be a gap before infilling
    from methods.metrics_tree_infilling.gaps import flag_gap_nodes
    from methods.metrics_tree_infilling.io_metrics import make_design
    from methods.metrics_tree_infilling.mob.glmtree import GapTree
    Xd, fnd, Zd, spec = make_design(sm_d, df_d, cfg)
    Xt, _, Zt, _ = make_design(sm_t, df_t, cfg, spec=spec)
    y_t = df_t["judgement"].to_numpy(float)
    tree0 = GapTree(cfg).fit(Xd, df_d["judgement"].to_numpy(float), Zd, fnd)
    assert len(flag_gap_nodes(tree0, Xt, y_t, Zt, cfg)) >= 1, "expected a planted gap node"

    result = run_infill(df_d, df_t, metrics, sm_d, sm_t, cfg, _fake_proposer, _fake_judge_scorer,
                        log=lambda *a, **k: None)

    kept = [r for r in result.records if r.status == "kept"]
    assert any(r.name == "has_widget" for r in kept), \
        f"hidden feature not kept; records={[(r.name, r.status) for r in result.records]}"
    rec = next(r for r in kept if r.name == "has_widget")
    assert rec.gap_drop_fraction > 0.1
    assert result.final_gap_count < len(flag_gap_nodes(tree0, Xt, y_t, Zt, cfg)) + 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
