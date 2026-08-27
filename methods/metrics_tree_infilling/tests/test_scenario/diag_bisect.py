"""Diagnostic bisect: why did the LIVE smoke (n=500) miss song/glow?

The offline oracle test PASSES at n=2400 with ``test_discovery._config``. The LIVE smoke
(``smoke_infill_live.py``) FAILS at n=500 with a different cfg AND the glm-5.2 proposer+judge.
Three confounds are bundled in that failure: corpus size, cfg tightness, and the LLM seam.

This script removes the LLM and runs the ORACLE across a 2x2 to attribute the failure:

  H_LLM  : the glm-5.2 proposer/judge is the culprit
           -> oracle PASSES at A (n=500, smoke_cfg)
  H_SIZE : n=500 is too small for stable splits / enough wrong-contrast items
           -> oracle FAILS at A but PASSES at C (n=2400, smoke_cfg)
  H_CFG  : the smoke cfg (max_outer_rounds=2, max_depth=4, min_node_size=30) is too tight
           -> oracle FAILS at A but PASSES at B (n=500, test_cfg)

All four conditions are free (oracle, no model). Run from the repo root:

    PYTHONPATH=methods python -m metrics_tree_infilling.tests.test_scenario.diag_bisect
"""

from __future__ import annotations

import time

import numpy as np

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.io_metrics import discover_test_split, materialize
from methods.metrics_tree_infilling.loop import run_infill

from .generate import build_corpus
from .metrics import companion_code
from .oracle import oracle_judge_scorer, oracle_proposer


def smoke_cfg() -> InfillConfig:
    """Mirrors smoke_infill_live.py exactly (the failing condition)."""
    return InfillConfig(
        n_permutations=199, min_node_size=30, max_depth=4, random_seed=0,
        max_outer_rounds=2, max_features_per_round=4, reliability_sample_size=40,
        gap_deviance_per_item=1.20, gap_auc_threshold=0.55, contrastive_pairs_k=6,
        include_text_length_in_z=False,
    )


def test_cfg() -> InfillConfig:
    """Mirrors test_discovery._config (the passing condition)."""
    return InfillConfig(
        n_permutations=199, min_node_size=40, max_depth=5, random_seed=0,
        max_outer_rounds=5, reliability_sample_size=60,
        gap_deviance_per_item=1.20, gap_auc_threshold=0.55, contrastive_pairs_k=6,
        include_text_length_in_z=False,
    )


def run(n: int, make_cfg) -> dict:
    cfg = make_cfg()
    df, _ = build_corpus(n=n, seed=7)
    df_d, df_t = discover_test_split(df, cfg)
    metrics = companion_code()
    sm_d = materialize(metrics, df_d, cfg, oracle_judge_scorer)
    sm_t = materialize(metrics, df_t, cfg, oracle_judge_scorer)
    res = run_infill(df_d, df_t, metrics, sm_d, sm_t, cfg,
                     oracle_proposer, oracle_judge_scorer, log=lambda *a, **k: None)
    kept = [(r.name, round(float(r.coverage), 3) if np.isfinite(r.coverage) else None)
            for r in res.records if r.status == "kept"]
    dropped = [(r.name, r.status) for r in res.records if r.status != "kept"]
    names = " ".join(n.lower() for n, _ in kept)
    return dict(
        n=n, rounds=res.rounds, final_gaps=res.final_gap_count,
        kept=kept, dropped=dropped,
        song=("song" in names or "melod" in names),
        glow=("glow" in names or "lumin" in names),
    )


CONDITIONS = [
    ("A  n=500  smoke_cfg", 500, smoke_cfg),
    ("B  n=500  test_cfg ", 500, test_cfg),
    ("C  n=2400 smoke_cfg", 2400, smoke_cfg),
    ("D  n=2400 test_cfg ", 2400, test_cfg),
]


if __name__ == "__main__":
    for label, n, mk in CONDITIONS:
        t0 = time.time()
        r = run(n, mk)
        ok = "OK" if (r["song"] and r["glow"]) else "MISS"
        print(f"\n=== {label}   [{ok}]   ({time.time() - t0:.0f}s) ===")
        print(f"  rounds={r['rounds']}  final_gaps={r['final_gaps']}  "
              f"song={r['song']}  glow={r['glow']}")
        print(f"  kept:    {r['kept']}")
        if r["dropped"]:
            print(f"  dropped: {r['dropped']}")
