"""End-to-end discovery test on the creature-dossier scenario (offline, deterministic).

The published Companion Code explains grove creatures but is silent in the marsh and cavern,
where two tacit aesthetic norms rule: sweet SONG (the larger marsh) and bioluminescent GLOW
(the smaller cavern). Two further attributes (color, limbs) are described in every dossier but
never affect the verdict — decoys.

This asserts the loop:
  1. recovers BOTH tacit norms (song and glow),
  2. is NOT fooled into "discovering" a decoy,
  3. reads off measured generality as coverage that matches the true region sizes and ranks
     song (broad) above glow (narrow).

Run live (real LLM proposer + judge) instead of the oracle:
    PYTHONPATH=methods python -m metrics_tree_infilling.tests.test_scenario.test_discovery --live
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.gaps import flag_gap_nodes
from methods.metrics_tree_infilling.io_metrics import make_design, materialize, discover_test_split
from methods.metrics_tree_infilling.mob.glmtree import GapTree
from methods.metrics_tree_infilling.loop import run_infill
from methods.metrics_tree_infilling.contrast import build_contrast
from . import world, oracle
from .generate import build_corpus
from .metrics import companion_code
from .oracle import oracle_judge_scorer, oracle_proposer

HERE = Path(__file__).resolve().parent


def _load_corpus() -> pd.DataFrame:
    fp = HERE / "corpus.csv"
    if fp.exists():
        return pd.read_csv(fp)
    corpus, _ = build_corpus(n=2400, seed=7)
    return corpus


def _config(**kw) -> InfillConfig:
    base = dict(
        n_permutations=199, min_node_size=40, max_depth=5, random_seed=0,
        max_outer_rounds=5, reliability_sample_size=60,
        gap_deviance_per_item=1.20, gap_auc_threshold=0.55,
        contrastive_pairs_k=6, include_text_length_in_z=False,
    )
    base.update(kw)
    return InfillConfig(**base)


def _run(cfg):
    df = _load_corpus()
    df_d, df_t = discover_test_split(df, cfg)
    metrics = companion_code()
    sm_d = materialize(metrics, df_d, cfg, oracle_judge_scorer)
    sm_t = materialize(metrics, df_t, cfg, oracle_judge_scorer)
    return run_infill(df_d, df_t, metrics, sm_d, sm_t, cfg,
                      oracle_proposer, oracle_judge_scorer, log=lambda *a, **k: None)


def test_both_tacit_norms_discovered_and_decoys_rejected():
    result = _run(_config())
    kept = {r.name.lower() for r in result.records if r.status == "kept"}
    assert any("glow" in n or "luminous" in n for n in kept), kept
    assert any("song" in n or "melodious" in n for n in kept), kept
    # the decoys must never be "discovered"
    assert not any(("color" in n or "azure" in n or "limb" in n) for n in kept), kept
    assert result.final_gap_count == 0


def test_coverage_ranks_song_above_glow_and_matches_regions():
    result = _run(_config())
    kept = {r.name: r for r in result.records if r.status == "kept"}
    glow = next(r for n, r in kept.items() if "glow" in n.lower())
    song = next(r for n, r in kept.items() if "song" in n.lower())
    # measured generality: the broad marsh norm covers more than the narrow cavern norm
    assert song.coverage > glow.coverage, (song.coverage, glow.coverage)
    # and each coverage is in the neighborhood of its true region size (cavern .20, marsh .34)
    assert 0.10 <= glow.coverage <= 0.32, glow.coverage
    assert 0.22 <= song.coverage <= 0.46, song.coverage


def test_oracle_has_a_real_choice_among_distractors():
    """In the cavern contrast the oracle considers glow + song + the two decoys, and must
    pick glow by genuine label-separation (not by having only one option)."""
    cfg = _config()
    df = _load_corpus()
    df_d, df_t = discover_test_split(df, cfg)
    metrics = companion_code()
    sm_d = materialize(metrics, df_d, cfg, oracle_judge_scorer)
    sm_t = materialize(metrics, df_t, cfg, oracle_judge_scorer)
    Xd, fnd, Zd, spec = make_design(sm_d, df_d, cfg)
    Xt, _, Zt, _ = make_design(sm_t, df_t, cfg, spec=spec)
    y_t = df_t["judgement"].to_numpy(float)
    tree = GapTree(cfg).fit(Xd, df_d["judgement"].to_numpy(float), Zd, fnd)
    gap = flag_gap_nodes(tree, Xt, y_t, Zt, cfg)[0]
    c = build_contrast(tree, gap, df_d, Xd, df_d["judgement"].to_numpy(float), cfg,
                       np.random.default_rng(0))
    pool = [a for a in world.ATTRIBUTES if a not in world.KNOWN_ATTRS]
    assert set(pool) == {"glow", "song", "color", "limbs"}, pool   # 4 real candidates
    best = oracle._best_separating_attr(c.wrong_pos[:6], c.wrong_neg[:6])
    assert best is not None and best[0] in ("glow", "song"), best   # picks a real norm, not a decoy


def _run_live():
    from methods.metrics_tree_infilling.feature_gen import make_proposer
    from methods.metrics_tree_infilling.io_metrics import make_vllm_judge_scorer
    cfg = _config(proposer_backend="anthropic", materialize_backend="anthropic",
                  materialize_model="claude-sonnet-4-20250514")
    df = _load_corpus()
    df_d, df_t = discover_test_split(df, cfg)
    metrics = companion_code()
    judge = make_vllm_judge_scorer(cfg)
    sm_d = materialize(metrics, df_d, cfg, judge)
    sm_t = materialize(metrics, df_t, cfg, judge)
    result = run_infill(df_d, df_t, metrics, sm_d, sm_t, cfg, make_proposer(cfg), judge)
    print("LIVE kept:", [(r.name, round(r.coverage, 2)) for r in result.records if r.status == "kept"])


if __name__ == "__main__":
    import sys
    if "--live" in sys.argv:
        _run_live()
    else:
        raise SystemExit(pytest.main([__file__, "-v", "-s"]))
