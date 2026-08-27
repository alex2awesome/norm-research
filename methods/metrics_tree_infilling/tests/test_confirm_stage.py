"""Confirm stage (winner's-curse control, MCC §4.4): fresh-seed repeated-CV replication +
Nadeau-Bengio-corrected Bonferroni significance. Two properties matter:
  (a) a genuinely informative metric passes the confirm stage (power at planted-signal size);
  (b) a noise metric that sneaks past a permissive primary gate is KILLED by confirm —
      exactly the failure mode of the 2026-07-02 runs (CV survivor died on test)."""

import json

import numpy as np
import pandas as pd

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.global_infill import (
    _confirm_stage, _nb_corrected_p, run_global_infill)
from metrics_tree_infilling.io_metrics import MetricSpec

from .test_global_infill import _mk, oracle_judge, oracle_proposer


# ---- unit: the corrected test itself ---------------------------------------------------

def test_nb_corrected_p_signal_vs_noise():
    rng = np.random.default_rng(0)
    strong = rng.normal(0.05, 0.01, size=25)      # consistent positive fold gains
    noise = rng.normal(0.0, 0.03, size=25)        # zero-centered fold gains
    assert _nb_corrected_p(strong, n_folds=5) < 1e-3
    assert _nb_corrected_p(noise, n_folds=5) > 0.05


def test_nb_corrected_p_edge_cases():
    assert np.isnan(_nb_corrected_p(np.array([0.1, 0.2]), n_folds=5))   # too few diffs
    assert _nb_corrected_p(np.array([0.02] * 10), n_folds=5) < 1e-10    # ~zero variance, +mean
    assert _nb_corrected_p(np.array([-0.02] * 10), n_folds=5) > 1 - 1e-10  # ~zero var, -mean


def test_nb_correction_is_more_conservative_than_naive():
    """The NB variance inflation must make p LARGER than the naive paired t-test p."""
    from scipy import stats
    rng = np.random.default_rng(1)
    d = rng.normal(0.01, 0.02, size=25)
    p_nb = _nb_corrected_p(d, n_folds=5)
    p_naive = float(stats.ttest_1samp(d, 0.0, alternative="greater").pvalue)
    assert p_nb > p_naive


def test_confirm_stage_separates_planted_from_noise_column():
    rng = np.random.default_rng(2)
    n = 400
    signal = rng.uniform(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-3.0 * (signal - 0.5)))).astype(int)
    X_base = rng.uniform(size=(n, 1))                      # uninformative bank
    X_sig = np.column_stack([X_base[:, 0], signal])        # bank + real signal
    X_noise = np.column_stack([X_base[:, 0], rng.uniform(size=n)])

    conf_sig = _confirm_stage(X_base, X_sig, y, n_repeats=3, base_seed=0)
    conf_noise = _confirm_stage(X_base, X_noise, y, n_repeats=3, base_seed=0)
    assert conf_sig["auc_gain"] > 0.05
    assert conf_sig["p_auc"] < 0.05 / 20                   # clears Bonferroni at m=20
    assert not (np.isfinite(conf_noise["p_auc"]) and conf_noise["p_auc"] < 0.05 / 20)
    assert conf_sig["n_diffs"] == 15                       # 3 repeats x 5 folds


# ---- integration: the gate inside run_global_infill ------------------------------------

def _cfg(**kw):
    base = dict(random_seed=0, min_auc_gain=0.01, tau_redundant=0.9,
                viability_min_applicability=0.1, viability_min_std=0.05,
                acceptance_eval="cv", confirm_n_repeats=3, gate_alpha=0.05,
                gate_bonferroni_m=20,
                text_column="text", label_column="judgement", id_column="id")
    base.update(kw)
    return InfillConfig(**base)


def test_planted_metric_survives_confirm_stage():
    df_d, sm_d, y_d = _mk(400, seed=1)
    df_g, sm_g, y_g = _mk(200, seed=2)
    base = [MetricSpec(metric_id="m0", name="known_quality", description="known", kind="judge")]
    res = run_global_infill(sm_d, df_d, y_d, sm_g, df_g, y_g, base, _cfg(),
                            judge_scorer=oracle_judge, proposer=oracle_proposer,
                            max_rounds=3, patience=2)
    kept = [l for l in res.ledgers if l.status == "kept"]
    assert kept, f"planted metric killed by confirm: {[l.status for l in res.ledgers]}"
    led = kept[0]
    assert led.confirm_m == 20
    assert np.isfinite(led.confirm_p_auc) and led.confirm_p_auc <= 0.05 / 20
    assert led.confirm_auc_gain >= 0.01


def test_noise_past_permissive_primary_gate_dies_in_confirm():
    """min_auc_gain=-1 lets ANY finite gain through the primary gate — the confirm stage's
    significance leg must be the one that kills the noise proposal."""
    df_d, sm_d, y_d = _mk(400, seed=5)
    df_g, sm_g, y_g = _mk(200, seed=6)
    base = [MetricSpec(metric_id="m0", name="known_quality", description="known", kind="judge")]

    def noise_proposer(prompt):
        if "reverse-engineering" in prompt:
            return None
        return json.dumps({"candidates": [{"name": "coin_flip", "description": "irrelevant",
                                           "rubric": "YES if the text exists."}]})

    res = run_global_infill(sm_d, df_d, y_d, sm_g, df_g, y_g, base,
                            _cfg(min_auc_gain=-1.0),
                            judge_scorer=oracle_judge, proposer=noise_proposer,
                            max_rounds=2, patience=2)
    assert not [l for l in res.ledgers if l.status == "kept"]
    confirm_drops = [l for l in res.ledgers if l.status.startswith("dropped:confirm")]
    assert confirm_drops, f"expected a confirm-stage drop, got: {[l.status for l in res.ledgers]}"
    assert confirm_drops[0].confirm_m == 20


def test_confirm_stage_off_preserves_legacy_behavior():
    df_d, sm_d, y_d = _mk(400, seed=1)
    df_g, sm_g, y_g = _mk(200, seed=2)
    base = [MetricSpec(metric_id="m0", name="known_quality", description="known", kind="judge")]
    res = run_global_infill(sm_d, df_d, y_d, sm_g, df_g, y_g, base,
                            _cfg(confirm_n_repeats=0),
                            judge_scorer=oracle_judge, proposer=oracle_proposer,
                            max_rounds=3, patience=2)
    kept = [l for l in res.ledgers if l.status == "kept"]
    assert kept
    assert kept[0].confirm_m == 0 and np.isnan(kept[0].confirm_p_auc)
