"""E0 calibration tests for the V-information estimator (vinfo.py).

The kill-switch: on synthetic cells with a KNOWN analytic I_V the estimator must recover it
(within CI) before any real number is trusted. Also documents the residual K=5 upward bias
(why small I_V is unresolvable at 5 passes) and checks the qualitative guarantees: noise->0,
deterministic->near-max, monotone in #recovered criteria, range-collapse reads low.
"""
from __future__ import annotations

import numpy as np
import pytest

from methods.metric_implementer.vinfo import (
    _h_bits, analytic_iv, binary_soft_channel_mi, bootstrap_iv, cell_report, channel_synthetic,
    degenerate_flags, fixed_target_channel_certificate, iv_from_passes, iv_from_probs,
    iv_transmission, planted_probs, simulate_cell, target_channel_ceiling,
)

CLEAN = dict(n_items=400, n_passes=30)   # enough to make MM bias small + CI meaningful


def test_soft_readout_is_exact_analytic():
    # iv_from_probs on the true per-item probabilities equals analytic_iv by construction.
    rng = np.random.default_rng(0)
    q = rng.random(200)
    assert abs(iv_from_probs(q)["iv_mm"] - analytic_iv(q)) < 1e-9


@pytest.mark.parametrize("kind,kw", [
    ("deterministic", {}),
    ("k_of_K", dict(k=1, K=4)),
    ("k_of_K", dict(k=2, K=4)),
])
def test_recovers_analytic(kind, kw):
    # Faithfulness: point estimate within tolerance of the known analytic I_V. A small residual
    # UPWARD bias survives at finite passes (MM cannot correct items where only one outcome was
    # observed), so we do NOT require the sampling CI to bracket truth -- we bound |bias| and
    # check the bias has the expected sign (>= -tol).
    rep = channel_synthetic(kind, seed=1, **CLEAN, **kw)
    true = rep["analytic_iv"]
    assert abs(rep["iv_mm"] - true) < 0.04, (kind, kw, rep["iv_mm"], true)
    assert rep["iv_mm"] >= true - 0.01, (kind, kw, rep["iv_mm"], true)   # bias is upward
    assert rep["ci_lo"] <= rep["iv_mm"] <= rep["ci_hi"], rep             # CI brackets the estimate


def test_noise_is_zero_with_enough_passes():
    rep = channel_synthetic("noise", seed=2, **CLEAN)
    assert rep["analytic_iv"] == pytest.approx(0.0, abs=1e-9)
    assert rep["iv_mm"] < 0.03, rep


def test_deterministic_is_near_max():
    # balanced near-deterministic metric -> recovered info close to H(0.5)=1 bit.
    rep = channel_synthetic("deterministic", seed=3, eps=0.02, **CLEAN)
    assert rep["iv_mm"] > 0.7, rep
    assert rep["h_marg"] > 0.95, rep        # uses full range


def test_tracks_analytic_across_k():
    # The estimator must FOLLOW the ground truth across a sweep (here I_V actually DECREASES with k:
    # under the OR construction more criteria skew the base rate, lowering H(p_bar)). We assert the
    # estimate matches analytic per-k and reproduces its ordering -- not a hard-coded direction.
    reps = [channel_synthetic("k_of_K", seed=4, k=k, K=4, **CLEAN) for k in (1, 2, 3)]
    est = [r["iv_mm"] for r in reps]
    true = [r["analytic_iv"] for r in reps]
    for e, t in zip(est, true):
        assert abs(e - t) < 0.04, (e, t)
    assert np.argsort(est).tolist() == np.argsort(true).tolist(), (est, true)


def test_range_collapse_reads_low():
    # recovery collapses toward base even if a latent metric could discriminate -> low I_V.
    rep = channel_synthetic("compressed", seed=5, eps=0.03, **CLEAN)
    assert rep["iv_mm"] < 0.05, rep


def test_miller_madow_reduces_upward_bias_at_k5():
    # At 5 passes the plug-in MI is upward-biased on a high-entropy (noise) cell; MM corrects
    # it DOWN toward the true 0. Residual bias remains -> small I_V is unresolvable at K=5.
    rng = np.random.default_rng(6)
    q = planted_probs("noise", 400, rng, base=0.5)
    V = simulate_cell(q, 5, rng)
    rep = iv_from_passes(V)
    assert rep["iv_plugin"] > rep["iv_mm"] > 0.0          # MM pulls down, doesn't overshoot
    assert rep["iv_mm"] < 0.12                            # bounded residual floor at K=5
    # at 30 passes the same cell collapses to ~0
    V30 = simulate_cell(q, 30, np.random.default_rng(6))
    assert iv_from_passes(V30)["iv_mm"] < rep["iv_mm"]


def test_inapplicable_passes_dropped():
    # NaN entries (inapplicable) are ignored per item; an item with all-NaN drops out.
    rng = np.random.default_rng(7)
    q = planted_probs("k_of_K", 100, rng, k=1, K=4)
    V = simulate_cell(q, 8, rng)
    V[:10, :] = np.nan                                   # 10 fully-inapplicable items
    V[10:20, 4:] = np.nan                                # partial applicability
    rep = iv_from_passes(V)
    assert rep["n_items"] == 90
    assert np.isfinite(rep["iv_mm"])


def test_degenerate_flags():
    # a collapsed (constant-verdict) cell -> no range -> flagged, I_V ~ 0.
    V = np.ones((60, 5))
    rep = cell_report(V, n_boot=200, min_items=20)
    assert "collapsed_no_range" in rep["flags"] or "near_constant_verdict" in rep["flags"]
    assert (not np.isfinite(rep["iv_mm"])) or rep["iv_mm"] < 0.02

    short = cell_report(np.array([[1, 0], [0, 1]], dtype=float), n_boot=50, min_items=20)
    assert "few_items" in short["flags"]


def test_bootstrap_ci_brackets_point_estimate():
    rep = channel_synthetic("k_of_K", seed=8, k=2, K=4, **CLEAN)
    assert rep["ci_lo"] <= rep["iv_mm"] <= rep["ci_hi"]
    assert rep["ci_hi"] - rep["ci_lo"] < 0.2             # not absurdly wide at n=400


@pytest.mark.parametrize("e", [0.0, 0.1, 0.3])
def test_transmission_recovers_bsc(e):
    # recovered = m's verdict through a binary symmetric channel (flip prob e); balanced labels ->
    # I(m; recovered) = 1 - H_b(e). The transmission estimator must recover it.
    rng = np.random.default_rng(0)
    N, R = 400, 6
    y = (rng.random(N) < 0.5).astype(float)
    flips = rng.random((N, R)) < e
    recovered = np.where(flips, 1 - y[:, None], y[:, None]).astype(float)
    rep = iv_transmission(recovered, y, seed=1)
    analytic = 1.0 - _h_bits(e)
    assert abs(rep["iv_mm"] - analytic) < 0.06, (e, rep["iv_mm"], analytic)


def test_transmission_independent_is_zero():
    # recovered independent of m -> I(m; recovered) ~ 0 (this is the peer-review signature).
    rng = np.random.default_rng(3)
    N, R = 400, 6
    y = (rng.random(N) < 0.5).astype(float)
    recovered = (rng.random((N, R)) < 0.5).astype(float)
    rep = iv_transmission(recovered, y, seed=2)
    assert rep["iv_mm"] < 0.05, rep


def test_transmission_degenerate_labels():
    # constant m -> no label variance -> transmission undefined/flagged.
    rep = iv_transmission(np.zeros((40, 5)), np.ones(40), seed=0)
    assert rep.get("error") == "degenerate_labels"


def test_fixed_target_dpi_same_f_and_same_distribution():
    rng = np.random.default_rng(21)
    q = rng.uniform(0.02, 0.98, 500)
    p = rng.uniform(0.02, 0.98, 500)
    cert = fixed_target_channel_certificate(q, p)
    assert cert["valid"]
    assert cert["shannon"]["dpi_ok"] and cert["tvd"]["dpi_ok"]
    assert cert["shannon"]["R"] <= min(cert["shannon"]["T_target"],
                                          cert["shannon"]["T_candidate"]) + 1e-10
    assert cert["tvd"]["R"] <= min(cert["tvd"]["T_target"],
                                      cert["tvd"]["T_candidate"]) + 1e-10
    assert cert["tightness_established"] is False


def test_fixed_target_deterministic_perfect_readout_attains_ceiling():
    q = np.r_[np.zeros(100), np.ones(100)]
    cert = fixed_target_channel_certificate(q, q)
    assert cert["shannon"]["R"] == pytest.approx(1.0)
    assert cert["shannon"]["T_target"] == pytest.approx(1.0)
    assert cert["tvd"]["R"] == pytest.approx(0.5)
    assert cert["tvd"]["T_target"] == pytest.approx(0.5)
    mi = binary_soft_channel_mi(q, q)
    assert mi["tvd"] == pytest.approx(mi["tvd_cov"])


def test_target_population_ceiling_is_one_sided_and_explicitly_scoped():
    q = np.tile([0.1, 0.9], 150)
    cap = target_channel_ceiling(q, delta=0.05)
    truth_tvd = 0.4
    truth_shannon = 1.0 - _h_bits(0.1)
    assert cap["population"]["tvd_upper"] >= truth_tvd
    assert cap["population"]["shannon_upper"] >= truth_shannon
    # A pass matrix has extra probability-estimation uncertainty and must not receive this bound.
    sampled = (np.random.default_rng(2).random((300, 4)) < q[:, None]).astype(float)
    no_bound = target_channel_ceiling(sampled, delta=0.05)
    assert no_bound["population"] is None and "population_error" in no_bound


def test_population_gap_requires_frozen_candidate_and_exact_probabilities():
    q = np.tile([0.05, 0.95], 300)
    p = q.copy()
    blocked = fixed_target_channel_certificate(q, p, population_delta=0.05)
    assert blocked["population"] is None and "population_error" in blocked
    issued = fixed_target_channel_certificate(q, p, population_delta=0.05,
                                              candidate_frozen=True)
    assert issued["population"]["confidence"] == pytest.approx(0.95)
    assert issued["population"]["tvd_gap_upper"] >= 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
