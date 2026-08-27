"""Validation-by-construction for unseen_value_scaling: synthetic curves with KNOWN alpha /
ceiling, and a planted-rate de-bias recovery under a deliberately skewed sampler."""
import numpy as np
import pytest

from methods.metric_implementer.experiments import unseen_value_scaling as uvs


def test_power_law_recovers_known_alpha():
    m = np.arange(1, 200, dtype=float)
    y = 2.5 * m ** 0.42                                  # planted c=2.5, alpha=0.42
    fit = uvs.fit_power_law(m, y, n_boot=200, seed=1)
    assert fit["ok"]
    assert abs(fit["alpha"] - 0.42) < 0.01
    assert abs(fit["c"] - 2.5) < 0.05
    assert fit["r2_linear"] > 0.999
    lo, hi = fit["alpha_ci"]
    assert lo <= 0.42 <= hi


def test_power_law_recovers_alpha_with_noise():
    rng = np.random.default_rng(0)
    m = np.arange(1, 300, dtype=float)
    y = 4.0 * m ** 0.6 * (1 + 0.03 * rng.standard_normal(len(m)))
    fit = uvs.fit_power_law(m, y, n_boot=300, seed=2)
    assert fit["ok"]
    assert abs(fit["alpha"] - 0.6) < 0.03
    lo, hi = fit["alpha_ci"]
    assert lo <= 0.6 <= hi


def test_saturating_recovers_ceiling():
    m = np.arange(1, 400, dtype=float)
    y = 50.0 * (1 - np.exp(-m / 40.0))                   # planted y_inf=50, tau=40
    fit = uvs.fit_saturating(m, y, n_boot=100, seed=3)
    assert fit["ok"]
    assert abs(fit["y_inf"] - 50.0) < 1.0
    assert abs(fit["tau"] - 40.0) < 2.0
    assert fit["r2_linear"] > 0.999


def test_model_selection_picks_power_for_power_data():
    m = np.arange(1, 250, dtype=float)
    y = 3.0 * m ** 0.85                                  # alpha near 1 -> should NOT read as saturating
    cmp = uvs.compare_scaling_forms(m, y, n_boot=100, seed=4)
    assert cmp["aic_winner"] == "power"
    assert cmp["tail_test"]["winner"] == "power"
    assert cmp["verdict"] == "power"


def test_model_selection_picks_saturating_for_saturating_data():
    m = np.arange(1, 400, dtype=float)
    y = 80.0 * (1 - np.exp(-m / 30.0))
    cmp = uvs.compare_scaling_forms(m, y, n_boot=100, seed=5)
    assert cmp["aic_winner"] == "saturating"
    assert cmp["tail_test"]["winner"] == "saturating"
    assert cmp["verdict"] == "saturating"


def test_extrapolate_power_matches_closed_form():
    m = np.arange(1, 200, dtype=float)
    y = 2.0 * m ** 0.5
    fit = uvs.fit_power_law(m, y, n_boot=0)
    ex = uvs.extrapolate(fit, m_observed=199.0, horizons=[1.0, 100.0])
    # cumulative value at 199+100=299 should match 2*299^0.5
    assert abs(ex["cumulative_value"][1] - 2.0 * 299 ** 0.5) < 0.5
    # marginal per draw is positive and decreasing for alpha<1
    assert ex["marginal_per_draw_at_horizon"][0] > ex["marginal_per_draw_at_horizon"][1]


def test_effective_sample_size_bounds():
    assert abs(uvs.effective_sample_size(np.ones(100)) - 100.0) < 1e-6      # uniform -> full n
    # one dominant weight -> ESS near 1
    w = np.concatenate([[1000.0], np.ones(99)])
    assert uvs.effective_sample_size(w) < 5.0


def test_ip_novelty_rate_is_flagged_diagnostic_not_estimator():
    """HONEST test (the old one falsely claimed 'recovers base rate'). The IP novelty rate is a
    self-flagging diagnostic: it must (a) declare itself invalid for missing mass, (b) report
    ess_frac, and (c) NOT be trusted as a point estimate. Under a heavily skewed sampler its weights
    concentrate (low ess_frac), which is exactly the signal it exists to raise."""
    rng = np.random.default_rng(7)
    ids, prop = [], []
    for _ in range(500):
        ids.append(int(rng.integers(0, 50))); prop.append(0.9)
    for k in range(500):                                   # decade-spanning propensities -> UNEQUAL
        ids.append(1000 + k); prop.append(float(10 ** rng.uniform(-3, -1)))  # weights vary 10x-1000x
    res = uvs.inverse_propensity_novelty_rate(ids, prop)
    assert res["ok"]
    assert res["valid_missing_mass_estimator"] is False
    assert "warning" in res
    assert 0.0 <= res["ess_frac"] <= 1.0
    assert res["weights_stable"] is False                  # skewed sampler -> unstable, correctly flagged


def test_ip_novelty_rate_inert_when_all_singletons():
    """When every type is a singleton (M0->1 regime), the IP rate must equal the raw rate identically
    (both 1.0) — documenting that the branch does nothing in the frequent real-data regime."""
    ids = list(range(300))                                 # all distinct
    prop = [0.3] * 300
    res = uvs.inverse_propensity_novelty_rate(ids, prop)
    assert res["ip_novelty_rate"] == pytest.approx(res["raw_novelty_rate"]) == pytest.approx(1.0)


def test_heaps_unit_bootstrap_wider_than_fit_stability():
    """The real sampling CI (resampling underlying increments) must be WIDER than the fit-stability CI
    (resampling curve points) on a noisy cumulative discovery curve."""
    rng = np.random.default_rng(3)
    inc = (rng.random(120) < 0.45).astype(int)             # 0/1 discovery increments
    curve = np.cumsum(inc).astype(float)
    m = np.arange(1, len(curve) + 1, dtype=float)
    fit = uvs.fit_power_law(m, curve, n_boot=500)
    assert fit["ci_kind"] == "fit_stability"
    ub = uvs.heaps_unit_bootstrap_ci(inc, n_boot=500)
    assert ub["ok"] and ub["ci_kind"] == "sampling_moving_block"
    w_fit = fit["alpha_ci"][1] - fit["alpha_ci"][0]
    w_unit = ub["alpha_ci"][1] - ub["alpha_ci"][0]
    assert w_unit > w_fit


def test_audit_anchor_flags_adaptive_over_coverage():
    """Audit (i.i.d.) subset has many singletons; mined subset was steered to revisit types
    (few singletons). The anchor should report mined_minus_audit < 0 (over-claimed coverage)."""
    rng = np.random.default_rng(11)
    ids, is_audit = [], []
    for k in range(300):                                  # audit: all distinct -> high f1/N
        ids.append(("aud", k)); is_audit.append(True)
    mined_types = rng.integers(0, 20, 300)                # mined: only 20 types, heavily repeated
    for t in mined_types:
        ids.append(("mined", int(t))); is_audit.append(False)
    res = uvs.audit_anchored_missing_mass(ids, is_audit)
    assert res["ok"]
    assert res["audit"]["m0"] > 0.9
    assert res["mined"]["m0"] < 0.2
    assert res["mined_minus_audit"] < 0


def test_value_coverage_slope_is_value_per_discovery():
    """Construct S(m) and V(m) with a KNOWN per-discovery value: V = beta * S (each new type worth a
    constant beta). Then dV/dS must equal beta everywhere and the exponent must read rarity-neutral."""
    m = np.arange(1, 200, dtype=float)
    S = 3.0 * m ** 0.5
    beta = 0.4
    V = beta * S
    vc = uvs.value_coverage_curve(m, S, V, richness=S[-1])
    dvds = np.asarray(vc["value_per_new_discovery_at_S"])
    assert np.allclose(dvds, beta, atol=1e-6)
    assert vc["coverage"][-1] == pytest.approx(1.0)


def test_value_coverage_exponent_sign():
    # value grows slower than count -> diminishing (head-concentrated)
    d = uvs.value_coverage_exponent(alpha_count=0.6, alpha_value=0.4)
    assert d["value_coverage_exponent"] == pytest.approx(-0.2)
    assert "diminishing" in d["regime"]
    # value grows faster -> tail gems
    d2 = uvs.value_coverage_exponent(alpha_count=0.4, alpha_value=0.7)
    assert "tail-gems" in d2["regime"]


def test_debias_report_warns_without_signals():
    ids = [1, 2, 2, 3, 3, 3]
    rep = uvs.debias_report(ids)
    assert "warning" in rep
    assert rep["naive_good_turing_m0"] == pytest.approx(1 / 6)
