"""CPU tests for prompt_space_bracket — planted positives/negatives for each bracket arm.

Discipline mirrors test_value_certificate: every instrument must (i) fire on a planted case it is
designed to detect, (ii) stay quiet on a null, and (iii) never violate the DPI chain.
"""
import numpy as np
import pytest

from methods.metric_implementer import vinfo
from methods.metric_implementer.experiments import prompt_space_bracket as psb


def _flip(x, p, rng):
    f = rng.random(len(x)) < p
    return np.where(f, 1 - x, x).astype(float)


# ------------------------------------------------ (a) reliability ceiling


def test_reliability_soft_algebra_exact():
    # two-level exact q: T = H(0.5) - mean H(q_i) = 1 - H(0.9)
    q = np.array([0.9] * 200 + [0.1] * 200)
    out = psb.reliability_ceiling(q)
    expect = 1.0 - float(vinfo._h_bits(0.9))
    assert abs(out["T_soft"] - expect) < 1e-9
    assert out["T_hard_HM"] == pytest.approx(1.0, abs=1e-9)   # hard threshold degenerates to H(M)
    assert out["noise_charged_bits"] == pytest.approx(float(vinfo._h_bits(0.9)), abs=1e-9)
    assert out["certified"] is False


def test_reliability_pass_matrix_blocks_population_and_reports_self_half():
    rng = np.random.default_rng(0)
    q_true = np.array([0.9] * 150 + [0.1] * 150)
    passes = (rng.random((300, 8)) < q_true[:, None]).astype(float)
    out = psb.reliability_ceiling(passes, delta=0.05)
    assert out["population"] is None and out["population_error"]      # pass-estimated -> blocked
    assert out["R_self_split_half"] is not None
    # self-transmission is positive and cannot exceed either half's own cap (DPI, algebraic)
    assert 0.05 < out["R_self_split_half"] <= out["self_cert"]["shannon"]["dpi_upper"] + 1e-9


# ------------------------------------------------ (b) frozen head


def test_frozen_head_removes_pure_noise_optimism():
    rng = np.random.default_rng(1)
    q = (rng.random(240) < 0.5).astype(float)
    sigs = (rng.random((40, 240)) < 0.5).astype(float)      # 40 independent noise units
    out = psb.frozen_head(sigs, q, n_select=5)
    assert out["R_selection_mean"] > out["R_frozen_mean"]   # optimism exists on the selection half
    assert out["R_frozen_mean"] < 0.05                      # and freezing removes it
    assert out["certified"] is False


def test_frozen_head_keeps_planted_signal():
    rng = np.random.default_rng(2)
    q = (rng.random(300) < 0.5).astype(float)
    planted = _flip(q, 0.05, rng)
    noise = (rng.random((30, 300)) < 0.5).astype(float)
    sigs = np.vstack([planted[None, :], noise])
    out = psb.frozen_head(sigs, q, n_select=4)
    assert all(0 in r["selected"] for r in out["runs"])     # planted unit found in both directions
    truth = np.mean([psb._trans_bits(q[m], planted[m]) for m in psb._halves(300)])
    assert out["R_frozen_mean"] > 0.5 * truth               # frozen value retains real signal


# ------------------------------------------------ (c) EVT endpoint


def test_evt_bounded_sample_finds_finite_endpoint():
    rng = np.random.default_rng(3)
    v = 0.8 * rng.beta(2.0, 1.2, size=600)                  # right endpoint exactly 0.8
    out = psb.evt_endpoint(v, seed=0)
    assert out["valid"] and out["certified"] is False
    assert out["xi"] < 0 and out["endpoint"] is not None
    assert 0.7 < out["endpoint"] < 1.0
    assert out["endpoint"] >= out["max_observed"] - 1e-9 or out["endpoint"] > 0.75


def test_evt_heavy_tail_reports_no_finite_endpoint():
    rng = np.random.default_rng(4)
    v = rng.pareto(2.0, size=600)                           # GPD xi = 0.5 > 0
    out = psb.evt_endpoint(v, seed=0)
    assert out["valid"]
    assert out["endpoint"] is None or out["xi"] > -0.05
    if out["boot_frac_finite_endpoint"] is not None:
        assert out["boot_frac_finite_endpoint"] < 0.5


def test_evt_refuses_tiny_samples():
    assert psb.evt_endpoint([0.1, 0.2, 0.3])["valid"] is False


# ------------------------------------------------ (d) joint ladder + span residual


def test_joint_ladder_xor_interaction_detected_and_dpi_holds():
    rng = np.random.default_rng(5)
    u0 = (rng.random(400) < 0.5).astype(float)
    u1 = (rng.random(400) < 0.5).astype(float)
    q = _flip(np.logical_xor(u0 > 0.5, u1 > 0.5).astype(float), 0.03, rng)
    noise = (rng.random((4, 400)) < 0.5).astype(float)
    sigs = np.vstack([u0, u1, noise])
    out = psb.joint_combiner_ceiling(sigs, q)
    assert out["dpi_ok"]                                     # every rung inside the certified cap
    assert out["rungs_mean"]["linear"] < 0.15                # F1 is parity-blind
    assert out["rungs_mean"]["lookup"] > 0.4                 # free g finds the configuration
    assert out["interaction_bits"] > 0.3                     # the gestalt readout fires


def test_joint_ladder_linear_rule_shows_no_interaction():
    rng = np.random.default_rng(6)
    u0 = (rng.random(400) < 0.5).astype(float)
    q = _flip(u0, 0.05, rng)
    noise = (rng.random((5, 400)) < 0.5).astype(float)
    out = psb.joint_combiner_ceiling(np.vstack([u0, noise]), q)
    assert out["dpi_ok"]
    assert out["interaction_bits"] < 0.08                    # nothing configural to find


def test_span_residual_separates_spanned_from_channel():
    rng = np.random.default_rng(7)
    u0 = rng.random(400)
    u1 = rng.random(400)
    sigs = np.vstack([u0, u1])
    inside = np.clip(0.6 * u0 + 0.4 * u1 + rng.normal(0, 0.02, 400), 0, 1)
    z = (rng.random(400) < 0.5).astype(float)                # latent target axis, independent of units
    q = _flip(z, 0.05, rng)
    outside = np.clip(z + rng.normal(0, 0.05, 400), 0, 1)    # candidate tracking z, outside the span
    r_in = psb.span_residual(sigs, inside, target=q)
    r_out = psb.span_residual(sigs, outside, target=q)
    assert r_in["r2_heldout_mean"] > 0.9
    assert r_in["channel_gap_bits_mean"] < 0.05
    assert r_out["r2_heldout_mean"] < 0.2
    assert r_out["channel_gap_bits_mean"] > 0.3              # alignment the span cannot reproduce


# ------------------------------------------------ assembly


def test_bracket_report_orders_the_arms():
    rng = np.random.default_rng(8)
    q_true = np.where(rng.random(240) < 0.5, 0.85, 0.15)
    passes = (rng.random((240, 6)) < q_true[:, None]).astype(float)
    unit = _flip((q_true > 0.5).astype(float), 0.1, rng)
    sigs = np.vstack([unit, (rng.random((6, 240)) < 0.5).astype(float)])
    row = psb.bracket_report(sigs, passes, achieved_pool=0.5 * rng.beta(2, 2, 300),
                             candidates={"cand": unit})
    b = row["bracket_bits"]
    assert 0.0 <= b["achieved_lower"] <= b["dpi_upper"] + 1e-9   # the bracket is ordered
    assert row["certified"] is False and row["evt"]["certified"] is False
