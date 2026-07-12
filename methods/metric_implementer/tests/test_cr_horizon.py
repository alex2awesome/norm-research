"""CPU tests for cr_horizon (CR-2). Slow-ish (~2 min): each test scores a full planted world.

The load-bearing test is the PURE-NOISE null: v1 of the estimator manufactured ~0.6 bits of
pair-synergy out of positively-clipped CMI noise on a LINEAR world; the significance gates must
keep the null raw_gap near zero, or the instrument is vacuous (estimate always at the H cap).
"""
import numpy as np
import pytest

from methods.metric_implementer.experiments import cr_horizon as crh

FAST = dict(top_k=6, n_rand_pairs=60, n_null=20, suspect_cap=30, topk_pairs=8)


def test_planted_world_truth_analytics():
    rng = np.random.default_rng(0)
    w = crh.planted_world(rng, rule="linear3", flip=0.0)
    # POOL truth (noisy units) < concept truth; at zero label noise concept truth = H(M)
    assert w["truth"] < w["truth_concept"] + 1e-9
    assert w["truth_concept"] == pytest.approx(w["H_M"], abs=2e-2)   # empirical base-rate wobble
    w2 = crh.planted_world(rng, rule="xor2", flip=0.05, guarantee_pair=True)
    assert 0.0 < w2["truth"] < w2["truth_concept"] < w2["H_M"]
    # enumeration sanity: pool truth at zero unit noise equals concept truth
    assert crh.pool_truth_bits("linear3", 0.05, 0.0) == pytest.approx(
        crh.planted_world(np.random.default_rng(1), rule="linear3")["truth_concept"], abs=2e-2)


def test_pure_noise_pool_stays_quiet():
    # no true units at all: every unit independent of M -> gates must hold raw_gap near zero and
    # the estimate must NOT ride to the H cap (the v1 phantom-bits failure mode)
    rng = np.random.default_rng(1)
    n = 300
    M = (rng.random(n) < 0.5).astype(float)
    sigs = (rng.random((60, n)) < 0.5).astype(float)
    cert = crh.cr2_certificate(sigs, M, ["iid"] * 60, **FAST)
    d = max(cert["directions"], key=lambda x: x["raw_gap"])
    assert d["raw_gap"] < 0.2
    assert not cert["at_H_cap"]
    assert cert["horizon_estimate"] < 0.7 * d["H_M"]


def test_xor_pair_priced_by_chain():
    rng = np.random.default_rng(11)
    w = crh.planted_world(rng, rule="xor2", guarantee_pair=True)
    cert = crh.cr2_certificate(w["sigs"], w["M"], w["tags"], **FAST)
    d = max(cert["directions"], key=lambda x: x["chain_total"])
    assert w["hidden_true"] == 0
    assert d["chain_total"] > 0.05                        # the pair machinery fires
    est2 = min(d["value_frozen"] + 2.0 * d["raw_gap"] + d["slack"], d["H_M"])
    assert est2 >= 0.8 * w["truth"]                       # coverage reachable at modest lambda


def test_linear_world_estimate_ordered_and_capped():
    rng = np.random.default_rng(3)
    w = crh.planted_world(rng, rule="linear3")
    cert = crh.cr2_certificate(w["sigs"], w["M"], w["tags"], **FAST)
    for d in cert["directions"]:
        assert 0.0 <= d["value_frozen"] <= d["estimate"] + 1e-9
        assert d["estimate"] <= d["H_M"] + 1e-9
        assert d["G1"] >= 0 and d["pair_seen"] >= 0 and d["pair_unseen"] >= 0
    assert cert["certified"] is False


def test_audit_flags_violation():
    cert = {"horizon_estimate": 0.5}
    audit = crh.audit_no_violation(cert, {"ok_obs": 0.4, "bad_obs": 0.62})
    assert not audit["ok"] and "bad_obs" in audit["violations"]
    audit2 = crh.audit_no_violation(cert, {"a": 0.1, "b": 0.5})
    assert audit2["ok"]


def test_concept_horizon_paths():
    rng = np.random.default_rng(4)
    out = crh.concept_horizon(0.6 * rng.beta(2.0, 1.2, 300))
    assert out["n_instantiations"] == 300 and out["certified"] is False
    assert out["evt"]["valid"]
    small = crh.concept_horizon([0.1, 0.2])
    assert small["error"] == "too_few_instantiations"


def test_lambda_fit_reports_tightness_and_vacuity():
    # lam is chosen on train coverage; the credential is TEST coverage (out-of-sample).
    rows = []
    for split in ("train", "test"):
        rows += [
            dict(split=split, in_scope=True, truth=0.5, H_M=1.0, value_frozen=0.4, raw_gap=0.2, slack=0.05),
            dict(split=split, in_scope=True, truth=0.8, H_M=1.0, value_frozen=0.3, raw_gap=0.25, slack=0.05),
            dict(split=split, in_scope=False, blind_spot="parity3:truncation", truth=0.9, H_M=1.0,
                 value_frozen=0.1, raw_gap=0.05, slack=0.02),
        ]
    fit = crh.fit_lambda(rows, target=0.95, lam_grid=(1.0, 2.0, 4.0))
    assert fit["lam"] == 2.0                       # lam=1 covers 1/2 in-scope, lam=2 covers 2/2
    blk = fit["test"]["in_scope"]                  # credential reported on TEST
    assert blk["coverage"] == 1.0 and "tightness_mean" in blk and "vacuity_rate" in blk
    assert "blind:parity3:truncation" in fit["test"]
