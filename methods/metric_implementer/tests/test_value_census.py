"""Planted-ground-truth tests for the §12.3 VALUE-CENSUS (experiments/value_census.py), per metric M_i.
All CPU, anchor-free (M_i is the planted metric verdict — never an aggregate Y).

Gates the value-census construction the way test_alpha_probe gates §12.1a: on planted data with a KNOWN
answer, the estimators must satisfy their invariants (value rarefaction monotone & ends at Σv_s; MV_0 =
Σ_singleton v_s/N; i_binary exact for independent/perfectly-coupled) and the headline behavior must hold
— a high-recovery CORE captured many times + a zero-recovery singletons TAIL ⇒ α_V ≪ α (the breadth
gap): many expressible criteria, few recover M_i."""
from __future__ import annotations

import numpy as np

from methods.metric_implementer.experiments.value_census import (
    additive_value, hill_tail_index, i_binary, submodular_value, value_census,
    value_missing_mass, value_rarefaction,
)
from methods.metric_implementer.experiments.alpha_probe import alpha_probe


def test_i_binary_independent_and_coupled():
    rng = np.random.default_rng(0)
    M_i = (rng.random(400) > 0.5).astype(int)                  # the metric's verdict (planted)
    X_indep = (rng.random(400) > 0.5).astype(int)
    assert i_binary(M_i, X_indep) < 0.05                       # independent ⇒ ≈0 bits
    assert abs(i_binary(M_i, M_i) - 1.0) < 0.02                # σ = M_i ⇒ recovers all H(M_i)=1 bit
    assert abs(i_binary(M_i, 1 - M_i) - 1.0) < 0.02            # monotone flip still full recovery
    assert i_binary(M_i, np.zeros(400, int)) == 0.0            # constant predictor ⇒ 0


def test_value_rarefaction_monotone_ends_at_sum():
    # 3 species: v=(2,1,0.5) bits, captured (40,10,2)×; E[V(m)] monotone ↑, V(N)=Σv_s
    n_s = np.array([40, 10, 2])
    v_s = np.array([2.0, 1.0, 0.5])
    N = int(n_s.sum())
    ms, V = value_rarefaction(n_s, v_s, N, n_points=20)
    assert ms[0] == 1 and ms[-1] == N
    assert np.all(np.diff(V) >= -1e-9)                       # non-decreasing in m
    assert abs(V[-1] - v_s.sum()) < 1e-6                     # full sample captures all value
    assert V[0] < V[-1]                                      # grows with m


def test_value_missing_mass_singleton_formula():
    n_s = np.array([1, 1, 1, 5, 8])                          # 3 singletons
    v_s = np.array([0.4, 0.2, 0.1, 1.5, 0.8])
    N = int(n_s.sum())
    mm = value_missing_mass(n_s, v_s, N, b_cap=1.5, delta=0.05, frozen_value_map=True)
    assert abs(mm["MV0"] - (0.4 + 0.2 + 0.1) / N) < 1e-9     # value-weighted Good–Turing
    assert mm["cert_hi"] >= mm["MV0"]                        # certificate above the point estimate
    assert mm["B"] == 1.5
    assert mm["certificate_valid"]


def test_additive_value_core_vs_noise():
    rng = np.random.default_rng(1)
    n_p = 200
    M_i = (rng.random(n_p) > 0.5).astype(int)                 # metric verdict
    # species 0: verdict = M_i (perfectly recovers it); species 1: random (uninformative)
    sig_core = (M_i * 0.9 + 0.05)[None, :] + rng.normal(0, 0.005, (4, n_p))   # 4 captures, same species
    sig_noise = rng.uniform(0.1, 0.9, (4, n_p))                             # 4 captures, random
    sigs = np.vstack([sig_core, sig_noise])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    sp, n_s, v_s = additive_value(sigs, labels, M_i)
    assert len(sp) == 2
    core_v = v_s[0] if n_s[0] == 4 else v_s[1]
    noise_v = v_s[1] if n_s[0] == 4 else v_s[0]
    assert core_v > 0.8                                       # core ≈ H(M_i)=1 bit
    assert noise_v < 0.2                                      # noise ≈ 0 bits


def test_hill_tail_index_orders_light_vs_heavy():
    rng = np.random.default_rng(2)
    light = rng.uniform(0.9, 1.0, 500)                       # bounded ⇒ light tail ⇒ small/negative ξ
    heavy = np.exp(rng.exponential(scale=3.0, size=500))     # log-normal ⇒ heavy tail ⇒ large ξ
    assert hill_tail_index(light) < hill_tail_index(heavy)


def test_submodular_redundancy_gap_nonneg_and_bounded():
    # two perfectly-correlated species (both ≈ M_i): additive counts both, submodular counts one
    rng = np.random.default_rng(3)
    n_p = 200
    M_i = (rng.random(n_p) > 0.5).astype(int)
    pat = M_i * 0.9 + 0.05
    sigs = np.vstack([pat[None, :] + rng.normal(0, 0.005, (5, n_p)),
                      pat[None, :] + rng.normal(0, 0.005, (5, n_p))])   # 2 species, near-identical
    labels = np.array([0]*5 + [1]*5)
    sub = submodular_value(sigs, labels, M_i, top_k=20, eps=0.001)
    assert sub["R_full"] >= 0
    assert sub["redundancy_gap_selected"] >= -1e-6           # additive ≥ submodular (redundancy ≥ 0)


def test_value_census_core_plus_tail_gives_small_alpha_V_large_breadth_gap():
    """§12.3 prior: a high-recovery core (captured many×, recovers M_i) + a zero-recovery singletons tail
    ⇒ α_i (behavior) ≈ high (the tail is many distinct species) but α_{V,i} ≪ α_i (only the core recovers
    M_i) ⇒ a large, positive breadth gap α_i − α_{V,i}."""
    rng = np.random.default_rng(7)
    n_p = 300
    M_i = (rng.random(n_p) > 0.5).astype(int)
    core = M_i * 0.9 + 0.05
    sigs, tags = [], []
    for _ in range(30):                                       # ONE core species, captured 30×
        sigs.append(core + rng.normal(0, 0.005, n_p)); tags.append("A")
    for _ in range(80):                                       # 80 zero-recovery singletons (random)
        sigs.append(rng.uniform(0.1, 0.9, n_p)); tags.append("B")
    sigs = np.array(sigs)

    beh = alpha_probe(sigs, tags, tau=0.05, compute_cmi=False)
    rep = value_census(sigs, M_i, tau=0.05, submodular_top_k=40)
    alpha_i, alpha_V = beh["alpha_terminal"], rep["alpha_V_terminal"]
    gap = alpha_i - alpha_V
    assert alpha_i > 0.7                                      # many distinct species ⇒ high behavior α_i
    assert alpha_V < alpha_i - 0.3                            # recovery grows much slower ⇒ α_{V,i} ≪ α_i
    assert gap > 0.3                                          # the breadth gap (many criteria, few recover M_i)
    assert rep["MV0"] >= 0.0                                  # singleton recovery-mass (tail ≈ 0 here)
    assert rep["value_certificate_valid"] is False
    assert rep["value_cert_hi"] is None
    # the value map is dominated by the core species (top-v_s species is the core)
    assert rep["top_value_species"][0]["v_additive"] > 0.5


def test_value_missing_mass_requires_cap_and_independent_freezing():
    import numpy as np
    from methods.metric_implementer.experiments.value_census import value_missing_mass
    n_s = np.array([1, 1, 2, 3])
    v_s = np.array([0.2, 0.1, 0.05, 0.02])
    descriptive = value_missing_mass(n_s, v_s, 50, delta=0.05, b_cap=0.9)
    certified = value_missing_mass(
        n_s, v_s, 50, delta=0.05, b_cap=0.9, frozen_value_map=True)
    assert descriptive["B"] == 0.9 and descriptive["b_cap_predeclared"]
    assert descriptive["cert_hi"] is None and not descriptive["certificate_valid"]
    assert certified["cert_hi"] is not None and certified["certificate_valid"]


def test_value_missing_mass_rejects_anti_conservative_cap():
    import pytest
    with pytest.raises(ValueError, match="smaller than an observed"):
        value_missing_mass(np.array([1]), np.array([0.4]), 10, b_cap=0.2)
