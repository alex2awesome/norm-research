"""Planted-ground-truth tests for the orthogonalization machinery (experiments/orthogonalize.py).

These gate the §6.5/§6.6/§6.7c/§6.8 construction the way test_vinfo gates the estimator: on planted
data with a KNOWN answer the filter must drop redundant paraphrases and keep orthogonal units, the
adversarial probe must distinguish a pure-noise fragment (saturated) from a true hidden M-driver (not
saturated), and the permutation test must justify the set-abstraction only when σ²_subset ≫ σ²_perm.
"""
from __future__ import annotations

import numpy as np

from methods.metric_implementer.experiments.orthogonalize import (
    adversarial_saturation, orthogonalization_filter, permutation_order_test,
    shannon_cmi_surrogate, submodular_tail_bound,
)

N = 300


def _planted(seed=0):
    rng = np.random.default_rng(seed)
    base = (rng.random(N) < 0.5).astype(float)
    orth = (rng.random(N) < 0.5).astype(float)
    k1 = (rng.random(N) < 0.5).astype(float)
    k2 = (rng.random(N) < 0.5).astype(float)
    k3 = (rng.random(N) < 0.5).astype(float)
    h = (rng.random(N) < 0.5).astype(float)
    w = 0.70 * k1 + 0.40 * k2 + 0.18 * k3 + 0.60 * h + 0.05 * rng.standard_normal(N)
    M = (w > np.median(w)).astype(int)
    return base, orth, np.column_stack([k1, k2, k3]), h, M, rng


def test_filter_drops_redundant_keeps_orthogonal():
    base, orth, _, _, _, _ = _planted()
    signals = np.column_stack([base, base.copy(), base.copy(), base.copy(), orth])
    filt = orthogonalization_filter(signals, seed=0)
    assert set(filt["dropped"]) == {1, 2, 3}, filt          # the 3 exact copies
    assert set(filt["kept"]) == {0, 4}, filt                # base + the orthogonal signal


def test_shannon_cmi_redundant_high_orthogonal_low():
    base, orth, _, _, _, _ = _planted()
    # a copy of base is fully explained by base; an orthogonal signal is not.
    assert shannon_cmi_surrogate(base.copy(), base[:, None]) > 0.5
    assert shannon_cmi_surrogate(orth, base[:, None]) < 0.05


def test_tail_bound_decays_and_is_nonneg():
    _, _, Omega, _, M, _ = _planted()
    tb = submodular_tail_bound(M, Omega)
    assert tb["R_full"] > 0
    assert tb["tail_bound"] >= -1e-9
    assert tb["tail_bound"] <= max(tb["marginal_gains"]) + 1e-9
    # decreasing-weight latents -> the first greedy gain dominates the last
    assert tb["marginal_gains"][0] >= tb["marginal_gains"][-1] - 1e-9


def test_adversarial_noise_saturated_hidden_not():
    _, _, Omega, h, M, rng = _planted()
    noise = (rng.random(N) < 0.5).astype(float)
    assert adversarial_saturation(M, Omega, noise[:, None], seed=0)["saturated"]
    assert not adversarial_saturation(M, Omega, h[:, None], seed=0)["saturated"]


def test_permutation_test_justifies_set_abstraction():
    rng = np.random.default_rng(1)
    R_subset = np.array([0.05, 0.18, 0.31, 0.42, 0.49, 0.27, 0.11])
    R_perm = 0.42 + 0.005 * rng.standard_normal(20)
    pt = permutation_order_test(R_subset, R_perm)
    assert pt["set_abstraction_justified"]
    assert pt["ratio"] > 1.0


def test_permutation_test_flags_order_sensitivity():
    # when order jitter rivals subset spread, the abstraction is NOT justified
    pt = permutation_order_test([0.40, 0.41, 0.42], [0.10, 0.30, 0.50, 0.70])
    assert not pt["set_abstraction_justified"]
