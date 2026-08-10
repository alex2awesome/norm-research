"""Validation-by-construction for the exchangeable joint-value rarefaction (the design-valid
repair of the greedy V_best curve) and the paired front-loading statistic.

Planted constructions with KNOWN structure:
  * k latent bits define M_i; criteria = each bit duplicated r times (+ flip noise) + pure-noise
    criteria. The joint curve must saturate near the k-bit joint value while discovery keeps
    counting noise species -> D > 0.
  * M_i independent of all criteria -> V ~ 0 everywhere and the statistic declines to answer.
  * probe bootstrap WITH replacement must not leak: group-disjoint folds keep duplicated probes on
    one side, so pure-noise criteria stay at ~0 value even under resampling.
"""
import numpy as np
import pytest

from methods.metric_implementer.experiments import unseen_value_scaling as uvs
from methods.metric_implementer.experiments.value_census import (
    _species_bin_signatures, exchangeable_joint_value, joint_value_bits)


def _planted_pool(n_probes=240, k=3, dup=10, n_noise=40, flip=0.05, seed=0):
    """k informative latent bits -> M_i = majority vote; criteria = dup noisy copies of each bit
    plus n_noise pure-noise criteria. Species labels = one species per distinct source (copies of
    the same bit are the same species; each noise criterion its own species)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(k, n_probes))
    M_i = (bits.sum(0) > k / 2).astype(int) if k % 2 == 1 else (bits.sum(0) >= k / 2).astype(int)
    sigs, labels = [], []
    for b in range(k):
        for _ in range(dup):
            noise = rng.random(n_probes) < flip
            sigs.append(np.where(noise, 1 - bits[b], bits[b]).astype(float))
            labels.append(b)
    for j in range(n_noise):
        sigs.append(rng.integers(0, 2, n_probes).astype(float))
        labels.append(k + j)
    return np.asarray(sigs), np.asarray(labels), M_i


def test_joint_saturates_and_value_frontloads_vs_discovery():
    sigs, labels, M_i = _planted_pool()
    _, _, sp_bin = _species_bin_signatures(sigs, labels)
    out = exchangeable_joint_value(sp_bin, labels, M_i, n_points=8, n_subsets=12, seed=1)
    V, S, m = np.asarray(out["V_mean"]), np.asarray(out["S_mean"]), np.asarray(out["m"])
    # informative species are heavily duplicated -> a quarter of the pool already captures most value
    q = int(np.searchsorted(m, len(labels) // 4))
    assert V[-1] > 0.2                                    # recovers a real chunk of H(M_i)
    assert V[min(q, len(V) - 1)] > 0.7 * V[-1]            # saturation: 25% of draws -> >70% of value
    # discovery keeps finding noise species, value does not -> front-loading strictly positive
    fs = uvs.value_frontloading_stat(m, V, S)
    assert fs["ok"] and fs["D"] > 0.05


def test_paired_curves_shapes_and_monotone_mean():
    sigs, labels, M_i = _planted_pool(n_probes=160, n_noise=20, seed=2)
    _, _, sp_bin = _species_bin_signatures(sigs, labels)
    out = exchangeable_joint_value(sp_bin, labels, M_i, n_points=6, n_subsets=8, seed=3)
    V, S = np.asarray(out["V_draws"]), np.asarray(out["S_draws"])
    assert V.shape == S.shape == (len(out["m"]), 8)
    # mean discovery is exactly monotone; mean value monotone within OOF noise
    assert np.all(np.diff(out["S_mean"]) >= 0)
    assert np.all(np.diff(out["V_mean"]) >= -0.05)


def test_independent_target_gives_zero_value_and_declined_stat():
    rng = np.random.default_rng(4)
    sigs = rng.integers(0, 2, size=(50, 200)).astype(float)
    labels = np.arange(50)
    M_i = rng.integers(0, 2, 200)                        # independent of every criterion
    _, _, sp_bin = _species_bin_signatures(sigs, labels)
    out = exchangeable_joint_value(sp_bin, labels, M_i, n_points=5, n_subsets=6, seed=5)
    assert max(out["V_mean"]) < 0.08                     # OOF keeps pure noise near zero bits
    fs = uvs.value_frontloading_stat(out["m"], np.zeros(5), out["S_mean"])
    assert not fs["ok"]                                  # degenerate value -> no regime call


def test_probe_bootstrap_group_folds_do_not_leak():
    rng = np.random.default_rng(6)
    X = rng.integers(0, 2, size=(120, 40)).astype(float)  # 40 noise columns, overfit-prone
    y = rng.integers(0, 2, 120)
    boot = rng.integers(0, 120, 120)                      # resample WITH replacement (duplicates)
    v = joint_value_bits(y[boot], X[boot], groups=boot)   # group folds: copies never straddle folds
    assert v < 0.1                                        # leakage would report large fake bits


def test_frontloading_stat_signs():
    m = np.arange(1, 11, dtype=float)
    sat = 1 - np.exp(-m / 1.5)                            # fast-saturating "value"
    lin = m / m[-1]                                       # linear "discovery"
    d_pos = uvs.value_frontloading_stat(m, sat, lin)
    d_zero = uvs.value_frontloading_stat(m, lin, lin)
    assert d_pos["ok"] and d_pos["D"] > 0.2
    assert d_zero["ok"] and abs(d_zero["D"]) < 1e-12
