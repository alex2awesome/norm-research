"""CPU-safe tests: residualizer recovers a planted tacit component; stop-rule fires on noise;
exchange-rate contour logic."""
import numpy as np
import pytest

from methods.tacit_channels.channels.eval.tally_exchange_rate import (
    iso_rho_contours, n_from_tag,
)
from methods.tacit_channels.channels.peer_review.residualize_outcome import fit_residuals


def _rows(n, tacit_weight, noise, seed=11):
    rng = np.random.default_rng(seed)
    comp_a = rng.normal(size=n)       # articulable metric 1
    comp_b = rng.normal(size=n)       # articulable metric 2
    tacit = rng.normal(size=n)        # the unobserved subjective component
    holistic = 2.0 * comp_a + 1.0 * comp_b + tacit_weight * tacit \
        + rng.normal(scale=noise, size=n)
    return ([{"item_id": f"p{i}", "holistic": float(holistic[i]),
              "components": {"a": float(comp_a[i]), "b": float(comp_b[i])}}
             for i in range(n)], tacit)


def test_residual_recovers_planted_tacit_component():
    rows, tacit = _rows(500, tacit_weight=1.5, noise=0.1)
    _beta, _fitted, resid, r2 = fit_residuals(rows, ["a", "b"])
    assert 0.5 < r2 < 0.95            # articulable part explains a lot but not everything
    corr = np.corrcoef(resid, tacit)[0, 1]
    assert corr > 0.9                 # residual IS the planted tacit component


def test_residual_is_noise_when_no_tacit_component():
    rows, tacit = _rows(500, tacit_weight=0.0, noise=0.5)
    _beta, _fitted, resid, r2 = fit_residuals(rows, ["a", "b"])
    assert abs(np.corrcoef(resid, tacit)[0, 1]) < 0.15
    # residual variance ~= noise variance -> a stop-rule with reliability ~1-noise catches it
    assert np.var(resid) == pytest.approx(0.25, rel=0.3)


def test_n_from_tag():
    assert n_from_tag("base") == 0
    assert n_from_tag("lora_n32") == 32
    assert n_from_tag("lora_nXX") == -1


def test_iso_rho_contours_prefers_cheapest_dose():
    rows = [
        # articulation axis (N=0): rich arm reaches .8, cheap arm only .6
        {"cell_id": "c", "intervention": "base", "n_examples": 0, "arm_id": "cheap",
         "added_words": 10, "is_control": False, "adverse_rho": 0.62},
        {"cell_id": "c", "intervention": "base", "n_examples": 0, "arm_id": "rich",
         "added_words": 120, "is_control": False, "adverse_rho": 0.81},
        # intervention axis (name arm): n8 reaches .55, n32 reaches .82
        {"cell_id": "c", "intervention": "lora_n8", "n_examples": 8, "arm_id": "name",
         "added_words": 0, "is_control": False, "adverse_rho": 0.55},
        {"cell_id": "c", "intervention": "lora_n32", "n_examples": 32, "arm_id": "name",
         "added_words": 0, "is_control": False, "adverse_rho": 0.82},
        # controls never count
        {"cell_id": "c", "intervention": "base", "n_examples": 0, "arm_id": "ctrl",
         "added_words": 120, "is_control": True, "adverse_rho": 0.99},
    ]
    contours = {(c["rho_level"]): c for c in iso_rho_contours(rows, levels=(0.6, 0.8))}
    assert contours[0.6]["min_articulation_words"] == 10
    assert contours[0.6]["min_intervention_n"] == 32   # n8 only reached .55
    assert contours[0.8]["min_articulation_words"] == 120
    assert contours[0.8]["min_intervention_n"] == 32
