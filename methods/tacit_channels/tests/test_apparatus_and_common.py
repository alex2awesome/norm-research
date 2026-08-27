"""CPU-safe tests: apparatus indirection, bank parsing, stats helpers, splits."""
import json

import numpy as np
import pytest

from methods.tacit_channels import _apparatus
from methods.tacit_channels.channels.common import (
    cell_stats, is_conditioned_rescue, load_grid, parse_bank_cells, spearman,
    stable_split,
)


def test_apparatus_lazy_symbols_resolve():
    assert callable(_apparatus.score_prompt)
    assert callable(_apparatus.score_declared_binary)
    assert callable(_apparatus.load_domain_items)
    assert callable(_apparatus.sha256_file)
    assert _apparatus.APPARATUS_ROOT == "methods.codability.experiments"


def test_apparatus_unknown_symbol_raises():
    with pytest.raises(AttributeError):
        _apparatus.not_a_real_symbol


def test_parse_bank_cells_handles_stringified_arms_with_nan(tmp_path):
    bank = {
        "schema": "test", "cells": [{
            "id": "TB::humor::x", "domain": "humor", "construct": "Test construct",
            "arms": ("[{'id': 'name', 'channel': 'sparse', 'control_for': None, "
                     "'meta_ratio': nan, "
                     "'forms': [{'id': 'canonical', 'prompt': 'Test construct'}]}]"),
        }],
    }
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    cells = parse_bank_cells(str(path))
    arms = cells["TB::humor::x"]["arms"]
    assert isinstance(arms, list) and arms[0]["id"] == "name"
    assert np.isnan(arms[0]["meta_ratio"])


def test_spearman_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.normal(size=40), rng.normal(size=40)
        assert spearman(a, b) == pytest.approx(
            float(scipy_stats.spearmanr(a, b).statistic), abs=1e-9)
    # with ties
    a = np.array([1, 1, 2, 3, 3, 3, 4.0])
    b = np.array([2, 1, 1, 3, 4, 4, 5.0])
    assert spearman(a, b) == pytest.approx(
        float(scipy_stats.spearmanr(a, b).statistic), abs=1e-9)


def test_stable_split_deterministic_and_balanced():
    keys = [f"item_{i}" for i in range(4000)]
    first = [stable_split(k) for k in keys]
    assert first == [stable_split(k) for k in keys]  # deterministic
    frac = sum(1 for s in first if s == "train") / len(first)
    assert 0.76 < frac < 0.84  # ~0.8


def _write_grid(tmp_path, job, domain, rows):
    """rows: list of (meta_dict, vector)."""
    d = tmp_path / job
    d.mkdir(exist_ok=True)
    np.savez_compressed(
        d / f"grid_{domain}_test_rep0.npz",
        scores=np.vstack([v for _m, v in rows]),
        meta=np.array([json.dumps(m) for m, _v in rows], dtype=object))


def _meta(cell, arm, form="canonical", control_for=None):
    return {"cell_id": cell, "arm_id": arm, "form": form, "control_for": control_for,
            "added_content_word_count": 10}


def test_cell_stats_and_conditioned_rescue(tmp_path):
    rng = np.random.default_rng(7)
    target = rng.normal(size=60)
    # executor name-only = weak; articulation arm = strong; control = weak
    weak = target + rng.normal(scale=3.0, size=60)
    strong = target + rng.normal(scale=0.3, size=60)
    ctrl = rng.normal(size=60)
    _write_grid(tmp_path, "tgt", "humor", [(_meta("c1", "name"), target)])
    _write_grid(tmp_path, "exe", "humor", [
        (_meta("c1", "name"), weak),
        (_meta("c1", "source_definition"), strong),
        (_meta("c1", "control_inert_definition", control_for="source_definition"), ctrl),
    ])
    tgt, _ = load_grid(str(tmp_path), "tgt", "humor")
    exe, emeta = load_grid(str(tmp_path), "exe", "humor")
    stats = cell_stats(tgt, exe, emeta, "c1")
    assert stats["best_arm"] == "source_definition"
    assert stats["best_rho"] > 0.9
    assert stats["beats_controls"] is True
    assert stats["gap"] > 0.10
    assert is_conditioned_rescue(stats)

    # same cell but executor name already matches target -> gap fails -> not a rescue
    _write_grid(tmp_path, "exe2", "humor", [
        (_meta("c1", "name"), target + rng.normal(scale=0.05, size=60)),
        (_meta("c1", "source_definition"), strong),
    ])
    exe2, emeta2 = load_grid(str(tmp_path), "exe2", "humor")
    stats2 = cell_stats(tgt, exe2, emeta2, "c1")
    assert not is_conditioned_rescue(stats2)  # native gap < .10 -> window closed
