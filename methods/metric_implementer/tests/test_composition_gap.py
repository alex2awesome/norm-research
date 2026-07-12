"""Planted tests for the Δ_comp composition-gap arm (experiments/composition_gap.py) and the
holistic-coverage flag on adversarial_saturation. All CPU, no model. The planted world: a composed
prompt that carries value BEYOND the certified head must read Δ_comp_beyond > 0; a composed prompt
that merely restates the checklist must read ≈ 0 (composition measured, never assumed)."""
from __future__ import annotations

import numpy as np

from methods.metric_implementer.experiments.composition_gap import (
    compose_checklist_prompts, delta_comp, holistic_probe_prompts)
from methods.metric_implementer.experiments.orthogonalize import adversarial_saturation
from methods.metric_implementer.experiments.value_census import i_binary


def test_compose_and_holistic_builders_deterministic_and_complete():
    crits = ["the tone stays consistent across scenes", "the ending resolves the central arc"]
    v1 = compose_checklist_prompts(crits, "narrative craft", seed=3)
    v2 = compose_checklist_prompts(crits, "narrative craft", seed=3)
    assert [x["id"] for x in v1] == [x["id"] for x in v2] and len(v1) == 6   # deterministic
    assert any("persona" in x["id"] for x in v1)                             # gestalt frame present
    assert all("tone stays consistent" in x["text"] for x in v1)             # criteria stated
    hp = holistic_probe_prompts(crits, "narrative craft", extra_prompts=["<gepa-mined prompt>"])
    ids = [h["id"] for h in hp]
    assert "pointer_name" in ids and "persona_gestalt" in ids and "gepa_0" in ids
    assert len(hp) >= 6


def test_delta_comp_planted_beyond_vs_restated_checklist():
    rng = np.random.default_rng(0)
    n = 400
    u = (rng.uniform(0, 1, n) > 0.5).astype(int)          # the component the unit pool carries
    h = (rng.uniform(0, 1, n) > 0.5).astype(int)          # hidden component NO unit carries
    half = np.arange(n) < n // 2
    M = np.where(half, u, h)                              # the practice uses both
    S_cols = u[:, None].astype(float)                     # the certified head = u only
    opt = float(i_binary(M, u))
    comp_full = np.where(rng.uniform(0, 1, n) < 0.05, 1 - M, M).astype(float)  # gestalt sees BOTH
    comp_null = np.where(rng.uniform(0, 1, n) < 0.05, 1 - u, u).astype(float)  # restated checklist
    d = delta_comp(np.vstack([comp_full, comp_null]), M, S_cols, opt,
                   variant_ids=["full", "null"])
    per = {p["id"]: p for p in d["per_variant"]}
    assert per["full"]["v_beyond"] > 0.15                 # value beyond the certified head — the
    assert per["null"]["v_beyond"] < 0.05                 # exact quantity ε cannot bound
    assert d["composition_carries_value"] and d["best_variant_beyond"] == "full"
    assert d["delta_comp_total"] > 0.1                    # composed beats OPT_Ω in total too


def test_delta_comp_null_when_units_exhaust_the_practice():
    rng = np.random.default_rng(2)
    n = 400
    u = (rng.uniform(0, 1, n) > 0.5).astype(int)
    M = u.copy()                                          # the checklist IS the practice
    comp = np.where(rng.uniform(0, 1, n) < 0.05, 1 - u, u).astype(float)
    d = delta_comp(comp, M, u[:, None].astype(float), float(i_binary(M, u)))
    assert d["delta_comp_beyond"] < 0.05
    assert not d["composition_carries_value"]
    assert d["delta_comp_total"] < 0.05                   # composing adds ~nothing (can be < 0)


def test_adversarial_saturation_covers_composition_flag():
    rng = np.random.default_rng(1)
    M = (rng.uniform(0, 1, 300) > 0.5).astype(float)
    Xo = (rng.uniform(0, 1, (300, 3)) > 0.5).astype(float)
    Xp = (rng.uniform(0, 1, (300, 2)) > 0.5).astype(float)
    r_unit = adversarial_saturation(M, Xo, Xp, probe_kinds=["unit", "unit"])
    assert r_unit["covers_composition"] is False          # checklist-channel saturation only
    r_hol = adversarial_saturation(M, Xo, Xp, probe_kinds=["unit", "holistic"])
    assert r_hol["covers_composition"] is True
    r_none = adversarial_saturation(M, Xo, Xp)            # kinds unknown → not claimed
    assert r_none["covers_composition"] is False and r_none["probe_kinds"] is None
