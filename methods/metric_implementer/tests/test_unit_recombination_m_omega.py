"""Contract + geometry tests for `unit_recombination_m_omega` (M_ω v2, D2 reconstruction side).

Pins the properties run_r2_recovery.py --momega-v2 and the D2 superset argument depend on:
(a) drop-in return contract — a superset of `gepa_discriminative_m_omega`'s keys, with the same
    trajectory row shape the call site indexes (t[0], t[2], t[3], t[4]);
(b) the children-metrics unit source actually reaches the compile (the D2-novel source);
(c) the no-units-help path ships the GEPA winner VERBATIM (the superset/no-regret guarantee);
(d) unit-pool provenance counts, marginal ordering, select/confirm split accounting;
(e) a failed LLM unit suggestion degrades to trajectory+children units with a RuntimeWarning,
    never an exception.

No GPU, no network: fakes in the style of test_m_omega_gepa_official.py; ``gepa.optimize`` is
monkeypatched with a stub that drives the real adapter.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

import gepa

from methods.metric_implementer.experiments.m_omega_gepa import (
    _discrimination_score,
    unit_recombination_m_omega,
)

COMPONENT = "m_omega_criterion"
SEED_CRITERION = "SKEWED: does the excerpt contain at least one word?"
WINNER_CRITERION = "BALANCED: does the excerpt land on an even index in the pool?"
CHILD_UNIT = "CHILD_GOLD: the excerpt uses concrete sensory detail in at least one clause."
LLM_UNIT = "LLM_UNIT: prefer excerpts whose verbs are specific rather than generic."
TEXTS = [f"excerpt body item_{index}" for index in range(16)]


def _parse(prompt: str):
    rubric = re.search(r"Criterion:\n(.*?)\n\nText:", prompt, re.DOTALL).group(1)
    text = re.search(r"\n\nText:\n(.*?)\n\nDoes the text", prompt, re.DOTALL).group(1)
    return rubric, int(text.strip().split("item_")[-1])


class _FakeExecutor:
    """Verdict is a deterministic function of (rubric content, item index).

    - rubric containing CHILD_GOLD  -> YES iff index even   (balanced: the dominating unit)
    - rubric containing BALANCED    -> YES iff index % 8 == 1 (skewed: the GEPA winner)
    - otherwise (the seed)          -> YES iff index % 8 != 0 (very skewed)
    With `units_hurt=True`, ANY compiled candidate (contains "refinements") scores constant YES,
    so every unit's marginal is negative.
    """

    def __init__(self, units_hurt: bool = False):
        self.units_hurt = units_hurt
        self.n_scored = 0

    def generate_batch(self, prompts, system=None, max_tokens=4, temperature=0.7, seed=0):
        out = []
        for p in prompts:
            rubric, index = _parse(p)
            if self.units_hurt and "refinements" in rubric:
                out.append("YES")
                continue
            if "CHILD_GOLD" in rubric:
                out.append("YES" if index % 2 == 0 else "NO")
            elif "BALANCED" in rubric:
                out.append("YES" if index % 8 == 1 else "NO")
            else:
                out.append("YES" if index % 8 != 0 else "NO")
        self.n_scored += len(prompts)
        return out


class _FakeReviser:
    """Reflection LM: proposes the winner criterion; answers the unit-suggestion ask with JSON
    (or garbage when `break_json=True`, to exercise the degraded path)."""

    def __init__(self, break_json: bool = False):
        self.break_json = break_json
        self.prompts = []

    def generate_batch(self, prompts, system=None, max_tokens=1200, temperature=0.9, seed=None):
        self.prompts.extend(prompts)
        out = []
        for p in prompts:
            if "JSON array" in p:
                out.append("no brackets here at all" if self.break_json else f'["{LLM_UNIT}"]')
            else:
                out.append(WINNER_CRITERION)
        return out


def _install_fake_optimize(monkeypatch):
    def fake_optimize(**kwargs):
        adapter, trainset = kwargs["adapter"], list(kwargs["trainset"])
        seed_text = str(next(iter(kwargs["seed_candidate"].values())))
        adapter.evaluate(trainset, {COMPONENT: seed_text}, capture_traces=True)
        kwargs["reflection_lm"]("propose an improved criterion")
        adapter.evaluate(trainset, {COMPONENT: WINNER_CRITERION}, capture_traces=False)
        return SimpleNamespace(best_candidate={COMPONENT: WINNER_CRITERION})

    monkeypatch.setattr(gepa, "optimize", fake_optimize)


def _run(monkeypatch, *, units_hurt=False, break_json=False, children=(CHILD_UNIT,), **kwargs):
    _install_fake_optimize(monkeypatch)
    executor, reviser = _FakeExecutor(units_hurt=units_hurt), _FakeReviser(break_json=break_json)
    result = unit_recombination_m_omega(
        executor, reviser, SEED_CRITERION, TEXTS, "story",
        children=list(children), **kwargs,
    )
    return result, executor, reviser


# --------------------------------------------------------------------------------------------
# (a) drop-in return contract
# --------------------------------------------------------------------------------------------
def test_return_contract_is_superset_of_gepa_contract(monkeypatch):
    result, _, _ = _run(monkeypatch)
    gepa_keys = {"optimized_prompt", "pyes", "mean", "std", "base_rate", "discrimination",
                 "trajectory"}
    assert gepa_keys <= set(result)
    assert "units" in result
    # exactly what run_r2_recovery.run_one consumes:
    float(result["std"]); float(result["base_rate"])  # noqa: E702 — the call site casts
    for t in result["trajectory"]:
        assert len(t) == 5
        t[0], float(t[2]), float(t[3]), float(t[4])
    assert len(result["pyes"]) == len(TEXTS)


def test_trajectory_row0_seed_last_row_shipped(monkeypatch):
    result, _, _ = _run(monkeypatch)
    assert result["trajectory"][0][1] == SEED_CRITERION
    assert result["trajectory"][-1][1] == result["optimized_prompt"]


# --------------------------------------------------------------------------------------------
# (b) the children-metrics source reaches the compile
# --------------------------------------------------------------------------------------------
def test_children_unit_dominates_and_is_compiled(monkeypatch):
    result, _, _ = _run(monkeypatch)
    units = result["units"]
    assert units["n_children"] == 1
    assert CHILD_UNIT in units["compiled_units"]
    assert "CHILD_GOLD" in result["optimized_prompt"]
    assert result["optimized_prompt"].startswith(WINNER_CRITERION)  # init-from-GEPA-winner
    assert not units["fell_back_to_init"]
    # the child unit's marginal is labeled with its source and tops the ranking
    assert units["marginals_top10"][0]["source"] == "children"
    assert units["marginals_top10"][0]["delta"] > 0


def test_compiled_beats_init_on_select_slice(monkeypatch):
    result, _, _ = _run(monkeypatch)
    units = result["units"]
    assert units["compiled_select_discrimination"] > units["init_select_discrimination"]


# --------------------------------------------------------------------------------------------
# (c) superset guarantee: no helpful unit => the GEPA winner ships verbatim
# --------------------------------------------------------------------------------------------
def test_no_units_help_ships_gepa_winner_verbatim(monkeypatch):
    result, _, _ = _run(monkeypatch, units_hurt=True)
    assert result["optimized_prompt"] == WINNER_CRITERION
    assert result["units"]["n_compiled"] == 0
    assert result["units"]["compiled_units"] == []


# --------------------------------------------------------------------------------------------
# (d) pool provenance + split accounting
# --------------------------------------------------------------------------------------------
def test_pool_counts_and_split_accounting(monkeypatch):
    result, _, _ = _run(monkeypatch)
    units = result["units"]
    assert units["n_llm"] == 1
    assert units["n_trajectory"] >= 2          # seed + winner clauses at minimum
    assert units["n_pool_used"] <= 40
    assert units["n_select"] + units["n_confirm"] == len(TEXTS)
    assert units["n_confirm"] == round(0.25 * len(TEXTS))
    deltas = [m["delta"] for m in units["marginals_top10"]]
    assert deltas == sorted(deltas, reverse=True)


def test_final_stats_are_canonical_over_full_pool(monkeypatch):
    result, executor, _ = _run(monkeypatch)
    # recompute the canonical statistic from the returned verdict vector
    assert result["discrimination"] == pytest.approx(
        _discrimination_score(np.asarray(result["pyes"], dtype=float)))


# --------------------------------------------------------------------------------------------
# (e) degraded LLM-suggestion path
# --------------------------------------------------------------------------------------------
def test_llm_suggestion_failure_warns_and_continues(monkeypatch):
    with pytest.warns(RuntimeWarning, match="unit suggestion failed"):
        result, _, _ = _run(monkeypatch, break_json=True)
    units = result["units"]
    assert units["n_llm"] == 0
    assert units["n_children"] == 1
    assert CHILD_UNIT in units["compiled_units"]   # children source still carries the compile
