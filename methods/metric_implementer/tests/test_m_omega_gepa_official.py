"""Contract + faithfulness tests for the official-GEPA M_ω search (`m_omega_gepa`).

The in-house rounds/mutations loop was deprecated 2026-07-19 (user directive D1); the verbatim copy
lives in ``experiments/archive/inhouse_m_omega_gepa_deprecated.py`` and
``gepa_discriminative_m_omega`` now drives official ``gepa.optimize`` through a ``GEPAAdapter``.
``run_r2_recovery.py --gepa-m-omega`` was deliberately NOT edited, so these tests pin the things it
depends on:

(a) the returned dict has exactly the fields the in-house loop returned — asserted by RUNNING the
    archived loop against the same fake backends and comparing key sets, not against a hand-copied
    literal;
(b) the per-instance search signal ``_per_instance_discrimination`` is RANK-EQUIVALENT to the
    canonical pool objective ``_discrimination_score`` on binary verdicts, so routing the search
    through GEPA's per-instance Pareto machinery does not change the estimand;
(c) the archived module still imports (and still runs).

No GPU and no network: the executor/reviser are fakes in the established style of
tests/test_v14_campaign.py, and ``gepa.optimize`` is monkeypatched in the style of
tests/test_official_gepa_tune.py (the fake optimizer drives the real adapter so the adapter surface
is genuinely exercised).
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

import gepa

from methods.metric_implementer.experiments import m_omega_gepa
from methods.metric_implementer.experiments.archive import inhouse_m_omega_gepa_deprecated as archived
from methods.metric_implementer.experiments.m_omega_gepa import (
    _discrimination_score,
    _per_instance_discrimination,
    gepa_discriminative_m_omega,
)

COMPONENT = "m_omega_criterion"
SEED_CRITERION = "SKEWED: does the excerpt contain at least one word?"
WINNER_CRITERION = "BALANCED: does the excerpt land on an even index in the pool?"
TEXTS = [f"excerpt body item_{index}" for index in range(16)]


class _FakeExecutor:
    """Planted executor: verdict is a deterministic function of (criterion, item index).

    The seed criterion yields base-rate 7/8 (skewed, low discrimination); the winner criterion
    yields base-rate 1/2 (perfectly balanced ⇒ std .5, discrimination .5).
    """

    def __init__(self):
        self.n_scored = 0

    def generate_batch(self, prompts, system=None, max_tokens=4, temperature=0.7, seed=0):
        output = []
        for prompt in prompts:
            rubric = re.search(r"Criterion:\n(.*?)\n\nText:", prompt, re.DOTALL).group(1)
            text = re.search(r"\n\nText:\n(.*?)\n\nDoes the text", prompt, re.DOTALL).group(1)
            index = int(text.strip().split("item_")[-1])
            balanced = rubric.strip().startswith("BALANCED")
            output.append("YES" if (index % 2 == 0 if balanced else index % 8 != 0) else "NO")
        self.n_scored += len(prompts)
        return output


class _FakeReviser:
    """Reflection LM stand-in: always proposes the balanced criterion."""

    def __init__(self):
        self.prompts = []

    def generate_batch(self, prompts, system=None, max_tokens=1200, temperature=0.9, seed=None):
        self.prompts.extend(prompts)
        return [WINNER_CRITERION] * len(prompts)


def _install_fake_optimize(monkeypatch, captured):
    """Monkeypatch gepa.optimize with a stub that still drives the real adapter."""

    def fake_optimize(**kwargs):
        captured["kwargs"] = kwargs
        adapter = kwargs["adapter"]
        trainset = list(kwargs["trainset"])
        seed_text = str(next(iter(kwargs["seed_candidate"].values())))
        # 1. seed evaluated on the full valset, with traces (what real gepa does first)
        seed_batch = adapter.evaluate(trainset, {COMPONENT: seed_text}, capture_traces=True)
        captured["seed_batch"] = seed_batch
        captured["reflective"] = adapter.make_reflective_dataset(
            {COMPONENT: seed_text}, seed_batch, [COMPONENT],
        )
        # 2. reflection LM round-trip
        captured["reflection_out"] = kwargs["reflection_lm"]("propose an improved criterion")
        # 3. winner evaluated on the full valset
        captured["winner_batch"] = adapter.evaluate(
            trainset, {COMPONENT: WINNER_CRITERION}, capture_traces=False,
        )
        # 4. degenerate proposal on a reflection minibatch (rejection branch)
        captured["degenerate_batch"] = adapter.evaluate(
            trainset[:3], {COMPONENT: "no"}, capture_traces=True,
        )
        return SimpleNamespace(best_candidate={COMPONENT: WINNER_CRITERION})

    monkeypatch.setattr(gepa, "optimize", fake_optimize)


def _run_official(monkeypatch, **kwargs):
    captured: dict = {}
    _install_fake_optimize(monkeypatch, captured)
    executor, reviser = _FakeExecutor(), _FakeReviser()
    result = gepa_discriminative_m_omega(
        executor, reviser, SEED_CRITERION, TEXTS, "story", **kwargs,
    )
    return result, captured, executor, reviser


# --------------------------------------------------------------------------------------------
# (a) return contract
# --------------------------------------------------------------------------------------------

def test_return_contract_matches_archived_inhouse_loop(monkeypatch):
    """The official-GEPA return dict must be field-identical to the archived in-house loop's."""
    official, _captured, _ex, _rev = _run_official(monkeypatch, rounds=2, n_mutations=2)
    legacy = archived.gepa_discriminative_m_omega(
        _FakeExecutor(), _FakeReviser(), SEED_CRITERION, TEXTS, "story",
        rounds=1, n_mutations=2,
    )

    assert set(official) == set(legacy)
    assert set(official) == {
        "optimized_prompt", "pyes", "mean", "std", "base_rate", "discrimination", "trajectory",
    }
    for key in ("optimized_prompt", "mean", "std", "base_rate", "discrimination"):
        assert type(official[key]) is type(legacy[key]), key
    assert isinstance(official["pyes"], np.ndarray) and isinstance(legacy["pyes"], np.ndarray)
    assert len(official["pyes"]) == len(TEXTS)
    # trajectory rows keep the 5-tuple shape run_r2_recovery.py indexes as t[0], t[2], t[3], t[4]
    assert all(len(row) == 5 for row in official["trajectory"])
    assert {len(row) for row in official["trajectory"]} == {len(row) for row in legacy["trajectory"]}


def test_run_r2_recovery_consumption_shape(monkeypatch):
    """Exactly the fields/coercions run_r2_recovery.run_one performs on the returned dict."""
    result, _captured, _ex, _rev = _run_official(monkeypatch)

    body = result["optimized_prompt"]
    gepa_info = {
        "gepa_std": float(result["std"]),
        "gepa_base_rate": float(result["base_rate"]),
        "optimized_prompt": str(result["optimized_prompt"])[:200],
        "opt_trajectory": [(row[0], float(row[2]), float(row[3]), float(row[4]))
                           for row in (result.get("trajectory") or [])],
    }
    assert body == WINNER_CRITERION
    # planted winner is balanced -> base_rate .5, std .5, discrimination .5
    assert gepa_info["gepa_base_rate"] == pytest.approx(0.5)
    assert gepa_info["gepa_std"] == pytest.approx(0.5)
    assert result["discrimination"] == pytest.approx(0.5)
    assert len(gepa_info["opt_trajectory"]) >= 2


def test_trajectory_reports_canonical_statistic_for_seed_and_every_candidate(monkeypatch):
    """Per-instance signal drives the SEARCH; the canonical pool statistic drives the REPORT."""
    result, _captured, executor, _rev = _run_official(monkeypatch)
    trajectory = result["trajectory"]

    assert trajectory[0][1] == SEED_CRITERION, "row 0 must be the seed"
    assert trajectory[-1][1] == result["optimized_prompt"], "last row must be the returned prompt"
    prompts = [row[1] for row in trajectory]
    assert WINNER_CRITERION in prompts

    # every reported statistic is the canonical pool-level _discrimination_score, recomputed here
    # straight from the executor rather than trusted from the search.
    for _index, prompt, std, base_rate, discrimination in trajectory:
        pyes = m_omega_gepa._score_binary_sampled(executor, prompt, TEXTS, 600)
        assert base_rate == pytest.approx(float(np.nanmean(pyes)))
        assert std == pytest.approx(float(np.nanstd(pyes)))
        assert discrimination == pytest.approx(_discrimination_score(pyes))

    # seed is skewed (7/8 YES) and must score strictly below the balanced winner
    assert trajectory[0][4] < result["discrimination"]


def test_budget_mapping_and_override(monkeypatch):
    """rounds * n_mutations * len(texts) -> max_metric_calls; explicit override wins."""
    _result, captured, _ex, _rev = _run_official(monkeypatch, rounds=3, n_mutations=4)
    assert captured["kwargs"]["max_metric_calls"] == 3 * 4 * len(TEXTS)
    assert set(captured["kwargs"]["seed_candidate"]) == {COMPONENT}
    assert captured["kwargs"]["adapter"] is not None
    assert callable(captured["kwargs"]["reflection_lm"])
    assert len(captured["kwargs"]["trainset"]) == len(TEXTS)
    assert captured["kwargs"]["valset"] is not None, "valset must be the pool (faithful selection)"

    _result2, captured2, _ex2, _rev2 = _run_official(
        monkeypatch, rounds=3, n_mutations=4, max_metric_calls=37,
    )
    assert captured2["kwargs"]["max_metric_calls"] == 37


def test_adapter_surface(monkeypatch):
    """Per-instance scores, reflective feedback + CONSTRAINT, and the degenerate-candidate branch."""
    _result, captured, _ex, reviser = _run_official(monkeypatch)

    seed_batch = captured["seed_batch"]
    assert len(seed_batch.scores) == len(TEXTS)
    assert len(seed_batch.trajectories) == len(TEXTS)
    # search signal is the MAD decomposition, not the pool statistic
    seed_pyes = np.array([1.0 if index % 8 != 0 else 0.0 for index in range(len(TEXTS))])
    assert seed_batch.scores == pytest.approx(_per_instance_discrimination(seed_pyes))

    feedback = captured["reflective"][COMPONENT][0]["Feedback"]
    assert "self-contained evaluation criterion" in feedback, "CONSTRAINT line must reach the LM"
    assert "do not add meta-commentary" in feedback.lower()
    assert "do not copy in examples, exemplars" in feedback.lower()
    assert "base_rate=" in feedback, "measured base-rate must reach the LM"
    assert SEED_CRITERION in feedback, "incumbent criterion (_mutation_prompt intent)"
    assert "score=" in feedback, "near-0.5 failure items (_select_failures)"

    assert captured["reflection_out"] == WINNER_CRITERION
    assert reviser.prompts, "reflection_lm must route through the reviser backend"

    # empty/too-short candidates are scored -1.0 rather than silently dropped
    assert captured["degenerate_batch"].scores == [-1.0, -1.0, -1.0]


def test_inert_kwargs_warn_but_do_not_break(monkeypatch):
    """mutation_mode/fewshot_examples are accepted for signature compatibility and ignored."""
    with pytest.warns(RuntimeWarning, match="inert under official GEPA"):
        result, _captured, _ex, _rev = _run_official(
            monkeypatch, mutation_mode="fewshot", fewshot_examples=(["a"], ["b"]),
        )
    assert result["optimized_prompt"] == WINNER_CRITERION


def test_empty_pool_rejected(monkeypatch):
    captured: dict = {}
    _install_fake_optimize(monkeypatch, captured)
    with pytest.raises(ValueError):
        gepa_discriminative_m_omega(_FakeExecutor(), _FakeReviser(), SEED_CRITERION, [], "story")


# --------------------------------------------------------------------------------------------
# (b) rank-equivalence of the per-instance search signal
# --------------------------------------------------------------------------------------------

def _mean_per_instance(pyes: np.ndarray) -> float:
    return float(np.mean(_per_instance_discrimination(pyes)))


def test_per_instance_signal_is_rank_equivalent_to_discrimination_score():
    """s_i = |p_i - p_bar| - .5|p_bar - .5| ranks binary candidates exactly as std - .5|mean - .5|.

    For binary verdicts MAD = 2*p_bar*(1-p_bar) and std = sqrt(p_bar*(1-p_bar)); with
    d = |p_bar - .5| both reduce to strictly decreasing functions of d, so the orderings coincide.
    Vectors here are built with DISTINCT d (base rates all on one side of .5, since BOTH statistics
    are symmetric about .5 — see test_objective_is_symmetric_about_half) so the ordering is strict.
    """
    rng = np.random.default_rng(11)
    vectors = []
    for n_yes in (1, 2, 3, 4, 5, 6):  # distinct |p_bar - .5| on a 12-item pool
        vector = np.zeros(12)
        vector[rng.permutation(12)[:n_yes]] = 1.0
        vectors.append(vector)
    assert len(vectors) >= 5
    assert len({abs(float(v.mean()) - 0.5) for v in vectors}) == len(vectors)

    canonical = np.array([_discrimination_score(v) for v in vectors])
    per_instance = np.array([_mean_per_instance(v) for v in vectors])
    assert np.array_equal(np.argsort(canonical), np.argsort(per_instance))
    # Spearman == 1 (no ties, so plain rank correlation)
    ranks_c = np.argsort(np.argsort(canonical)).astype(float)
    ranks_p = np.argsort(np.argsort(per_instance)).astype(float)
    assert np.corrcoef(ranks_c, ranks_p)[0, 1] == pytest.approx(1.0)


def test_objective_is_symmetric_about_half():
    """Both statistics depend on the base rate only through d = |p_bar - .5|.

    So p_bar and 1 - p_bar are GENUINE ties (an inherited property of `_discrimination_score`, not
    an artifact of the per-instance decomposition). Pinned here because it is why the pairwise
    agreement check below has to tolerate ties.
    """
    for n_yes in (1, 2, 3, 4, 5):
        low = np.array([1.0] * n_yes + [0.0] * (12 - n_yes))
        high = 1.0 - low
        assert _discrimination_score(low) == pytest.approx(_discrimination_score(high))
        assert _mean_per_instance(low) == pytest.approx(_mean_per_instance(high))


def test_per_instance_signal_pairwise_agreement_on_random_binary_vectors():
    """Every pairwise comparison agrees, up to exact ties (float epsilon)."""
    rng = np.random.default_rng(0)
    vectors = [
        (rng.random(int(rng.integers(4, 40))) < float(rng.uniform(0.02, 0.98))).astype(float)
        for _ in range(60)
    ]
    canonical = [_discrimination_score(v) for v in vectors]
    per_instance = [_mean_per_instance(v) for v in vectors]
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            delta_c = canonical[i] - canonical[j]
            delta_p = per_instance[i] - per_instance[j]
            if abs(delta_c) < 1e-12 or abs(delta_p) < 1e-12:
                # tied under one statistic must be tied under the other
                assert abs(delta_c) < 1e-12 and abs(delta_p) < 1e-12
                continue
            assert np.sign(delta_c) == np.sign(delta_p)


def test_per_instance_signal_nan_handling():
    """NaN verdicts score -1.0 and are excluded from p_bar; all-NaN degrades to all -1.0."""
    pyes = np.array([1.0, 0.0, np.nan, 1.0, 0.0])
    scores = _per_instance_discrimination(pyes)
    assert scores[2] == -1.0
    valid = pyes[~np.isnan(pyes)]
    p_bar = float(np.mean(valid))
    for index in (0, 1, 3, 4):
        assert scores[index] == pytest.approx(abs(pyes[index] - p_bar) - 0.5 * abs(p_bar - 0.5))
    assert _per_instance_discrimination(np.array([np.nan, np.nan])) == [-1.0, -1.0]


# --------------------------------------------------------------------------------------------
# (c) archive integrity
# --------------------------------------------------------------------------------------------

def test_archived_inhouse_loop_still_imports_and_runs():
    assert callable(archived.gepa_discriminative_m_omega)
    assert callable(archived._fewshot_block)          # loop-only helper moved out of the live module
    assert "DEPRECATED 2026-07-19" in (archived.__doc__ or "")
    assert not hasattr(m_omega_gepa, "_fewshot_block"), "few-shot operator must live in the archive"

    result = archived.gepa_discriminative_m_omega(
        _FakeExecutor(), _FakeReviser(), SEED_CRITERION, TEXTS, "story",
        rounds=1, n_mutations=2,
    )
    # the archived loop finds the same balanced winner via its own hand-rolled search
    assert result["optimized_prompt"] == WINNER_CRITERION
    assert result["discrimination"] == pytest.approx(0.5)


def test_archive_reuses_live_scoring_primitives():
    """The archive must not fork a second copy of the scoring/reporting primitives."""
    assert archived._score_binary_sampled is m_omega_gepa._score_binary_sampled
    assert archived._compute_stats is m_omega_gepa._compute_stats
    assert archived._select_failures is m_omega_gepa._select_failures
    assert archived._mutation_prompt is m_omega_gepa._mutation_prompt
