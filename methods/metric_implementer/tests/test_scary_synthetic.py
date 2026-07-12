"""Unit tests for the planted `is_scary` synthetic testbed — zero LLM spend, deterministic.

Covers: corpus balance + confound-freeness, the planted cue round-trip (judge recovers the
label), coverage monotonicity, seed-metric + feature-extraction sanity, the deterministic
backend's routing, and one offline GEPA run end-to-end.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from methods.metric_implementer.config import (BudgetCaps, ImplementerConfig,
                                                apply_task_preset)
from methods.metric_implementer.features import extract_prompt_features
from methods.metric_implementer.measures import compute_scorecard
from methods.metric_implementer.judges import PromptJudge
from methods.metric_implementer.optimizer import improve
from methods.metric_implementer.registry import Registry
from methods.metric_implementer.synthetic_examples.test_metric_scary import (
    build_dataset as BD, cues, scary_metric as SM)
from methods.metric_implementer.synthetic_examples.test_metric_scary.scary_judge import (
    JUDGE_TAG, ScaryFakeBackend, scary_roles)


# ---- 1. dataset balance + confound control ----------------------------------------------

def test_dataset_balanced_and_confound_free():
    recs = BD.build(n_examples=200, seed=1)
    assert len(recs) == 200
    assert sum(r["label"] for r in recs) == 100                # exact 50/50

    # setting and #characters identically distributed across the two labels (no confound)
    by_setting = collections.Counter((r["setting"], r["label"]) for r in recs)
    for s in BD.SETTINGS:
        assert by_setting[(s, 1)] == by_setting[(s, 0)]
    by_nchar = collections.Counter((r["n_characters"], r["label"]) for r in recs)
    for c in BD.CHAR_COUNTS:
        assert by_nchar[(c, 1)] == by_nchar[(c, 0)]

    # matched pairs share setting + char count; differ only in scariness
    pairs = collections.defaultdict(dict)
    for r in recs:
        pairs[r["pair_id"]][r["label"]] = r
    for p in pairs.values():
        assert p[0]["setting"] == p[1]["setting"]
        assert p[0]["n_characters"] == p[1]["n_characters"]

    # length is not a tell: mean word counts within a couple of words
    wc = {lab: [r["n_words"] for r in recs if r["label"] == lab] for lab in (0, 1)}
    assert abs(np.mean(wc[1]) - np.mean(wc[0])) < 3.0

    # planted label is exactly the cue signal; ids unique
    assert all(cues.is_scary_label(r["text"]) == r["label"] for r in recs)
    assert len({r["id"] for r in recs}) == len(recs)


def test_build_rejects_indivisible_n():
    with pytest.raises(ValueError):
        BD.build(n_examples=37)


def test_build_is_deterministic():
    a = BD.build(40, seed=3)
    b = BD.build(40, seed=3)
    assert [r["text"] for r in a] == [r["text"] for r in b]


# ---- 2. planted cue round-trip ----------------------------------------------------------

def test_seed_and_reference_coverage():
    assert cues.coverage(SM.SEED_PROMPT) == {"DREAD"}          # crude seed: one cue
    assert cues.coverage(SM.REFERENCE_RUBRIC) == set(cues.CATEGORIES)   # reference: all five


def test_coverage_monotone_recovery():
    recs = BD.build(80, seed=2)
    scary = [r["text"] for r in recs if r["label"] == 1][:20]
    calm = [r["text"] for r in recs if r["label"] == 0][:20]
    for t in scary:
        assert cues.planted_score(SM.REFERENCE_RUBRIC, t) >= cues.planted_score(SM.SEED_PROMPT, t)
        assert cues.planted_score(SM.REFERENCE_RUBRIC, t) > 0
    for t in calm:
        assert cues.planted_score(SM.REFERENCE_RUBRIC, t) == 0.0
        assert cues.planted_score(SM.SEED_PROMPT, t) == 0.0
    # a rubric naming no concrete cue cannot tell -> 0.5
    assert cues.planted_score("Rate the overall literary merit.", scary[0]) == 0.5


def test_reference_rubric_recovers_label_via_judge():
    recs = BD.build(120, seed=5)
    texts = [r["text"] for r in recs]
    labels = np.array([r["label"] for r in recs])
    cfg = ImplementerConfig(); apply_task_preset(cfg, "creative-writing")
    cfg.judge_model = JUDGE_TAG
    pj = PromptJudge(ScaryFakeBackend(role="judge"), cfg)
    scores, ap = pj.score(SM.reference_artifact(), texts)
    assert ap.all()
    # perfect rank separation of scary vs non-scary
    pos, neg = scores[labels == 1], scores[labels == 0]
    auc = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert auc == 1.0


# ---- 3. seed metric + feature extraction -------------------------------------------------

def test_seed_metric_shape():
    art = SM.seed_artifact()
    assert art.metric_id == "is_scary" and art.kind == "prompt"
    assert art.description and art.invariances
    assert art.violates(BudgetCaps(instruction_tokens=400)) == []


def test_reference_rubric_features():
    f = extract_prompt_features(SM.REFERENCE_RUBRIC)
    assert f["is_decomposed"] == 1.0                # numbered, multi-criterion
    assert f["has_scale_anchors"] == 1.0
    assert f["numbered_item_count"] >= 5
    assert f["has_aggregation_rule"] == 1.0


# ---- 4. deterministic backend routing ----------------------------------------------------

def test_backend_judge_and_counterfactual_routes():
    be = ScaryFakeBackend()
    scary = next(r["text"] for r in BD.build(40, seed=0) if r["label"] == 1)

    # EXHIBITS edit adds a not-yet-present cue; LACKS strips them all
    up = be.generate(f"Rewrite the following story so that it clearly EXHIBITS the property: "
                     f"x.\n```\n{scary}\n```")
    down = be.generate(f"Rewrite the following story so that it clearly LACKS the property: "
                       f"x.\n```\n{scary}\n```")
    assert len(cues.fires(up)) > len(cues.fires(scary))
    assert len(cues.fires(down)) == 0


def test_backend_reviser_advances_coverage():
    be = ScaryFakeBackend()
    import json
    raw = be.generate("You are improving the RUBRIC ...\nCURRENT RUBRIC:\n"
                      + SM.SEED_PROMPT + "\n\nMEASURED PROBLEMS:\n- none")
    obj = json.loads(raw)
    assert obj["operator"] in ("CLARIFY", "ANCHOR", "EDGE", "DECOMPOSE", "MECHANIZE")
    assert cues.coverage(SM.SEED_PROMPT) < cues.coverage(obj["rubric"])   # strictly grew


# ---- 5. offline GEPA end-to-end ----------------------------------------------------------

def _small_cfg(tmp_path) -> ImplementerConfig:
    cfg = ImplementerConfig(output_dir=str(tmp_path))
    apply_task_preset(cfg, "creative-writing")
    cfg.task = "test_metric_scary"
    cfg.judge_model = JUDGE_TAG
    cfg.n_reliability_items = 10
    cfg.reliability_passes = 2
    cfg.n_reconstruct_label_items = 16
    cfg.n_reconstruct_shown = 8
    cfg.n_reconstruct_behavioral = 12
    cfg.n_cf_base_texts = 4
    cfg.n_consistency_items = 6
    cfg.n_oracle_items = 10
    cfg.n_mutations = 1
    return cfg


def test_offline_gepa_end_to_end(tmp_path):
    cfg = _small_cfg(tmp_path)
    recs = BD.build(80, seed=4)
    texts, ids = [r["text"] for r in recs], [r["id"] for r in recs]
    roles = scary_roles()
    reg = Registry(cfg.registry_dir())

    summary = improve(SM.seed_artifact(), texts, roles, cfg, reg,
                      caps=BudgetCaps(instruction_tokens=400, optimizer_rounds=2),
                      rounds=2, run_id="t", data_ids=ids, log=lambda *a, **k: None)

    assert summary["rounds"] == 2
    assert np.isfinite(summary["best_fidelity_acceptance"])
    versions = reg.versions("is_scary", "prompt")
    assert len(versions) >= 2                                  # seed + >=1 mutation
    cards = reg.scorecards("is_scary")
    assert len(cards) >= 2

    # the optimizer articulated cues the seed lacked -> coverage grew, no spend
    best_body = reg.get_version("is_scary", summary["best_version"], "prompt")["body"]
    assert cues.coverage(SM.SEED_PROMPT) <= cues.coverage(best_body)
    assert roles.total_cost() == 0.0


# ---- 6. the three new optimizers run offline + tag the registry --------------------------

@pytest.mark.parametrize("name", ["evoprompt", "protegi", "ape"])
def test_new_optimizer_offline(tmp_path, name):
    from methods.metric_implementer import optimizers as OPT
    from methods.metric_implementer.mining import assemble_table
    cfg = _small_cfg(tmp_path)
    recs = BD.build(80, seed=name.__hash__() % 7 * 0 + 4)   # fixed seed, deterministic
    texts, ids = [r["text"] for r in recs], [r["id"] for r in recs]
    roles = scary_roles()
    reg = Registry(cfg.registry_dir())

    summary = OPT.OPTIMIZERS[name](
        SM.seed_artifact(), texts, roles, cfg, reg,
        caps=BudgetCaps(instruction_tokens=400, optimizer_rounds=2), rounds=2,
        run_id=f"t_{name}", data_ids=ids, log=lambda *a, **k: None)

    assert summary["optimizer"] == name
    assert np.isfinite(summary["best_fidelity_acceptance"])
    assert roles.total_cost() == 0.0
    # the population is registered AND tagged with this optimizer
    df = assemble_table(cfg.registry_dir(), task="scary")
    assert (df["optimizer"] == name).any()
    assert len(df[df["optimizer"] == name]) >= 2
    # the optimizer articulated cues the seed lacked
    best = reg.get_version("is_scary", summary["best_version"], "prompt")["body"]
    assert cues.coverage(SM.SEED_PROMPT) <= cues.coverage(best)


# ---- 7. TVD-MI estimator + scorecard recording ------------------------------------------

def test_tvd_mi_properties():
    from methods.metric_implementer.vinfo import tvd_mi, tvd_mi_passes
    rng = np.random.default_rng(0)
    a = rng.random(60)
    assert tvd_mi(a, a.copy()) > 0.6                          # perfect dependence
    assert tvd_mi(a, np.full(60, 0.5)) == 0.0                # CONSTANT view -> 0 (anti-gaming)
    ind = np.mean([tvd_mi(rng.random(60), rng.random(60), seed=i) for i in range(40)])
    assert ind < 0.15                                        # independent -> ~0 in expectation
    assert tvd_mi_passes(rng.random((60, 4))) < 0.15        # pure-noise passes
    two = np.where(rng.random(60) < 0.7, 0.33, 0.0)         # coarse 2-level metric, median-tie
    # faithful recovery reads MODERATE (a balanced split divides the majority value) but NOT the
    # spurious 0 the old `x > median` gave; clearly above the independent floor (~0.03).
    assert tvd_mi(two, two + rng.normal(0, 0.02, 60)) > 0.15


def test_scorecard_records_tvd_mi(tmp_path):
    cfg = _small_cfg(tmp_path)
    cfg.n_reconstruct_label_items = 40
    cfg.n_reconstruct_behavioral = 40
    cfg.n_reliability_items = 24
    cfg.n_consistency_items = 16
    cfg.reliability_passes = 3
    texts = [r["text"] for r in BD.build(160, seed=2)]
    roles = scary_roles()
    card = compute_scorecard(SM.reference_artifact(), texts, roles, cfg,
                             np.random.default_rng(0))
    assert np.isfinite(card["tvd_recovery"]) and card["tvd_recovery"] > 0.4
    for ch in ("reliability", "consistency", "reconstruction"):
        assert "tvd_mi" in card[ch]


# ---- 8. shared-distortion mitigations: cross-family executor + format perturbation ------

def test_format_perturb_preserves_meaning():
    from methods.metric_implementer.measures import format_perturb
    body = SM.REFERENCE_RUBRIC
    pert = format_perturb(body, 3)
    assert pert != body                                  # the layout actually changed
    assert cues.coverage(pert) == cues.coverage(body)    # same cue categories named -> meaning kept


def test_cross_executor_used_for_reconstruction(tmp_path):
    cfg = _small_cfg(tmp_path)
    cfg.n_reconstruct_label_items = 30
    cfg.n_reconstruct_behavioral = 30
    texts = [r["text"] for r in BD.build(120, seed=2)]
    roles = scary_roles()
    base = compute_scorecard(SM.reference_artifact(), texts, roles, cfg,
                             np.random.default_rng(0))

    class _ConstExec:                                    # a "different family" that scores constant
        model = "fake/const-family"

        def __init__(self):
            from methods.metric_implementer.backends import CallStats
            self.stats = CallStats()

        def generate_batch(self, prompts, **kw):
            return ['{"score": 0.5, "applicable": true}'] * len(prompts)

        def generate(self, p, **kw):
            return '{"score": 0.5, "applicable": true}'

    roles.cross_executor = _ConstExec()
    swapped = compute_scorecard(SM.reference_artifact(), texts, roles, cfg,
                                np.random.default_rng(0))
    assert base["reconstruction"]["tvd_mi"] > 0.2        # original executor recovers the signal
    assert swapped["reconstruction"]["tvd_mi"] == 0.0    # constant cross-executor -> 0 (path used)


def test_reliability_format_perturb_runs(tmp_path):
    cfg = _small_cfg(tmp_path)
    cfg.reliability_passes = 3
    cfg.reliability_format_perturb = True
    texts = [r["text"] for r in BD.build(80, seed=1)]
    roles = scary_roles()
    card = compute_scorecard(SM.reference_artifact(), texts, roles, cfg,
                             np.random.default_rng(0))
    # the planted judge is format-invariant (scores by cue keywords), so a meaning-preserving
    # format perturbation does NOT break reliability -> TVD-MI stays high.
    assert card["reliability"]["tvd_mi"] > 0.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
