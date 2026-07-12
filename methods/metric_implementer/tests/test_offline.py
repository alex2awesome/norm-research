"""Offline plumbing tests — fake backends, zero LLM spend.

Covers: complexity measurement, budget-cap enforcement, registry round-trip + ledger,
CodeJudge sandbox, scorecard shape, and one optimizer round end-to-end with a scripted
reviser.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from methods.metric_implementer.artifact import MetricArtifact, measure_code, measure_prompt
from methods.metric_implementer.backends import Roles
from methods.metric_implementer.config import BudgetCaps, ImplementerConfig
from methods.metric_implementer.judges import CodeJudge
from methods.metric_implementer.measures import compute_scorecard, convergence
from methods.metric_implementer.optimizer import improve
from methods.metric_implementer.registry import Registry
from methods.metric_implementer.trial.trial_metrics import trial_metrics


class FakeBackend:
    """Deterministic stand-in for LLMBackend. Routes by prompt content."""

    def __init__(self, model="fake", role="fake"):
        self.model, self.role = model, role
        from methods.metric_implementer.backends import CallStats
        self.stats = CallStats()

    def generate(self, prompt, system=None, max_tokens=600, validate=None,
                 temperature=None):
        self.stats.n_calls += 1
        if "articulate the single rule" in prompt or "hidden evaluator" in prompt:
            return json.dumps({"rule": "names are long",
                               "rubric": "Score 1 if most variable names exceed 3 chars."})
        if "RULE A" in prompt:
            return json.dumps({"match": 4, "difference": "emphasis"})
        if "Rewrite the following solution" in prompt:
            code = prompt.split("```")[1]
            if "EXHIBITS" in prompt:
                return code.replace("a =", "accumulated_total =")
            if "LACKS" in prompt:
                return code.replace("count", "q").replace("result", "z")
            return code  # suspect cell: hold target
        if "improving the RUBRIC" in prompt:
            return json.dumps({"operator": "CLARIFY",
                               "rubric": "Score names: 1.0 if descriptive multiword "
                                         "identifiers dominate; 0.0 if opaque.",
                               "rationale": "tighten"})
        if "improving a Python implementation" in prompt:
            return json.dumps({"operator": "CODE_REVISE",
                               "code": "def score(text):\n    return 0.5\n",
                               "rationale": "simplify"})
        # judge: score by mean identifier length (deterministic, content-sensitive)
        import re
        body = prompt.split("SOLUTION:")[-1]
        names = re.findall(r"\b([a-zA-Z_]{1,30})\s*=", body)
        if not names:
            return json.dumps({"score": 0.5, "applicable": True})
        s = min(1.0, sum(len(n) for n in names) / (6.0 * len(names)))
        return json.dumps({"score": round(s, 2), "applicable": True})

    def generate_batch(self, prompts, system=None, max_tokens=600, validate=None,
                       temperature=None):
        return [self.generate(p, system, max_tokens, validate, temperature)
                for p in prompts]


def fake_roles() -> Roles:
    f = FakeBackend
    return Roles(judge=f(role="judge"), reviser=f(role="reviser"),
                 reconstructor=f(role="reconstructor"),
                 acceptance_reconstructor=f(role="acceptance_reconstructor"),
                 generator=f(role="generator"),
                 acceptance_generator=f(role="acceptance_generator"),
                 grader=f(role="grader"), oracle=f(role="oracle"))


TEXTS = [
    "data_values = [1, 2, 3]\nrunning_total = 0\nfor i in data_values:\n"
    "    running_total += i\nprint(running_total)",
    "a = [1]\nb = 0\nfor i in a:\n    b += i\nprint(b)",
    "char_counts = {}\nfor ch in 'ab':\n    char_counts[ch] = char_counts.get(ch, 0) + 1",
    "x = input()\ny = x[::-1]\nprint(y)",
] * 6


def small_cfg(tmp_path) -> ImplementerConfig:
    cfg = ImplementerConfig(output_dir=str(tmp_path))
    cfg.n_reliability_items = 6
    cfg.reliability_passes = 2
    cfg.n_reconstruct_label_items = 10
    cfg.n_reconstruct_shown = 6
    cfg.n_reconstruct_behavioral = 6
    cfg.n_cf_base_texts = 2
    cfg.n_consistency_items = 4
    cfg.n_convergence_items = 8
    cfg.n_mutations = 1
    cfg.default_rounds = 1
    return cfg


def test_complexity_measures():
    p = measure_prompt("Score this.\n- check names\n- check comments\nEXAMPLE 1:\nfoo")
    assert p["clause_count"] >= 2 and p["n_fewshots"] == 1 and p["instruction_tokens"] > 5
    c = measure_code("def score(text):\n    if text:\n        return 1.0\n    return None\n")
    assert c["code_loc"] == 4 and c["code_cyclomatic"] == 2 and c["code_functions"] == 1


def test_budget_caps_enforced():
    art = MetricArtifact("m", "prompt", "word " * 400, name="m")
    assert "instruction_tokens" in art.violates(BudgetCaps(instruction_tokens=100))
    assert art.violates(BudgetCaps()) == []


def test_registry_roundtrip(tmp_path):
    reg = Registry(tmp_path / "registry")
    art = MetricArtifact("m1", "prompt", "Score things.", name="M1", description="d")
    reg.register_metric("m1", "M1", "d")
    vid = reg.create_version(art, operator="INIT", parent_version=None, run_id="r0",
                             optimizer_round=0, data_budget_ids=["a", "b"],
                             caps_in_force=None, models={"judge": "x"})
    assert vid == "v000"
    rec = reg.get_version("m1", vid, "prompt")
    assert rec["budget"]["data_budget_n"] == 2
    assert rec["budget"]["instruction_tokens"] > 0
    art2 = reg.artifact_from_version(rec)
    assert art2.body == art.body
    vid2 = reg.create_version(art, operator="CLARIFY", parent_version=vid, run_id="r0",
                              optimizer_round=1, data_budget_ids=["a"],
                              caps_in_force=None, models={})
    assert vid2 == "v001"
    reg.set_head("m1", "prompt", vid2, "r0")
    assert reg.head("m1", "prompt") == "v001"
    events = [e["event"] for e in reg.ledger()]
    assert events == ["METRIC_REGISTERED", "VERSION_CREATED", "VERSION_CREATED", "ACCEPTED"]


def test_code_judge_sandbox():
    _, code_art = trial_metrics()[0]
    s, ap = CodeJudge().score(code_art, TEXTS[:4])
    assert ap[:3].all() and np.isfinite(s[ap]).all()
    assert (s[ap] >= 0).all() and (s[ap] <= 1).all()
    bad = MetricArtifact("bad", "code", "def score(text):\n    raise ValueError\n")
    s2, ap2 = CodeJudge().score(bad, TEXTS[:2])
    assert not ap2.any()


def test_scorecard_shape(tmp_path):
    cfg = small_cfg(tmp_path)
    roles = fake_roles()
    rng = np.random.default_rng(0)
    prompt_art, code_art = trial_metrics()[0]
    card = compute_scorecard(prompt_art, TEXTS, roles, cfg, rng)
    for key in ("reliability", "reconstruction", "counterfactual", "consistency",
                "fidelity_scalar", "complexity"):
        assert key in card
    assert np.isfinite(card["fidelity_scalar"])
    conv = convergence(prompt_art, code_art, TEXTS[:8], roles, cfg)
    assert "spearman" in conv and len(conv["disagreement_queue"]) >= 0


def test_optimizer_one_round(tmp_path):
    cfg = small_cfg(tmp_path)
    roles = fake_roles()
    reg = Registry(cfg.registry_dir())
    prompt_art, _ = trial_metrics()[0]
    summary = improve(prompt_art, TEXTS, roles, cfg, reg,
                      caps=BudgetCaps(instruction_tokens=500), rounds=1,
                      run_id="t", log=lambda *a, **k: None)
    assert summary["rounds"] == 1
    versions = reg.versions(prompt_art.metric_id, "prompt")
    assert len(versions) >= 2                       # seed + >=1 mutation
    assert versions[1]["lineage"]["operator"] in (
        "CLARIFY", "FEWSHOT+", "ANCHOR", "EDGE", "PRUNE", "DECOMPOSE", "?")
    cards = reg.scorecards(prompt_art.metric_id)
    assert len(cards) >= 2
    events = [e["event"] for e in reg.ledger()]
    assert "RUN_STARTED" in events and "RUN_FINISHED" in events


def test_triad_planted_structure():
    """Thin metric (variance in items) vs thick (variance in item x judge interaction):
    words_share must order them correctly; planted ambiguous items must land in the hard
    stratum; D-study Erho2 must rise with panel size."""
    import pandas as pd
    from metric_implementer import triad as T

    rng = np.random.default_rng(3)
    judges, items, passes = ["a", "b", "c"], [f"i{k}" for k in range(24)], 3

    def make_table(thick: bool, ambiguous=("i0", "i1")):
        item_val = {i: rng.uniform(0.1, 0.9) for i in items}
        jbias = {(i, j): rng.uniform(-0.35, 0.35) for i in items for j in judges}
        rows = []
        for j in judges:
            for p in range(passes):
                for i in items:
                    mu = item_val[i] + (jbias[(i, j)] if thick else 0.0)
                    noise = 0.30 if i in ambiguous else 0.03
                    rows.append({"judge": j, "pass": p, "item_id": i,
                                 "score": float(np.clip(mu + rng.normal(0, noise), 0, 1))})
        return pd.DataFrame(rows)

    thin_t, thick_t = make_table(False), make_table(True)
    vc_thin, vc_thick = T.variance_components(thin_t), T.variance_components(thick_t)
    assert vc_thin["words_share"] > 0.85
    assert vc_thick["words_share"] < vc_thin["words_share"] - 0.2

    ct = T.crowdtruth(thin_t)
    assert 0.0 < ct["aqs"] <= 1.0
    lab = T.strata(ct["uqs"])
    assert lab["i0"] == "hard" and lab["i1"] == "hard"   # planted ambiguity detected

    ds = {(d["n_judges"], d["n_passes"]): d["E_rho2"] for d in T.d_study(vc_thick)}
    assert ds[(5, 3)] > ds[(1, 1)]                       # bigger panel -> more generalizable

    order = T.stratified_order(items, ct["uqs"], frac_hard=0.9,
                               rng=np.random.default_rng(0))
    hard_ids = {i for i, s in lab.items() if s == "hard"}
    assert len(set(order[:6]) & hard_ids) >= 3           # head is hard-oversampled
    assert set(order) == set(items)                      # nothing lost

    # anchor agreement: a perfect anchor = item_val rank order
    anch_rows = [{"judge": "anchor", "pass": 0, "item_id": i,
                  "score": float(thin_t[thin_t.item_id == i]["score"].mean())}
                 for i in items]
    aa = T.anchor_agreement(pd.concat([thin_t, pd.DataFrame(anch_rows)]), lab)
    assert aa["a"]["overall"]["spearman"] > 0.8


def test_triad_long_table_schema(tmp_path):
    """The long table is self-describing: judge model x pass x (prompt, prompt-iteration)
    x item x score x #datapoints — prompt-identity and data-budget columns on every row,
    so concatenated runs form the evolution panel without joins."""
    from metric_implementer import triad as T
    from metric_implementer.artifact import MetricArtifact

    art = MetricArtifact(metric_id="m", kind="prompt", body="Score X.", name="m",
                         description="d", invariances=[])

    class _FakeBackend:
        model = "fake/judge-1b"

        def __init__(self):
            self.stats = type("S", (), {"cost_usd": 0.0})()

        def generate_batch(self, prompts, **kw):
            return ['{"score": 0.5, "applicable": true}'] * len(prompts)

    cfg = type("C", (), {"max_text_chars": 4000})()
    table = T.build_long_table(art, ["t1", "t2"], ["i1", "i2"],
                               {"j1": _FakeBackend()}, cfg, passes=2,
                               version_id="v003", n_data=60)
    assert set(table.columns) >= {"judge", "model", "pass", "item_id", "score",
                                  "metric_id", "version_id", "body_hash", "n_data",
                                  "run_ts"}
    assert (table["version_id"] == "v003").all()
    assert (table["n_data"] == 60).all()
    assert (table["body_hash"] == art.body_hash).all()
    assert len(table) == 4                                # 1 judge x 2 passes x 2 items


def test_head_per_judge_tier(tmp_path):
    """Prompts are optimized FOR a judge: HEAD is keyed by (kind, tier); tiers never
    overwrite each other, and the legacy un-tiered pointer still works."""
    from metric_implementer.artifact import MetricArtifact

    reg = Registry(tmp_path / "reg")
    art = MetricArtifact(metric_id="m", kind="prompt", body="Score X.", name="m",
                         description="d", invariances=[])
    reg.register_metric("m", "m", "d")
    v0 = reg.create_version(art, operator="INIT", parent_version=None, run_id="r",
                            optimizer_round=None, data_budget_ids=["a"],
                            caps_in_force=None, models={})
    reg.set_head("m", "prompt", v0, "r", judge_tier="meta-llama/llama-3.1-8b-instruct")
    reg.set_head("m", "prompt", "v001", "r", judge_tier="qwen/qwen-2.5-72b-instruct")
    reg.set_head("m", "code", "v000", "r")                 # legacy un-tiered
    assert reg.head("m", "prompt", "meta-llama/llama-3.1-8b-instruct") == v0
    assert reg.head("m", "prompt", "qwen/qwen-2.5-72b-instruct") == "v001"
    assert reg.head("m", "prompt", "some/other-model") is None
    assert reg.head("m", "code") == "v000"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
