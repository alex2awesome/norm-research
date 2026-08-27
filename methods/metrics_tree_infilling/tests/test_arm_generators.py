"""The two ported method arms (autometrics_iterative, metric_tree) as generator arms:
conditioning-strategy ports through the SHARED proposer executor. Checks each arm's
distinctive mechanism (failure pairs + memory + self-critique; partition-conditioned
gap-fill) and that both flow through the common acceptance gate end-to-end."""

import json

import numpy as np

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.contrast import Contrast
from metrics_tree_infilling.generators import (
    autometrics_iterative_generator, metric_tree_generator)
from metrics_tree_infilling.global_infill import run_global_infill
from metrics_tree_infilling.io_metrics import MetricSpec

from .test_global_infill import _mk, oracle_judge


def _cfg(**kw):
    base = dict(random_seed=0, min_auc_gain=0.01,
                viability_min_applicability=0.1, viability_min_std=0.05,
                text_column="text", label_column="judgement", id_column="id")
    base.update(kw)
    return InfillConfig(**base)


def _contrast():
    return Contrast(node_id="GLOBAL",
                    wrong_pos=["good story about a zephyr"], wrong_neg=["dull story"],
                    pairs=[("good story about a zephyr", "dull story")],
                    wrong_disc_idx=np.array([0, 1]), right_disc_idx=np.array([2]),
                    n_wrong=2)


# ---- autometrics_iterative arm ---------------------------------------------------------

def test_ami_arm_parses_list_output_and_tags_generator():
    calls = []

    def proposer(prompt):
        calls.append(prompt)
        if "verdict whether it is" in prompt:
            return json.dumps({"verdicts": ["substantive"]})
        return json.dumps([{"name": "Zephyr_Specificity",
                            "rubric": {"yes": "names the zephyr wind concretely",
                                       "no": "no concrete wind imagery"},
                            "scale": "binary"}])

    gen = autometrics_iterative_generator("evaluating short fiction", k=4)
    props = gen(_contrast(), ["known metric"], _cfg(), proposer)
    assert len(props) == 1
    assert props[0].generator == "autometrics_iterative"
    assert "zephyr" in props[0].rubric.lower()
    assert props[0].n_examples == 2                       # 1 pair = 2 items
    assert "PAIR" in calls[0] and "known metric" in calls[0]


def test_ami_arm_self_critique_drops_superficial():
    def proposer(prompt):
        if "verdict whether it is" in prompt:
            return json.dumps({"verdicts": ["superficial: length proxy", "substantive"]})
        return json.dumps([
            {"name": "Word_Count", "rubric": {"yes": "long", "no": "short"}, "scale": "binary"},
            {"name": "Evidence_Depth", "rubric": {"yes": "cites concrete evidence",
                                                  "no": "assertion only"}, "scale": "binary"}])

    gen = autometrics_iterative_generator("task", k=4)
    props = gen(_contrast(), [], _cfg(), proposer)
    assert [p.name for p in props] == ["Evidence_Depth"]


def test_ami_arm_iteration_memory_dedups_and_reaches_prompt():
    prompts = []

    def proposer(prompt):
        prompts.append(prompt)
        if "verdict whether it is" in prompt:
            return json.dumps({"verdicts": ["substantive"]})
        return json.dumps([{"name": "Same_Metric",
                            "rubric": {"yes": "x", "no": "y"}, "scale": "binary"}])

    gen = autometrics_iterative_generator("task", k=4)
    first = gen(_contrast(), [], _cfg(), proposer)
    assert len(first) == 1
    second = gen(_contrast(), [], _cfg(), proposer)
    assert second == []                                   # hash-dedup vs own memory
    assert "PREVIOUSLY PROPOSED BY YOU" in prompts[-1]
    assert "Same_Metric" in prompts[-1]


def test_ami_arm_no_pairs_returns_empty():
    gen = autometrics_iterative_generator("task")
    assert gen(None, [], _cfg(), lambda p: "") == []


# ---- metric_tree arm --------------------------------------------------------------------

def _mt_fixture(n=200, seed=0):
    """Bank with one label-informative column + one noise column; y mixed within cells."""
    rng = np.random.default_rng(seed)
    informative = rng.uniform(size=n)
    noise = rng.uniform(size=n)
    y = (rng.uniform(size=n) < 0.2 + 0.6 * (informative > 0.5)).astype(int)
    texts = [f"story {i} " + ("zephyr wind" if rng.uniform() < 0.4 else "calm day")
             for i in range(n)]
    bank = np.column_stack([informative, noise])
    return texts, y, bank, ["informative_metric", "noise_metric"]


def test_metric_tree_arm_partition_context_and_tag():
    texts, y, bank, names = _mt_fixture()
    prompts = []

    def proposer(prompt):
        prompts.append(prompt)
        return json.dumps([{"name": "Cell_Specific_Craft",
                            "rubric": {"yes": "shows cell-specific craft",
                                       "no": "generic"}, "scale": "binary"}])

    gen = metric_tree_generator("evaluating fiction", texts, y, bank, names, k=2, seed=0)
    props = gen(None, ["known desc"], _cfg(), proposer)
    assert props and props[0].generator == "metric_tree"
    p = prompts[0]
    assert "WHAT THIS PARTITION IS" in p
    assert "informative_metric" in p
    assert "BLOCKLIST" in p and "known desc" in p
    assert "ACCEPTANCE RATE" in p


def test_metric_tree_arm_round_robins_cells():
    texts, y, bank, names = _mt_fixture()
    prompts = []

    def proposer(prompt):
        prompts.append(prompt)
        return json.dumps([{"name": f"M{len(prompts)}",
                            "rubric": {"yes": "a", "no": "b"}, "scale": "binary"}])

    gen = metric_tree_generator("t", texts, y, bank, names, k=1, seed=0)
    gen(None, [], _cfg(), proposer)
    gen(None, [], _cfg(), proposer)
    assert len(prompts) == 2
    ctx = lambda s: s.split("WHAT THIS PARTITION IS ===")[1].split("Given that")[0]
    assert ctx(prompts[0]) != ctx(prompts[1])             # different cells across rounds


def test_metric_tree_arm_pure_cells_returns_empty():
    """When every partition cell is label-pure (rate outside [0.2, 0.8]) there is nothing to
    gap-fill — the arm must stay silent rather than propose on degenerate cells."""
    n = 100
    informative = np.linspace(0, 1, n)
    y = (informative > 0.5).astype(int)                   # perfectly separated -> pure cells
    bank = np.column_stack([informative, informative])
    gen = metric_tree_generator("t", ["x"] * n, y, bank, ["a", "b"], seed=0)
    assert gen(None, [], _cfg(), lambda p: "") == []


# ---- both arms end-to-end through the gate ----------------------------------------------

def test_both_arms_flow_through_gate_and_accept_planted_metric():
    df_d, sm_d, y_d = _mk(400, seed=1)
    df_g, sm_g, y_g = _mk(200, seed=2)
    base = [MetricSpec(metric_id="m0", name="known_quality", description="known", kind="judge")]

    zephyr = json.dumps([{"name": "zephyr_marker",
                          "rubric": {"yes": "the text mentions a zephyr wind",
                                     "no": "no zephyr"}, "scale": "binary"}])

    def proposer(prompt):
        if "verdict whether it is" in prompt:
            return json.dumps({"verdicts": ["substantive"]})
        if "reverse-engineering" in prompt:
            return json.dumps({"rubric": "Text mentions the zephyr wind marker."})
        return zephyr

    for factory, tag in [
        (lambda: autometrics_iterative_generator("fiction", k=2), "autometrics_iterative"),
        (lambda: metric_tree_generator("fiction", df_d["text"].tolist(), y_d,
                                       sm_d.levels, sm_d.metric_names, k=2, seed=0),
         "metric_tree"),
    ]:
        res = run_global_infill(sm_d, df_d, y_d, sm_g, df_g, y_g, list(base), _cfg(),
                                judge_scorer=oracle_judge, proposer=proposer,
                                max_rounds=3, patience=2, measure_reconstruction=False,
                                proposal_fn=factory())
        kept = [l for l in res.ledgers if l.status == "kept"]
        assert kept, f"{tag}: planted metric not accepted: {[l.status for l in res.ledgers]}"
        assert kept[0].generator == tag
