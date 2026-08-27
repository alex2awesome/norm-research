"""Generator arms: all three produce Proposals through the common interface, tagged by arm,
and the unconditional/label arms drive run_global_infill end-to-end offline."""

import json

import numpy as np

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.generators import (
    label_contrast_generator, residual_generator, unconditional_generator)
from metrics_tree_infilling.global_infill import run_global_infill
from metrics_tree_infilling.io_metrics import MetricSpec, ScoreMatrix
from metrics_tree_infilling.tests.test_global_infill import _mk, oracle_judge


def proposer_zephyr(prompt):
    return json.dumps({"candidates": [{
        "name": "zephyr_marker", "description": "Mentions a zephyr wind",
        "rubric": "YES if the text mentions a zephyr; NO otherwise."}]})


def test_arms_emit_tagged_proposals():
    cfg = InfillConfig(text_column="text", label_column="judgement", id_column="id")
    texts = ["a"] * 10 + ["b"] * 10
    y = np.array([1] * 10 + [0] * 10)
    for gen, arm in [(unconditional_generator("creative writing", k=2), "unconditional"),
                     (label_contrast_generator(texts, y, seed=0), "label_contrast")]:
        props = gen(None, ["known thing"], cfg, proposer_zephyr)
        assert props and all(p.generator == arm for p in props)
        assert all(p.name and p.rubric for p in props)
    assert residual_generator()(None, [], cfg, proposer_zephyr) == []   # residual needs contrast


def test_unconditional_arm_through_gate():
    cfg = InfillConfig(random_seed=0, min_auc_gain=0.01, min_bits_gain=0.005,
                       viability_min_applicability=0.1, viability_min_std=0.05,
                       text_column="text", label_column="judgement", id_column="id")
    df_d, sm_d, y_d = _mk(400, seed=1)
    df_g, sm_g, y_g = _mk(200, seed=2)
    base = [MetricSpec(metric_id="m0", name="known_quality", description="known", kind="judge")]
    res = run_global_infill(sm_d, df_d, y_d, sm_g, df_g, y_g, base, cfg,
                            judge_scorer=oracle_judge, proposer=proposer_zephyr,
                            max_rounds=3, patience=2, measure_reconstruction=False,
                            proposal_fn=unconditional_generator("stories", k=2))
    kept = [l for l in res.ledgers if l.status == "kept"]
    assert kept and kept[0].generator == "unconditional"
    assert kept[0].bits_gain >= cfg.min_bits_gain          # bits gate enforced
    assert len(res.guard_bits_trajectory) == len(res.guard_auc_trajectory)
