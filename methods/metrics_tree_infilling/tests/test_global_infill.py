"""Offline test of the global (tree-free) infilling loop: plant a corpus-wide missing signal,
give the oracle proposer the ability to articulate it, and check that the loop (a) accepts it,
(b) raises guard AUC, (c) fills all three ledger tracks, and (d) reconstruction round-trips."""

import json

import numpy as np
import pandas as pd
import pytest

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.global_infill import run_global_infill
from metrics_tree_infilling.io_metrics import MetricSpec, ScoreMatrix


def _mk(n, seed, p_signal=0.55):
    """Corpus where y = known_metric OR hidden marker 'zephyr' (the plantable deficit)."""
    rng = np.random.default_rng(seed)
    known = rng.uniform(size=n)
    hidden = rng.uniform(size=n) < 0.4
    logit = 2.0 * (known - 0.5) + 2.2 * (hidden.astype(float) - 0.4)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    texts = [("the zephyr wind blew. " if h else "a calm day passed. ") + f"story {i}"
             for i, h in enumerate(hidden)]
    df = pd.DataFrame({"id": np.arange(n).astype(str), "text": texts, "judgement": y})
    sm = ScoreMatrix(levels=known.reshape(-1, 1), applicable=np.ones((n, 1), bool),
                     metric_ids=["m0"], metric_names=["known_quality"], roles=["both"])
    return df, sm, y


def oracle_judge(metrics, texts):
    """Scores 'zephyr'-rubrics by marker presence; anything else 0.5."""
    lv = np.zeros((len(texts), len(metrics)))
    ap = np.ones((len(texts), len(metrics)), bool)
    for j, m in enumerate(metrics):
        probe = (m.guidance or m.description or m.name).lower()
        if "zephyr" in probe or "wind" in probe:
            lv[:, j] = np.array(["zephyr" in t for t in texts], float)
        else:
            lv[:, j] = 0.5 + 0.01 * np.arange(len(texts)) % 2
    return lv, ap


def oracle_proposer(prompt):
    if "reverse-engineering" in prompt:   # reconstruction call
        return json.dumps({"rubric": "Text mentions the zephyr wind marker."})
    return json.dumps({"candidates": [{
        "name": "zephyr_marker",
        "description": "Mentions a zephyr wind",
        "rubric": "YES if the text mentions a zephyr; NO otherwise."}]})


def test_global_infill_accepts_planted_metric_and_fills_ledger():
    cfg = InfillConfig(random_seed=0, min_auc_gain=0.01, tau_redundant=0.9,
                       viability_min_applicability=0.1, viability_min_std=0.05,
                       text_column="text", label_column="judgement", id_column="id")
    df_d, sm_d, y_d = _mk(400, seed=1)
    df_g, sm_g, y_g = _mk(200, seed=2)
    base = [MetricSpec(metric_id="m0", name="known_quality", description="known", kind="judge")]

    res = run_global_infill(sm_d, df_d, y_d, sm_g, df_g, y_g, base, cfg,
                            judge_scorer=oracle_judge, proposer=oracle_proposer,
                            max_rounds=3, patience=2)

    kept = [l for l in res.ledgers if l.status == "kept"]
    assert len(kept) >= 1, f"planted metric not accepted: {[l.status for l in res.ledgers]}"
    led = kept[0]
    # value: guard AUC must have risen by the gate
    assert led.auc_gain >= cfg.min_auc_gain
    assert res.guard_auc_trajectory[-1] > res.guard_auc_trajectory[0]
    # track 1: data-to-develop
    assert led.n_proposal_examples > 0
    assert set(led.data_curve) == {"0.25", "0.5", "1.0"}
    assert np.isfinite(led.min_train_frac)
    # track 2: applicability on both splits
    assert 0.99 <= led.applicability_discover <= 1.0
    assert 0.99 <= led.applicability_guard <= 1.0
    # track 3: reconstruction round-trips (oracle rubric contains the marker -> agreement 1.0)
    assert led.reconstruction_agreement > 0.9
    assert "zephyr" in led.reconstruction_rubric.lower()


def test_global_infill_stops_on_useless_proposals():
    cfg = InfillConfig(random_seed=0, min_auc_gain=0.01,
                       viability_min_applicability=0.1, viability_min_std=0.05,
                       text_column="text", label_column="judgement", id_column="id")
    df_d, sm_d, y_d = _mk(300, seed=3)
    df_g, sm_g, y_g = _mk(150, seed=4)
    base = [MetricSpec(metric_id="m0", name="known_quality", description="known", kind="judge")]

    def dud_proposer(prompt):
        if "reverse-engineering" in prompt:
            return None
        return json.dumps({"candidates": [{"name": "noise", "description": "irrelevant",
                                           "rubric": "YES if the text exists."}]})

    res = run_global_infill(sm_d, df_d, y_d, sm_g, df_g, y_g, base, cfg,
                            judge_scorer=oracle_judge, proposer=dud_proposer,
                            max_rounds=5, patience=2)
    assert not [l for l in res.ledgers if l.status == "kept"]
    # patience must have cut the loop short of max_rounds proposals
    assert len(res.ledgers) <= 3
