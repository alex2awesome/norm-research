"""Offline tests for the scaled batch-scoring path — FakeVLLM, no GPU, zero spend.

Covers: vLLM backend interface + validate/retry, mega-batch demux correctness, long-table
schema + writer round-trip, and the scale orchestrator MEASURE phase end-to-end.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from methods.metric_implementer.batch_scoring import (
    LONGTABLE_COLUMNS, LongTableWriter, ScoreJob, batch_score_many)
from methods.metric_implementer.config import ImplementerConfig, apply_task_preset
from methods.metric_implementer.trial.trial_metrics import trial_metrics
from methods.metric_implementer.vllm_backend import (
    _ENGINE_CACHE,
    FakeVLLM,
    OfflineVLLM,
    make_judge_backend,
)


def _cfg(fake=True):
    cfg = ImplementerConfig()
    cfg.vllm_fake = fake
    return cfg


def test_make_judge_backend_returns_fake_when_flagged():
    be = make_judge_backend("any/model", _cfg(fake=True))
    assert isinstance(be, FakeVLLM)
    assert be.model == "any/model"


def test_offline_vllm_treats_none_lfs_home_as_sk3_default(monkeypatch):
    captured = {}
    fake_vllm = types.ModuleType("vllm")

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return object()

    fake_vllm.LLM = fake_llm
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setenv("HOME", "/afs/cs.stanford.edu/u/alexspan")
    cfg = ImplementerConfig()
    cfg.vllm_lfs_home = None
    _ENGINE_CACHE.clear()
    try:
        OfflineVLLM._engine("not/a/local/model", cfg)
    finally:
        _ENGINE_CACHE.clear()
    assert captured["model"] == "not/a/local/model"
    assert os.environ["HOME"] == "/lfs/skampere3/0/alexspan"


def test_fake_vllm_deterministic_and_ordered():
    be = make_judge_backend("m", _cfg())
    a = be.generate_batch(["alpha", "beta", "gamma"])
    b = be.generate_batch(["alpha", "beta", "gamma"])
    assert a == b                      # deterministic
    assert len(a) == 3
    assert be.stats.n_prompts == 6


def test_validate_retry_runs_failed_subset():
    cfg = _cfg()
    cfg.max_retries = 3
    be = make_judge_backend("m", cfg)
    calls = {"n": 0}
    # reject everything once, then accept -> exercises the retry-the-failed-subset path
    def validate(s):
        calls["n"] += 1
        return calls["n"] > 3
    out = be.generate_batch(["a", "b", "c"], validate=validate)
    assert len(out) == 3
    assert be.stats.n_retries >= 1


def test_mega_batch_demux_shapes_and_rows(tmp_path):
    cfg = ImplementerConfig()
    cfg.vllm_fake = True
    be = make_judge_backend("fake/7b", cfg)
    texts = [f"solution {i} body" for i in range(5)]
    ids = [f"it{i}" for i in range(5)]
    jobs = []
    for k, (p, c) in enumerate(trial_metrics()):
        jobs.append(ScoreJob(artifact=p, version_id=f"v{k:03d}", texts=texts, item_ids=ids,
                             operator=["INIT", "MECHANIZE", "FEWSHOT+"][k], round=k,
                             token_cap=120, dataset="d", task="code-review"))
    w = LongTableWriter(out_dir=tmp_path, run_id="t", flush_every=10_000)
    sm, am = batch_score_many(jobs, be, cfg, run_id="t", passes=2, writer=w)
    w.flush()
    # one generate flush for the whole union (3 jobs x 5 items x 2 passes = 30 prompts)
    assert be.stats.n_calls == 1 and be.stats.n_prompts == 30
    for k in range(3):
        assert sm[f"v{k:03d}"].shape == (2, 5)
    df = LongTableWriter.load(tmp_path, "t")
    assert len(df) == 30
    assert list(df.columns) == LONGTABLE_COLUMNS
    assert set(df.operator) == {"INIT", "MECHANIZE", "FEWSHOT+"}
    assert df.applicable.all()                       # FakeVLLM always applicable
    assert df.score.between(0, 1).all()


def test_longtable_writer_parts_and_reload(tmp_path):
    w = LongTableWriter(out_dir=tmp_path, run_id="r", flush_every=4)
    # add in chunks so the threshold triggers multiple flushes (each add flushes the whole
    # buffer once it crosses flush_every)
    for i in range(10):
        w.add([{c: (i if c == "pass" else "x") for c in LONGTABLE_COLUMNS}])
    w.flush()              # final flush of the remainder
    df = LongTableWriter.load(tmp_path, "r")
    assert len(df) == 10
    parts = list(tmp_path.glob("r__part*.parquet"))
    assert len(parts) >= 2                            # at least one threshold flush + remainder


def test_scale_measure_smoke(tmp_path):
    """MEASURE phase over the pilot manifest with FakeVLLM, seeds only — exercises registry
    seed creation + per-tier mega-batch + long-table streaming. out_root isolates all I/O."""
    from methods.metric_implementer import scale, manifest as M
    m = M.pilot_manifest("test_measure")
    m.tiers = ["fake/8b", "fake/70b"]
    m.n_items = 4
    m.passes = 2
    res = scale.score_all(m, fake=True, include_history=False, out_root=str(tmp_path),
                          log=lambda *a: None)
    # 3 datasets x (3 metrics + 1 empty-rubric BASELINE control) x 2 tiers x 4 items x 2 passes
    assert res["long_rows"] == 192
    df = LongTableWriter.load(str(tmp_path / "longtable"), "test_measure")
    assert (df.operator == "BASELINE").sum() == 48       # 3 datasets x 2 tiers x 4 items x 2 passes
    assert (df.operator != "BASELINE").sum() == 144      # the 3x3 metric grid
    assert df.judge_model.nunique() == 2
    assert df.task.nunique() == 3
