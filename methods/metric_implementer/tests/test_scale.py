"""Offline tests for the scaled batch-scoring path — FakeVLLM, no GPU, zero spend.

Covers: vLLM backend interface + validate/retry, mega-batch demux correctness, long-table
schema + writer round-trip, and the scale orchestrator MEASURE phase end-to-end.
"""

from __future__ import annotations

import os
import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from methods.metric_implementer.batch_scoring import (
    LONGTABLE_COLUMNS, LongTableWriter, ScoreJob, batch_score_many)
from methods.metric_implementer.config import ImplementerConfig, apply_task_preset
from methods.metric_implementer.trial.trial_metrics import trial_metrics
from methods.metric_implementer.vllm_backend import (
    CHOICE_READOUT_ID,
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
    assert be.choice_readout_id != CHOICE_READOUT_ID


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


class _BinaryTokenizer:
    all_special_ids = [0]
    _encoded = {
        "YES": [101], "NO": [102], "MANY": [201, 202], "ALIAS": [101],
        "1": [301], "2": [302], "3": [303], "4": [304],
    }
    _decoded = {
        0: "<eos>", 101: "YES", 102: "NO", 201: "MA", 202: "NY",
        301: "1", 302: "2", 303: "3", 304: "4",
    }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self._encoded[text]

    def decode(self, token_ids, **_kwargs):
        return "".join(self._decoded[token_id] for token_id in token_ids)

    def apply_chat_template(self, messages, **_kwargs):
        return "CHAT:" + messages[-1]["content"]


class _ConstrainedBinaryEngine:
    """Mock an engine whose unconstrained favorite would be `0`, outside the allowed support."""

    def __init__(self, omit_neg=False):
        self.tokenizer = _BinaryTokenizer()
        self.params = None
        self.omit_neg = omit_neg

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, texts, params):
        self.params = params if isinstance(params, list) else [params] * len(texts)
        outputs = []
        for sampling in self.params:
            assert sampling.allowed_token_ids == [101, 102]
            # These are deliberately unnormalized full-vocabulary masses. Their ratio is 1:4;
            # token `0` may own the other 0.90, but cannot displace the two requested logprobs.
            row = {101: SimpleNamespace(logprob=math.log(0.02))}
            if not self.omit_neg:
                row[102] = SimpleNamespace(logprob=math.log(0.08))
            outputs.append(SimpleNamespace(
                outputs=[SimpleNamespace(logprobs=[row], text="NO")]))
        return outputs


class _ConstrainedChoiceEngine(_ConstrainedBinaryEngine):
    def __init__(self, omit_id=None):
        super().__init__()
        self.omit_id = omit_id

    def generate(self, texts, params):
        self.params = params if isinstance(params, list) else [params] * len(texts)
        outputs = []
        for sampling in self.params:
            assert sampling.allowed_token_ids == [301, 302, 303, 304]
            masses = {301: 0.001, 302: 0.002, 303: 0.003, 304: 0.004}
            row = {
                token_id: SimpleNamespace(logprob=math.log(mass))
                for token_id, mass in masses.items()
                if token_id != self.omit_id
            }
            outputs.append(SimpleNamespace(
                outputs=[SimpleNamespace(logprobs=[row], text="4")]))
        return outputs


def _install_binary_vllm_mock(monkeypatch, engine, *, supports_allowed_tokens=True):
    fake_vllm = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            if not supports_allowed_tokens and "allowed_token_ids" in kwargs:
                raise TypeError("unexpected keyword argument 'allowed_token_ids'")
            self.__dict__.update(kwargs)

    fake_vllm.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(
        OfflineVLLM, "_engine", classmethod(lambda _cls, _model, _cfg: engine))


def test_constrained_binary_readout_is_total_and_exact_when_other_tokens_dominate(monkeypatch):
    engine = _ConstrainedBinaryEngine()
    _install_binary_vllm_mock(monkeypatch, engine)
    backend = OfflineVLLM("executor", "judge", _cfg(fake=False), temperature=0.0)

    scores = backend.score_binary_constrained(
        ["first", "second"], pos="YES", neg="NO", seed=[17, 23])

    assert scores == pytest.approx([0.2, 0.2])
    assert all(param.allowed_token_ids == [101, 102] for param in engine.params)
    assert all(param.temperature == 0.0 and param.max_tokens == 1 and param.logprobs == 2
               for param in engine.params)
    assert [param.seed for param in engine.params] == [17, 23]
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    ("pos", "neg", "message"),
    [
        ("MANY", "NO", "exactly one token"),
        ("YES", "YES", "same token id"),
        (" YES", "NO", "unpadded literal"),
    ],
)
def test_constrained_binary_readout_rejects_invalid_label_pairs(
        monkeypatch, pos, neg, message):
    engine = _ConstrainedBinaryEngine()
    _install_binary_vllm_mock(monkeypatch, engine)
    backend = OfflineVLLM("executor", "judge", _cfg(fake=False), temperature=0.0)

    with pytest.raises(ValueError, match=message):
        backend.score_binary_constrained(["probe"], pos=pos, neg=neg)


def test_constrained_binary_readout_fails_closed_if_engine_omits_a_label(monkeypatch):
    engine = _ConstrainedBinaryEngine(omit_neg=True)
    _install_binary_vllm_mock(monkeypatch, engine)
    backend = OfflineVLLM("executor", "judge", _cfg(fake=False), temperature=0.0)

    with pytest.raises(RuntimeError, match="omitted allowed token ids"):
        backend.score_binary_constrained(["probe"], pos="YES", neg="NO")


def test_constrained_readouts_fail_closed_when_vllm_lacks_allowed_token_support(monkeypatch):
    engine = _ConstrainedBinaryEngine()
    _install_binary_vllm_mock(monkeypatch, engine, supports_allowed_tokens=False)
    backend = OfflineVLLM("executor", "judge", _cfg(fake=False), temperature=0.0)

    with pytest.raises(RuntimeError, match="allowed_token_ids"):
        backend.score_binary_constrained(["probe"], pos="YES", neg="NO")
    with pytest.raises(RuntimeError, match="allowed_token_ids"):
        backend.score_choices(["probe"], ["1", "2", "3", "4"])


def test_constrained_choices_return_the_stable_full_declared_posterior(monkeypatch):
    engine = _ConstrainedChoiceEngine()
    _install_binary_vllm_mock(monkeypatch, engine)
    backend = OfflineVLLM("reconstructor", "judge", _cfg(fake=False), temperature=0.0)
    assert backend.choice_readout_id == CHOICE_READOUT_ID

    rows = backend.score_choices(
        ["menu one", "menu two"], ["1", "2", "3", "4"], seed=[31, 37])

    assert np.asarray(rows) == pytest.approx(np.tile([0.1, 0.2, 0.3, 0.4], (2, 1)))
    assert np.asarray(rows).sum(axis=1) == pytest.approx([1.0, 1.0])
    assert all(param.logprobs == 4 for param in engine.params)
    assert [param.seed for param in engine.params] == [31, 37]


def test_constrained_choices_fail_closed_if_any_declared_option_is_missing(monkeypatch):
    engine = _ConstrainedChoiceEngine(omit_id=303)
    _install_binary_vllm_mock(monkeypatch, engine)
    backend = OfflineVLLM("reconstructor", "judge", _cfg(fake=False), temperature=0.0)

    with pytest.raises(RuntimeError, match=r"omitted allowed token ids \[303\]"):
        backend.score_choices(["menu"], ["1", "2", "3", "4"])


def test_constrained_choices_reject_a_multitoken_declared_option(monkeypatch):
    engine = _ConstrainedChoiceEngine()
    _install_binary_vllm_mock(monkeypatch, engine)
    backend = OfflineVLLM("reconstructor", "judge", _cfg(fake=False), temperature=0.0)

    with pytest.raises(ValueError, match="exactly one token"):
        backend.score_choices(["menu"], ["1", "MANY"])


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
