"""Tests for the two-backend real-test architecture: the z.ai backend registry, the mixed-roles
constructor (vLLM judge + GLM reviser/reconstructor), the recon_channel two-backend split, and the
run_real_test dry-run orchestration. Zero GPU, zero API spend."""

from __future__ import annotations

import numpy as np

from methods.metric_implementer.backends import BACKENDS, LLMBackend, make_roles_mixed, Roles, CallStats
from methods.metric_implementer.config import ImplementerConfig
from methods.metric_implementer.vllm_backend import FakeVLLM, make_judge_backend


def test_backend_registry_has_all_endpoints():
    """The registry must include all three endpoints. z.ai has TWO: 'zai' (PaaS OpenAI-compat,
    pay-per-token) and 'zai_anthropic' (Anthropic-format, subscription-free, glm-5→glm-5.2)."""
    assert "openrouter" in BACKENDS
    assert "zai" in BACKENDS
    assert "zai_anthropic" in BACKENDS
    assert BACKENDS["zai"]["url"] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert BACKENDS["zai"]["format"] == "openai"
    assert BACKENDS["zai_anthropic"]["url"] == "https://api.z.ai/api/anthropic/v1/messages"
    assert BACKENDS["zai_anthropic"]["format"] == "anthropic"
    assert BACKENDS["zai_anthropic"]["key"].endswith("z-ai-api-key.txt")
    # back-compat shims still resolve
    from methods.metric_implementer.backends import OPENROUTER_URL, KEY_PATHS
    assert OPENROUTER_URL == BACKENDS["openrouter"]["url"]
    assert set(KEY_PATHS) == {"openrouter", "zai", "zai_anthropic"}


def test_unknown_backend_raises_clearly():
    import pytest
    cfg = ImplementerConfig(); cfg.backend = "bogus"
    with pytest.raises(ValueError, match="unknown backend"):
        LLMBackend("any/model", "reviser", cfg)


def test_make_roles_mixed_vllm_judge_glm_reviser(monkeypatch, tmp_path):
    """The role split: judge = the vLLM target X (passed in), reviser/reconstructor = GLM (API).
    A dummy z.ai key is staged so the GLM LLMBackend constructs without a real key (no API call)."""
    dummy = tmp_path / "zai-api-key.txt"; dummy.write_text("dummy-test-key")
    import methods.metric_implementer.backends as B
    monkeypatch.setitem(B.BACKENDS, "zai", {**B.BACKENDS["zai"], "key": str(dummy)})
    cfg = ImplementerConfig()
    judge = make_judge_backend("fake/8b", cfg, 0.7)          # vLLM FakeVLLM = target X
    roles = make_roles_mixed(judge, strong_model="glm-4.6", strong_backend="zai", base_cfg=cfg)
    assert roles.judge is judge                                # X is the SAME object passed in
    assert roles.reviser.model == "glm-4.6"
    assert roles.reconstructor.model == "glm-4.6"
    assert roles.reviser.cfg.backend == "zai"                 # strong roles hit the z.ai endpoint
    assert roles.judge.model == "fake/8b"                     # judge stays vLLM (X)


def test_recon_channel_accepts_recon_backend():
    """run_metric's two-backend split: executor X + reconstructor (GLM). Signature only — the
    plumbing test (GLM induces, X re-executes) is the run_real_test dry-run below."""
    import inspect
    from methods.metric_implementer.recon_channel import run_metric, _mcq_recon
    assert "recon_backend" in inspect.signature(run_metric).parameters
    assert "recon" in inspect.signature(_mcq_recon).parameters


def test_run_real_test_dry_run_orchestration(tmp_path):
    """End-to-end dry-run: Phase A (GEPA, FakeVLLM X + mock GLM) → prose p̂ extracted → GLM-decompose
    (mock → 3 criteria) → orthogonalize → certificate, with BOTH I(M,M_ω) and I(M_ω,M_s) measured.
    Wires together with no GPU/API; the small MI values are fake-backend artifacts, not wiring
    failures."""
    from methods.metric_implementer.experiments.run_real_test import main
    main(["--tasks", "creative-writing", "--dry-run", "--n-items", "24",
          "--budget", "2", "--rounds", "1", "--large-k", "15",
          "--out-dir", str(tmp_path / "rt")])
    import json
    summary = json.load(open(tmp_path / "rt" / "summary.json"))
    assert len(summary) == 1
    s = summary[0]
    assert s["task"] == "creative-writing"
    # Phase B reaches the certificate (no error): mode set, K>=2, decomposition gap measured
    assert "mode" in s, f"expected certificate to run, got: {s}"
    assert s["K"] >= 2
    assert "I_M_Momega" in s.get("decomposition", {}), \
        f"expected I(M,M_ω) in decomposition, got: {s.get('decomposition')}"
