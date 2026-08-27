"""Role-based LLM backends with cost accounting and detect-bad-output retry.

Default backend is OpenRouter (key: ``~/.openrouter-api-key.txt``) with cheap Llama models —
this is the *development* path. Large-scale runs swap ``base_url`` to a local vLLM server or
move to offline-batch vLLM on sk3; nothing above this module changes.

Every response's ``usage.cost`` (OpenRouter reports it) is accumulated per role and flushed
to the run ledger, so spend is part of the experimental record.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import urllib.request

# Backend registry: each API provider is one entry. `cfg.backend` selects one; `format` picks the
# request/response shape ("openai" = /chat/completions; "anthropic" = /v1/messages). Add a provider
# here by adding one dict entry; nothing else changes.
BACKENDS: Dict[str, Dict[str, str]] = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions",
                   "key": "~/.openrouter-api-key.txt", "format": "openai"},
    # z.ai PaaS OpenAI-compatible endpoint — PAY-PER-TOKEN (prepaid balance; subscription does NOT cover).
    "zai": {"url": "https://api.z.ai/api/paas/v4/chat/completions",
            "key": "~/.z-ai-api-key.txt", "format": "openai"},
    # z.ai Anthropic-compatible endpoint — SUBSCRIPTION-FREE under the GLM Coding Plan. glm-5 → glm-5.2.
    # This is the strong-model path for GEPA reviser + reconstruction (prompt-optimality real-test).
    "zai_anthropic": {"url": "https://api.z.ai/api/anthropic/v1/messages",
                      "key": "~/.z-ai-api-key.txt", "format": "anthropic"},
    # OpenAI direct (SALT-lab key on sk3) — added 2026-08-12 as a diverse proposer family after
    # OpenRouter went 402 (out of credits). Proposer-only use; judging stays on local executors.
    "openai": {"url": "https://api.openai.com/v1/chat/completions",
               "key": "~/.openai-salt-lab-key.txt", "format": "openai"},
}

# Back-compat shims (old imports / callers may reference these names).
OPENROUTER_URL = BACKENDS["openrouter"]["url"]
KEY_PATHS = {k: v["key"] for k, v in BACKENDS.items()}


def _read_key(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; known: {sorted(BACKENDS)}")
    # z.ai has two accounts we toggle by monthly quota (primary exhausts ~month-end). Mirror
    # glm_cluster._key: an env override (file OR literal key), else prefer the alexander-spangher
    # account, then the primary, then the registry default. The registry default path
    # (~/.z-ai-api-key.txt) does not exist on disk, so without this the zai backends KeyError.
    if backend in ("zai", "zai_anthropic"):
        env = os.environ.get("ZAI_KEY_FILE") or os.environ.get("GLMCLUSTER_KEY")
        if env:
            ep = Path(os.path.expanduser(env))
            return ep.read_text().strip() if ep.exists() else env.strip()
        for cand in ("~/.z-ai-api-key-alexander-spangher.txt",
                     "~/.z-ai-api-key-spangher.txt", BACKENDS[backend]["key"]):
            p = Path(os.path.expanduser(cand))
            if p.exists():
                return p.read_text().strip()
        raise FileNotFoundError("no z.ai key (set ZAI_KEY_FILE or create "
                                "~/.z-ai-api-key-alexander-spangher.txt)")
    p = Path(os.path.expanduser(BACKENDS[backend]["key"]))
    return p.read_text().strip()


@dataclass
class CallStats:
    n_calls: int = 0
    n_retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def as_dict(self) -> dict:
        return dict(n_calls=self.n_calls, n_retries=self.n_retries,
                    prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                    cost_usd=round(self.cost_usd, 6))


class LLMBackend:
    """Synchronous-feeling batched chat client. ``generate_batch`` fans out with a
    bounded semaphore; per-call retry re-samples on transport errors or when the
    caller-supplied ``validate`` rejects the output (never distorts sampling params)."""

    def __init__(self, model: str, role: str, cfg, temperature: Optional[float] = None):
        self.model = model
        self.role = role
        self.cfg = cfg
        self.temperature = cfg.other_temperature if temperature is None else temperature
        self.stats = CallStats()
        self._key = _read_key(cfg.backend)

    # -- single blocking call (used inside the async fan-out) --------------------------
    def _call_once(self, prompt: str, system: Optional[str], max_tokens: int,
                   temperature: float) -> str:
        fmt = BACKENDS[self.cfg.backend].get("format", "openai")
        url = BACKENDS[self.cfg.backend]["url"]
        if fmt == "anthropic":
            # Anthropic Messages API (z.ai /api/anthropic, glm-5→glm-5.2, subscription-free):
            # system is a top-level field, messages are user/assistant only, max_tokens REQUIRED,
            # auth via x-api-key, anthropic-version header.
            body_dict = {"model": self.model, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": prompt}]}
            if system:
                body_dict["system"] = system
            if temperature is not None:
                body_dict["temperature"] = temperature
            req = urllib.request.Request(
                url, data=json.dumps(body_dict).encode(), method="POST",
                headers={"x-api-key": self._key,
                         "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_s) as r:
                obj = json.loads(r.read().decode())
            if obj.get("type") == "error" or "error" in obj:
                raise RuntimeError(f"{self.cfg.backend} API error: {obj.get('error') or obj}")
            self._tally(obj.get("usage") or {})
            content = obj.get("content") or []
            return (content[0]["text"] if content and isinstance(content[0], dict) else "")
        # OpenAI-compatible (openrouter, z.ai PaaS)
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body_d = {"model": self.model, "messages": msgs,
                  "max_tokens": max_tokens, "temperature": temperature}
        if os.environ.get("OSL_REASONING_OFF"):
            # hybrid thinkers (kimi-k2.5, qwen3-32b/max) burn max_tokens=8 on reasoning ->
            # empty content; OpenRouter normalizes this off per-provider (2026-07-09)
            body_d["reasoning"] = {"enabled": False}
        body = json.dumps(body_d).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_s) as r:
            obj = json.loads(r.read().decode())
        if "error" in obj:                       # API-level error (e.g. z.ai PaaS prepaid balance)
            raise RuntimeError(f"{self.cfg.backend} API error: {obj['error']}")
        self._tally(obj.get("usage") or {})
        return obj["choices"][0]["message"]["content"] or ""

    def _tally(self, usage):
        """Accumulate token usage from a call's `usage` dict (field names differ by API)."""
        self.stats.n_calls += 1
        self.stats.prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.stats.completion_tokens += int(usage.get("completion_tokens")
                                            or usage.get("output_tokens") or 0)
        self.stats.cost_usd += float(usage.get("cost") or 0.0)

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 600,
                 validate: Optional[Callable[[str], bool]] = None,
                 temperature: Optional[float] = None,
                 retry_temp_bump: float = 0.0) -> str:
        # `retry_temp_bump` (default 0.0 = unchanged behavior for all existing callers) raises the
        # sampling temperature on each *resample* so a deterministic bad/empty draw (common at temp 0)
        # gets a genuinely different sample instead of the identical failure. First attempt is never
        # bumped. This is the "retry a different draw" rule (never repetition_penalty).
        temp = self.temperature if temperature is None else temperature
        last = ""
        for attempt in range(self.cfg.max_retries):
            t = temp + (retry_temp_bump * attempt if retry_temp_bump else 0.0)
            try:
                last = self._call_once(prompt, system, max_tokens, t)
            except Exception:
                self.stats.n_retries += 1
                time.sleep(1.5 * (attempt + 1))
                continue
            if validate is None or validate(last):
                return last
            self.stats.n_retries += 1   # bad output -> resample (bumped temp if retry_temp_bump>0)
        return last

    def generate_batch(self, prompts: List[str], system: Optional[str] = None,
                       max_tokens: int = 600,
                       validate: Optional[Callable[[str], bool]] = None,
                       temperature: Optional[float] = None,
                       seed: Optional[int] = None,
                       retry_temp_bump: float = 0.0) -> List[str]:
        # `seed` is accepted for signature parity with the vLLM judge backend
        # (vllm_backend.generate_batch honors it via SamplingParams.seed) so callers like
        # recon_channel.induce_* can target EITHER backend uniformly. It is NOT forwarded: the
        # Anthropic/z.ai HTTP API has no seed field, so recovery diversity comes from `temperature`
        # (induce_free uses 0.9). Dropping it is the documented fix for the recon-via-API blocker.
        sem = asyncio.Semaphore(self.cfg.llm_concurrency)

        async def one(p):
            async with sem:
                return await asyncio.to_thread(
                    self.generate, p, system, max_tokens, validate, temperature, retry_temp_bump)

        async def all_():
            return await asyncio.gather(*[one(p) for p in prompts])

        return asyncio.run(all_())


_ROLE_NAMES = ("judge", "reviser", "reconstructor", "acceptance_reconstructor",
               "generator", "acceptance_generator", "grader", "oracle", "cross_executor")


@dataclass
class Roles:
    """The full cast. Acceptance-time roles are different model families from the judge.
    ``oracle`` is the strong reference judge (strong-proposer/weak-judge asymmetry).
    ``cross_executor`` (optional) is a DIFFERENT-FAMILY, SAME-TIER judge used to re-execute the
    reconstructed rubric, so reconstruction TVD-MI does not count bias the original judge and a
    same-model executor would share. None -> the original judge executes (in-loop / 1-GPU default;
    set it only in a measure pass where a second same-tier model can be resident)."""
    judge: LLMBackend
    reviser: LLMBackend
    reconstructor: LLMBackend
    acceptance_reconstructor: LLMBackend
    generator: LLMBackend
    acceptance_generator: LLMBackend
    grader: LLMBackend
    oracle: Optional[LLMBackend] = None
    cross_executor: Optional[LLMBackend] = None

    def stats(self) -> Dict[str, dict]:
        return {name: getattr(self, name).stats.as_dict()
                for name in _ROLE_NAMES if getattr(self, name) is not None}

    def total_cost(self) -> float:
        return sum(getattr(self, n).stats.cost_usd
                   for n in _ROLE_NAMES if getattr(self, n) is not None)


def make_roles(cfg, judge_model: Optional[str] = None) -> Roles:
    """``judge_model`` overrides the judge tier (the judge-capability scaling axis);
    every other role keeps its configured model."""
    return Roles(
        judge=LLMBackend(judge_model or cfg.judge_model, "judge", cfg,
                         cfg.judge_temperature),
        reviser=LLMBackend(cfg.reviser_model, "reviser", cfg),
        reconstructor=LLMBackend(cfg.reconstructor_model, "reconstructor", cfg),
        acceptance_reconstructor=LLMBackend(
            cfg.acceptance_reconstructor_model, "acceptance_reconstructor", cfg),
        generator=LLMBackend(cfg.generator_model, "generator", cfg),
        acceptance_generator=LLMBackend(
            cfg.acceptance_generator_model, "acceptance_generator", cfg),
        grader=LLMBackend(cfg.grader_model, "grader", cfg),
        oracle=(LLMBackend(cfg.oracle_model, "oracle", cfg, temperature=0.3)
                if getattr(cfg, "n_oracle_items", 0) else None),
        cross_executor=(LLMBackend(cfg.cross_executor_model, "cross_executor", cfg,
                                   cfg.judge_temperature)
                        if getattr(cfg, "cross_executor_model", None) else None),
    )


def make_roles_mixed(judge, *, strong_model: str, strong_backend: str = "zai", base_cfg=None,
                     judge_cfg=None) -> Roles:
    """The role split for the real-test architecture (prompt-optimality theory, 2026-06):

      * **judge** = a vLLM resident backend (the TARGET model X — the executor E under study),
        passed in already constructed (from ``vllm_backend.make_judge_backend``). X scores.
      * **reviser / reconstructor / acceptance_reconstructor / generator / grader / oracle** =
        a STRONG model via an API backend (GLM-4.6 on z.ai by default; Sonnet later). GLM
        iterates the GEPA prompt and induces the reconstruction — the power work.

    ``strong_model``: the API slug (e.g. ``"glm-4.6"``). ``strong_backend``: registry key
    (``"zai"``). ``base_cfg``: a config whose fields (judge_temperature, etc.) the strong-role
    LLMBackends inherit; its ``backend`` is OVERRIDDEN to ``strong_backend`` so they hit the right
    endpoint. ``judge_cfg``: the config the passed-in vLLM judge was built with (kept for reference;
    the judge object already carries its own cfg). Returns a ``Roles`` with the two backends mixed.
    """
    import copy
    if base_cfg is None:
        raise ValueError("make_roles_mixed needs base_cfg (the strong-role config template)")
    scfg = copy.copy(base_cfg)
    scfg.backend = strong_backend            # strong roles -> GLM/z.ai endpoint
    sm = strong_model
    return Roles(
        judge=judge,                         # vLLM target model X (passed in)
        reviser=LLMBackend(sm, "reviser", scfg),
        reconstructor=LLMBackend(sm, "reconstructor", scfg),
        acceptance_reconstructor=LLMBackend(sm, "acceptance_reconstructor", scfg),
        generator=LLMBackend(sm, "generator", scfg),
        acceptance_generator=LLMBackend(sm, "acceptance_generator", scfg),
        grader=LLMBackend(sm, "grader", scfg),
        oracle=(LLMBackend(sm, "oracle", scfg, temperature=0.3)
                if getattr(base_cfg, "n_oracle_items", 0) else None),
    )


def parse_json_obj(s: str) -> Optional[dict]:
    if not s:
        return None
    lo, hi = s.find("{"), s.rfind("}")
    if lo == -1 or hi <= lo:
        return None
    try:
        obj = json.loads(s[lo:hi + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
