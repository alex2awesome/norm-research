"""Offline-batch vLLM backend — the scaling path (sk3, GPU). Drop-in for ``LLMBackend``.

Why this exists: the OpenRouter path in ``backends.py`` is per-call HTTP (dev only, and as of
2026-06-12 we are sk3-only). At scale we load ONE model resident in the process and feed it the
largest possible prompt list per ``LLM.generate`` call (vLLM continuous-batching schedules them
internally). The model loads ONCE per tier and is reused across every metric, candidate, round,
and pass — that amortization (not HTTP concurrency) is the win.

Two implementations behind one interface:
- ``OfflineVLLM``  : real, imports vllm lazily, resident-model singleton keyed by model id.
- ``FakeVLLM``     : deterministic, no GPU, for offline tests and dry-runs (hash->score).

Both expose ``generate_batch(prompts, system, max_tokens, validate, temperature)`` with the SAME
signature as ``LLMBackend.generate_batch``, so ``judges.PromptJudge`` and everything above it run
unchanged. ``validate``/retry is applied post-hoc: vLLM has no per-item resample, so we re-run the
*failed subset* once at a fresh seed (detect-bad-output-and-retry, never repetition_penalty —
[[feedback_no_repetition_penalty_retry_instead]]).

sk3 env quirks are honored in ``OfflineVLLM._engine`` (HOME pinned to /lfs so nohup keeps AFS
tokens; FLASHINFER MoE off for Qwen-FP8; Qwen3 enable_thinking=False; gpu_memory_utilization 0.93;
kv-cache fp8 optional). Confirm exact values against the repo's canonical recipe before a big run.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class CallStats:
    n_calls: int = 0           # = number of generate() flushes
    n_prompts: int = 0
    n_retries: int = 0         # prompts re-run because validate() rejected them
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0      # 0 on local vLLM; kept for ledger-schema parity with OpenRouter

    def as_dict(self) -> dict:
        return dict(n_calls=self.n_calls, n_prompts=self.n_prompts,
                    n_retries=self.n_retries, prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens, cost_usd=round(self.cost_usd, 6))


# Resident engines keyed by model id — load once per process, reuse across all work.
# (Across TIERS we run ONE model per PROCESS — `scale measure --tier` — so process exit frees the
# GPU; vLLM pre-allocates ~90% for KV cache and does not release it on in-process model switch.)
_ENGINE_CACHE: Dict[str, object] = {}


def _resolve_model_path(model: str) -> str:
    """Map a HF hub id -> its local snapshot dir if cached, so vLLM loads OFFLINE without hub-id
    resolution (which fails in the shared sk3 cache: config.json resolves but transformers reports
    'not cached' — a .no_exist / revision artifact). Returns the id unchanged if already a path or
    not found. Verified 2026-06-13: unlocks Qwen2.5/Qwen3/Mixtral that fail by-id offline."""
    if os.path.isdir(model):
        return model
    hub = os.path.join(os.environ.get("HF_HOME") or
                       os.path.expanduser("~/.cache/huggingface"), "hub")
    d = os.path.join(hub, "models--" + model.replace("/", "--"))
    try:
        commit = open(os.path.join(d, "refs", "main")).read().strip()
        snap = os.path.join(d, "snapshots", commit)
        if os.path.isdir(snap):
            return snap
    except OSError:
        pass
    return model


class _BaseVLLM:
    """Shared interface + the validate/retry-the-failed-subset wrapper."""

    def __init__(self, model: str, role: str, cfg, temperature: Optional[float] = None):
        self.model = model
        self.role = role
        self.cfg = cfg
        self.temperature = (getattr(cfg, "other_temperature", 0.7)
                            if temperature is None else temperature)
        self.stats = CallStats()

    # subclasses implement the raw flush over a list of (system, prompt) -> list[str]
    def _flush(self, prompts: List[str], system: Optional[str], max_tokens: int,
               temperature: float, seed: int | Sequence[int]) -> List[str]:
        raise NotImplementedError

    def generate_batch(self, prompts: List[str], system: Optional[str] = None,
                       max_tokens: int = 600,
                       validate: Optional[Callable[[str], bool]] = None,
                       temperature: Optional[float] = None,
                       seed: int | Sequence[int] = 0) -> List[str]:
        if not prompts:
            return []
        temp = self.temperature if temperature is None else temperature
        outs = self._flush(prompts, system, max_tokens, temp, seed=seed)
        self.stats.n_calls += 1
        self.stats.n_prompts += len(prompts)
        if validate is not None:
            bad = [i for i, o in enumerate(outs) if not validate(o)]
            for attempt in range(1, max(1, getattr(self.cfg, "max_retries", 3))):
                if not bad:
                    break
                if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
                    retry_seed = [int(seed[i]) + 1_000_003 * attempt for i in bad]
                else:
                    retry_seed = int(seed) + 1_000_003 * attempt
                redo = self._flush([prompts[i] for i in bad], system, max_tokens,
                                   temp, seed=retry_seed)
                self.stats.n_calls += 1
                self.stats.n_retries += len(bad)
                still = []
                for j, i in enumerate(bad):
                    outs[i] = redo[j]
                    if not validate(redo[j]):
                        still.append(i)
                bad = still
        return outs

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 600,
                 validate: Optional[Callable[[str], bool]] = None,
                 temperature: Optional[float] = None) -> str:
        return self.generate_batch([prompt], system, max_tokens, validate, temperature)[0]


class OfflineVLLM(_BaseVLLM):
    """Real resident-model vLLM. Imports vllm lazily so the module loads on a laptop."""

    @classmethod
    def _engine(cls, model: str, cfg):
        if model in _ENGINE_CACHE:
            return _ENGINE_CACHE[model]
        # sk3 env (matches the repo's 137 canonical recipes): pin HOME to /lfs BEFORE importing
        # vllm/HF (nohup AFS-token safety [[feedback_sk3_afs_tokens]]); disable FlashInfer
        # version check (required across sk3 scripts); MoE-FP8 safety for Qwen.
        # ImplementerConfig declares this field as None, so getattr(..., default)
        # does not apply. Treat None as "use the sk3 runtime default" explicitly.
        lfs_home = getattr(cfg, "vllm_lfs_home", None) or "/lfs/skampere3/0/alexspan"
        if lfs_home:
            os.environ["HOME"] = lfs_home
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from vllm import LLM
        kwargs = dict(
            model=_resolve_model_path(model),
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL")
                                         or getattr(cfg, "vllm_gpu_mem_util", 0.90)),
            max_model_len=getattr(cfg, "vllm_max_model_len", 8192),
            dtype=getattr(cfg, "vllm_dtype", "auto"),   # 'auto' keeps FP8 weights / picks BF16
            trust_remote_code=True,
            enable_prefix_caching=True,                  # shared judge-system prefix reuse
            tensor_parallel_size=getattr(cfg, "vllm_tp_size", 1),  # 1 GPU default (cluster rule)
        )
        # kv_cache_dtype='fp8' produces '!!!!' garbage on FP8 ckpts without calibrated attn
        # scales — leave at 'auto' (BF16 kv) unless explicitly set on a validated model.
        kv = getattr(cfg, "vllm_kv_cache_dtype", None)
        if kv:
            kwargs["kv_cache_dtype"] = kv
        mm = getattr(cfg, "vllm_limit_mm", None)
        if mm:
            kwargs["limit_mm_per_prompt"] = mm
        bs = os.environ.get("VLLM_BLOCK_SIZE")
        if bs:
            kwargs["block_size"] = int(bs)  # FlashInfer head_size-256 bug (gemma-2) needs 32/64
        if os.environ.get("VLLM_ENFORCE_EAGER"):
            kwargs["enforce_eager"] = True  # skip compile/cudagraph (gemma-3 init-hang diagnosis)
        eng = LLM(**kwargs)
        _ENGINE_CACHE[model] = eng
        return eng

    def _flush(self, prompts, system, max_tokens, temperature, seed):
        from vllm import SamplingParams
        eng = self._engine(self.model, self.cfg)
        tok = eng.get_tokenizer()
        # build chat-formatted strings; enable_thinking=False for Qwen3
        # ([[project_articulation_star_smoke_status]]). apply_chat_template returns a STRING
        # here (tokenize=False) — avoids the transformers5 BatchEncoding len() trap
        # ([[feedback_transformers5_chat_template_len]]).
        texts = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": p}]
            try:
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
            except TypeError:                       # tokenizer without enable_thinking kwarg
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(s)
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [int(s) for s in seed]
            if len(seeds) != len(texts):
                raise ValueError(f"got {len(seeds)} seeds for {len(texts)} prompts")
            sp = [SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=s)
                  for s in seeds]
        else:
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=int(seed))
        outs = eng.generate(texts, sp)
        # vLLM preserves input order; map back defensively by request_id if present
        return [o.outputs[0].text if o.outputs else "" for o in outs]

    def score_binary(self, prompts: List[str], system: Optional[str] = None,
                     pos: str = "YES", neg: str = "NO",
                     seed: int | Sequence[int] = 0) -> List[float]:
        """CONTINUOUS score in [0,1] = P(pos) over a 1-token {pos,neg} answer, read from logprobs.
        Bypasses JSON-format failure entirely — this is the U3 continuous-fidelity readout AND the
        GAN detector (P(AI)). Returns nan when neither token appears in the top logprobs."""
        import math
        from vllm import SamplingParams
        eng = self._engine(self.model, self.cfg)
        tok = eng.get_tokenizer()
        texts = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": p}]
            try:
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
            except TypeError:
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(s)
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [int(s) for s in seed]
            if len(seeds) != len(texts):
                raise ValueError(f"got {len(seeds)} seeds for {len(texts)} prompts")
            sp = [SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=s)
                  for s in seeds]
        else:
            sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=int(seed))
        outs = eng.generate(texts, sp)
        self.stats.n_calls += 1
        self.stats.n_prompts += len(prompts)
        res = []
        for o in outs:
            lps = (o.outputs[0].logprobs[0] if o.outputs and o.outputs[0].logprobs else {}) or {}
            ppos = pneg = 0.0
            for _tid, L in lps.items():
                t = (getattr(L, "decoded_token", "") or "").strip().upper()
                if t == pos.upper():
                    ppos += math.exp(L.logprob)
                elif t == neg.upper():
                    pneg += math.exp(L.logprob)
            res.append(ppos / (ppos + pneg) if (ppos + pneg) > 0 else float("nan"))
        return res

    def score_choices(self, prompts: List[str], choices: Sequence[str],
                      system: Optional[str] = None,
                      seed: int | Sequence[int] = 0) -> List[List[float]]:
        """Normalized first-token probabilities over a small declared choice vocabulary.

        The caller must prompt for one of the literal single-token choices (MCQ uses digits). Values
        are normalized over the declared alternatives, so they are a lower-variance replacement for
        repeatedly sampling A/B/C/D. A row is all-NaN if a choice token falls outside vLLM's returned
        top-logprob set; callers fail closed or fall back to sampled choices rather than imputing it.
        """
        import math
        from vllm import SamplingParams

        labels = [str(choice).strip() for choice in choices]
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("score_choices needs at least two unique literal choices")
        eng = self._engine(self.model, self.cfg)
        tok = eng.get_tokenizer()
        texts = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": p}]
            try:
                rendered = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(rendered)
        n_logprobs = max(20, len(labels) * 4)
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [int(s) for s in seed]
            if len(seeds) != len(texts):
                raise ValueError(f"got {len(seeds)} seeds for {len(texts)} prompts")
            params = [SamplingParams(
                temperature=0.0, max_tokens=1, logprobs=n_logprobs, seed=s) for s in seeds]
        else:
            params = SamplingParams(
                temperature=0.0, max_tokens=1, logprobs=n_logprobs, seed=int(seed))
        outputs = eng.generate(texts, params)
        self.stats.n_calls += 1
        self.stats.n_prompts += len(prompts)
        rows = []
        for output in outputs:
            logprobs = (output.outputs[0].logprobs[0]
                        if output.outputs and output.outputs[0].logprobs else {}) or {}
            masses = {label: 0.0 for label in labels}
            for candidate in logprobs.values():
                token = (getattr(candidate, "decoded_token", "") or "").strip()
                if token in masses:
                    masses[token] += math.exp(candidate.logprob)
            values = [masses[label] for label in labels]
            total = sum(values)
            rows.append([value / total for value in values] if total > 0.0
                        else [float("nan")] * len(labels))
        return rows


class FakeVLLM(_BaseVLLM):
    """Deterministic, GPU-free, role-routing stub. Returns the right JSON shape for whichever
    role the prompt belongs to (judge / reviser-mutation / reconstruction / grader), so a full
    SEARCH→MEASURE dry-run exercises the GEPA loop, mints operator-labeled iterations, and fills
    the long table — all with zero spend. Scores are a hash of the prompt (stable per seed)."""

    _OPS = ["CLARIFY", "MECHANIZE", "FEWSHOT+", "ANCHOR", "EDGE", "PRUNE", "DECOMPOSE"]

    def _flush(self, prompts, system, max_tokens, temperature, seed):
        out = []
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [int(s) for s in seed]
            if len(seeds) != len(prompts):
                raise ValueError(f"got {len(seeds)} seeds for {len(prompts)} prompts")
        else:
            seeds = [int(seed)] * len(prompts)
        for p, item_seed in zip(prompts, seeds):
            h = int(hashlib.sha256((p + str(item_seed)).encode()).hexdigest(), 16)
            score = round((h % 1000) / 999.0, 3)
            if "mutation operator" in p or "revised rubric" in p:
                op = self._OPS[h % len(self._OPS)]
                out.append(json.dumps({"operator": op,
                                       "rubric": f"[{op} v{item_seed}] revised scoring rubric: "
                                                 f"rate the property on a 0-1 scale.",
                                       "rationale": "fake mutation"}))
            elif "revised implementation" in p or "score(text)" in p:
                out.append(json.dumps({"operator": "CODE_REVISE",
                                       "code": "def score(text):\n    return min(1.0, len(text)/5000)",
                                       "rationale": "fake code"}))
            elif "attributions" in p:
                out.append(json.dumps({"attributions": ["AMBIGUOUS_PROMPT"]}))
            elif "articulate the single rule" in p or '"rule"' in p:
                out.append(json.dumps({"rule": "scores on the stated property",
                                       "rubric": "rate 0-1 on the property"}))
            elif "RULE A" in p or '"match"' in p:
                out.append(json.dumps({"match": 1 + h % 5, "difference": "fake"}))
            elif '{"critique"' in p:                       # ProTeGi textual gradient
                out.append(json.dumps({"critique": "the rubric is too vague; name concrete checks"}))
            elif 'Respond ONLY with JSON: {"rubric"' in p:  # EvoPrompt/ProTeGi/APE proposal
                op = self._OPS[h % len(self._OPS)]
                out.append(json.dumps({"rubric": f"[{op} v{item_seed}] rate the property on a 0-1 "
                                                 f"scale using concrete checks for it."}))
            else:
                out.append(f'{{"score": {score}, "applicable": true}}')
        return out

    def score_binary(self, prompts, system=None, pos="YES", neg="NO", seed=0):
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [int(s) for s in seed]
            if len(seeds) != len(prompts):
                raise ValueError(f"got {len(seeds)} seeds for {len(prompts)} prompts")
        else:
            seeds = [int(seed)] * len(prompts)
        out = []
        for p, item_seed in zip(prompts, seeds):
            h = int(hashlib.sha256((p + str(item_seed)).encode()).hexdigest(), 16)
            out.append(round((h % 1000) / 999.0, 3))
        return out

    def score_choices(self, prompts, choices, system=None, seed=0):
        labels = [str(choice) for choice in choices]
        rows = []
        for prompt in prompts:
            raw = np.array([
                1 + int(hashlib.sha256((prompt + "|" + label).encode()).hexdigest(), 16) % 1000
                for label in labels
            ], dtype=float)
            rows.append((raw / raw.sum()).tolist())
        return rows

    # FakeVLLM is also a valid stand-in for reviser/reconstructor/grader roles, which call
    # generate() (single) rather than generate_batch(); both route through _flush.


def make_judge_backend(model: str, cfg, temperature: Optional[float] = None) -> _BaseVLLM:
    """Factory: FakeVLLM when cfg.vllm_fake (tests/dry-run), else OfflineVLLM."""
    cls = FakeVLLM if getattr(cfg, "vllm_fake", False) else OfflineVLLM
    return cls(model, "judge", cfg, temperature)
