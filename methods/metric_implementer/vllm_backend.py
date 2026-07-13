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

# Stable semantic protocol identifier for evidence/cache manifests. Any change to label
# validation, constrained support, or posterior extraction must advance this value.
CHOICE_READOUT_ID = "allowed-exact-single-token-choice-processed-posterior-v2"
FAKE_CHOICE_READOUT_ID = "fake-hash-choice-probabilities-v1"
CR3_BINARY_READOUT_ID = "rubric-first-pyes-allowed-two-token-processed-content-seed-v3"
FAKE_CR3_BINARY_READOUT_ID = "fake-synthetic-binary-signature-v1"


def _resolve_model_path(model: str, *, home: str | None = None) -> str:
    """Map a HF hub id -> its local snapshot dir if cached, so vLLM loads OFFLINE without hub-id
    resolution (which fails in the shared sk3 cache: config.json resolves but transformers reports
    'not cached' — a .no_exist / revision artifact). Returns the id unchanged if already a path or
    not found. Verified 2026-06-13: unlocks Qwen2.5/Qwen3/Mixtral that fail by-id offline."""
    if os.path.isdir(model):
        return model
    cache_home = os.environ.get("HF_HOME")
    if cache_home is None:
        cache_home = os.path.join(home or os.path.expanduser("~"), ".cache", "huggingface")
    hub = os.path.join(cache_home, "hub")
    d = os.path.join(hub, "models--" + model.replace("/", "--"))
    try:
        commit = open(os.path.join(d, "refs", "main")).read().strip()
        snap = os.path.join(d, "snapshots", commit)
        if os.path.isdir(snap):
            return snap
    except OSError:
        pass
    return model


def model_revision_id(model: str, *, home: str | None = None) -> str:
    """Resolve the immutable local snapshot revision used by an offline worker."""
    resolved = os.path.abspath(_resolve_model_path(model, home=home))
    if os.path.isdir(resolved) and os.path.basename(os.path.dirname(resolved)) == "snapshots":
        return os.path.basename(resolved)
    return resolved if os.path.isdir(resolved) else str(_resolve_model_path(model, home=home))


def _single_token_label_id(tokenizer, label: str) -> int:
    """Resolve an exact output label to one non-special tokenizer token or fail closed."""
    literal = str(label)
    if not literal or literal != literal.strip():
        raise ValueError(f"binary label must be a nonempty, unpadded literal: {label!r}")
    try:
        token_ids = tokenizer.encode(literal, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(literal)
    token_ids = list(token_ids)
    if len(token_ids) != 1:
        raise ValueError(
            f"binary label {literal!r} must encode as exactly one token; got {token_ids}")
    token_id = int(token_ids[0])
    if token_id in set(getattr(tokenizer, "all_special_ids", ()) or ()):
        raise ValueError(f"binary label {literal!r} resolves to special token id {token_id}")
    try:
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    if str(decoded) != literal:
        raise ValueError(
            f"binary label {literal!r} does not round-trip through token id {token_id}: "
            f"{decoded!r}")
    return token_id


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

    choice_readout_id = CHOICE_READOUT_ID

    @classmethod
    def _engine(cls, model: str, cfg):
        if model in _ENGINE_CACHE:
            return _ENGINE_CACHE[model]
        # sk3 env (matches the repo's 137 canonical recipes): pin HOME to /lfs BEFORE importing
        # vllm/HF (nohup AFS-token safety [[feedback_sk3_afs_tokens]]); disable FlashInfer
        # version check (required across sk3 scripts); MoE-FP8 safety for Qwen.
        # The orchestrator pins METRIC_IMPLEMENTER_LFS_HOME to its declared worker-home before
        # this process imports vLLM. Keep the sk3 fallback for older entry points.
        lfs_home = (
            getattr(cfg, "vllm_lfs_home", None)
            or os.environ.get("METRIC_IMPLEMENTER_LFS_HOME")
            or "/lfs/skampere3/0/alexspan"
        )
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
            # vLLM otherwise returns top-k raw model probabilities from before
            # allowed_token_ids is applied. CR3 needs the complete posterior on
            # the masked YES/NO or MCQ support.
            logprobs_mode="processed_logprobs",
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

    def score_binary_constrained(self, prompts: List[str], system: Optional[str] = None,
                                 pos: str = "YES", neg: str = "NO",
                                 seed: int | Sequence[int] = 0) -> List[float]:
        """Return the exact first-token conditional probability ``P(pos | {pos, neg})``.

        Unlike the legacy top-logprob readout above, this path constrains vLLM's first-token
        support to two declared token IDs before its log-softmax.  It therefore remains total when
        an unconstrained model would prefer ``0``, ``1``, or prose.  Labels must be distinct exact
        single-token literals; invalid labels or incomplete engine evidence raise rather than
        producing a NaN or an imputed score.
        """
        import math
        from vllm import SamplingParams

        if not prompts:
            return []
        eng = self._engine(self.model, self.cfg)
        tok = eng.get_tokenizer()
        pos_id = _single_token_label_id(tok, pos)
        neg_id = _single_token_label_id(tok, neg)
        if pos_id == neg_id:
            raise ValueError(
                f"binary labels {pos!r} and {neg!r} resolve to the same token id {pos_id}")
        allowed_ids = [pos_id, neg_id]

        texts = []
        for prompt in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            try:
                rendered = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                rendered = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            texts.append(rendered)

        def sampling_params(item_seed: int):
            try:
                return SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    logprobs=2,
                    seed=int(item_seed),
                    allowed_token_ids=allowed_ids,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "installed vLLM lacks SamplingParams.allowed_token_ids; "
                    "constrained binary scoring cannot run safely") from exc

        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [int(item_seed) for item_seed in seed]
            if len(seeds) != len(texts):
                raise ValueError(f"got {len(seeds)} seeds for {len(texts)} prompts")
            params = [sampling_params(item_seed) for item_seed in seeds]
        else:
            params = sampling_params(int(seed))
        outputs = eng.generate(texts, params)
        self.stats.n_calls += 1
        self.stats.n_prompts += len(prompts)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} constrained binary outputs for {len(prompts)} prompts")

        scores: List[float] = []
        for row_index, output in enumerate(outputs):
            logprobs = (
                output.outputs[0].logprobs[0]
                if output.outputs and output.outputs[0].logprobs
                else None
            )
            if not logprobs:
                raise RuntimeError(
                    f"constrained binary output {row_index} has no first-token logprobs")
            missing = [token_id for token_id in allowed_ids if token_id not in logprobs]
            if missing:
                raise RuntimeError(
                    f"constrained binary output {row_index} omitted allowed token ids {missing}")
            pos_logprob = float(logprobs[pos_id].logprob)
            neg_logprob = float(logprobs[neg_id].logprob)
            if not (math.isfinite(pos_logprob) and math.isfinite(neg_logprob)):
                raise RuntimeError(
                    f"constrained binary output {row_index} returned non-finite label logprobs")
            # Stable two-class normalization. vLLM already normalizes after applying the allowed
            # token mask; repeating the ratio here makes the intended conditional explicit and is
            # robust to an engine returning logits shifted by a shared constant.
            if pos_logprob >= neg_logprob:
                score = 1.0 / (1.0 + math.exp(neg_logprob - pos_logprob))
            else:
                ratio = math.exp(pos_logprob - neg_logprob)
                score = ratio / (1.0 + ratio)
            if not math.isfinite(score):
                raise RuntimeError(
                    f"constrained binary output {row_index} produced a non-finite score")
            scores.append(score)
        return scores

    def score_choices(self, prompts: List[str], choices: Sequence[str],
                      system: Optional[str] = None,
                      seed: int | Sequence[int] = 0) -> List[List[float]]:
        """Exact first-token probabilities conditional on the declared choice vocabulary.

        Protocol: ``CHOICE_READOUT_ID``. Every choice must be a distinct exact single-token
        literal (CR3 MCQ uses digits). vLLM
        masks all other tokens before its log-softmax, and every declared token must be present in
        the returned evidence. This never imputes zero probability for a top-k omission: invalid
        labels, unsupported vLLM versions, and incomplete outputs fail closed.
        """
        import math
        from vllm import SamplingParams

        if not prompts:
            return []
        labels = [str(choice) for choice in choices]
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("score_choices needs at least two unique literal choices")
        eng = self._engine(self.model, self.cfg)
        tok = eng.get_tokenizer()
        token_ids = [_single_token_label_id(tok, label) for label in labels]
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("score_choices literals must resolve to distinct token ids")
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
        def sampling_params(item_seed: int):
            try:
                return SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    logprobs=len(token_ids),
                    seed=int(item_seed),
                    allowed_token_ids=token_ids,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "installed vLLM lacks SamplingParams.allowed_token_ids; "
                    "constrained choice scoring cannot run safely") from exc

        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [int(s) for s in seed]
            if len(seeds) != len(texts):
                raise ValueError(f"got {len(seeds)} seeds for {len(texts)} prompts")
            params = [sampling_params(item_seed) for item_seed in seeds]
        else:
            params = sampling_params(int(seed))
        outputs = eng.generate(texts, params)
        self.stats.n_calls += 1
        self.stats.n_prompts += len(prompts)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} constrained choice outputs for {len(prompts)} prompts")
        rows = []
        for row_index, output in enumerate(outputs):
            logprobs = (
                output.outputs[0].logprobs[0]
                if output.outputs and output.outputs[0].logprobs
                else None
            )
            if not logprobs:
                raise RuntimeError(
                    f"constrained choice output {row_index} has no first-token logprobs")
            missing = [token_id for token_id in token_ids if token_id not in logprobs]
            if missing:
                raise RuntimeError(
                    f"constrained choice output {row_index} omitted allowed token ids {missing}")
            values = [float(logprobs[token_id].logprob) for token_id in token_ids]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(
                    f"constrained choice output {row_index} returned non-finite label logprobs")
            shift = max(values)
            masses = [math.exp(value - shift) for value in values]
            total = math.fsum(masses)
            if not math.isfinite(total) or total <= 0.0:
                raise RuntimeError(
                    f"constrained choice output {row_index} produced invalid total mass")
            row = [mass / total for mass in masses]
            if not all(math.isfinite(value) for value in row):
                raise RuntimeError(
                    f"constrained choice output {row_index} produced non-finite probabilities")
            rows.append(row)
        return rows


class FakeVLLM(_BaseVLLM):
    """Deterministic, GPU-free, role-routing stub. Returns the right JSON shape for whichever
    role the prompt belongs to (judge / reviser-mutation / reconstruction / grader), so a full
    SEARCH→MEASURE dry-run exercises the GEPA loop, mints operator-labeled iterations, and fills
    the long table — all with zero spend. Scores are a hash of the prompt (stable per seed)."""

    _OPS = ["CLARIFY", "MECHANIZE", "FEWSHOT+", "ANCHOR", "EDGE", "PRUNE", "DECOMPOSE"]
    # Deliberately not CHOICE_READOUT_ID: this hash stub is useful for dry-runs but is not
    # evidence from the constrained vLLM token posterior and must never receive a bound-grade tag.
    choice_readout_id = FAKE_CHOICE_READOUT_ID

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

    def score_binary_constrained(self, prompts, system=None, pos="YES", neg="NO", seed=0):
        """CPU stand-in for the bound-grade CR3 readout; always finite and deterministic."""
        if not str(pos).strip() or not str(neg).strip() or str(pos) == str(neg):
            raise ValueError("constrained binary labels must be two distinct nonempty literals")
        return self.score_binary(prompts, system=system, pos=pos, neg=neg, seed=seed)

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
