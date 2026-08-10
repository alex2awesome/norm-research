"""LoRA-aware fork of the frozen teacher-forced YES/NO readout.

WHY A FORK: the frozen `score_declared_binary` (score_adaptive_ostensive_orbits.py) calls
`engine.generate(...)` directly, so adapter routing cannot reach it without editing a file
that live frozen manifests hash-pin. This fork is byte-equivalent in semantics, with exactly
one addition: `**lora_kwargs` on the generate call (obtained from the backend's
`_maybe_lora()`, which returns {} when no adapter is configured).

DRIFT GUARD: `UPSTREAM_SHA256` pins the source of the frozen function at fork time
(2026-07-22). `check_upstream_drift()` raises if the frozen implementation ever changes,
so the fork can never silently diverge. The zero-adapter acceptance test
(score_with_adapter --acceptance-test) is the runtime equivalence gate.
"""
from __future__ import annotations

import hashlib
import inspect
from collections.abc import Sequence

import numpy as np

from methods.tacit_channels import _apparatus

# sha256 of inspect.getsource(score_declared_binary) at fork time — refreshed only by a
# deliberate re-fork after reviewing the upstream change.
UPSTREAM_SHA256 = None  # computed lazily on first check; pinned value recorded by tests


def upstream_source_sha256() -> str:
    fn = _apparatus.score_declared_binary
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def check_upstream_drift(pinned_sha256: str) -> None:
    observed = upstream_source_sha256()
    if observed != pinned_sha256:
        raise RuntimeError(
            "frozen score_declared_binary changed upstream since this fork was cut "
            f"(observed {observed[:12]}, pinned {pinned_sha256[:12]}). Re-review the fork "
            "before scoring anything with an adapter.")


def score_declared_binary_lora(backend, prompts: list[str], *, pos: str = "YES",
                               neg: str = "NO", seed: int | Sequence[int] = 0,
                               expected_token_ids: dict[str, int] | None = None) -> np.ndarray:
    """Exact conditional P(pos) from teacher-forced pos/neg continuation likelihoods,
    with optional LoRA adapter routing. Fork of the frozen implementation — see module doc."""
    prompt_seeds = None
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
        prompt_seeds = [int(value) for value in seed]
        if len(prompt_seeds) != len(prompts):
            raise ValueError(f"got {len(prompt_seeds)} seeds for {len(prompts)} prompts")
    if backend.__class__.__name__ == "FakeVLLM":
        values = np.asarray(backend.score_binary(
            prompts, pos=pos, neg=neg,
            seed=prompt_seeds if prompt_seeds is not None else seed), float)
        if values.shape != (len(prompts),):
            raise ValueError(
                "declared-binary backend returned an invalid output shape: "
                f"observed={values.shape}, expected={(len(prompts),)}")
        return values
    from vllm import SamplingParams

    engine = backend._engine(backend.model, backend.cfg)
    tokenizer = engine.get_tokenizer()
    labels = [pos, neg]
    token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(values) != 1 for values in token_ids):
        raise ValueError(f"declared labels are not single tokens: {dict(zip(labels, token_ids))}")
    runtime_token_ids = [values[0] for values in token_ids]
    if expected_token_ids is not None:
        frozen = [int(expected_token_ids[label]) for label in labels]
        if runtime_token_ids != frozen:
            raise ValueError(
                "runtime declared-label token ids differ from the frozen manifest: "
                f"observed={dict(zip(labels, runtime_token_ids))} "
                f"expected={dict(zip(labels, frozen))}")
    rendered_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        rendered_prompts.append(rendered)
    scoring_texts = [f"{rendered}{label}"
                     for rendered in rendered_prompts for label in labels]
    if prompt_seeds is None:
        params = SamplingParams(
            temperature=0.0, max_tokens=1, prompt_logprobs=0, seed=int(seed))
    else:
        scoring_seeds = [ps for ps in prompt_seeds for _label in labels]
        params = [SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0,
                                 seed=item_seed) for item_seed in scoring_seeds]
    # === the single forked-in change: optional LoRA routing ===
    lora_kwargs = getattr(backend, "_maybe_lora", dict)()
    outputs = engine.generate(scoring_texts, params, **lora_kwargs)
    # ===========================================================
    if len(outputs) != len(scoring_texts):
        raise ValueError(
            "teacher-forced engine returned an invalid output count: "
            f"observed={len(outputs)}, expected={len(scoring_texts)}")
    backend.stats.n_calls += 1
    backend.stats.n_prompts += len(scoring_texts)
    log_probabilities = []
    for output_index, output in enumerate(outputs):
        expected = runtime_token_ids[output_index % len(labels)]
        actual = int(output.prompt_token_ids[-1])
        if actual != expected:
            raise ValueError(
                f"continuation tokenization changed: expected {expected}, observed {actual}")
        prompt_logprobs = output.prompt_logprobs or []
        if not prompt_logprobs or not prompt_logprobs[-1]:
            raise ValueError("teacher-forced continuation logprob is missing")
        entry = prompt_logprobs[-1].get(actual)
        if entry is None:
            raise ValueError(
                f"teacher-forced logprob mapping omits actual continuation token {actual}")
        log_probabilities.append(float(entry.logprob))
    log_probabilities = np.asarray(log_probabilities, float).reshape(len(prompts), 2)
    log_odds = np.clip(
        log_probabilities[:, 0] - log_probabilities[:, 1], -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-log_odds))
