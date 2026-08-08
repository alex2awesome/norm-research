#!/usr/bin/env python3
"""Standalone minimal repro/fix probe for the Magistral-Small-2509 vLLM 0.17 init hang.

Usage: python magfix_test.py <mode>
  mode = "mistral"      -> tokenizer_mode="mistral", config_format="mistral",
                            load_format="mistral"  (pure mistral-native single-file load;
                            avoids the ambiguity of BOTH consolidated.safetensors AND the
                            10-shard HF-format safetensors being present in the same dir)
  mode = "safetensors"  -> load_format="safetensors" (force HF sharded-format load,
                            default tokenizer_mode/config_format)
  mode = "eager"        -> vLLM defaults but enforce_eager=True (rules out a cudagraph-
                            capture/compile hang rather than a weight/tokenizer-loading hang)

GPU/env discipline (hard rules for this box): caller (bash wrapper) is responsible for
CUDA_DEVICE_ORDER=PCI_BUS_ID / CUDA_VISIBLE_DEVICES=6 / HOME / HF_HOME / HF_HUB_OFFLINE /
TMPDIR and for the pre-launch `nvidia-smi` free-memory gate. We ALSO defensively pin the
same env vars here (matching vllm_backend.OfflineVLLM._engine's own pattern of setting HOME
before importing vllm), so this script is safe to run standalone.
"""
from __future__ import annotations

import os
import sys
import time

MODEL_DIR = (
    "/lfs/skampere3/0/shared_hf_cache/models--mistralai--Magistral-Small-2509/"
    "snapshots/a31cc96ab10cf19bc42c628fedf1e359e0853c49"
)

# --- defensive env pinning (mirrors vllm_backend.OfflineVLLM._engine) ---------------------
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")  # reassigned from GPU6 -> GPU3 (coordinator,
                                                     # GPU6 permanently held by another session's
                                                     # engine, PID 2186213); wrapper still exports
                                                     # this explicitly on every launch
os.environ.setdefault("HOME", "/lfs/skampere3/0/alexspan")
os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/lfs/skampere3/0/alexspan/tmp")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("mistral", "safetensors", "eager"):
        print(f"usage: {sys.argv[0]} <mistral|safetensors|eager>", file=sys.stderr)
        return 2
    mode = sys.argv[1]

    log(f"mode={mode} model_dir={MODEL_DIR}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')} "
        f"HOME={os.environ.get('HOME')} HF_HOME={os.environ.get('HF_HOME')} "
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")

    if not os.path.isdir(MODEL_DIR):
        log(f"FATAL: model dir does not exist: {MODEL_DIR}")
        return 1

    log("importing vllm ...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    log(f"vllm imported in {time.time() - t0:.1f}s")

    kwargs = dict(
        model=MODEL_DIR,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        dtype="auto",
        trust_remote_code=True,
        enable_prefix_caching=True,
        logprobs_mode="processed_logprobs",
        tensor_parallel_size=1,
    )
    if mode == "mistral":
        kwargs.update(tokenizer_mode="mistral", config_format="mistral",
                      load_format="mistral")
    elif mode == "safetensors":
        kwargs.update(load_format="safetensors")
    elif mode == "eager":
        kwargs.update(enforce_eager=True)

    log(f"LLM(**kwargs) kwargs (minus model)={ {k: v for k, v in kwargs.items() if k != 'model'} }")
    t0 = time.time()
    try:
        eng = LLM(**kwargs)
    except Exception:
        log(f"LLM(**kwargs) RAISED after {time.time() - t0:.1f}s")
        raise
    log(f"LLM(**kwargs) SUCCEEDED in {time.time() - t0:.1f}s")

    prompts = [
        "Say the single word: hello",
        "What is 2 + 2? Answer with just the number.",
        "Name the capital of France in one word.",
    ]
    tok = eng.get_tokenizer()
    texts = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(s)

    sp = SamplingParams(temperature=0.0, max_tokens=32, seed=0)
    log("generate() sanity check on 3 prompts ...")
    t0 = time.time()
    outs = eng.generate(texts, sp)
    log(f"generate() done in {time.time() - t0:.1f}s")
    for p, o in zip(prompts, outs):
        text = o.outputs[0].text if o.outputs else ""
        log(f"PROMPT: {p!r}\n  -> {text!r}")

    log(f"RESULT mode={mode} STATUS=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
