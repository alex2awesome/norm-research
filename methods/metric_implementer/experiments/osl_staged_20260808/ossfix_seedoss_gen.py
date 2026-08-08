"""FAILURE 1 GPU diagnostic: ByteDance-Seed/Seed-OSS-36B-Instruct.

Standalone script (does NOT modify the repo). Protocol:
  1. hard-rule env vars set before any torch/vllm import
  2. GPU7 memory gate (<2000 MiB) enforced right before engine construction
  3. load model standalone via vllm.LLM, apply chat template to 5 test prompts, generate 64
     tokens, print raw output -> explains WHY first-token logprob never lands on YES/NO
  4. import methods.metric_implementer.vllm_backend, build an OfflineVLLM via
     make_judge_backend on the SAME resident engine (engine cache is keyed by model path, so
     this reuses the already-loaded weights) and call score_binary_gen on 20 prompts -> measured
     parse (non-nan) rate for the EXISTING implementation
  5. supplementary probe (clearly labeled): what apply_chat_template(thinking_budget=0) would
     look like piped through the same generate+parse logic, to support a concrete fix
     recommendation without editing the backend
"""
import os
import re
import sys
import time

# ---- hard rules: set before importing torch/vllm ----
# NOTE: coordinator redirected physical GPU7 -> GPU5 mid-task (2026-08-07): GPU7 is pinned by
# another session's train_grl.py and will not free soon. GPU5 is this task's target now.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ["HF_HOME"] = "/lfs/skampere3/0/shared_hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TMPDIR"] = "/lfs/skampere3/0/alexspan/tmp"
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, "/lfs/skampere3/0/alexspan/outputs/osl_multi")

from ossfix_prompts import FIVE_TEST_PROMPTS, TWENTY_PROMPTS  # noqa: E402

MODEL = "ByteDance-Seed/Seed-OSS-36B-Instruct"


def gpu_mem_used_mib(gpu_index=5):
    import subprocess
    out = subprocess.check_output([
        "nvidia-smi", f"--query-gpu=memory.used", "--format=csv,noheader,nounits",
        "-i", str(gpu_index)]).decode().strip()
    return int(out.splitlines()[0].strip())


def require_gpu_free(gpu_index=5, threshold_mib=2000):
    used = gpu_mem_used_mib(gpu_index)
    print(f"[gate] GPU{gpu_index} memory.used = {used} MiB (threshold {threshold_mib} MiB)")
    if used >= threshold_mib:
        print(f"[gate] ABORT: GPU{gpu_index} not free, refusing to launch engine.")
        sys.exit(3)


def section(t):
    print("\n" + "=" * 20 + f" {t} " + "=" * 20, flush=True)


def main():
    require_gpu_free()

    from vllm import LLM, SamplingParams

    section("loading Seed-OSS-36B-Instruct (bf16, tp=1, GPU7)")
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="auto",
        trust_remote_code=True,
        enable_prefix_caching=True,
        logprobs_mode="processed_logprobs",
        tensor_parallel_size=1,
    )
    print(f"[load] engine ready in {time.time()-t0:.1f}s")
    tok = llm.get_tokenizer()

    # ---- Step A: raw 64-token generations with the backend's CURRENT template call
    # (enable_thinking=False, which we already showed via CPU diag is a no-op for this model:
    # thinking_budget defaults to -1 = UNBOUNDED, so the model is free to open <seed:think>
    # with no cap) ----
    section("STEP A: raw 64-token generations, enable_thinking=False (== current battery call)")
    texts = []
    for p in FIVE_TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(s)
    sp64 = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5)
    outs = llm.generate(texts, sp64)
    for i, o in enumerate(outs):
        gen = o.outputs[0].text if o.outputs else ""
        first_lp = (o.outputs[0].logprobs[0] if o.outputs and o.outputs[0].logprobs else {}) or {}
        top_toks = sorted(
            [(getattr(L, "decoded_token", ""), L.logprob) for L in first_lp.values()],
            key=lambda x: -x[1])[:5]
        print(f"\n--- prompt {i} ---")
        print(f"first-token top-5 logprobs: {top_toks}")
        print(f"raw 64-token generation: {gen!r}")

    # ---- Step B: the EXISTING score_binary_gen, via the actual repo backend ----
    section("STEP B: methods.metric_implementer.vllm_backend.make_judge_backend + "
            "score_binary_gen on 20 prompts (default kwargs: thinking=False, max_gen_tokens=1024)")
    from methods.metric_implementer import vllm_backend as vb

    class Cfg:
        vllm_gpu_mem_util = 0.85
        vllm_max_model_len = 4096
        vllm_dtype = "auto"
        vllm_tp_size = 1
        vllm_fake = False
        vllm_lfs_home = "/lfs/skampere3/0/alexspan"
        other_temperature = 0.0
        max_retries = 1

    # seed the engine cache with the engine we already built above so make_judge_backend's
    # OfflineVLLM._engine() reuses it instead of constructing a second LLM() (would OOM/hang).
    # cache key = the raw model string passed to OfflineVLLM (== MODEL here, no lora), matching
    # _engine()'s `cache_key = model` when no lora_path is set -- NOT the resolved snapshot path.
    vb._ENGINE_CACHE[MODEL] = llm

    backend = vb.make_judge_backend(MODEL, Cfg())
    t0 = time.time()
    scores, raws = backend.score_binary_gen(
        TWENTY_PROMPTS, thinking=False, max_gen_tokens=1024, return_texts=True)
    dt = time.time() - t0
    n_nan = sum(1 for s in scores if s != s)  # nan != nan
    print(f"[score_binary_gen] {len(TWENTY_PROMPTS)} prompts in {dt:.1f}s, "
          f"nan_rate={n_nan/len(scores):.3f} ({n_nan}/{len(scores)})")
    for i, (s, raw) in enumerate(zip(scores, raws)):
        has_close = "</seed:think>" in raw
        print(f"\n--- prompt {i} score={s} has_</seed:think>_tag={has_close} "
              f"gen_len_chars={len(raw)} ---")
        print(f"raw[:300]={raw[:300]!r}")
        print(f"raw[-300:]={raw[-300:]!r}")

    # ---- Step C: supplementary probe -- what if we pass thinking_budget=0 (Seed-OSS's REAL
    # toggle, not exposed by score_binary_gen) and apply the SAME post-</think>-tag parse logic
    # that score_binary_gen uses, but corrected to split on '</seed:think>' instead of
    # '</think>'? This does NOT touch the repo -- it is a manual reimplementation here only,
    # to determine the exact fix. ----
    section("STEP C (probe, not the real backend): thinking_budget=0 + '</seed:think>' split")
    texts_budget0 = []
    for p in TWENTY_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     thinking_budget=0)
        texts_budget0.append(s)
    sp = SamplingParams(temperature=0.0, max_tokens=256, seed=0)
    outs2 = llm.generate(texts_budget0, sp)
    P, N = "YES", "NO"
    n_nan2 = 0
    for i, o in enumerate(outs2):
        txt = (o.outputs[0].text if o.outputs else "") or ""
        # corrected split: real closing tag is </seed:think>, not </think>
        tail = txt.rsplit("</seed:think>", 1)[-1]
        toks = re.findall(r"[A-Z]+", tail.upper())
        verdict = next((t for t in toks if t in (P, N)), None)
        if verdict is None:
            n_nan2 += 1
        if i < 5:
            print(f"\n--- budget0 prompt {i} verdict={verdict} ---")
            print(f"raw[:300]={txt[:300]!r}")
    print(f"\n[probe: thinking_budget=0 + </seed:think> split] "
          f"nan_rate={n_nan2/len(outs2):.3f} ({n_nan2}/{len(outs2)})")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
