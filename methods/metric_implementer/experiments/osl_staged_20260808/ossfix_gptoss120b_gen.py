"""gpt-oss-120b fit + load check (coordinator addendum, 2026-08-07).

CPU-side finding already established (see logs/ossfix_120b_cpu_check.log):
  - shared_hf_cache/models--openai--gpt-oss-120b (top-level, NO hub/ prefix) is the
    STALE/INCOMPLETE download: only 8 metadata files, 0 weight shards, 8 .incomplete blobs.
  - shared_hf_cache/hub/models--openai--gpt-oss-120b (hub/-prefixed) is COMPLETE: 15/15
    safetensors shards present, model.safetensors.index.json declares total_size =
    65,248,815,744 bytes (~65.25 GB) and the sum of on-disk shard sizes matches (65.25 GB) ==
    all shards intact, none missing/truncated. tokenizer.json/tokenizer_config.json/
    special_tokens_map.json all present. mxfp4 quantization (same quant_method as gpt-oss-20b;
    self_attn/router/embed/lm_head stay unquantized). The `du -shL` 183GB figure on the hub/
    snapshot INCLUDES two alternate checkpoint copies vLLM never touches: original/ (61GB, raw
    OpenAI checkpoint format) and metal/ (61GB, Apple Metal on-device quant) -- neither is
    loaded by transformers/vLLM's from_pretrained.
  - methods.metric_implementer.vllm_backend._resolve_model_path("openai/gpt-oss-120b") already
    resolves to the COMPLETE hub/ copy (its root-search order checks HF_HOME/hub/ BEFORE bare
    HF_HOME, and hub/ now has a valid refs/main+snapshot) -- no repo change needed for this.
  - 65.25 GB of weights fits a single 183GB B200 with ~118GB headroom for KV cache/activations/
    vLLM overhead -- well under the coordinator's ~150GB "needs TP=2" threshold. PLAUSIBLY FITS
    -> proceed with the load test below.

This script: ONE timeout-1800 load test + a 5-prompt Harmony-parse check, same rules as the
other two ossfix_* scripts (GPU5, env vars, <2000MiB gate). Runs LAST, after seed-oss and
gpt-oss-20b. Standalone, does not modify the repo.
"""
import os
import re
import sys
import time

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

from ossfix_prompts import FIVE_TEST_PROMPTS  # noqa: E402

MODEL = "openai/gpt-oss-120b"


def gpu_mem_used_mib(gpu_index=5):
    import subprocess
    out = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
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
    from methods.metric_implementer.vllm_backend import _resolve_model_path
    resolved = _resolve_model_path(MODEL)
    print(f"[resolve] {MODEL} -> {resolved}")
    assert "hub/models--openai--gpt-oss-120b" in resolved, (
        f"resolution did NOT pick the complete hub/ copy: {resolved}")

    require_gpu_free()

    from vllm import LLM, SamplingParams

    section("loading gpt-oss-120b (mxfp4, ~65GB weights, tp=1, GPU5) -- LOAD TEST")
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

    texts = []
    for p in FIVE_TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(s)

    section("STEP A: raw 64-token gen, DEFAULT skip_special_tokens (matches score_binary_gen)")
    sp_default = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5)
    outs = llm.generate(texts, sp_default)
    for i, o in enumerate(outs):
        gen = o.outputs[0].text if o.outputs else ""
        print(f"\n--- prompt {i} (skip_special_tokens=True/default) ---")
        print(f"raw 64-token generation: {gen!r}")

    section("STEP B: same prompts, skip_special_tokens=False (reveals Harmony structure)")
    sp_raw = SamplingParams(temperature=0.0, max_tokens=64, skip_special_tokens=False)
    outs_raw = llm.generate(texts, sp_raw)
    for i, o in enumerate(outs_raw):
        gen = o.outputs[0].text if o.outputs else ""
        print(f"\n--- prompt {i} (skip_special_tokens=False) ---")
        print(f"raw 64-token generation: {gen!r}")

    section("STEP C: methods.metric_implementer.vllm_backend score_binary_gen, 5 prompts "
            "(default kwargs, matches gpt-oss-20b test)")
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

    vb._ENGINE_CACHE[MODEL] = llm

    backend = vb.make_judge_backend(MODEL, Cfg())
    t0 = time.time()
    scores, raws = backend.score_binary_gen(
        FIVE_TEST_PROMPTS, thinking=False, max_gen_tokens=1024, return_texts=True)
    dt = time.time() - t0
    n_nan = sum(1 for s in scores if s != s)
    print(f"[score_binary_gen] {len(FIVE_TEST_PROMPTS)} prompts in {dt:.1f}s, "
          f"nan_rate={n_nan/len(scores):.3f} ({n_nan}/{len(scores)})")
    for i, (s, raw) in enumerate(zip(scores, raws)):
        print(f"\n--- prompt {i} score={s} gen_len_chars={len(raw)} ---")
        print(f"raw[:200]={raw[:200]!r}")
        print(f"raw[-200:]={raw[-200:]!r}")

    section("STEP D (probe): skip_special_tokens=False + '<|channel|>final<|message|>' split")
    sp = SamplingParams(temperature=0.0, max_tokens=512, seed=0, skip_special_tokens=False)
    texts5 = []
    for p in FIVE_TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts5.append(s)
    outs2 = llm.generate(texts5, sp)
    P, N = "YES", "NO"
    n_nan2 = 0
    for i, o in enumerate(outs2):
        txt = (o.outputs[0].text if o.outputs else "") or ""
        FINAL_MARK = "<|channel|>final<|message|>"
        if FINAL_MARK in txt:
            final = txt.rsplit(FINAL_MARK, 1)[-1]
            for stop in ("<|return|>", "<|end|>", "<|call|>"):
                final = final.split(stop, 1)[0]
        else:
            final = txt
        toks = re.findall(r"[A-Z]+", final.upper())
        verdict = next((t for t in toks if t in (P, N)), None)
        if verdict is None:
            n_nan2 += 1
        print(f"\n--- probe prompt {i} verdict={verdict} has_final_channel={FINAL_MARK in txt} ---")
        print(f"raw[:250]={txt[:250]!r}")
        print(f"raw[-250:]={txt[-250:]!r}")
    print(f"\n[probe: skip_special_tokens=False + final-channel split] "
          f"nan_rate={n_nan2/len(outs2):.3f} ({n_nan2}/{len(outs2)})")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
