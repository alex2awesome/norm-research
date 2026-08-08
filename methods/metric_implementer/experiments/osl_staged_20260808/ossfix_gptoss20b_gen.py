"""FAILURE 2 GPU diagnostic: openai/gpt-oss-20b (Harmony channel format).

Standalone script (does NOT modify the repo). Protocol:
  1. hard-rule env vars set before any torch/vllm import
  2. GPU7 memory gate (<2000 MiB) enforced right before engine construction
  3. load model standalone via vllm.LLM, apply chat template to 5 test prompts, generate 64
     tokens, print raw output twice: once with vLLM's DEFAULT skip_special_tokens (True) and
     once with skip_special_tokens=False, to show exactly what the Harmony channel markup
     looks like / whether it survives into `.text`
  4. import methods.metric_implementer.vllm_backend, build an OfflineVLLM via
     make_judge_backend on the SAME resident engine and call the EXISTING score_binary_gen on
     20 prompts -> measured parse (non-nan) rate for the implementation AS SHIPPED
  5. supplementary probe (clearly labeled, not the real backend): manual generate with
     skip_special_tokens=False + a parse rule keyed on the actual Harmony delimiter
     '<|channel|>final<|message|>' ... '<|return|>' / '<|end|>' / '<|call|>'
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

MODEL = "openai/gpt-oss-20b"


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
    require_gpu_free()

    from vllm import LLM, SamplingParams

    section("loading gpt-oss-20b (mxfp4, tp=1, GPU7)")
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

    # ---- Step A: raw 64-token gen, vLLM DEFAULT skip_special_tokens (== what score_binary_gen
    # currently gets, since it never overrides SamplingParams.skip_special_tokens) ----
    section("STEP A: raw 64-token gen, DEFAULT skip_special_tokens (score_binary_gen's actual "
            "SamplingParams)")
    sp_default = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5)
    print(f"SamplingParams default skip_special_tokens = {sp_default.skip_special_tokens}")
    outs = llm.generate(texts, sp_default)
    for i, o in enumerate(outs):
        gen = o.outputs[0].text if o.outputs else ""
        first_lp = (o.outputs[0].logprobs[0] if o.outputs and o.outputs[0].logprobs else {}) or {}
        top_toks = sorted(
            [(getattr(L, "decoded_token", ""), L.logprob) for L in first_lp.values()],
            key=lambda x: -x[1])[:5]
        print(f"\n--- prompt {i} (skip_special_tokens=True/default) ---")
        print(f"first-token top-5 logprobs: {top_toks}")
        print(f"raw 64-token generation: {gen!r}")

    # ---- Step B: SAME prompts, skip_special_tokens=False, to reveal the real Harmony markup ----
    section("STEP B: raw 64-token gen, skip_special_tokens=False (reveals Harmony structure)")
    sp_raw = SamplingParams(temperature=0.0, max_tokens=64, skip_special_tokens=False)
    outs_raw = llm.generate(texts, sp_raw)
    for i, o in enumerate(outs_raw):
        gen = o.outputs[0].text if o.outputs else ""
        print(f"\n--- prompt {i} (skip_special_tokens=False) ---")
        print(f"raw 64-token generation: {gen!r}")

    # ---- Step C: the EXISTING score_binary_gen, via the actual repo backend ----
    section("STEP C: methods.metric_implementer.vllm_backend.make_judge_backend + "
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

    vb._ENGINE_CACHE[MODEL] = llm  # reuse the already-loaded engine; cache key == raw model str

    backend = vb.make_judge_backend(MODEL, Cfg())
    t0 = time.time()
    scores, raws = backend.score_binary_gen(
        TWENTY_PROMPTS, thinking=False, max_gen_tokens=1024, return_texts=True)
    dt = time.time() - t0
    n_nan = sum(1 for s in scores if s != s)
    print(f"[score_binary_gen] {len(TWENTY_PROMPTS)} prompts in {dt:.1f}s, "
          f"nan_rate={n_nan/len(scores):.3f} ({n_nan}/{len(scores)})")
    for i, (s, raw) in enumerate(zip(scores, raws)):
        print(f"\n--- prompt {i} score={s} gen_len_chars={len(raw)} ---")
        print(f"raw[:200]={raw[:200]!r}")
        print(f"raw[-200:]={raw[-200:]!r}")

    # ---- Step D: supplementary probe (not the real backend) -- generate with
    # skip_special_tokens=False and parse on the REAL Harmony delimiter:
    # '<|channel|>final<|message|>' ... up to the first of '<|return|>' / '<|end|>' / '<|call|>'
    section("STEP D (probe): skip_special_tokens=False + '<|channel|>final<|message|>' split")
    sp = SamplingParams(temperature=0.0, max_tokens=512, seed=0, skip_special_tokens=False)
    texts20 = []
    for p in TWENTY_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts20.append(s)
    outs2 = llm.generate(texts20, sp)
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
            final = txt  # no final channel emitted (e.g. truncated in analysis)
        toks = re.findall(r"[A-Z]+", final.upper())
        verdict = next((t for t in toks if t in (P, N)), None)
        if verdict is None:
            n_nan2 += 1
        if i < 5:
            print(f"\n--- probe prompt {i} verdict={verdict} has_final_channel="
                  f"{FINAL_MARK in txt} ---")
            print(f"raw[:250]={txt[:250]!r}")
            print(f"raw[-250:]={txt[-250:]!r}")
    print(f"\n[probe: skip_special_tokens=False + final-channel split] "
          f"nan_rate={n_nan2/len(outs2):.3f} ({n_nan2}/{len(outs2)})")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
