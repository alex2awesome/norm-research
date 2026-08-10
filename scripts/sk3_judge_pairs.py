"""LLM same/different judge for rubric pairs, via vLLM on sk3 (BF16 Llama-70B).

Scores each pair in judge_pool.jsonl on the calibrated 0/1/2 sameness scale
(judge_prompt.py): 2=same criterion, 1=related but different, 0=unrelated.
One pass; reused at every tau in the FP/FN sweep.

Usage:
  CUDA_VISIBLE_DEVICES=N python sk3_judge_pairs.py --input judge_pool.jsonl \\
    --output judge_verdicts.jsonl --model <bf16-llama-path>
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ["HF_HOME"] = "/lfs/skampere3/0/alexspan/hf_cache"
os.environ["HF_MODULES_CACHE"] = "/lfs/skampere3/0/alexspan/hf_cache/modules"
os.environ.setdefault("VLLM_CACHE_ROOT", "/lfs/skampere3/0/alexspan/.cache/vllm")
os.environ.setdefault("TRITON_CACHE_DIR", "/lfs/skampere3/0/alexspan/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/lfs/skampere3/0/alexspan/.cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompt import SYSTEM, build_user, salvage, JUDGE_VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    print(f"judging {len(rows)} pairs | prompt={JUDGE_VERSION}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem_util, kv_cache_dtype="auto",
              dtype="auto", tensor_parallel_size=args.tp, max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=400)

    convos = [[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": build_user(
                   r["task"], r["canonical_a"], r["canonical_b"])}]
              for r in rows]
    t0 = time.perf_counter()
    outs = llm.chat(convos, sp)
    dt = time.perf_counter() - t0

    ok = 0
    with open(args.output, "w") as f:
        for r, o in zip(rows, outs):
            raw = o.outputs[0].text if o.outputs else ""
            p = salvage(raw)
            score = None
            if p is not None and "score" in p:
                try:
                    score = int(p["score"])
                except Exception:
                    score = None
            if score in (0, 1, 2):
                ok += 1
                f.write(json.dumps({**r, "score": score,
                                    "judge_reasoning": p.get("reasoning", "")}) + "\n")
            else:
                f.write(json.dumps({**r, "score": None, "raw": raw}) + "\n")
    print(f"{ok}/{len(rows)} judged ok in {dt:.0f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
