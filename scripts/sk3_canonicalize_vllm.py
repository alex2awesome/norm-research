"""Canonicalize leaf rubrics with Llama-3.3-70B-FP8 via vLLM (offline, on sk3).

Reuses the calibrated v5 prompt machinery from canonicalize_leaves.py. Run one
process per GPU and shard the leaves with --shard/--of. Co-locates with other
jobs at --gpu-mem-util 0.7.

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python sk3_canonicalize_vllm.py \\
    --input _sk3_leaf_input.jsonl --output canon_shard0.jsonl --shard 0 --of 4
"""
from __future__ import annotations

import os

# sk3: a nohup job loses its AFS token, so the AFS home (~) becomes unreadable
# and anything touching ~/.cache, ~/.triton etc. dies with PermissionError.
# Pin HOME and all caches to /lfs BEFORE importing anything that resolves "~".
os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/alexspan/hf_cache")
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
from canonicalize_leaves import build_system_prompt, salvage_json, PROMPT_VERSION

MODEL = "/lfs/skampere3/0/alexspan/merged_models/Llama-3.3-70B-FP8-with-tokenizer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: first N only")
    ap.add_argument("--gpu-mem-util", type=float, default=0.7)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    rows = [r for i, r in enumerate(rows) if i % args.of == args.shard]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[shard {args.shard}/{args.of}] {len(rows)} leaves | prompt={PROMPT_VERSION}",
          flush=True)

    from vllm import LLM, SamplingParams
    # FP8 weights are fine, but this checkpoint ships no q/k/v scaling factors,
    # so kv_cache_dtype="fp8" produces garbage ("!!!!" output). Use BF16 KV cache.
    llm = LLM(model=args.model, max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem_util,
              kv_cache_dtype="auto", dtype="auto",
              enforce_eager=args.enforce_eager,
              max_num_seqs=args.max_num_seqs, tensor_parallel_size=args.tp)
    sp = SamplingParams(temperature=0.0, max_tokens=1200)

    convos = [[{"role": "system", "content": build_system_prompt(r["task"])},
               {"role": "user", "content": f'RUBRIC: {r["name"]}'}] for r in rows]
    t0 = time.perf_counter()
    outs = llm.chat(convos, sp)
    dt = time.perf_counter() - t0

    results, ok = [], 0
    for r, o in zip(rows, outs):
        raw = o.outputs[0].text if o.outputs else ""
        parsed = salvage_json(raw)
        if parsed is not None and "canonical" in parsed:
            c = parsed["canonical"]
            ok += 1
            results.append({**r, "ok": True,
                            "canonical": c.strip() if isinstance(c, str) else None,
                            "reasoning": parsed.get("reasoning", ""),
                            "off_topic": bool(parsed.get("off_topic", False))})
        else:
            results.append({**r, "ok": False, "error": "json_salvage_failed",
                            "raw": raw})
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[shard {args.shard}] {ok}/{len(results)} ok in {dt:.0f}s "
          f"-> {args.output}", flush=True)


if __name__ == "__main__":
    main()
