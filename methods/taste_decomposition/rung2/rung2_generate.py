#!/usr/bin/env python3
"""RUNG 2, stage G — candidate generation (design frozen 2026-08-21:
notes/2026-08-21__rung12_design_gap_consequences.md §2.3).

Per cell: draw N held-out prompts from the cell's certified honest frame by
STABLE HASH over the prompt/group id (never a seeded shuffle of a growing
list), generate K candidates per prompt with ONE mid-tier open generator,
and write a candidates CSV shaped like the cell's ext-scoring input so the
frozen bank scorer (stage0_score_ext_gemma pattern) and the dense arbiter
consume it unchanged.

Generator (recorded per design §2.3 at smoke time): Gemma-4-31b-it, the
model already cached on sk3. The generator shares a family with the bank
judge; per design §2.6 this cancels in first order because both selection
policies choose from the SAME candidate pool.

Smoke mode: --n-prompts 10 --k 4, plumbing only (never read smoke direction).

Run on sk3, ONE GPU from the allowed set (3+7 banned through ~08-24):
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=4 GPU_MEM_UTIL=0.37 \
    $HOME/envs/gemma4/bin/python rung2_generate.py --cell cw_community \
    --n-prompts 10 --k 4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
GEMMA4 = os.environ.get(
    "GEMMA4_PATH",
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb",
)
OUT = REPO / "methods/taste_decomposition/rung2"
SEED = 20260821

# cell -> (population csv, prompt-id col, prompt col, real-item col, gen instruction, max_new_tokens)
CELLS = {
    "cw_community": dict(
        pop=REPO / "methods/taste_decomposition/closure/cw_community/cw_honest_population.csv",
        gid="prompt_id", prompt="prompt", real="story",
        sys=("You are a skilled fiction writer on r/WritingPrompts. Write ONE complete "
             "short story responding to the writing prompt below. Write only the story "
             "text - no title, no preamble, no notes."),
        # 1500 tokens ~= 6.5k chars: spans the real story range (p10/p50/p90 =
        # 1160/3078/6206 chars) so length isn't mechanically capped below the
        # population's upper tail. Smoke ran at 900; raised before the full run.
        max_new=1500,
    ),
}


def stable_pick(ids, n):
    """Deterministic prompt selection: sort by sha256(id), take first n."""
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--n-prompts", type=int, default=150)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = CELLS[args.cell]

    rows = list(csv.DictReader(open(cfg["pop"])))
    by_gid = {}
    for r in rows:
        by_gid.setdefault(r[cfg["gid"]], r)
    picked = stable_pick(list(by_gid), args.n_prompts)
    print(f"[{args.cell}] {len(by_gid)} prompts in frame -> {len(picked)} picked "
          f"(stable-hash), K={args.k}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.49")),
              max_model_len=4096, enforce_eager=True,
              # text-only task: without this, vLLM profiles the multimodal
              # encoder cache (6 max-size VIDEO items) and starves the KV cache
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=cfg["max_new"],
                        n=args.k, seed=SEED)

    tag = args.tag or ("smoke" if args.n_prompts <= 20 else "full")
    out_csv = OUT / f"rung2_candidates_{args.cell}_{tag}.csv"
    done = set()
    if out_csv.exists():                      # resume: skip prompts already on disk
        done = {r["prompt_id"] for r in csv.DictReader(open(out_csv))}
        print(f"resume: {len(done)} prompts already written", flush=True)
    todo = [g for g in picked if g not in done]

    n_short, CHUNK = 0, 50
    mode = "a" if done else "w"
    with open(out_csv, mode, newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["cand_id", "prompt_id", "prompt", "story", "k_index", "n_chars"])
        for c0 in range(0, len(todo), CHUNK):   # chunked, flushed per batch rule
            chunk = todo[c0:c0 + CHUNK]
            convs = [[{"role": "system", "content": cfg["sys"]},
                      {"role": "user", "content": by_gid[g][cfg["prompt"]]}]
                     for g in chunk]
            t0 = time.time()
            outs = llm.chat(convs, sp)
            for g, o in zip(chunk, outs):
                for k, comp in enumerate(o.outputs):
                    txt = comp.text.strip()
                    if len(txt) < 200:
                        n_short += 1
                    w.writerow([f"{g}_gen{k:02d}", g, by_gid[g][cfg["prompt"]], txt,
                                k, len(txt)])
            f.flush()
            print(f"chunk {c0//CHUNK + 1}: {len(chunk)} prompts in "
                  f"{time.time()-t0:.0f}s", flush=True)
    meta = dict(cell=args.cell, tag=tag, generator=GEMMA4, seed=SEED,
                n_prompts=len(picked), k=args.k, temperature=1.0, top_p=0.95,
                max_new_tokens=cfg["max_new"], n_short_lt200=n_short,
                design="notes/2026-08-21__rung12_design_gap_consequences.md §2.3",
                prompt_ids=picked)
    (OUT / f"rung2_candidates_{args.cell}_{tag}.meta.json").write_text(
        json.dumps(meta, indent=2))
    print(f"wrote {out_csv} ({len(picked)}x{args.k} candidates, {n_short} short)",
          flush=True)


if __name__ == "__main__":
    main()
