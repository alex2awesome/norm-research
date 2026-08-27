#!/usr/bin/env python3
"""Fork B — Llama-70B consistency-ensemble runs for R1.

Runs N R1 passes on a single task with per-seed shuffling of the cluster
anchor order. vLLM is loaded ONCE; all N runs go through one batched chat().

Reuses sk3_build_r1.py's prompts/parsing/assembly — does NOT modify that
file (per user directive that Llama prompts must stay intact for
re-runnability on sk3).

Output:
  /lfs/skampere3/0/alexspan/norm_embed/match_out/fork_b/seed_<n>/
      r1_families_<task>.json
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm_embed")

import sk3_build_r1
from sk3_build_r1 import (FORMS, MATCH_OUT, MODEL_BASE, TASK_INFO,
                          assemble_per_task, build_messages, cluster_data,
                          load_task, make_batches)


TASK = os.environ.get("FORK_B_TASK", "peer-review")
N_SEEDS = int(os.environ.get("FORK_B_N_SEEDS", "5"))
BATCH_SIZE = int(os.environ.get("FORK_B_BATCH_SIZE", "40"))
MAX_MODEL_LEN = int(os.environ.get("FORK_B_MAX_MODEL_LEN", "16384"))
OUT_BASE = MATCH_OUT / "fork_b"
OUT_BASE.mkdir(exist_ok=True, parents=True)


def main():
    print(f"=== Fork B: task={TASK} N_seeds={N_SEEDS} bs={BATCH_SIZE} ===",
          flush=True)

    # Load embeddings + clusters once
    forms = [json.loads(l) for l in FORMS.open()
             if json.loads(l)["task"] == TASK]
    rows, emb = load_task(TASK, forms)
    if rows is None:
        print(f"ERR: no embeddings for {TASK}", flush=True)
        return
    cl = json.loads((MATCH_OUT / f"clusters_{TASK}.json").read_text())
    reps, centroids, members = cluster_data(rows, emb, cl)
    cids_canonical = list(reps.keys())
    print(f"loaded {len(cids_canonical)} clusters", flush=True)

    # Build per-seed batches (shuffled cids -> different anchors)
    all_jobs = []  # (seed, bi, batch, messages, pre_singletons)
    per_seed_pre_sing = {}
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        cids = list(cids_canonical)
        rng.shuffle(cids)
        batches, pre_sing = make_batches(cids, centroids, BATCH_SIZE)
        per_seed_pre_sing[seed] = pre_sing
        print(f"  seed {seed}: {len(batches)} batches  (pre_singletons={len(pre_sing)})",
              flush=True)
        for bi, batch in enumerate(batches):
            msgs = build_messages(TASK, batch, reps, members, step_b=False)
            all_jobs.append((seed, bi, batch, msgs))

    print(f"total jobs across {N_SEEDS} seeds: {len(all_jobs)}", flush=True)
    print(f"loading vLLM (Llama-70B-FP8 BF16-cast)...", flush=True)
    from vllm import LLM, SamplingParams
    model_dir = MODEL_BASE + "/" + sorted(os.listdir(MODEL_BASE))[0]
    llm = LLM(model=model_dir, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
              enforce_eager=False)
    sampling = SamplingParams(temperature=0.0, max_tokens=3500)
    print("submitting all jobs...", flush=True)
    outputs = llm.chat([m for _, _, _, m in all_jobs], sampling, use_tqdm=True)

    # Group outputs by seed
    by_seed = defaultdict(list)  # seed -> [(bi, batch, text)]
    for (seed, bi, batch, _), out in zip(all_jobs, outputs):
        by_seed[seed].append((bi, batch, out.outputs[0].text))

    # Assemble + save per-seed
    for seed in sorted(by_seed):
        items = by_seed[seed]
        pre_sing = per_seed_pre_sing[seed]
        all_fams, cluster_appearances = assemble_per_task(items, reps,
                                                          pre_sing)
        out_dir = OUT_BASE / f"seed_{seed}"
        out_dir.mkdir(exist_ok=True)
        families = []
        for fi, f in enumerate(all_fams):
            families.append({
                "family_id": fi,
                "name": f.get("name", ""),
                "description": f.get("description", ""),
                "cluster_ids": list(f.get("cluster_ids") or []),
            })
        out_path = out_dir / f"r1_families_{TASK}.json"
        out_path.write_text(json.dumps({
            "task": TASK,
            "seed": seed,
            "method": "fork_b_llama_consistency",
            "batch_size": BATCH_SIZE,
            "n_clusters": len(cids_canonical),
            "n_families": len(families),
            "families": families,
        }, indent=1))
        print(f"seed {seed}: {len(families)} families -> {out_path}",
              flush=True)

    print("=== Fork B DONE ===", flush=True)


if __name__ == "__main__":
    main()
