#!/usr/bin/env python3
"""Fork B v2 — exercise the right variance sources.

User insight: Fork B v1 only varied which cluster gets to be the FIRST anchor
(random shuffle of cids). That changes anchor identity but not LoRA-bge
neighborhoods, so 5 seeds gave only ~2% variance. The real variance sources
are (a) different anchor PERSPECTIVES and (b) LLM temperature.

This script runs a 2×3 grid:
  anchors ∈ {farthest-point-0, farthest-point-1}  (2 maximally-different seeds)
  temperature ∈ {0.0, 0.5, 0.7}

= 6 runs. vLLM loaded once. ~6 × 5 min inference = ~30 min after model load.

Farthest-point selection picks two starting anchors that are maximally far in
LoRA-bge centroid space, so each anchor pulls in a different "region" of the
cluster space first. The cover-once propagation from each anchor produces a
different batch order, which is structurally different (not just a permutation
of the baseline order).

Output:
  /lfs/skampere3/0/alexspan/norm_embed/match_out/fork_b_v2/
      anchor{i}_temp{t}/r1_families_<task>.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm_embed")
import sk3_build_r1
from sk3_build_r1 import (FORMS, MATCH_OUT, MODEL_BASE, TASK_INFO,
                          assemble_per_task, build_messages, cluster_data,
                          load_task, make_batches)


TASK = os.environ.get("FORK_B_V2_TASK", "peer-review")
BATCH_SIZE = int(os.environ.get("FORK_B_V2_BATCH_SIZE", "40"))
MAX_MODEL_LEN = int(os.environ.get("FORK_B_V2_MAX_MODEL_LEN", "16384"))
N_ANCHORS = int(os.environ.get("FORK_B_V2_N_ANCHORS", "2"))
TEMPERATURES = [float(t) for t in
                os.environ.get("FORK_B_V2_TEMPS", "0.0,0.5,0.7").split(",")]
OUT_BASE = MATCH_OUT / "fork_b_v2"
OUT_BASE.mkdir(exist_ok=True, parents=True)


def farthest_point_anchors(centroids_dict, n_anchors):
    """Pick n_anchors cluster ids via farthest-point sampling on centroids."""
    cids = list(centroids_dict.keys())
    vecs = np.stack([centroids_dict[c] for c in cids])
    # Seed with the cluster closest to the global mean
    mean = vecs.mean(0)
    mean /= (np.linalg.norm(mean) + 1e-9)
    dists_to_mean = 1.0 - vecs @ mean
    first_idx = int(np.argmin(dists_to_mean))
    chosen_idx = [first_idx]
    while len(chosen_idx) < n_anchors:
        chosen_vecs = vecs[chosen_idx]
        sims = vecs @ chosen_vecs.T  # (N, k)
        max_sim_to_chosen = sims.max(axis=1)
        dist_to_nearest_chosen = 1.0 - max_sim_to_chosen
        # Exclude already chosen
        for ci in chosen_idx:
            dist_to_nearest_chosen[ci] = -1.0
        next_idx = int(np.argmax(dist_to_nearest_chosen))
        chosen_idx.append(next_idx)
    return [cids[i] for i in chosen_idx]


def main():
    print(f"=== Fork B v2: task={TASK} bs={BATCH_SIZE} ===", flush=True)
    print(f"    n_anchors={N_ANCHORS}, temperatures={TEMPERATURES}", flush=True)

    forms = [json.loads(l) for l in FORMS.open()
             if json.loads(l)["task"] == TASK]
    rows, emb = load_task(TASK, forms)
    cl = json.loads((MATCH_OUT / f"clusters_{TASK}.json").read_text())
    reps, centroids, members = cluster_data(rows, emb, cl)
    cids_canonical = list(reps.keys())
    print(f"loaded {len(cids_canonical)} L0 clusters", flush=True)

    # Farthest-point anchors
    anchors = farthest_point_anchors(centroids, N_ANCHORS)
    print(f"farthest-point anchors: {anchors}", flush=True)

    # Build per-anchor batches (each anchor forces its own first batch)
    per_anchor_batches = {}
    per_anchor_pre_sing = {}
    for ai, anchor_cid in enumerate(anchors):
        # Force anchor_cid to be the FIRST cid in the list -> first batch
        cids = [anchor_cid] + [c for c in cids_canonical if c != anchor_cid]
        batches, pre_sing = make_batches(cids, centroids, BATCH_SIZE)
        per_anchor_batches[ai] = batches
        per_anchor_pre_sing[ai] = pre_sing
        print(f"  anchor {ai} (cid={anchor_cid}): {len(batches)} batches",
              flush=True)

    # Build job list: (anchor_idx, temp, batch_idx, batch, messages)
    all_jobs = []  # later assigned to specific (anchor, temp) bucket
    for ai in range(N_ANCHORS):
        for temp in TEMPERATURES:
            for bi, batch in enumerate(per_anchor_batches[ai]):
                msgs = build_messages(TASK, batch, reps, members,
                                      step_b=False)
                all_jobs.append((ai, temp, bi, batch, msgs))
    print(f"total jobs across {N_ANCHORS} anchors × {len(TEMPERATURES)} temps: "
          f"{len(all_jobs)}", flush=True)

    # vLLM loaded once
    print("loading vLLM (Llama-70B)...", flush=True)
    from vllm import LLM, SamplingParams
    model_dir = MODEL_BASE + "/" + sorted(os.listdir(MODEL_BASE))[0]
    llm = LLM(model=model_dir, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=MAX_MODEL_LEN,
              enforce_eager=False)

    # Submit per temperature (vLLM requires same sampling params across batch)
    by_run = defaultdict(list)  # (anchor_idx, temp) -> [(bi, batch, text)]
    for temp in TEMPERATURES:
        jobs_for_temp = [(ai, tp, bi, batch, msgs)
                         for ai, tp, bi, batch, msgs in all_jobs
                         if tp == temp]
        print(f"\n--- temp={temp}: {len(jobs_for_temp)} jobs ---", flush=True)
        sampling = SamplingParams(temperature=temp, max_tokens=3500,
                                  seed=42 if temp > 0 else None)
        outputs = llm.chat([m for _, _, _, _, m in jobs_for_temp],
                           sampling, use_tqdm=True)
        for (ai, tp, bi, batch, _), out in zip(jobs_for_temp, outputs):
            by_run[(ai, tp)].append((bi, batch, out.outputs[0].text))

    # Per-run assemble + save
    for (ai, temp), items in by_run.items():
        items.sort(key=lambda x: x[0])
        pre_sing = per_anchor_pre_sing[ai]
        all_fams, _cluster_app = assemble_per_task(items, reps, pre_sing)
        out_dir = OUT_BASE / f"anchor{ai}_temp{temp}"
        out_dir.mkdir(exist_ok=True, parents=True)
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
            "anchor_idx": ai,
            "anchor_cid": anchors[ai],
            "temperature": temp,
            "batch_size": BATCH_SIZE,
            "n_clusters": len(cids_canonical),
            "n_families": len(families),
            "families": families,
        }, indent=1))
        print(f"  anchor{ai} temp={temp}: {len(families)} families -> {out_path}",
              flush=True)

    print("=== Fork B v2 DONE ===", flush=True)


if __name__ == "__main__":
    main()
