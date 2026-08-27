#!/usr/bin/env python3
"""RUNG 2, generation v2 — ADDENDUM D (frozen 2026-08-24): diverse pools.

3 generator families x 6 conditions x 1 sample per prompt, same 150
stable-hash prompts as v1. One FAMILY per process (wrapper loops); chunked +
resumable per family. Real stories are NOT handled here — the readout mixes
them in from the honest population.

Usage (sk3, one allowed GPU {4,5,6}):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 GPU_MEM_UTIL=0.30 \
    $HOME/envs/gemma4/bin/python rung2_generate_v2.py --family llama8b
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
OUT = REPO / "methods/taste_decomposition/rung2"
CACHE = "/lfs/skampere3/0/shared_hf_cache"
SEED = 20260824

FAMILIES = {
    "llama8b": (f"{CACHE}/models--meta-llama--Llama-3.1-8B-Instruct", {}),
    "qwen14b": (f"{CACHE}/models--Qwen--Qwen2.5-14B-Instruct", {}),
    # Mistral-Small-2501 ships no HF-fast tokenizer -> vLLM mistral mode
    "mistral24b": (f"{CACHE}/models--mistralai--Mistral-Small-24B-Instruct-2501",
                   {"tokenizer_mode": "mistral", "config_format": "mistral",
                    "load_format": "mistral"}),
    "phi4": (f"{CACHE}/models--microsoft--phi-4", {}),   # fallback third family
}

BASE = ("Write ONE complete short story responding to the writing prompt "
        "below. Write only the story text - no title, no preamble, no notes.")
CONDITIONS = {
    "plain":      (BASE, 1.0),
    "human":      ("You are a real human redditor on r/WritingPrompts - write like a "
                   "human, not like an AI. " + BASE, 1.0),
    "veryhuman":  ("You are a very human writer: idiosyncratic voice, small "
                   "imperfections, personal texture, the kind of story only a specific "
                   "person would tell. Do not write like an AI. " + BASE, 1.0),
    "casual":     ("You are a redditor dashing off a fun reply on your phone because "
                   "the prompt grabbed you. " + BASE, 1.0),
    "literary":   ("You are a practiced fiction writer attempting something polished "
                   "and ambitious for this prompt. " + BASE, 1.0),
    "hightemp":   (BASE, 1.3),
}


def snapshot(path):
    snaps = sorted((Path(path) / "snapshots").iterdir())
    assert snaps, f"no snapshot in {path}"
    return str(snaps[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--n-prompts", type=int, default=150)
    ap.add_argument("--prompt-list", default=None,
                    help="json with prompt_ids (E2 frame); default = v1 meta list")
    ap.add_argument("--out-tag", default="", help="suffix for output CSV")
    a = ap.parse_args()

    if a.prompt_list:
        picked = json.load(open(a.prompt_list))["prompt_ids"]
    else:
        meta = json.load(open(OUT / "rung2_candidates_cw_community_full.meta.json"))
        picked = meta["prompt_ids"][:a.n_prompts]
    pop = REPO / "methods/taste_decomposition/closure/cw_community/cw_honest_population.csv"
    prompt_txt = {}
    for r in csv.DictReader(open(pop)):
        prompt_txt.setdefault(r["prompt_id"], r["prompt"])
    assert all(p in prompt_txt for p in picked)

    tag = f"_{a.out_tag}" if a.out_tag else ""
    out_csv = OUT / f"rung2v2_candidates_{a.family}{tag}.csv"
    done = set()
    if out_csv.exists():
        done = {(r["prompt_id"], r["condition"]) for r in csv.DictReader(open(out_csv))}
        print(f"resume: {len(done)} rows exist", flush=True)

    jobs = [(p, c) for p in picked for c in CONDITIONS if (p, c) not in done]
    if not jobs:
        print("nothing to do"); return
    print(f"[{a.family}] {len(jobs)} generations", flush=True)

    from vllm import LLM, SamplingParams
    path, extra = FAMILIES[a.family]
    llm = LLM(model=snapshot(path), dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.30")),
              max_model_len=4096, enforce_eager=True,
              trust_remote_code=True, **extra)

    mode = "a" if done else "w"
    with open(out_csv, mode, newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["cand_id", "prompt_id", "family", "condition", "prompt",
                        "story", "n_chars"])
        CH = 300
        for c0 in range(0, len(jobs), CH):
            chunk = jobs[c0:c0 + CH]
            convs, sps = [], []
            for p, c in chunk:
                sysmsg, temp = CONDITIONS[c]
                seed = int(hashlib.md5(f"{SEED}:{a.family}:{c}:{p}".encode())
                           .hexdigest(), 16) % (2**31)
                convs.append([{"role": "system", "content": sysmsg},
                              {"role": "user", "content": prompt_txt[p]}])
                sps.append(SamplingParams(temperature=temp, top_p=0.95,
                                          max_tokens=1500, seed=seed))
            t0 = time.time()
            outs = llm.chat(convs, sps)
            for (p, c), o in zip(chunk, outs):
                txt = o.outputs[0].text.strip()
                w.writerow([f"{p}_{a.family}_{c}", p, a.family, c,
                            prompt_txt[p], txt, len(txt)])
            f.flush()
            print(f"chunk {c0//CH+1}: {len(chunk)} in {time.time()-t0:.0f}s",
                  flush=True)
    print(f"V2_GEN_DONE {a.family}", flush=True)


if __name__ == "__main__":
    main()
