#!/usr/bin/env python3
"""GEPA Stage 2: score the K=3 rephrasing variants per targeted criterion on a
bounded, deterministic FIT+MINE probe subset (never MONITOR/TEST), Gemma-4-31B
offline-batch vLLM.  Label-blind: `judgement` is never loaded into a prompt.

Outputs gepa_probe_scores.npz with the same {X, cids} shape gepa_phrasing.py's
`select` subcommand expects.

Run on sk3, ONE GPU:
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=<n> $HOME/envs/gemma4/bin/python gepa_probe_score.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
HERE = REPO / "methods/taste_decomposition/closure/cw_community"
GEMMA4 = os.environ.get(
    "GEMMA4_PATH",
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb",
)
PROBE_N = 600
SEED = 20260809

# import the exact prompt-construction / truncation / token-parsing logic used by
# the round scorer, so the probe scores live on the identical instrument.
import score_round_gemma as S  # noqa: E402


def main():
    import pandas as pd

    pop = pd.read_csv(HERE / "cw_population_with_splits.csv").fillna(
        {"prompt": "", "story": ""})
    fm = pop[pop.split == "fit_mine"].reset_index(drop=True)

    # stable-hash probe sample of FIT+MINE only (never MONITOR/TEST), deterministic
    def h(x):
        return int(hashlib.sha256(f"gepa-probe|{x}".encode()).hexdigest(), 16) / (1 << 256)
    fm["_h"] = fm["prompt_id"].astype(str).map(h)
    probe = fm.sort_values("_h").head(PROBE_N).reset_index(drop=True)

    variants = json.loads((HERE / "gepa_variants.json").read_text())
    blocks = [f"CRITERION: {v['name']}\nDESCRIPTION: {v['instruction']}\n\n"
              f"Answer with one token:" for v in variants]
    k = len(blocks)
    print(f"[gepa-probe] rows={len(probe)} variants={k} prompts={len(probe) * k}",
          flush=True)

    convs = []
    for r in probe.itertuples():
        c = S.ctx(r.prompt, r.story)
        for b in blocks:
            convs.append([{"role": "user", "content": f"{S.SYS_CW}\n\n{c}\n\n{b}"}])

    out_path = HERE / "gepa_probe_scores.npz"
    ckpt = HERE / "gepa_probe_scores_parts"
    ckpt.mkdir(parents=True, exist_ok=True)
    chunk = 27000

    def done(n):
        return all((ckpt / f"main_{i}.npy").exists() for i in range(0, n, chunk))

    llm = None
    if not done(len(convs)):
        from vllm import LLM, SamplingParams
        llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.90,
                  max_model_len=6144, enable_prefix_caching=True,
                  trust_remote_code=True, max_num_seqs=256)
        sp = SamplingParams(temperature=0.0, max_tokens=6)

    vals = []
    for i in range(0, len(convs), chunk):
        f = ckpt / f"main_{i}.npy"
        if f.exists():
            vals.append(np.load(f))
            print(f"[main] {min(i + chunk, len(convs))}/{len(convs)} (cached)", flush=True)
            continue
        outs = llm.chat(convs[i:i + chunk], sp)
        v = np.array([S.parse_tok(o.outputs[0].text) for o in outs], dtype=float)
        np.save(f, v)
        vals.append(v)
        print(f"[main] {min(i + chunk, len(convs))}/{len(convs)}", flush=True)
    X = np.concatenate(vals).reshape(len(probe), k)

    np.savez_compressed(out_path, X=X,
                        cids=np.array([v["cid"] for v in variants], dtype=object),
                        parent_cids=np.array([v["parent_cid"] for v in variants],
                                             dtype=object),
                        probe_ids=probe.id.values.astype(object))
    print("GEPA_PROBE_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
