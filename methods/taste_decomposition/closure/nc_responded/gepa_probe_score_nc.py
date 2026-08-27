#!/usr/bin/env python3
"""GEPA Stage 2 for N&C RESPONDED: score K=3 rephrasing variants per targeted
criterion on a bounded, deterministic FIT+MINE probe subset, Gemma-4-31B
offline-batch vLLM.  0-10 scale, identical instrument to score_round_gemma.py
(SYS prompt, 4,000-char truncation).

Run on sk3, ONE GPU:
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=<n> $HOME/envs/gemma4/bin/python gepa_probe_score_nc.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
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
HERE = REPO / "methods/taste_decomposition/closure/nc_responded"
GEMMA4 = os.environ.get(
    "GEMMA4_PATH",
    "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb",
)
PROBE_N = 600
TRUNC = 4000

SYS = (
    "You are an expert regulatory analyst reviewing PUBLIC COMMENTS submitted on proposed "
    "federal rules. You are given one public comment and ONE criterion. Decide how strongly "
    "the comment, on its own evidence, exhibits that criterion. Answer with EXACTLY ONE "
    "token:\n"
    "  an integer from 0 to 10, where 0 = not at all and 10 = to the fullest degree\n"
    "  NA = the comment gives no evidence bearing on this criterion\n"
    "Judge the criterion as literally described, not whether the comment's position is "
    "correct and not whether the agency will respond to it. Output only the token."
)


def parse_tok(t):
    t = (t or "").strip()
    low = t.lower()
    if low.startswith("na") or "n/a" in low:
        return np.nan
    m = re.search(r"\d+", t)
    if not m:
        return np.nan
    v = float(m.group(0))
    return v if 0.0 <= v <= 10.0 else np.nan


def main():
    import pandas as pd

    pop = pd.read_csv(HERE / "nc_responded_population.csv")
    fm = pop[pop.split == "fit_mine"].reset_index(drop=True)

    def h(x):
        return int(hashlib.sha256(f"gepa-probe-nc|{x}".encode()).hexdigest(), 16) / (1 << 256)

    idcol = "doc_id" if "doc_id" in fm.columns else fm.columns[0]
    fm["_h"] = fm[idcol].astype(str).map(h)
    probe = fm.sort_values("_h").head(PROBE_N).reset_index(drop=True)

    variants = json.loads((HERE / "gepa_variants_nc.json").read_text())
    blocks = [f"CRITERION: {v['name']}\nINSTRUCTION: {v['instruction']}\n\n"
              f"Answer with one token:" for v in variants]
    k = len(blocks)
    print(f"[gepa-probe-nc] rows={len(probe)} variants={k} prompts={len(probe) * k}",
          flush=True)

    text_col = "text" if "text" in probe.columns else "comment_text"
    convs = []
    for r in probe.itertuples():
        t = str(getattr(r, text_col))[:TRUNC]
        for b in blocks:
            convs.append([{"role": "user", "content": f"{SYS}\n\nCOMMENT:\n{t}\n\n{b}"}])

    out_path = HERE / "gepa_probe_scores_nc.npz"
    ckpt = HERE / "gepa_probe_scores_nc_parts"
    ckpt.mkdir(parents=True, exist_ok=True)
    chunk = 27000

    def done(n):
        return all((ckpt / f"main_{i}.npy").exists() for i in range(0, n, chunk))

    llm = None
    if not done(len(convs)):
        from vllm import LLM, SamplingParams
        llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.55,
                  max_model_len=4096, enable_prefix_caching=True,
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
        v = np.array([parse_tok(o.outputs[0].text) for o in outs], dtype=float)
        np.save(f, v)
        vals.append(v)
        print(f"[main] {min(i + chunk, len(convs))}/{len(convs)}", flush=True)
    X = np.concatenate(vals).reshape(len(probe), k)

    np.savez_compressed(out_path, X=X,
                        cids=np.array([v["cid"] for v in variants], dtype=object),
                        parent_tags=np.array([v["parent_tag"] for v in variants],
                                             dtype=object),
                        probe_ids=probe[idcol].values.astype(object))
    print("GEPA_PROBE_SCORE_NC_DONE", flush=True)


if __name__ == "__main__":
    main()
