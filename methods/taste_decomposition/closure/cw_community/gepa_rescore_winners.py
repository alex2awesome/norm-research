#!/usr/bin/env python3
"""GEPA Stage 3 (CW): full-population (7,008 rows) Gemma-4-31B rescore of the
ACCEPTED winning variants only (gepa_winners.json, ACCEPTED=true), identical
instrument to score_round_gemma.py (SYS_CW, truncation, temperature 0).

Run on sk3, ONE GPU:
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=<n> $HOME/envs/gemma4/bin/python gepa_rescore_winners.py
"""
from __future__ import annotations

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

import score_round_gemma as S  # noqa: E402


def main():
    import pandas as pd

    winners = json.loads((HERE / "gepa_winners.json").read_text())
    accepted = [w for w in winners if w["ACCEPTED"]]
    variants = {v["cid"]: v for v in json.loads((HERE / "gepa_variants.json").read_text())}
    crits = [variants[w["best_variant"]] for w in accepted]
    print(f"[gepa-rescore] {len(crits)} accepted winners: "
          f"{[c['cid'] for c in crits]}", flush=True)

    pop = pd.read_csv(HERE / "cw_population_with_splits.csv").fillna(
        {"prompt": "", "story": ""})
    blocks = [f"CRITERION: {c['name']}\nDESCRIPTION: {c['instruction']}\n\n"
              f"Answer with one token:" for c in crits]
    k = len(blocks)
    print(f"[gepa-rescore] rows={len(pop)} criteria={k} prompts={len(pop) * k}",
          flush=True)

    convs = []
    for r in pop.itertuples():
        c = S.ctx(r.prompt, r.story)
        for b in blocks:
            convs.append([{"role": "user", "content": f"{S.SYS_CW}\n\n{c}\n\n{b}"}])

    out_path = HERE / "gepa_winners_scores.npz"
    ckpt = HERE / "gepa_winners_scores_parts"
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
    X = np.concatenate(vals).reshape(len(pop), k)

    collapse = []
    for j, c in enumerate(crits):
        col = X[:, j]
        fin = np.isfinite(col)
        vals_, cnts = (np.unique(col[fin], return_counts=True) if fin.any()
                      else (np.array([]), np.array([])))
        modal = float(cnts.max() / max(fin.sum(), 1)) if len(cnts) else 1.0
        collapse.append({"cid": c["cid"], "parent_cid": c["parent_cid"],
                         "criterion": c["name"],
                         "na_rate": float((~fin).mean()), "modal_share": modal,
                         "n_distinct": int(len(vals_)),
                         "COLLAPSED": bool(modal > 0.98 or len(vals_) < 2
                                           or (~fin).mean() > 0.90)})

    np.savez_compressed(out_path, X=X, ids=pop.id.values.astype(object),
                        cids=np.array([c["cid"] for c in crits], dtype=object),
                        parent_cids=np.array([c["parent_cid"] for c in crits],
                                             dtype=object),
                        names=np.array([c["name"] for c in crits], dtype=object))
    (out_path.with_suffix(".report.json")).write_text(json.dumps(
        {"n_rows": int(len(pop)), "n_criteria": k, "collapse": collapse}, indent=1))
    print("GEPA_RESCORE_DONE " + json.dumps(
        {"n_collapsed": sum(c["COLLAPSED"] for c in collapse)}), flush=True)


if __name__ == "__main__":
    main()
