#!/usr/bin/env python3
"""ADDENDUM G2 pass 1 — JUDGE-SCALE LADDER: score the REAL cw E-frame on the FROZEN (form-a) blocks of the
Gemma-chosen top-24 criteria under a specified judge model (one rung per
process; wrapper loops over rungs). Same judge, system prompt, truncation, parse,
anchors and collapse gates as the certified scorer (rung2_score_bank_cw.py
constants imported). Chunk-checkpointed, SIGTERM-safe.

Run on sk3, TP=2 on the allowed pair:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,6 TP=2 \
    $HOME/envs/gemma4/bin/python g1_score_forms_cw.py --util 0.25
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import importlib.util
import sys

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
HERE = REPO / "methods/taste_decomposition/rung2"
CWD_ = REPO / "methods/taste_decomposition/closure/cw_community"


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


S = _mod(HERE / "rung2_score_bank_cw.py", "r2score_g1")   # constants + helpers


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.25)
    ap.add_argument("--model", required=True, help="HF snapshot path of the judge")
    ap.add_argument("--rung", required=True, help="short tag, e.g. llama3b")
    ap.add_argument("--anchor-k", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=27000)
    a = ap.parse_args()

    forms = json.load(open(HERE / "g1_forms_cw.json"))
    blocks = [f["form_a_frozen"] for f in forms]
    tags = [(f["name"], "a") for f in forms]
    k = len(blocks)

    pop = pd.read_csv(CWD_ / "cw_honest_population.csv").fillna(
        {"prompt": "", "story": ""})
    print(f"[G1] rows={len(pop)} form-blocks={k} prompts={len(pop)*k}", flush=True)

    rng = random.Random(S.SEED + 777)
    pos = pop[pop.judgement == 1].to_dict("records")
    neg = pop[pop.judgement == 0].to_dict("records")
    arows, atags = [], []
    for _ in range(a.anchor_k):
        p, n = rng.choice(pos), rng.choice(neg)
        s = dict(n)
        s["story"] = S.scramble([str(p["story"])[:4000], str(n["story"])[:4000]], rng)
        for tag, r in (("anchor_pos", p), ("anchor_neg", n), ("anchor_scram", s)):
            arows.append(r)
            atags.append(tag)

    convs = [[{"role": "user",
               "content": f"{S.SYS_CW}\n\n{S.ctx(r.prompt, r.story)}\n\n{b}"}]
             for r in pop.itertuples() for b in blocks]
    aconvs = [[{"role": "user",
                "content": f"{S.SYS_CW}\n\n{S.ctx(r['prompt'], r['story'])}\n\n{b}"}]
              for r in arows for b in blocks]

    ckpt = HERE / f"g2_ladder_{a.rung}_parts"
    ckpt.mkdir(exist_ok=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=6144, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=256,
              tensor_parallel_size=int(os.environ.get("TP", "1")),
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    def run(cs, tag):
        vals = []
        for i in range(0, len(cs), a.chunk):
            f = ckpt / f"{tag}_{i}.npy"
            if f.exists():
                vals.append(np.load(f))
                continue
            outs = llm.chat(cs[i:i + a.chunk], sp)
            v = np.array([S.parse_tok(o.outputs[0].text) for o in outs], float)
            np.save(f, v)
            vals.append(v)
            print(f"[{tag}] {min(i+a.chunk,len(cs))}/{len(cs)}", flush=True)
        return np.concatenate(vals)

    X = run(convs, "main").reshape(len(pop), k)
    aX = run(aconvs, "anchors").reshape(len(arows), k)

    from sklearn.metrics import roc_auc_score
    t = np.array(atags)
    im = np.nanmean(aX, axis=1)
    pv, nv, sv = im[t == "anchor_pos"], im[t == "anchor_neg"], im[t == "anchor_scram"]
    battery = {
        "pos_vs_neg_auc": float(roc_auc_score([1]*len(pv)+[0]*len(nv),
                                              np.concatenate([pv, nv]))),
        "coherent_vs_scrambled_auc": float(roc_auc_score(
            [1]*(len(pv)+len(nv))+[0]*len(sv), np.concatenate([pv, nv, sv]))),
        "ordering_holds": bool(np.mean(pv) > np.mean(nv) > np.mean(sv)),
    }
    np.savez_compressed(HERE / f"g2_ladder_scores_cw_{a.rung}.npz",
                        X=X, anchor_X=aX,
                        ids=pop.id.values.astype(object),
                        y=pop.judgement.values.astype(int),
                        groups=pop.prompt_id.values.astype(str).astype(object),
                        form_names=np.array([f"{n}::{fm}" for n, fm in tags],
                                            dtype=object),
                        anchor_tags=t.astype(object))
    rep = {"n_rows": int(len(pop)), "n_form_blocks": k,
           "na_rate": float(np.isnan(X).mean()), "anchor_battery": battery,
           "design": "ADDENDUM G2 pass1", "rung": a.rung, "model": a.model}
    (HERE / f"g2_ladder_scores_cw_{a.rung}.report.json").write_text(json.dumps(rep, indent=1))
    print(f"G2_{a.rung}_REPORT " + json.dumps(rep), flush=True)
    print(f"G2_DONE {a.rung}", flush=True)


if __name__ == "__main__":
    main()
