#!/usr/bin/env python3
"""Extended blinded anchor battery for the Gemma-4-31B A-bank scoring runs.

The per-shard protocol carries only 3 blinded anchor rows (known positive /
random negative / scrambled nonsense). A single-draw check is noisy, and it
failed on some shards. This script scores K anchors per class with the same
judge, prompts and criteria, so the ordering can be checked at the instrument
level rather than one draw at a time.
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

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_va_gemma_banks as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="hashtagwars,style_invitational,creative_writing")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=6144)
    a = ap.parse_args()

    banks = {t: S.BUILDERS[t]() for t in a.tasks.split(",") if t}
    payload = {}
    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    for name, b in banks.items():
        rows, tags = [], []
        for j in range(a.k):
            trio = b["anchors"](900_000 + j)  # seeds disjoint from the shard anchors
            for r in trio:
                rows.append(r)
                tags.append(r["anchor_tag"])
        convs = []
        for r in rows:
            c = b["ctx"](r)
            for blk in b["blocks"]:
                convs.append([{"role": "user", "content": f"{b['sys']}\n\n{c}\n\n{blk}"}])
        print(f"[battery:{name}] {len(rows)} anchors x {len(b['blocks'])} = {len(convs)}",
              flush=True)
        outs = llm.chat(convs, sp)
        X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                     dtype=float).reshape(len(rows), len(b["blocks"]))
        item_mean = np.nanmean(X, axis=1)
        tags = np.array(tags)
        res = {"k_per_class": a.k}
        for t in ("anchor_pos", "anchor_neg", "anchor_scram"):
            v = item_mean[tags == t]
            res[t] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                      "values": [round(float(x), 4) for x in v]}
        from sklearn.metrics import roc_auc_score
        pv, nv = item_mean[tags == "anchor_pos"], item_mean[tags == "anchor_neg"]
        sv = item_mean[tags == "anchor_scram"]
        res["pos_vs_neg_auc"] = float(roc_auc_score(
            [1] * len(pv) + [0] * len(nv), np.concatenate([pv, nv])))
        res["coherent_vs_scrambled_auc"] = float(roc_auc_score(
            [1] * (len(pv) + len(nv)) + [0] * len(sv),
            np.concatenate([pv, nv, sv])))
        res["ordering_holds_on_means"] = bool(
            res["anchor_pos"]["mean"] > res["anchor_neg"]["mean"] > res["anchor_scram"]["mean"])
        payload[name] = res
        print(f"[battery:{name}] pos {res['anchor_pos']['mean']:.3f} / "
              f"neg {res['anchor_neg']['mean']:.3f} / scram {res['anchor_scram']['mean']:.3f} "
              f"| pos-vs-neg AUC {res['pos_vs_neg_auc']:.3f} "
              f"| coherent-vs-scram AUC {res['coherent_vs_scrambled_auc']:.3f}", flush=True)

    (S.OUT / "anchor_battery.json").write_text(json.dumps(payload, indent=1))
    print("BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
