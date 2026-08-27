#!/usr/bin/env python3
"""Gemma scoring of the 13 word-probe-derived channels on the full BBC corpus
(50,761 headlines) — the SPUR-BATTERY pilot's measurement step.  Conventions =
the cell's campaign scorer (score_gemma_maps): 0-10/NA single token, persona =
news editor, 300-char headline, chunked resumable, blinded anchors.
  CUDA_VISIBLE_DEVICES=N python score_wordprobe_channels.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round0_bbc as R0  # noqa: E402

GEMMA4 = ("/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/"
          "snapshots/3548789868c5356dbf307c98e6f609007b82b3eb")
OUT = HERE / "wordprobe_scores"
OUT.mkdir(exist_ok=True)
CHUNK = 4000
NUM = re.compile(r"\d+")

SYS = ("You are an experienced news editor performing a measurement task. You are "
       "given one news headline and ONE property. Decide how strongly the headline, "
       "on its own evidence, exhibits that property. Answer with EXACTLY ONE token:\n"
       "  an integer from 0 to 10, where 0 = not at all and 10 = to the fullest degree\n"
       "  NA = the headline gives no evidence bearing on this property\n"
       "Judge the property as literally described, not whether the headline is good "
       "overall. Output only the token.")


def parse_tok(t):
    t = (t or "").strip().lower()
    if t.startswith("na"):
        return np.nan
    m = NUM.search(t)
    if not m:
        return np.nan
    v = int(m.group())
    return float(v) / 10.0 if 0 <= v <= 10 else np.nan


def main():
    chans = json.loads((HERE / "wordprobe_channels.json").read_text())["channels"]
    blocks = [f"PROPERTY: {c['name']}\nDESCRIPTION: {c['description']}\n\n"
              "Answer with one token:" for c in chans]
    pop = pd.read_csv(R0.VA_DIR / "population.csv.gz")
    texts = [str(t)[:300] for t in pop.text.astype(str)]
    ids = pop.row_id.astype(str).tolist()
    y = pop.judgement.astype(int).values
    print(f"[wordprobe] {len(texts)} headlines x {len(blocks)} channels", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.70,
              max_model_len=2048, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    n_chunks = (len(texts) + CHUNK - 1) // CHUNK
    for k in range(n_chunks):
        outp = OUT / f"chunk{k}.npz"
        if outp.exists():
            continue
        sub = texts[k * CHUNK:(k + 1) * CHUNK]
        convs = [[{"role": "user", "content": f"{SYS}\n\nHEADLINE: {t}\n\n{blk}"}]
                 for t in sub for blk in blocks]
        outs = llm.chat(convs, sp)
        X = np.array([parse_tok(o.outputs[0].text) for o in outs],
                     dtype=float).reshape(len(sub), len(blocks))
        np.savez_compressed(outp, X=X,
                            row_id=np.array(ids[k * CHUNK:(k + 1) * CHUNK], dtype=object),
                            names=np.array([c["name"] for c in chans], dtype=object))
        print(f"[chunk {k}/{n_chunks}] NA={np.isnan(X).mean():.3f}", flush=True)

    # anchors: pos/neg by y + scrambled, K=40
    rng = random.Random(20260818)
    WORD = re.compile(r"[A-Za-z]+")
    pos_i = [i for i in range(len(y)) if y[i] == 1]
    neg_i = [i for i in range(len(y)) if y[i] == 0]
    anchors, tags = [], []
    for _ in range(40):
        p, n = texts[rng.choice(pos_i)], texts[rng.choice(neg_i)]
        toks = WORD.findall(p + " " + n)
        rng.shuffle(toks)
        anchors += [p, n, " ".join(toks[:20])]
        tags += ["pos", "neg", "scram"]
    convs = [[{"role": "user", "content": f"{SYS}\n\nHEADLINE: {t}\n\n{blk}"}]
             for t in anchors for blk in blocks]
    outs = llm.chat(convs, sp)
    Xa = np.array([parse_tok(o.outputs[0].text) for o in outs],
                  dtype=float).reshape(len(anchors), len(blocks))
    with np.errstate(invalid="ignore"):
        im = np.nanmean(Xa, axis=1)
    tags = np.array(tags)
    from sklearn.metrics import roc_auc_score
    pv, nv, sv = im[tags == "pos"], im[tags == "neg"], im[tags == "scram"]
    ok = lambda v: v[np.isfinite(v)]
    pv, nv, sv = ok(pv), ok(nv), ok(sv)
    batt = {"pos_mean": float(np.mean(pv)), "neg_mean": float(np.mean(nv)),
            "scram_mean": float(np.mean(sv)) if len(sv) else None,
            "pos_vs_neg_auc": float(roc_auc_score([1] * len(pv) + [0] * len(nv),
                                                  np.concatenate([pv, nv])))}
    (OUT / "anchor_battery.json").write_text(json.dumps(batt, indent=1))
    print(json.dumps(batt, indent=1), flush=True)
    print("WORDPROBE_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
