#!/usr/bin/env python3
"""GEPA STAGE 4b — corpus-wide rescore of the ACCEPTED rephrasings only.

Fairness rule, inherited from ../gepa_select_peer.py: accept/reject must be probe-vs-probe.
The incumbent's population-level modal share was measured over 16,000 items; a variant
measured on 600 items is not comparable to it. So this job re-scores the INCUMBENT
instruction on the probe subset too, in the same batch, with the same judge and the same
item view, and selection compares the two numbers that were produced under identical
conditions.

PROBE SUBSET: 600 rows drawn by stable sha256 of the row id (never a seeded shuffle),
restricted to FIT+MINE so no MONITOR row is read while an instrument is being chosen.
That is the same discipline as parent selection in `mixed_parents.py`: MONITOR is never
read for a design decision.

Item view and judge identical to `score_gemma_maps.py` (JOKE:\\n"<text>", persona = expert
comedy writer, one token 0-10 or NA), plus the standard K>=50/class anchor battery.

Run on sk3 via gpu_lane_runner.sh.  Usage:
  ./gpu_lane_runner.sh jokes_gepa_probe <log> 5 100000 $HOME/envs/gemma4/bin/python \\
      gepa_probe_score.py
"""
import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
GEMMA4 = ("/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/"
          "snapshots/3548789868c5356dbf307c98e6f609007b82b3eb")
CELL = "jokes_community"
PROBE_N = 0   # 0 = whole population
K_ANCHOR = 50
SEED = 20260806
NUM = re.compile(r"\d+")
PERSONA = "an expert comedy writer performing a measurement task"
NOUN = "JOKE"


def sys_prompt():
    return (f"You are {PERSONA}. You are given one {NOUN} and ONE criterion. Decide how "
            f"strongly the {NOUN.lower()}, on its own evidence, exhibits that criterion. "
            "Answer with EXACTLY ONE token:\n"
            "  an integer from 0 to 10, where 0 = not at all and 10 = to the fullest degree\n"
            f"  NA = the {NOUN.lower()} gives no evidence bearing on this criterion\n"
            "Judge the criterion as literally described, not whether the item is good "
            "overall. Output only the token.")


def parse_tok(t):
    t = (t or "").strip()
    if t.lower().startswith("na") or "n/a" in t.lower():
        return np.nan
    m = NUM.search(t)
    if not m:
        return np.nan
    v = float(m.group(0))
    return v if 0.0 <= v <= 10.0 else np.nan


def hash_unit(key):
    return int(hashlib.sha256(str(key).encode()).hexdigest(), 16) / float(1 << 256)


def scramble(texts, rng):
    words = " ".join(texts).split()
    rng.shuffle(words)
    return " ".join(words[:220])


def main():
    from vllm import LLM, SamplingParams

    var = {v["variant_id"]: v for v in
           json.loads((HERE / f"{CELL}_gepa_variants.json").read_text())["variants"]}
    win = json.loads((HERE / f"{CELL}_gepa_selection.json").read_text())["winners"]
    units = [{"uid": vid, "kind": "winner", "target_id": tid,
              "name": var[vid]["name"], "instruction": var[vid]["instruction"]}
             for tid, vid in win.items() if vid in var]

    with open(HERE / f"{CELL}_population.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    probe = rows            # whole population: these columns join the terminal bank
    print(f"[gepa] {len(units)} accepted rephrasings x {len(probe)} population rows",
          flush=True)

    rng = random.Random(SEED)
    pos = [r["text"] for r in probe if str(r["judgement"]) == "1"]
    neg = [r["text"] for r in probe if str(r["judgement"]) == "0"]
    a_texts, a_tags = [], []
    for _ in range(K_ANCHOR):
        p, n = rng.choice(pos), rng.choice(neg)
        a_texts += [p, n, scramble([p, n], rng)]
        a_tags += ["anchor_pos", "anchor_neg", "anchor_scram"]

    texts = [r["text"] for r in probe]
    all_texts = texts + a_texts
    SYS = sys_prompt()
    blocks = [f"CRITERION: {u['name']}\nINSTRUCTION: {u['instruction']}\n\n"
              "Answer with one token:" for u in units]

    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.60,
              max_model_len=4096, enable_prefix_caching=True, trust_remote_code=True,
              max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    convs = []
    for t in all_texts:
        item = f'{NOUN}:\n"{(t or "").strip()}"'
        for b in blocks:
            convs.append([{"role": "user", "content": f"{SYS}\n\n{item}\n\n{b}"}])
    print(f"[gepa] {len(convs)} prompts", flush=True)
    outs = llm.chat(convs, sp)
    X = np.array([parse_tok(o.outputs[0].text) for o in outs], dtype=float).reshape(
        len(all_texts), len(units))
    Xp, Xa = X[:len(texts)], X[len(texts):]

    np.savez_compressed(HERE / f"{CELL}_gepa_winner_scores.npz", X=Xp,
                        uid=np.array([u["uid"] for u in units], dtype=object),
                        kind=np.array([u["kind"] for u in units], dtype=object),
                        target_id=np.array([u["target_id"] for u in units], dtype=object),
                        probe_i=np.array([int(r["i"]) for r in probe]),
                        probe_id=np.array([r["id"] for r in probe], dtype=object),
                        Xanchor=Xa, anchor_tags=np.array(a_tags, dtype=object))

    from sklearn.metrics import roc_auc_score
    tags = np.array(a_tags)
    item = np.nanmean(Xa, axis=1)
    pv, nv, sv = item[tags == "anchor_pos"], item[tags == "anchor_neg"], item[tags == "anchor_scram"]
    anc = {"k_per_class": K_ANCHOR,
           "pos_vs_neg_auc": float(roc_auc_score([1]*len(pv)+[0]*len(nv), np.concatenate([pv, nv]))),
           "coherent_vs_scrambled_auc": float(roc_auc_score(
               [1]*(len(pv)+len(nv))+[0]*len(sv), np.concatenate([pv, nv, sv])))}
    anc["pass_scrambled"] = bool(anc["coherent_vs_scrambled_auc"] >= 0.70)
    (HERE / f"{CELL}_gepa_winner_report.json").write_text(json.dumps(
        {"n_units": len(units), "n_probe": len(probe), "anchors": anc,
         "rows_all_NA": int(np.isnan(Xp).all(axis=1).sum())}, indent=2))
    print("[gepa] ANCHORS " + json.dumps(anc), flush=True)
    print("GEPA_WINNER_DONE", flush=True)


if __name__ == "__main__":
    main()
