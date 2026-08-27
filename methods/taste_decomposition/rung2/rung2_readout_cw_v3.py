#!/usr/bin/env python3
"""RUNG 2 READOUT v3 — ADDENDUM D diverse pools (frozen 2026-08-24).

Pools: 18 generated (3 families x 6 conditions) + all real stories per
prompt (labels quarantined until this readout). Policies: SEL_bank (frozen
articulated selector), SEL_dense (certified dense), SEL_random. Readouts
R1-R4 of Addendum D; floor-based, no own-argmax terms; blindness AUC uses
the v1-clean convention flag (diagonal noted). CPU, mac.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SEED = 12345


def load(name, val):
    return {r["cand_id"]: float(r[val]) for r in csv.DictReader(open(HERE / name))}


import os
TAG = os.environ.get("R2TAG", "v2")
FILES = {
    "v2": ("rung2v2_bank_selector_scores_cw.csv", "rung2_dense_scores_cw_full_v2.csv",
           "dense_full_v2_prob", "rung2v2_pool_cw.csv", "rung2v2_real_labels_cw.json"),
    "e2": ("rung2_e2_bank_selector_scores_cw.csv", "rung2_dense_scores_cw_full_e2.csv",
           "dense_full_e2_prob", "rung2_e2_pool_cw.csv", "rung2_e2_real_labels_cw.json"),
}[TAG]
bank = load(FILES[0], "bank_score")
dF = load(FILES[1], FILES[2])
pool = list(csv.DictReader(open(HERE / FILES[3])))
ylab = json.load(open(HERE / FILES[4]))
meta = {r["cand_id"]: (r["family"], r["condition"]) for r in pool}
by_prompt = {}
for r in pool:
    by_prompt.setdefault(r["prompt_id"], []).append(r["cand_id"])
print(f"{len(by_prompt)} pools, sizes {min(map(len,by_prompt.values()))}-"
      f"{max(map(len,by_prompt.values()))}; scored bank/{len(bank)} dF/{len(dF)}")


def wrank(sc, ids, pick):
    s = np.array([sc[i] for i in ids])
    return float((s < sc[pick]).mean() + 0.5 * (s == sc[pick]).mean())


rows = []
for pid, ids in sorted(by_prompt.items()):
    sb = max(ids, key=lambda i: bank[i])
    sd = max(ids, key=lambda i: dF[i])
    sr = ids[int(hashlib.md5(f"rand::{pid}".encode()).hexdigest(), 16) % len(ids)]
    rows.append(dict(
        pid=pid, agree=sb == sd,
        lb=wrank(bank, ids, sd) - wrank(bank, ids, sr),
        ld=wrank(dF, ids, sb) - wrank(dF, ids, sr),
        pick_b=sb, pick_d=sd,
        b_real=meta[sb][0] == "real", d_real=meta[sd][0] == "real",
        r_real=meta[sr][0] == "real",
    ))

lb = np.array([r["lb"] for r in rows])
ld = np.array([r["ld"] for r in rows])
Ap = lb - ld
rng = np.random.default_rng(SEED)
boots = {"lb": [], "ld": [], "Ap": []}
n = len(rows)
for _ in range(2000):
    ix = rng.integers(0, n, n)
    for k, v in (("lb", lb), ("ld", ld), ("Ap", Ap)):
        boots[k].append(v[ix].mean())
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
      for k, v in boots.items()}

# R2: real-vs-generated separation per instrument
ids_all = [r["cand_id"] for r in pool]
is_real = np.array([meta[i][0] == "real" for i in ids_all])
r2 = {lab: float(roc_auc_score(is_real, [sc[i] for i in ids_all]))
      for lab, sc in (("bank", bank), ("certified_dense", dF))}

# R3: humanness gradient (mean scores by condition, rank-normalized in-corpus)
def rank01(v):
    a = np.array(v)
    return np.argsort(np.argsort(a)) / (len(a) - 1)


rk_b = dict(zip(ids_all, rank01([bank[i] for i in ids_all])))
rk_d = dict(zip(ids_all, rank01([dF[i] for i in ids_all])))
r3 = {}
for cond in ("plain", "human", "veryhuman", "casual", "literary", "hightemp", "real"):
    sel = [i for i in ids_all if meta[i][1] == cond]
    r3[cond] = {"n": len(sel),
                "bank_mean_rank": float(np.mean([rk_b[i] for i in sel])),
                "dense_mean_rank": float(np.mean([rk_d[i] for i in sel]))}

# R4: family separability (dense channel), generated only
r4 = {}
for fam in ("llama8b", "qwen14b", "phi4"):
    sel = np.array([meta[i][0] == fam for i in ids_all if meta[i][0] != "real"])
    sc = [dF[i] for i in ids_all if meta[i][0] != "real"]
    r4[fam] = float(roc_auc_score(sel, sc))

# picks: how often each policy reaches a REAL story; and when it does, its label
real_rate = float(np.mean(is_real))
def pick_stats(key_real, key_pick):
    picks = [r for r in rows if r[key_real]]
    labs = [ylab.get(r[key_pick]) for r in picks]
    labs = [l for l in labs if l is not None]
    return {"p_pick_real": float(np.mean([r[key_real] for r in rows])),
            "picked_real_label_mean": float(np.mean(labs)) if labs else None,
            "n_real_picks": len(picks)}

out = {
    "design": "ADDENDUM D readout, notes/2026-08-21__rung12_design_gap_consequences.md",
    "n_pools": n, "pool_real_base_rate": real_rate,
    "agreement_rate": float(np.mean([r["agree"] for r in rows])),
    "R1_floor_signals": {
        "dense_policy_on_bank_channel": {"mean": float(lb.mean()), "ci95": ci["lb"]},
        "bank_policy_on_dense_channel": {"mean": float(ld.mean()), "ci95": ci["ld"]},
        "ASYMMETRY_Aprime": {"mean": float(Ap.mean()), "ci95": ci["Ap"],
                             "p_gt_0": float(np.mean(np.array(boots["Ap"]) > 0))},
    },
    "R2_real_vs_generated_auc": r2,
    "R3_condition_mean_ranks": r3,
    "R4_family_auc_dense_channel": r4,
    "picks": {"bank_policy": pick_stats("b_real", "pick_b"),
              "dense_policy": pick_stats("d_real", "pick_d")},
}
(HERE / f"rung2_readout_cw_v3_{TAG}.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
