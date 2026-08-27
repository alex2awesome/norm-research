#!/usr/bin/env python3
"""RUNG 2, stage B-sel — fit the frozen bank recipe on REAL E rows and score
the generated candidates (design §2.4.1: "no refitting on generated data";
fullfit-on-E convention, seeds {0,1,2} mean, same grid family as Layer-1).

Inputs: closure round7_state.npz (real 144-col bank + y + groups),
rung2_bank_scores_cw.npz (candidate X/V in the same column order).
Output: rung2_bank_selector_scores_cw.csv (per-candidate articulated score)
+ report with a real-holdout sanity AUC (grouped 80/20 by prompt hash).

CPU only (mac).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CWD_ = HERE.parent / "closure" / "cw_community"

z7 = np.load(CWD_ / "round7_state.npz", allow_pickle=True)
bank_names = [str(s) for s in z7["bank_names"]]
Xr = z7["VA"].astype(float)
y = z7["y"].astype(int)
groups = np.array([str(g) for g in z7["groups"]])

import sys
SCORES = sys.argv[1] if len(sys.argv) > 1 else "rung2_bank_scores_cw.npz"
OUTCSV = sys.argv[2] if len(sys.argv) > 2 else "rung2_bank_selector_scores_cw.csv"
zc = np.load(HERE / SCORES, allow_pickle=True)
jn = [str(s) for s in zc["judge_names"]]
vn = [str(s) for s in zc["v_names"]]
cols = {n: zc["V"][:, i] for i, n in enumerate(vn)}
cols.update({n: zc["X"][:, i] for i, n in enumerate(jn)})
Xc = np.column_stack([cols[n] for n in bank_names])
cand_ids = [str(s) for s in zc["cand_ids"]]
prompt_ids = [str(s) for s in zc["prompt_ids"]]
print(f"real {Xr.shape} candidates {Xc.shape}", flush=True)

import importlib.util, sys


def _mod(p, alias):
    spec = importlib.util.spec_from_file_location(alias, str(p))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


L = _mod(CWD_ / "closure_lib_cw.py", "closure_lib_cw_r2")

# sanity: grouped 80/20 holdout by stable hash of prompt group
hold = np.array([int(hashlib.md5(f"r2sel::{g}".encode()).hexdigest(), 16) % 5 == 0
                 for g in groups])
from sklearn.metrics import roc_auc_score
_, nl_hold, _ = L.fit_predict_monitor(Xr[~hold], y[~hold], groups[~hold], Xr[hold])
sanity_auc = float(roc_auc_score(y[hold], nl_hold.mean(0)))
print(f"sanity grouped-holdout AUC (fit 80% real -> 20% real): {sanity_auc:.4f}",
      flush=True)

# the selector: fit on ALL real E rows, predict candidates
_, nl_c, picks = L.fit_predict_monitor(Xr, y, groups, Xc)
pred_c = nl_c.mean(0)
print("grid picks:", picks, flush=True)

import csv
with open(HERE / OUTCSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cand_id", "prompt_id", "bank_score"])
    for c, p, s in zip(cand_ids, prompt_ids, pred_c):
        w.writerow([c, p, float(s)])
rep = {"n_candidates": len(cand_ids), "n_features": Xc.shape[1],
       "sanity_grouped_holdout_auc": sanity_auc,
       "fit": "fullfit on all 7,008 real E rows, closure_lib fit_predict_monitor "
              "(seeds mean), frozen grid family",
       "design": "notes/2026-08-21__rung12_design_gap_consequences.md §2.4.1"}
(HERE / (OUTCSV.replace(".csv", ".report.json"))).write_text(json.dumps(rep, indent=2))
print("RUNG2_BANKSEL_REPORT " + json.dumps(rep), flush=True)
