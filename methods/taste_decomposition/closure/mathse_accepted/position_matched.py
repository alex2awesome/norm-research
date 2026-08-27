#!/usr/bin/env python3
"""Round-0 MATCHED-SAMPLING discount on the answer-position family.

The freeze triggers matched sampling once a nuisance channel's alone-AUC exceeds
.65.  On this cell the ARRIVAL-ORDER family reaches that on the full population
before a single Track-B channel has been proposed (joint grouped-OOF position
model: .6538 pooled full, .6140 on HONEST), so the trigger is armed at round 0 and
the matched estimator is reported alongside decile stratification from the start.

Matching pairs each positive row with the nearest-percentile negative row on the
control, within a .02 caliper, and reports the fraction of matched pairs each
instrument orders correctly.  Two controls are reported:
  * the joint position model score (the family in one scalar);
  * `is_first` exactly (an EXACT match: first-answer vs first-answer, later vs
    later) -- the cleanest form on this cell because the dominant channel is binary.

CPU only.  Usage: python3 position_matched.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
from discount_cumulative import matched_pair_auc

HERE = Path(__file__).resolve().parent


def exact_stratum_auc(y, score, key, min_n=25):
    """n-weighted AUC within exact levels of a discrete control."""
    num, tot, used = 0.0, 0, 0
    for v in np.unique(key):
        m = key == v
        if m.sum() < min_n or len(set(y[m].tolist())) < 2:
            continue
        num += m.sum() * roc_auc_score(y[m], score[m])
        tot += int(m.sum())
        used += 1
    return (float(num / tot) if tot else float("nan")), {"n_levels": used, "n_rows": tot}


def main():
    d = C.load()
    sp = json.loads((HERE / "mathse_accepted_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    monm = split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    y = d["y"]
    z = np.load(HERE / "mathse_accepted_r0_preds.npz", allow_pickle=True)
    va = z["va_nl"]
    pz = np.load(HERE / "mathse_accepted_position.npz", allow_pickle=True)
    joint = pz["joint"]
    is_first = (d["answer_position"] == 0).astype(int)

    out = {"cell": "mathse_accepted", "trigger": 0.65,
           "why_armed": "the observed arrival-order family already exceeds the matched-"
                        "sampling trigger before any Track-B channel exists",
           "position_alone_AUC": {
               "joint_model_pooled_full": float(roc_auc_score(y, joint)),
               "joint_model_HONEST": float(roc_auc_score(y[held], joint[held])),
               "is_first_pooled_full": float(roc_auc_score(y, is_first))}}

    for pop, m in (("HONEST", held), ("MONITOR", monm)):
        ym, dm, vm = y[m], d["dense"][m], va[m]
        rec = {"n": int(m.sum()),
               "pooled_T": float(roc_auc_score(ym, dm)),
               "pooled_VA": float(roc_auc_score(ym, vm))}
        rec["pooled_Delta"] = rec["pooled_T"] - rec["pooled_VA"]
        tv, ti = matched_pair_auc(ym, dm, joint[m])
        vv, vi = matched_pair_auc(ym, vm, joint[m])
        rec["matched_on_joint_position"] = {
            "T_adj": tv, "VA_adj": vv, "Delta_adj": tv - vv, "info_T": ti, "info_VA": vi}
        tE, iE = exact_stratum_auc(ym, dm, is_first[m])
        vE, _ = exact_stratum_auc(ym, vm, is_first[m])
        rec["exact_on_is_first"] = {"T_adj": tE, "VA_adj": vE, "Delta_adj": tE - vE, **iE}
        out[pop] = rec

    (HERE / "position_matched.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
