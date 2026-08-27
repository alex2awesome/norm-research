#!/usr/bin/env python3
"""Round-0 baseline for the two humor map cells: bank vs dense on the HONEST
(dense-held-out) population, on MONITOR, and on MONITOR_FULL, before any mining.

T is reported FOUR ways because these cells' dense arm is a 3-seed
dense-standard chain with select-on-eval, so eval is the SELECTED-ON half and
test is the selection-free half (the mirror image of the N&C responded cell,
whose chain selected on test):

  T_registry_eval_meanseedAUC  mean over seeds of the eval AUC -- the number the
                               registry carries (HW .6642 / SI .6343)
  T_test_meanseedAUC           the selection-free half
  T_HONEST_meanseedAUC         mean over seeds of the AUC on eval+test
  T_HONEST_ensemble            AUC of the mean-of-seeds probability, the row-level
                               score every downstream readout actually uses

CPU only.  Usage: python round0_context.py [cell ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent


def t_table(d, mask):
    y = d["y"][mask]
    P = d["dense_seeds"][mask]
    per = [L.auc(y, P[:, k]) for k in range(P.shape[1])]
    return {"n": int(mask.sum()), "per_seed": per, "mean_of_seed_AUC": float(np.mean(per)),
            "ensemble": L.auc(y, d["dense"][mask])}


def run(cell):
    d = C.load(cell)
    sp = json.loads((HERE / f"{cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    monf = np.array([r["in_monitor_full"] for r in sp["rows"]])
    y, g = d["y"], d["groups"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    ev, te = d["dense_split"] == "eval", d["dense_split"] == "test"

    r0 = L.fit_block([d["V"], d["A"]], fitm, monm, y, g)
    va = np.full(len(y), np.nan)
    va[fitm] = r0["oof_nl_fitmine"]
    va[monm] = r0["nl_mon"]
    lin = np.full(len(y), np.nan)
    lin[fitm] = r0["oof_lin_fitmine"]
    lin[monm] = r0["lin_mon"]

    # MONITOR_FULL: VA-honest only (fit on the complement inside FIT+MINE hashing)
    fitf = ~monf
    rF = L.fit_block([d["V"], d["A"]], fitf, monf, y, g, want_oof=False)

    out = {
        "cell": cell, "n": int(len(y)), "n_features_r0": r0["n_features"],
        "n_HONEST": int(held.sum()), "n_MONITOR": int(monm.sum()),
        "n_MONITOR_FULL": int(monf.sum()),
        "T": {"HONEST": t_table(d, held), "MONITOR": t_table(d, monm),
              "eval_selected_on": t_table(d, ev), "test_selection_free": t_table(d, te)},
        "VA_nl_HONEST": L.auc(y[held], va[held]),
        "VA_lin_HONEST": L.auc(y[held], lin[held]),
        "VA_nl_MONITOR": L.auc(y[monm], r0["nl_mon"]),
        "VA_lin_MONITOR": L.auc(y[monm], r0["lin_mon"]),
        "VA_nl_MONITOR_per_seed": [L.auc(y[monm], p) for p in r0["nl_mon_seeds"]],
        "VA_nl_OOF_fitmine": L.auc(y[fitm], r0["oof_nl_fitmine"]),
        "VA_nl_MONITOR_FULL": L.auc(y[monf], rF["nl_mon"]),
        "VA_lin_MONITOR_FULL": L.auc(y[monf], rF["lin_mon"]),
        "VA_nl_MONITOR_FULL_per_seed": [L.auc(y[monf], p) for p in rF["nl_mon_seeds"]],
        "layer1_ledger_VA_nl": d["layer1"]["ledger"]["VA_nl_mean"],
        "layer1_ledger_VA_lin": d["layer1"]["ledger"]["VA_lin"],
    }
    out["Delta_beyond_HONEST_ensembleT"] = out["T"]["HONEST"]["ensemble"] - out["VA_nl_HONEST"]
    out["Delta_beyond_HONEST_meanseedT"] = (out["T"]["HONEST"]["mean_of_seed_AUC"]
                                           - out["VA_nl_HONEST"])
    out["Delta_beyond_MONITOR_ensembleT"] = out["T"]["MONITOR"]["ensemble"] - out["VA_nl_MONITOR"]
    out["Delta_interact_HONEST"] = out["VA_nl_HONEST"] - out["VA_lin_HONEST"]
    out["ci_Delta_HONEST"] = L.group_boot_ci(y[held], d["dense"][held], va[held], g[held])
    np.savez(HERE / f"{cell}_r0_preds.npz", va_nl=va, va_lin=lin,
             nl_mon_seeds=r0["nl_mon_seeds"], monfull_nl=rF["nl_mon"])
    (HERE / f"{cell}_r0_context.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    todo = sys.argv[1:] or C.CELLS
    allr = {}
    for c in todo:
        allr[c] = run(c)
    p = HERE / "round0_context_all.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update(allr)
    p.write_text(json.dumps(prev, indent=1))
