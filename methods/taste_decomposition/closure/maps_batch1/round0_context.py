#!/usr/bin/env python3
"""Round-0 context table: the bank-vs-dense picture on each cell's HONEST
(dense-held-out) population and on MONITOR, before any mining.

This is the baseline every round-1 number is read against, and it is also the
apples-to-apples Delta_beyond for this batch: T and VA_nl on the SAME rows, both
out-of-sample (VA via grouped OOF inside FIT+MINE, held-out prediction on
MONITOR; T from the same-rows dense rescore restricted to dense eval/test).

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


def run(cell):
    d = C.load(cell)
    sp = json.loads((HERE / f"{cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, g, dense = d["y"], d["groups"], d["dense"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])

    r0 = L.fit_block([d["V"], d["A"]], fitm, monm, y, g)
    va = np.full(len(y), np.nan)
    va[fitm] = r0["oof_nl_fitmine"]
    va[monm] = r0["nl_mon"]
    lin = np.full(len(y), np.nan)
    lin[fitm] = r0["oof_lin_fitmine"]
    lin[monm] = r0["lin_mon"]

    out = {
        "cell": cell, "n": int(len(y)), "n_features_r0": r0["n_features"],
        "n_HONEST": int(held.sum()), "n_MONITOR": int(monm.sum()),
        "T_HONEST": L.auc(y[held], dense[held]),
        "VA_nl_HONEST": L.auc(y[held], va[held]),
        "VA_lin_HONEST": L.auc(y[held], lin[held]),
        "T_MONITOR": L.auc(y[monm], dense[monm]),
        "VA_nl_MONITOR": L.auc(y[monm], r0["nl_mon"]),
        "VA_lin_MONITOR": L.auc(y[monm], r0["lin_mon"]),
        "VA_nl_OOF_fitmine": L.auc(y[fitm], r0["oof_nl_fitmine"]),
        "VA_nl_MONITOR_per_seed": [L.auc(y[monm], p) for p in r0["nl_mon_seeds"]],
        "layer1_ledger_VA_nl": d["layer1"]["ledger"].get(
            "VA_nl_mean", d["layer1"]["ledger"].get("VA_nl")),
    }
    out["Delta_beyond_HONEST"] = out["T_HONEST"] - out["VA_nl_HONEST"]
    out["Delta_beyond_MONITOR"] = out["T_MONITOR"] - out["VA_nl_MONITOR"]
    out["Delta_interact_HONEST"] = out["VA_nl_HONEST"] - out["VA_lin_HONEST"]
    out["ci_Delta_HONEST"] = L.group_boot_ci(y[held], dense[held], va[held], g[held])
    np.savez(HERE / f"{cell}_r0_preds.npz", va_nl=va, va_lin=lin,
             nl_mon_seeds=r0["nl_mon_seeds"])
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
