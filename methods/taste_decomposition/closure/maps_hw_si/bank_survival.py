#!/usr/bin/env python3
"""Which FROZEN bank criteria survive the surface discount? (zero new judge calls)

For every Layer-1 A-bank rubric: alone-AUC on the HONEST rows, and alone-AUC
re-read inside deciles of the joint programmatic-V score.  A criterion whose
signal is length/format-borne collapses toward .5 under stratification; one that
measures something else does not.  This is the per-criterion version of the
cap_finalist "at chance within strata" test, applied to the bank instead of to
the dense model.

CPU only.  Usage: python bank_survival.py [cell ...]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import cells as C, closure_core as L

HERE = Path(__file__).resolve().parent

def run(cell):
    d = C.load(cell)
    sp = json.loads((HERE / f"{cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, g = d["y"], d["groups"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    rv = L.fit_block([d["V"]], fitm, monm, y, g)
    jv = np.full(len(y), np.nan); jv[fitm] = rv["oof_nl_fitmine"]; jv[monm] = rv["nl_mon"]
    st = L.decile_strata(jv[held], q=10)
    keep, med = L.clean_fit(d["A"][fitm])
    A = L.clean_apply(d["A"], keep, med)
    names = [d["a_names"][j] for j in keep]
    rows = []
    for k, n in enumerate(names):
        a = L.auc(y[held], A[held, k])
        adj, info = L.stratified_auc(y[held], A[held, k], st, min_n=20)
        rows.append({"criterion": n, "alone_AUC_HONEST": a,
                     "alone_AUC_within_V_strata": adj,
                     "shrinkage": abs(a - .5) - abs(adj - .5), **info})
    rows.sort(key=lambda r: -abs(r["alone_AUC_HONEST"] - .5))
    out = {"cell": cell, "n_HONEST": int(held.sum()), "joint_V_AUC_HONEST": L.auc(y[held], jv[held]),
           "n_criteria": len(rows),
           "n_above_.55_or_below_.45_pooled": sum(1 for r in rows if abs(r["alone_AUC_HONEST"]-.5) >= .05),
           "n_above_.55_or_below_.45_within_V_strata": sum(1 for r in rows if abs(r["alone_AUC_within_V_strata"]-.5) >= .05),
           "median_abs_dev_pooled": float(np.median([abs(r["alone_AUC_HONEST"]-.5) for r in rows])),
           "median_abs_dev_within_V": float(np.median([abs(r["alone_AUC_within_V_strata"]-.5) for r in rows])),
           "criteria": rows}
    (HERE / f"{cell}_bank_survival.json").write_text(json.dumps(out, indent=1))
    print(f"=== {cell}: {out['n_criteria']} bank criteria, joint V {out['joint_V_AUC_HONEST']:.3f}")
    print(f"    |AUC-.5|>=.05: {out['n_above_.55_or_below_.45_pooled']} pooled -> "
          f"{out['n_above_.55_or_below_.45_within_V_strata']} within V strata; "
          f"median |AUC-.5| {out['median_abs_dev_pooled']:.4f} -> {out['median_abs_dev_within_V']:.4f}")
    for r in rows[:8]:
        print(f"    {r['alone_AUC_HONEST']:.3f} -> {r['alone_AUC_within_V_strata']:.3f}  {r['criterion'][:50]}")
    return out

if __name__ == "__main__":
    todo = sys.argv[1:] or C.CELLS
    allr = {c: run(c) for c in todo}
    (HERE / "bank_survival_all.json").write_text(json.dumps(allr, indent=1))
