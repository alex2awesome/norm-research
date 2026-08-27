#!/usr/bin/env python3
"""ROUND-0, the "what does the stack actually lean on" line (code_v3 s6 shape),
math.SE vote-score cell.

Two questions this cell has to answer BEFORE any round runs, because two of the
brief's five upstream priors (answer length, LaTeX density) are ALREADY COLUMNS IN
THE V BLOCK:

  1. Is the dense arm's residual over the bank a LENGTH effect?  If Delta survives
     stratification on length deciles, no.
  2. Same for LaTeX density, and for the two jointly.

Because these channels are already articulated, they cannot be "discounted off
Delta" the way a mined Track-B channel can -- the bank reads them too.  What the
stratification tests is narrower and still worth knowing: whether the residual is
concentrated inside a length/LaTeX band.

Also reports the ABLATION direction the closure protocol allows on CPU: VA_nl refit
with the surface v_* columns REMOVED, so we can see how much of the articulated
instrument's .63-.65 is surface and how much is rubric.

CPU only.  Usage: OMP_NUM_THREADS=6 python3 length_stratification.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
import closure_core as L
from position_line import within_question_auc

HERE = Path(__file__).resolve().parent
SURFACE = set(C.V_ALREADY_ARTICULATED_SURFACE)


def main():
    d = C.load()
    sp = json.loads((HERE / "mathse_accepted_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    y, g = d["y"], np.array([str(x) for x in d["groups"]])
    z = np.load(HERE / "mathse_accepted_r0_preds.npz", allow_pickle=True)
    va = z["va_nl"]

    vn = d["v_names"]
    cols = {n: d["V"][:, k] for k, n in enumerate(vn)}
    out = {"cell": "mathse_accepted", "sklearn": C.sklearn_guard(),
           "note": "length and LaTeX are ALREADY V columns on this cell; stratifying on "
                   "them tests whether the residual is concentrated in a band, NOT whether "
                   "the bank is innocent of them."}

    strata = {
        "v_log_len_decile": L.decile_strata(cols["v_log_len"], q=10),
        "v_latex_density_decile": L.decile_strata(cols["v_latex_density"], q=10),
        "v_n_display_math_decile": L.decile_strata(cols["v_n_display_math"], q=10),
    }
    joint = (L.decile_strata(cols["v_log_len"], q=4) * 4
             + L.decile_strata(cols["v_latex_density"], q=4))
    strata["len_x_latex_4x4"] = joint

    for pop_name, m in (("MONITOR", monm), ("HONEST", held)):
        rec = {}
        ym, dm, vm = y[m], d["dense"][m], va[m]
        rec["pooled_T"] = float(roc_auc_score(ym, dm))
        rec["pooled_VA"] = float(roc_auc_score(ym, vm))
        rec["pooled_Delta"] = rec["pooled_T"] - rec["pooled_VA"]
        for name, st in strata.items():
            tA, info = L.stratified_auc(ym, dm, st[m], min_n=20)
            vA, _ = L.stratified_auc(ym, vm, st[m], min_n=20)
            rec[name] = {"T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **info}
        out[f"stratified_{pop_name}"] = rec

    # alone-AUC of the surface columns, on FIT+MINE only
    out["surface_alone_AUC_FITMINE"] = {
        n: float(roc_auc_score(y[fitm], cols[n][fitm])) for n in vn if n in SURFACE}

    # ABLATION: refit the bank without the surface v_* columns
    keep_v = [k for k, n in enumerate(vn) if n not in SURFACE]
    r_nosurf = L.fit_block([d["V"][:, keep_v], d["A"]], fitm, monm, y, d["groups"])
    r_aonly = L.fit_block([d["A"]], fitm, monm, y, d["groups"])
    r_vonly = L.fit_block([d["V"]], fitm, monm, y, d["groups"])
    out["ablations_MONITOR"] = {
        "VA_nl_full": float(roc_auc_score(y[monm], va[monm])),
        "VA_nl_no_surface_v": float(roc_auc_score(y[monm], r_nosurf["nl_mon"])),
        "A_only_nl": float(roc_auc_score(y[monm], r_aonly["nl_mon"])),
        "V_only_nl": float(roc_auc_score(y[monm], r_vonly["nl_mon"])),
        "T": float(roc_auc_score(y[monm], d["dense"][monm])),
        "n_surface_v_dropped": len(vn) - len(keep_v),
        "within_question": {
            "VA_nl_full": within_question_auc(y[monm], va[monm], g[monm])[0],
            "A_only_nl": within_question_auc(y[monm], r_aonly["nl_mon"], g[monm])[0],
            "V_only_nl": within_question_auc(y[monm], r_vonly["nl_mon"], g[monm])[0],
            "T": within_question_auc(y[monm], d["dense"][monm], g[monm])[0]},
    }
    (HERE / "length_stratification.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
