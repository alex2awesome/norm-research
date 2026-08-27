#!/usr/bin/env python3
"""Judge-free surface audit: what does the PROGRAMMATIC V block alone explain?

This is the zero-judge-call complement to the mined Track-B map, and on these two
cells it is not decoration: the FIT+MINE SHAP screen puts `v_char_count` (Style
Invitational, mean|SHAP| .442) and `v_word_count` (.213) three to four times above
the strongest judged criterion, so "the bank" on this cell is substantially a
length model wearing a rubric.

Readouts, all on populations where both instruments are out-of-sample:
  * alone-AUC of every V feature;
  * joint V model (grouped-OOF HistGB inside FIT+MINE, held-out on MONITOR);
  * decile-stratified T_adj / VA_adj / Delta_adj on the joint V score -- the
    cap_finalist "is the dense model at chance inside strata?" test;
  * stacked increment of dense and of the bank over the joint V model.

CPU only.  Usage: python vblock_audit.py [cell ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L
from readout import stack_oof

HERE = Path(__file__).resolve().parent


def run(cell):
    d = C.load(cell)
    sp = json.loads((HERE / f"{cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, g, dense = d["y"], d["groups"], d["dense"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])

    z = np.load(HERE / f"{cell}_r0_preds.npz")
    va = z["va_nl"]

    rv = L.fit_block([d["V"]], fitm, monm, y, g)
    jv = np.full(len(y), np.nan)
    jv[fitm] = rv["oof_nl_fitmine"]
    jv[monm] = rv["nl_mon"]

    keep, med = L.clean_fit(d["V"][fitm])
    Vc = L.clean_apply(d["V"], keep, med)
    names = [d["v_names"][j] for j in keep]
    feats = [{"feature": n, "alone_AUC_HONEST": L.auc(y[held], Vc[held, k]),
              "alone_AUC_FITMINE": L.auc(y[fitm], Vc[fitm, k])}
             for k, n in enumerate(names)]
    feats.sort(key=lambda r: -abs(r["alone_AUC_HONEST"] - .5))

    yh, dh, vh, jh = y[held], dense[held], va[held], jv[held]
    st = L.decile_strata(jh, q=10)
    tA, iA = L.stratified_auc(yh, dh, st, min_n=20)
    vA, _ = L.stratified_auc(yh, vh, st, min_n=20)

    s_vd = stack_oof([jh, dh], yh, g[held])
    s_vb = stack_oof([jh, vh], yh, g[held])
    out = {
        "cell": cell, "n_HONEST": int(held.sum()),
        "V_features": feats,
        "joint_V_model": {"n_features": rv["n_features"],
                          "AUC_HONEST": L.auc(yh, jh),
                          "AUC_MONITOR": L.auc(y[monm], rv["nl_mon"]),
                          "AUC_OOF_fitmine": L.auc(y[fitm], rv["oof_nl_fitmine"])},
        "stratified_on_joint_V_HONEST_q10": {
            "pooled_T": L.auc(yh, dh), "pooled_VA_nl": L.auc(yh, vh),
            "pooled_Delta": L.auc(yh, dh) - L.auc(yh, vh),
            "T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA},
        "stacked_increment_over_V": {
            "AUC_jointV": L.auc(yh, jh), "AUC_dense": L.auc(yh, dh),
            "AUC_bank_VA_nl": L.auc(yh, vh),
            "dense_increment_over_V": L.auc(yh, s_vd) - L.auc(yh, jh),
            "bank_increment_over_V": L.auc(yh, s_vb) - L.auc(yh, jh),
            "ci_dense_increment": L.group_boot_ci(yh, s_vd, jh, g[held]),
            "ci_bank_increment": L.group_boot_ci(yh, s_vb, jh, g[held])},
    }
    (HERE / f"{cell}_vblock_audit.json").write_text(json.dumps(out, indent=1))
    j = out["joint_V_model"]
    s = out["stratified_on_joint_V_HONEST_q10"]
    k = out["stacked_increment_over_V"]
    print(f"=== {cell}: joint V AUC HONEST {j['AUC_HONEST']:.4f} (MONITOR {j['AUC_MONITOR']:.4f})")
    print(f"    top V: " + ", ".join(f"{f['feature']} {f['alone_AUC_HONEST']:.3f}"
                                     for f in feats[:5]))
    print(f"    stratified on V: T {s['pooled_T']:.4f}->{s['T_adj']:.4f}  "
          f"VA {s['pooled_VA_nl']:.4f}->{s['VA_adj']:.4f}  "
          f"Delta {s['pooled_Delta']:+.4f}->{s['Delta_adj']:+.4f}")
    print(f"    stacked over V: dense {k['dense_increment_over_V']:+.4f} "
          f"(p={k['ci_dense_increment']['p_gt0']:.2f})  bank {k['bank_increment_over_V']:+.4f} "
          f"(p={k['ci_bank_increment']['p_gt0']:.2f})")
    return out


if __name__ == "__main__":
    todo = sys.argv[1:] or C.CELLS
    allr = {c: run(c) for c in todo}
    (HERE / "vblock_audit_all.json").write_text(json.dumps(allr, indent=1))
