#!/usr/bin/env python3
"""Round-1 mechanism diagnostics (supplementary to stage4_readout.py).

Three questions the headline Delta table cannot answer on its own:
 1. Do the 14 A-routed criteria carry ANY label signal on MONITOR by themselves?
 2. Did mining actually hit its target -- i.e. do the new criteria track the DENSE
    score (the label-blind mining target), not just y?
 3. Which individual criteria moved, so a later round knows what to keep.

CPU only.  Usage: python stage4_mechanism.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import closure_lib as L
from stage4_readout import build_blocks, fit_block

HERE = Path(__file__).resolve().parent


def main():
    pop, split, dsplit, XA, XB, a_ids, b_ids, summary = build_blocks()
    y, nt = pop["y"], pop["ntitle"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])
    same = monm & held
    same_in_mon = same[monm]

    dense = pd.read_csv(HERE / "peer_verdict_dense_preds.csv").set_index("i").loc[
        np.arange(len(y)), "dense_prob"
    ].values

    ra = fit_block([XA], fitm, monm, y, nt)
    r0 = fit_block([pop["V"], pop["A"]], fitm, monm, y, nt)
    r1 = fit_block([pop["V"], pop["A"], XA], fitm, monm, y, nt)

    ymon = y[monm]
    out = {
        "A_block_alone": {
            "n_features": ra["n_features"],
            "lin_MONITOR_all": L.auc(ymon, ra["lin_mon"]),
            "nl_MONITOR_all": L.auc(ymon, ra["nl_mon"]),
            "lin_MONITOR_samerows": L.auc(ymon[same_in_mon], ra["lin_mon"][same_in_mon]),
            "nl_MONITOR_samerows": L.auc(ymon[same_in_mon], ra["nl_mon"][same_in_mon]),
            "lin_OOF_fitmine_MINING_CONTAMINATED": L.auc(y[fitm], ra["oof_lin_fitmine"]),
            "nl_OOF_fitmine_MINING_CONTAMINATED": L.auc(y[fitm], ra["oof_nl_fitmine"]),
        }
    }

    # --- did mining hit its target?  rank agreement with the DENSE score ---------
    keepA, medA = L.clean_fit(XA[fitm])
    XAmon = L.clean_apply(XA[monm], keepA, medA)
    kept_a_ids = [a_ids[j] for j in keepA]
    dmon = dense[monm]
    out["dense_alignment_MONITOR"] = {
        "rho_round0_VAnl_vs_dense": float(spearmanr(r0["nl_mon"], dmon).statistic),
        "rho_round1_VAnl_vs_dense": float(spearmanr(r1["nl_mon"], dmon).statistic),
        "rho_Ablock_vs_dense": float(spearmanr(ra["nl_mon"], dmon).statistic),
        "note": "MONITOR rows; dense is in-sample for 943/1192 of them, so this is a "
                "rank-agreement diagnostic, not an honest generalisation number.",
    }
    dsame = dense[same]
    out["dense_alignment_MONITOR_samerows"] = {
        "n": int(same.sum()),
        "rho_round0_VAnl_vs_dense": float(spearmanr(r0["nl_mon"][same_in_mon], dsame).statistic),
        "rho_round1_VAnl_vs_dense": float(spearmanr(r1["nl_mon"][same_in_mon], dsame).statistic),
        "rho_Ablock_vs_dense": float(spearmanr(ra["nl_mon"][same_in_mon], dsame).statistic),
    }

    # --- per-criterion: alone-AUC vs y, and rank corr with dense -----------------
    prov = {p["blind_id"]: p["name"] for p in json.loads((HERE / "round1_proposals_provenance.json").read_text())}
    per = {}
    for k, cid in enumerate(kept_a_ids):
        col = XAmon[:, k]
        per[cid] = {
            "name": prov[cid],
            "alone_AUC_vs_y_MONITOR": L.auc(ymon, col),
            "rho_vs_dense_MONITOR": float(spearmanr(col, dmon).statistic),
        }
    out["per_A_criterion_MONITOR"] = per

    XBmon_keep, medB = L.clean_fit(XB[fitm])
    XBmon = L.clean_apply(XB[monm], XBmon_keep, medB)
    perb = {}
    for k, j in enumerate(XBmon_keep):
        cid = b_ids[j]
        perb[cid] = {
            "name": prov[cid],
            "alone_AUC_vs_y_MONITOR": L.auc(ymon, XBmon[:, k]),
            "rho_vs_dense_MONITOR": float(spearmanr(XBmon[:, k], dmon).statistic),
        }
    out["per_B_criterion_MONITOR"] = perb

    (HERE / "round1_mechanism.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
