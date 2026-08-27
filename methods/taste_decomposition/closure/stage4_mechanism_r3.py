#!/usr/bin/env python3
"""Round-3 mechanism diagnostics (supplementary to stage4_round2.py).

Questions the curve table cannot answer:
 1. Do the round-2 A criteria carry label signal alone, on MONITOR?
 2. Did round 2 move VA_nl's RANKING closer to the dense score (the label-blind
    mining target), as round 1 did?
 3. Did the interaction-shaped STEER pay off?  Round-2 Track A was deliberately
    ~2/3 composite; compare composite vs non-composite sub-blocks directly.
 4. Per-criterion alone-AUC and dense-alignment, for round-3 selection.

CPU only.  Usage: python stage4_mechanism_r2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import closure_lib as L
from stage4_readout import build_blocks, fit_block
from stage4_round3 import load_round_blocks

HERE = Path(__file__).resolve().parent


def main():
    pop, split, dsplit, XA1, XB1, a1_ids, b1_ids, summary = build_blocks()
    XA2, XB2, a2_ids, b2_ids, _ = load_round_blocks(2)
    XA3, XB3, a3_ids, b3_ids, _ = load_round_blocks(3)
    y, nt = pop["y"], pop["ntitle"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])
    same = monm & held
    same_in_mon = same[monm]
    ymon = y[monm]

    dense = pd.read_csv(HERE / "peer_verdict_dense_preds.csv").set_index("i").loc[
        np.arange(len(y)), "dense_prob"
    ].values
    dmon = dense[monm]

    track_a2 = json.loads((HERE / "round3_track_a.json").read_text())
    comp_by_src = {c["id"]: c.get("composite", False) for c in track_a2["criteria"]}
    prov2 = {p["blind_id"]: p for p in json.loads((HERE / "round3_proposals_provenance.json").read_text())}
    is_comp = np.array([comp_by_src.get(prov2[b]["src_id"], False) for b in a3_ids])

    out = {}

    # 1. round-2 A block alone
    ra2 = fit_block([XA3], fitm, monm, y, nt)
    out["A3_block_alone"] = {
        "n_criteria": len(a3_ids),
        "n_features": ra2["n_features"],
        "lin_MONITOR_all": L.auc(ymon, ra2["lin_mon"]),
        "nl_MONITOR_all": L.auc(ymon, ra2["nl_mon"]),
        "nl_MONITOR_samerows": L.auc(ymon[same_in_mon], ra2["nl_mon"][same_in_mon]),
        "nl_OOF_fitmine_MININGCONTAM": L.auc(y[fitm], ra2["oof_nl_fitmine"]),
    }

    # 3. composite vs non-composite sub-blocks (the steer test)
    for tag, mask in (("composite", is_comp), ("non_composite", ~is_comp)):
        if mask.sum() == 0:
            continue
        r = fit_block([XA3[:, mask]], fitm, monm, y, nt)
        out[f"A3_{tag}_alone"] = {
            "n_criteria": int(mask.sum()),
            "lin_MONITOR_all": L.auc(ymon, r["lin_mon"]),
            "nl_MONITOR_all": L.auc(ymon, r["nl_mon"]),
        }

    # 2. dense rank alignment across rounds
    banks = {
        "round0": [pop["V"], pop["A"]],
        "round1": [pop["V"], pop["A"], XA1],
        "round2": [pop["V"], pop["A"], XA1, XA2],
        "round3": [pop["V"], pop["A"], XA1, XA2, XA3],
    }
    rho = {}
    for k, v in banks.items():
        r = fit_block(v, fitm, monm, y, nt)
        rho[k] = {
            "rho_VAnl_vs_dense_MONITOR_all": float(spearmanr(r["nl_mon"], dmon).statistic),
            "rho_VAnl_vs_dense_samerows": float(
                spearmanr(r["nl_mon"][same_in_mon], dense[same]).statistic),
            "Delta_interact_MONITOR_all": L.auc(ymon, r["nl_mon"]) - L.auc(ymon, r["lin_mon"]),
        }
    rho["A3_block_vs_dense_MONITOR_all"] = float(spearmanr(ra2["nl_mon"], dmon).statistic)
    out["dense_alignment"] = rho

    # 4. per-criterion
    keepA, medA = L.clean_fit(XA3[fitm])
    XA2mon = L.clean_apply(XA3[monm], keepA, medA)
    per = {}
    for k, j in enumerate(keepA):
        cid = a3_ids[j]
        per[cid] = {
            "name": prov2[cid]["name"],
            "src_id": prov2[cid]["src_id"],
            "composite": bool(is_comp[j]),
            "alone_AUC_vs_y_MONITOR": L.auc(ymon, XA2mon[:, k]),
            "rho_vs_dense_MONITOR": float(spearmanr(XA2mon[:, k], dmon).statistic),
        }
    out["per_A3_criterion_MONITOR"] = per

    (HERE / "round3_mechanism.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_A3_criterion_MONITOR"}, indent=2))
    print()
    print("%-4s %-3s %-50s %8s %8s" % ("id", "cmp", "criterion", "aloneAUC", "rho_dns"))
    for k, v in sorted(per.items(), key=lambda t: -t[1]["rho_vs_dense_MONITOR"]):
        print("%-4s %-3s %-50s %8.4f %8.3f" % (
            k, "C" if v["composite"] else "-", v["name"][:50],
            v["alone_AUC_vs_y_MONITOR"], v["rho_vs_dense_MONITOR"]))


if __name__ == "__main__":
    main()
