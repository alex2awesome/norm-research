#!/usr/bin/env python3
"""GEPA Stage 4 (peer-verdict): swap ACCEPTED winners' full-population scores
into the nuisance-corrected 55-criterion round-4 bank and recompute Delta_beyond
on the 1,244-row honest (dense-held-out) population, old vs new.  CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import closure_lib as L
from stage4_readout import fit_block
from stage4_round4 import load_round_blocks

HERE = Path(__file__).resolve().parent


def main():
    pop = L.load_population()
    _, split, dsplit, mining = L.load_splits()
    y, nt = pop["y"], pop["ntitle"]
    fitm = split == "fit_mine"
    monm = split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])

    XA1, _, a1, _, _ = load_round_blocks(1)
    XA2, _, a2, _, _ = load_round_blocks(2)
    XA3, _, a3, _, _ = load_round_blocks(3)
    XA4, _, a4, _, _ = load_round_blocks(4)
    restraint_j = a2.index("P13")
    XA2c = np.delete(XA2, restraint_j, axis=1)
    a2c = [c for c in a2 if c != "P13"]

    tag_to_col = {}
    off = 0
    for rn, ids in ((1, a1), (2, a2c), (3, a3), (4, a4)):
        for k, cid in enumerate(ids):
            tag_to_col[f"r{rn}:{cid}"] = off + k
        off += len(ids)

    XA_surviving = np.column_stack([XA1, XA2c, XA3, XA4])
    assert XA_surviving.shape[1] == 55, XA_surviving.shape

    winners = json.loads((HERE / "gepa_winners_peer.json").read_text())
    accepted = [w for w in winners if w["ACCEPTED"]]
    wz = np.load(HERE / "gepa_winners_scores_peer.npz", allow_pickle=True)
    w_i = [int(x) for x in wz["i"]]
    assert w_i == list(range(len(y))), (
        "winner rescore row order != pop row order (peer_verdict_population.csv "
        "`i` is assumed arange-aligned with load_population(), as build_blocks() "
        "asserts elsewhere)")
    wrep = json.loads((HERE / "gepa_winners_scores_peer.report.json").read_text())
    w_collapsed = {c["parent_tag"] for c in wrep["collapse"] if c["COLLAPSED"]}

    XA_gepa = XA_surviving.copy()
    swapped, skipped = [], []
    for j, tag in enumerate(wz["parent_tags"]):
        tag = str(tag)
        if tag in w_collapsed:
            skipped.append(tag)
            continue
        col = tag_to_col[tag]
        # wz rows are in population.csv order (`rows` iterated directly), which is
        # also pop's row order since load_population preserves the source order --
        # verified by the row-count assertion below.
        XA_gepa[:, col] = wz["X"][:, j]
        swapped.append(tag)
    print(f"Swapped {len(swapped)} GEPA-accepted, non-collapsed winners: {swapped}")
    if skipped:
        print(f"SKIPPED (collapsed on full-population rescore): {skipped}")

    dense = pd.read_csv(HERE / "peer_verdict_dense_preds.csv").set_index("i").loc[
        np.arange(len(y)), "dense_prob"].values
    T_held = L.auc(y[held], dense[held])

    def honest(blocks):
        r = fit_block(blocks, fitm, monm, y, nt)
        va_full = np.full(len(y), np.nan)
        va_full[fitm] = r["oof_nl_fitmine"]
        va_full[monm] = r["nl_mon"]
        va_h = L.auc(y[held], va_full[held])
        return {"n_features": r["n_features"], "VA_nl_honest_1244": va_h,
                "Delta_honest_1244": T_held - va_h}

    r_all56 = honest([pop["V"], pop["A"], XA1, XA2, XA3, XA4])          # original pilot state
    r_55_pregepa = honest([pop["V"], pop["A"], XA_surviving])           # nuisance-corrected, PRE-GEPA
    r_55_postgepa = honest([pop["V"], pop["A"], XA_gepa])               # nuisance-corrected + GEPA

    out = {
        "T_held_1244": T_held,
        "n_targeted": len(winners), "n_accepted": len(accepted),
        "swapped_tags": swapped, "skipped_collapsed_tags": skipped,
        "original_pilot_56_incl_restraint": r_all56,
        "nuisance_corrected_55_pre_gepa": r_55_pregepa,
        "nuisance_corrected_55_post_gepa": r_55_postgepa,
    }
    out["movement_removal_only"] = (r_55_pregepa["Delta_honest_1244"]
                                    - r_all56["Delta_honest_1244"])
    out["movement_gepa_only"] = (r_55_postgepa["Delta_honest_1244"]
                                 - r_55_pregepa["Delta_honest_1244"])
    out["movement_total"] = (r_55_postgepa["Delta_honest_1244"]
                             - r_all56["Delta_honest_1244"])
    (HERE / "gepa_finalize_peer_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
