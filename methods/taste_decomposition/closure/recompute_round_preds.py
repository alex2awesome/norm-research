#!/usr/bin/env python3
"""Recompute the per-round VA_nl prediction VECTORS for the Layer-3 closure pilot.

The pilot's stage4_round4.py reported per-round AUCs but never persisted the
per-row VA_nl predictions.  The swap decomposition (notes/2026-08-06) needs the
vectors, so this script re-runs the identical estimator (frozen Layer-1 spec,
fit inside FIT+MINE only, VA_nl = mean over seeds {0,1,2}) for all five bank
states and writes the per-row predictions to round_preds_all.npz.

Reproduction check: the printed MONITOR / honest-level AUCs must match
round4_results.json exactly (same seeds, same deterministic GroupKFold).

CPU only, ~6 min.  Usage: python recompute_round_preds.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import closure_lib as L
from stage4_readout import build_blocks, fit_block
from stage4_round4 import load_round_blocks

HERE = Path(__file__).resolve().parent


def main():
    t0 = time.time()
    pop, split, dsplit, XA1, XB1, a1_ids, b1_ids, summary = build_blocks()
    XA2, _, a2_ids, _, _ = load_round_blocks(2)
    XA3, _, a3_ids, _, _ = load_round_blocks(3)
    XA4, _, a4_ids, _, _ = load_round_blocks(4)

    y, nt = pop["y"], pop["ntitle"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])

    dense = pd.read_csv(HERE / "peer_verdict_dense_preds.csv").set_index("i").loc[
        np.arange(len(y)), "dense_prob"
    ].values

    banks = {
        "round0": [pop["V"], pop["A"]],
        "round1": [pop["V"], pop["A"], XA1],
        "round2": [pop["V"], pop["A"], XA1, XA2],
        "round3": [pop["V"], pop["A"], XA1, XA2, XA3],
        "round4": [pop["V"], pop["A"], XA1, XA2, XA3, XA4],
    }

    out = {}
    check = {}
    for k, blocks in banks.items():
        t = time.time()
        r = fit_block(blocks, fitm, monm, y, nt)
        # honest per-row vector: OOF inside FIT+MINE, held-out preds on MONITOR
        va = np.full(len(y), np.nan)
        va[fitm] = r["oof_nl_fitmine"]
        va[monm] = r["nl_mon"]
        lin = np.full(len(y), np.nan)
        lin[fitm] = r["oof_lin_fitmine"]
        lin[monm] = r["lin_mon"]
        out[f"va_nl_{k}"] = va
        out[f"va_lin_{k}"] = lin
        check[k] = {
            "n_features": r["n_features"],
            "VA_nl_MONITOR_all": L.auc(y[monm], r["nl_mon"]),
            "VA_lin_MONITOR_all": L.auc(y[monm], r["lin_mon"]),
            "VA_nl_honest_level_heldout1244": L.auc(y[held], va[held]),
        }
        print(f"{k}: {check[k]}  ({time.time()-t:.0f}s)", flush=True)

    out["y"] = y
    out["ntitle"] = nt.astype(str)
    out["dense"] = dense
    out["held"] = held
    out["monitor"] = monm
    out["fit_mine"] = fitm
    np.savez_compressed(HERE / "round_preds_all.npz", **out)

    # verify against the canonical result file
    canon = json.loads((HERE / "round4_results.json").read_text())["rounds"]
    ok = True
    for k, v in check.items():
        for key in ("VA_nl_MONITOR_all", "VA_nl_honest_level_heldout1244"):
            d = abs(v[key] - canon[k][key])
            if d > 1e-9:
                ok = False
                print(f"MISMATCH {k}.{key}: {v[key]} vs {canon[k][key]} (d={d:.2e})")
    print("REPRODUCTION EXACT:" if ok else "REPRODUCTION FAILED", f"({time.time()-t0:.0f}s)")
    (HERE / "round_preds_all.report.json").write_text(
        json.dumps({"check": check, "reproduces_round4_results": ok}, indent=2)
    )


if __name__ == "__main__":
    main()
