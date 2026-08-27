#!/usr/bin/env python3
"""(dC+, dC-) SWAP readout per round (freeze: "Delta_r, (DeltaC+, DeltaC-) swap pair").

Pair algebra, verbatim from methods/taste_decomposition/closure/swap_analysis.py:
AUC on a binary population is the fraction of concordant (positive, negative) pairs,
so partition pairs by DENSE correctness and read the bank inside each cell:

    D+ : dense orders the responded comment above the unresponded one   (weight w+)
    D- : dense has it backwards                                          (weight w-)
    C+ = P(bank concordant with truth | D+)      C- = P(... | D-)

    AUC_bank  = w+ C+ + w0 C0 + w- C-
    agreement = w+ C+ + .5 w0 + w- (1 - C-)

So flat AUC with rising rank-agreement is the SWAP signature: the bank inherits the
dense model's ERRORS (C- falls) as fast as its INSIGHTS (C+ rises).  Both are read
here directly.

Population: the honest 1,904 dense-held-out rows (bank predictions out-of-sample
everywhere: grouped-OOF inside FIT+MINE, refit-and-predict on the MONITOR side).

Usage: python swap_readout.py --upto 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import nc_closure_lib as L
from readout import load_dense

HERE = Path(__file__).resolve().parent


def cells(va, dense, y):
    P, N = np.where(y == 1)[0], np.where(y == 0)[0]
    D = np.sign(dense[P][:, None] - dense[N][None, :])
    S = np.sign(va[P][:, None] - va[N][None, :])
    conc = (S > 0).astype(float) + 0.5 * (S == 0)
    out = {}
    for name, m in (("Dp", D > 0), ("Dm", D < 0), ("D0", D == 0)):
        w = m.sum()
        out[name] = {"w": int(w), "C": float(conc[m].mean()) if w else np.nan}
    tot = sum(v["w"] for v in out.values())
    return {
        "w_plus": out["Dp"]["w"] / tot, "w_minus": out["Dm"]["w"] / tot,
        "w_tie": out["D0"]["w"] / tot,
        "C_plus": out["Dp"]["C"], "C_minus": out["Dm"]["C"], "C_tie": out["D0"]["C"],
        "AUC_dense": out["Dp"]["w"] / tot + 0.5 * out["D0"]["w"] / tot,
        "AUC_bank": (out["Dp"]["w"] * out["Dp"]["C"] + out["Dm"]["w"] * out["Dm"]["C"]
                     + out["D0"]["w"] * (out["D0"]["C"] if out["D0"]["w"] else 0)) / tot,
        "agree_discordant": (out["Dp"]["w"] * out["Dp"]["C"]
                             + out["Dm"]["w"] * (1 - out["Dm"]["C"])
                             + 0.5 * out["D0"]["w"]) / tot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, required=True)
    a = ap.parse_args()

    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    y = pop["y"]
    heldout = np.isin(dsplit, ["eval", "test"])
    dense = load_dense()[heldout]
    yy = y[heldout]

    states = {}
    for r in range(0, a.upto + 1):
        p = HERE / f"state{r}_preds.npz"
        if not p.exists():
            continue
        va = np.load(p, allow_pickle=True)["nl_mean"][heldout]
        st = cells(va, dense, yy)
        st["spearman_vs_dense"] = float(spearmanr(va, dense).statistic)
        states[r] = st

    steps = []
    ks = sorted(states)
    for a_, b_ in zip(ks, ks[1:]):
        dCp = states[b_]["C_plus"] - states[a_]["C_plus"]
        dCm = states[b_]["C_minus"] - states[a_]["C_minus"]
        wp, wm = states[b_]["w_plus"], states[b_]["w_minus"]
        steps.append({
            "step": f"r{a_}->r{b_}",
            "dC_plus": dCp, "dC_minus": dCm,
            "insight_inheritance": dCp, "error_inheritance": -dCm,
            "error_minus_insight": (-dCm) - dCp,
            "contrib_AUC_from_Dplus": wp * dCp, "contrib_AUC_from_Dminus": wm * dCm,
            "dAUC_bank": states[b_]["AUC_bank"] - states[a_]["AUC_bank"],
            "d_agree_discordant": states[b_]["agree_discordant"] - states[a_]["agree_discordant"],
            "d_spearman": states[b_]["spearman_vs_dense"] - states[a_]["spearman_vs_dense"],
            "swap_signature": bool(dCp > 0 and dCm < 0),
        })

    out = {"population": "honest dense-held-out rows", "n": int(heldout.sum()),
           "states": states, "steps": steps}
    (HERE / f"round{a.upto}_swap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
