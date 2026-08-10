#!/usr/bin/env python3
"""Supplementary block decomposition: how much does each BLOCK of the scorecard buy?

Motivation specific to this cell.  Layer 1 reports V_nl = .709, A_nl = .658,
VA_nl = .724 -- i.e. the entire 198-rubric authored bank adds ~.015 AUC on top of 27
programmatic surface features.  That makes "how much do the mined criteria add" a
much sharper question here than on peer, where the bank carried most of the signal.
This script reads, on the SAME split and estimator as the closure curve:

  V            27 surface features alone
  A            the 198-rubric bank alone
  VA           both (= the closure curve's round-0 state)
  M_r          the mined A-routed criteria of rounds 1..r alone
  V + M_r      surface features plus mined criteria, WITHOUT the 198-rubric bank
  VA + M_r     the full round-r state (= the closure curve)

The V+M_r row is the one that answers "could the mined criteria have replaced the
bank", and M_r alone is the one that answers "what did this round's mining actually
measure".  Descriptive; not part of any stopping rule.

Usage: python block_decomposition.py --upto 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import nc_closure_lib as L
from readout import load_round_scores, load_dense

HERE = Path(__file__).resolve().parent


def fit_blocks(blocks, names, tag, pop, split, monitor_full, dsplit, dense):
    y, docket = pop["y"], pop["docket"]
    fm = split == "fit_mine"
    monf = monitor_full
    heldout = np.isin(dsplit, ["eval", "test"])

    X = np.column_stack(blocks)
    keep, meds = L.clean_fit(X[fm])
    Xc = L.clean_apply(X, keep, meds)
    if Xc.shape[1] == 0:
        return None
    Xfm, yfm, gfm = Xc[fm], y[fm], docket[fm]

    lin_oof = L.linear_oof(Xfm, yfm, gfm)
    nl_oof = np.mean([L.gbm_oof(Xfm, yfm, gfm, seed=s)[0] for s in L.SEEDS], axis=0)
    lin_m, nl_m, _ = L.fit_predict_monitor(Xfm, yfm, gfm, Xc[monf])

    lin_all = np.full(len(y), np.nan)
    nl_all = np.full(len(y), np.nan)
    lin_all[fm], nl_all[fm] = lin_oof, nl_oof
    lin_all[monf], nl_all[monf] = lin_m, nl_m.mean(0)

    out = {"tag": tag, "n_features": int(Xc.shape[1])}
    for label, mask in (("monitor_full", monf), ("honest", heldout)):
        out[label] = {"n": int(mask.sum()),
                      "lin": L.auc(y[mask], lin_all[mask]),
                      "nl": L.auc(y[mask], nl_all[mask])}
    out["honest"]["T"] = L.auc(y[heldout], dense[heldout])
    out["honest"]["Delta"] = out["honest"]["T"] - out["honest"]["nl"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, required=True)
    a = ap.parse_args()

    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    dense = load_dense()
    rounds = list(range(1, a.upto + 1))
    XM, _ = load_round_scores(rounds) if rounds else (None, [])

    specs = [("V", [pop["V"]]), ("A", [pop["A"]]), ("VA", [pop["V"], pop["A"]])]
    if XM is not None:
        specs += [(f"M1_{a.upto}", [XM]),
                  (f"V+M1_{a.upto}", [pop["V"], XM]),
                  (f"VA+M1_{a.upto}", [pop["V"], pop["A"], XM])]

    res = []
    for tag, blocks in specs:
        r = fit_blocks(blocks, None, tag, pop, split, monitor_full, dsplit, dense)
        if r:
            res.append(r)
            print(f"{tag:12s} feats={r['n_features']:4d}  MONITOR_FULL lin={r['monitor_full']['lin']:.4f} "
                  f"nl={r['monitor_full']['nl']:.4f}  honest nl={r['honest']['nl']:.4f} "
                  f"Delta={r['honest']['Delta']:.4f}", flush=True)

    (HERE / f"round{a.upto}_block_decomposition.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
