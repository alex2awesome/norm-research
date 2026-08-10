#!/usr/bin/env python3
"""Leave-one-company-out jackknife on Delta_beyond at ONE bank state.

`jackknife.py` refits every state 0..r, which on this cell is ~15 minutes per state.
Only the BEST-BANK-SO-FAR state needs the width statistic for the quotable line, so this
refits exactly that one state and reports:

  * Delta on HONEST under the declared T convention (mean over dense seeds of the AUC),
  * the leave-one-company-out jackknife over the 45 dense-held-out companies (mean, SE,
    range, pseudo-CI, most influential company),
  * the eval-only / test-only split (this cell selected the dense chain on EVAL, so TEST
    is the selection-free half),
  * the swap pair, and the within-company pair concordance.

CPU only.  Usage: python jack_state.py --round 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
import closure_core as L
import stage1_slice as S1
from readout import swap_pair

HERE = Path(__file__).resolve().parent


def T_of(d, mask):
    y = d["y"][mask]
    return float(np.mean([roc_auc_score(y, d["dense_seeds"][mask, j])
                          for j in range(d["dense_seeds"].shape[1])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    a = ap.parse_args()
    C.sklearn_guard()
    d = C.load()
    sp = json.loads((HERE / "press_verdict_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    y, g = d["y"], d["groups"]

    blocks, tags = S1.current_blocks(d, str(int(a.round) + 1))
    r = L.fit_block(blocks, fitm, monm, y, g)
    va = np.full(len(y), np.nan)
    va[fitm] = r["oof_nl_fitmine"]
    va[monm] = r["nl_mon"]

    def delta(mask):
        return T_of(d, mask) - L.auc(y[mask], va[mask])

    rows = []
    for c in sorted(set(g[held].tolist())):
        m = held & (g != c)
        if len(set(y[m].tolist())) < 2:
            continue
        rows.append({"dropped_company": c, "n_remaining": int(m.sum()), "Delta": delta(m)})
    dv = np.array([x["Delta"] for x in rows])
    n = len(dv)
    se = float(np.sqrt((n - 1) / n * ((dv - dv.mean()) ** 2).sum()))
    pooled = delta(held)
    worst = max(rows, key=lambda x: abs(x["Delta"] - pooled))

    out = {"cell": "press_verdict", "state": f"round {a.round}", "blocks": tags,
           "n_features": r["n_features"],
           "T_convention": "mean over dense seeds of AUC",
           "HONEST": {"n": int(held.sum()), "T": T_of(d, held),
                      "VA_nl": L.auc(y[held], va[held]), "Delta_beyond": pooled},
           "MONITOR": {"n": int(monm.sum()), "T": T_of(d, monm),
                       "VA_nl": L.auc(y[monm], va[monm]), "Delta_beyond": delta(monm)},
           "eval_only_selection_contaminated": {
               "n": int((d["dense_split"] == "eval").sum()),
               "Delta_beyond": delta(d["dense_split"] == "eval")},
           "test_only_SELECTION_FREE": {
               "n": int((d["dense_split"] == "test").sum()),
               "Delta_beyond": delta(d["dense_split"] == "test")},
           "jackknife": {"n_companies": n, "mean": float(dv.mean()), "SE": se,
                         "range": [float(dv.min()), float(dv.max())],
                         "pseudo_CI95": [float(dv.mean() - 1.96 * se),
                                         float(dv.mean() + 1.96 * se)],
                         "most_influential_company": worst["dropped_company"],
                         "Delta_without_it": worst["Delta"],
                         "leave_one_out": sorted(rows, key=lambda x: x["Delta"])},
           "swap_HONEST": swap_pair(y[held], d["dense"][held], va[held]),
           }
    # within-company pair concordance at this state
    num_t = num_v = tot = 0.0
    for c in sorted(set(g[held].tolist())):
        m = held & (g == c)
        yy = y[m]
        if yy.sum() == 0 or (yy == 0).sum() == 0:
            continue
        pi, ni = np.where(yy == 1)[0], np.where(yy == 0)[0]
        for name, s in (("t", d["dense"][m]), ("v", va[m])):
            diff = s[pi][:, None] - s[ni][None, :]
            conc = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
            if name == "t":
                num_t += conc
            else:
                num_v += conc
        tot += len(pi) * len(ni)
    out["within_company_HONEST"] = {"n_pairs": int(tot), "T": num_t / tot,
                                    "VA_nl": num_v / tot, "Delta": (num_t - num_v) / tot,
                                    "note": "uses the seed-ENSEMBLE dense vector, so compare "
                                            "against the ensemble-basis pooled Delta, not T"}
    (HERE / f"press_verdict_r{a.round}_jack_state.json").write_text(
        json.dumps(out, indent=1, default=float))
    j = out["jackknife"]
    print(json.dumps({k: v for k, v in out.items() if k != "jackknife"}, indent=1, default=float))
    print(f"\nJACKKNIFE {j['n_companies']} companies: Delta {pooled:+.4f} mean {j['mean']:+.4f} "
          f"SE {j['SE']:.4f} range [{j['range'][0]:+.4f}, {j['range'][1]:+.4f}] "
          f"CI [{j['pseudo_CI95'][0]:+.4f}, {j['pseudo_CI95'][1]:+.4f}] "
          f"most influential {j['most_influential_company']} -> {j['Delta_without_it']:+.4f}")


if __name__ == "__main__":
    main()
