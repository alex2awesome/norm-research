#!/usr/bin/env python3
"""Recovery-audit Q4(ii): would error-conditioned slices help?

The pilot slice rule shows proposers the top |dense rank - VA rank| rows.  The
proposed intervention shows instead the depleted stack's WORST-PREDICTED rows.
True error-conditioning needs the label, so it BREAKS the label-blind mining
protocol; computed here as an ORACLE DIAGNOSTIC that upper-bounds the whole
"better slices" route: if even the oracle slice does not surface the held-out
concepts' activity, no label-free slice rule can.

For each replicate:
  * oracle error slice = 30 worst-predicted positives (y=1, lowest depleted-VA
    OOF percentile) + 30 worst-predicted negatives (y=0, highest), mining rows;
  * compare row overlap with the actually-shown depletion slice;
  * per held-out concept, measure how much of the concept's column activity each
    slice surfaces: nonnull fraction on slice rows, and the concept's
    discrimination (rank-AUC of column vs y) restricted to slice rows,
    vs a random-mining-row baseline.
CPU only, uses stored preds (m3_depleted_preds.npz) - no refits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RMM = HERE.parent
CLOSURE = RMM.parent
sys.path.insert(0, str(RMM))
sys.path.insert(0, str(CLOSURE))
import closure_lib as L  # noqa: E402
from stage4_readout import build_blocks  # noqa: E402

cfg = json.loads((RMM / "m3_concepts.json").read_text())
z = np.load(RMM / "m3_depleted_preds.npz", allow_pickle=True)
y, dense = z["y"], z["dense"]
fitm, monm = z["fit_mine"], z["monitor"]

pop, split, dsplit, *_ = build_blocks()
A = pop["A"]
_, _, _, mining = L.load_splits()
mi = np.where(mining.astype(bool))[0]

rng = np.random.default_rng(7)
out = {"note": "oracle diagnostic; uses labels for row selection, which the sealed "
               "mining protocol forbids -- upper bound on the better-slices route",
       "replicates": {}}

for rep in ("rep1", "rep2", "rep3"):
    oof = z[f"oof_fitmine_{rep}"]
    va = np.full(len(y), np.nan)
    va[fitm] = oof
    # VA percentile within mining rows (same convention as the slice rule)
    v_rank = pd.Series(va[mi]).rank(pct=True).values
    ymi = y[mi]
    # oracle: worst-predicted positives (low VA pct) and negatives (high VA pct)
    pos = mi[ymi == 1][np.argsort(v_rank[ymi == 1])[:30]]
    neg = mi[ymi == 0][np.argsort(-v_rank[ymi == 0])[:30]]
    err_rows = set(map(int, np.concatenate([pos, neg])))
    # the slice the fleet actually read
    shown = {r["i"] for r in json.loads((RMM / f"slice_{rep}.json").read_text())}

    rrec = {"overlap_error_vs_shown": len(err_rows & shown),
            "n_slice": 60, "concepts": []}

    held_cols = {c["concept"]: c["footprint_columns"]
                 for c in cfg["replicate_detail"][rep]}
    for name, cols in held_cols.items():
        col = A[:, cols]
        nonnull = (~np.isnan(col)).any(axis=1)
        # concept score = mean over footprint (nan-aware)
        with np.errstate(all="ignore"):
            score = np.nanmean(col, axis=1)

        def surface(rows):
            rows = np.array(sorted(rows), dtype=int)
            nn = nonnull[rows]
            res = {"nonnull_frac": float(nn.mean())}
            rr = rows[nn & ~np.isnan(y[rows].astype(float))]
            if len(rr) >= 8 and len(set(y[rr])) == 2:
                res["auc_on_slice"] = L.auc(y[rr], score[rr])
                res["n_scored"] = int(len(rr))
            else:
                res["auc_on_slice"] = None
                res["n_scored"] = int(len(rr))
            return res

        rand_rows = rng.choice(mi, size=60, replace=False)
        rrec["concepts"].append({
            "concept": name,
            "nonnull_frac_all_mining": float(nonnull[mi].mean()),
            "shown_slice": surface(shown),
            "oracle_error_slice": surface(err_rows),
            "random60": surface(set(map(int, rand_rows))),
        })
    out["replicates"][rep] = rrec

# aggregate
agg = {"overlap_error_vs_shown": [out["replicates"][r]["overlap_error_vs_shown"]
                                  for r in ("rep1", "rep2", "rep3")]}
for key in ("shown_slice", "oracle_error_slice", "random60"):
    vals = [c[key]["nonnull_frac"] for r in out["replicates"].values() for c in r["concepts"]]
    agg[f"mean_nonnull_frac_{key}"] = float(np.mean(vals))
agg["mean_nonnull_frac_all_mining"] = float(np.mean(
    [c["nonnull_frac_all_mining"] for r in out["replicates"].values() for c in r["concepts"]]))
out["aggregate"] = agg

(HERE / "q4ii_error_slice.json").write_text(json.dumps(out, indent=1))
print(json.dumps(agg, indent=1))
for rep, rrec in out["replicates"].items():
    print(f"\n{rep}: overlap(error,shown)={rrec['overlap_error_vs_shown']}/60")
    for c in rrec["concepts"]:
        print(f"  {c['concept'][:55]:55s} nn_all={c['nonnull_frac_all_mining']:.2f} "
              f"shown={c['shown_slice']['nonnull_frac']:.2f} "
              f"oracle={c['oracle_error_slice']['nonnull_frac']:.2f} "
              f"rand={c['random60']['nonnull_frac']:.2f}")
