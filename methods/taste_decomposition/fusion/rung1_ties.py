#!/usr/bin/env python3
"""RUNG 1 ADDENDUM B — indistinguishability/tie readout (design doc, frozen
2026-08-22 before running).

Per cell: V-only, VA-bank and dense per-row scores on the E frame (same
frozen Layer-1 recipe as rung1_selection_regret). Rank-normalize each within
the cell; report, over the quantization grid q, how often each instrument
cannot separate the items it is asked to rank (grouped: top-1 not unique;
pairwise: |rank diff| <= q on pos-neg pairs), plus the dense-only-granularity
mass. Saves per-row scores to results/rung1_scores_<cell>.npz.

V block: closure cells expose round0_state['V']. Cells without it report
VA + dense only. ONE CELL PER PROCESS (cells.py module collision, 2026-08-22).

CPU only.  Usage: python3 rung1_ties.py --cell cw_community
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TD = HERE.parent
RESULTS = TD / "results"
CLOSURE = TD / "closure"


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


F2 = _mod(HERE / "f2_deconf.py", "f2_deconf_ties")
fit_arm = F2.fit_arm

QGRID = (0.0, 0.01, 0.02, 0.05)

# cell -> state npz exposing a V block aligned by ids. First pass: ONLY
# cw_community has a uniform round0_state.npz on this box; other campaigns
# store V differently (per-campaign extraction deferred — VA+dense ties are
# still reported everywhere).
V_STATE = {"cw_community": CLOSURE / "cw_community" / "round0_state.npz"}


def rank01(x):
    r = np.argsort(np.argsort(x, kind="stable"))
    return r / max(len(x) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    args = ap.parse_args()
    cell = args.cell
    t0 = time.time()

    meta, ids_E, y, groups, dense, t0col = F2.load_E(cell)
    a = F2.F2C.ADAPTERS[cell]()
    bank, nuis, join = F2.align(cell, a, ids_E, y, groups)
    y = np.asarray(y)

    r_va = fit_arm(meta["family"], bank, dense, y, groups)
    scores = {"va": r_va["_oof_VA_nl0"], "dense": np.asarray(dense, float)}

    vs = V_STATE.get(cell)
    if vs and vs.exists():
        z = np.load(vs, allow_pickle=True)
        sids = [str(s) for s in z["ids"]]
        pos = {s: i for i, s in enumerate(sids)}
        rows = [pos[i] for i in ids_E if i in pos]
        if len(rows) == len(ids_E):
            V = z["V"].astype(float)[rows]
            r_v = fit_arm(meta["family"], V, dense, y, groups)
            scores["v"] = r_v["_oof_VA_nl0"]
        else:
            print(f"  [{cell}] V block id coverage {len(rows)}/{len(ids_E)} — skipping V arm")

    R = {k: rank01(v) for k, v in scores.items()}
    np.savez(RESULTS / f"rung1_scores_{cell}.npz", ids=np.array(ids_E), y=y,
             groups=np.array([str(g) for g in groups]),
             **{f"score_{k}": v for k, v in scores.items()})

    gidx = {}
    for i, g in enumerate(np.asarray(groups)):
        gidx.setdefault(g, []).append(i)
    dec = [np.array(ix) for ix in gidx.values()
           if len(ix) >= 2 and 0 < y[ix].sum() < len(ix)]
    grouped = len(dec) > 1

    out = {"cell": cell, "mode": "grouped" if grouped else "pairwise",
           "n_E": int(len(y)), "n_decidable": len(dec),
           "instruments": sorted(R), "q_grid": list(QGRID),
           "design": "ADDENDUM B, notes/2026-08-21__rung12_design_gap_consequences.md",
           "ties": {}, "dense_only_granularity": {}}

    if grouped:
        for q in QGRID:
            row = {}
            for k, r in R.items():
                row[k] = float(np.mean([(r[ix] >= r[ix].max() - q).sum() >= 2
                                        for ix in dec]))
            out["ties"][str(q)] = row
        # dense-only granularity: groups where VA (and V if present) abstain
        # at q but dense's top-1 is unique — and whether dense's pick wins
        for q in (0.01, 0.02):
            arts = [k for k in ("v", "va") if k in R]
            silent = [ix for ix in dec
                      if all((R[k][ix] >= R[k][ix].max() - q).sum() >= 2 for k in arts)
                      and (R["dense"][ix] >= R["dense"][ix].max() - q).sum() == 1]
            hits = [int(y[ix[np.argmax(R["dense"][ix])]]) for ix in silent]
            out["dense_only_granularity"][str(q)] = dict(
                n_groups=len(silent), frac_of_decidable=len(silent) / len(dec),
                dense_top1_hit=float(np.mean(hits)) if hits else None)
    else:
        p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        for q in QGRID:
            row = {}
            for k, r in R.items():
                D = np.abs(r[p][:, None] - r[n][None, :])
                row[k] = float((D <= q).mean())
            out["ties"][str(q)] = row
        for q in (0.01, 0.02):
            arts = [k for k in ("v", "va") if k in R]
            M = np.ones((len(p), len(n)), bool)
            for k in arts:
                M &= np.abs(R[k][p][:, None] - R[k][n][None, :]) <= q
            Dd = R["dense"][p][:, None] - R["dense"][n][None, :]
            M &= np.abs(Dd) > q
            out["dense_only_granularity"][str(q)] = dict(
                n_pairs=int(M.sum()), frac_of_pairs=float(M.mean()),
                dense_hit=float((Dd[M] > 0).mean()) if M.any() else None)

    out["runtime_sec"] = time.time() - t0
    fp = RESULTS / f"rung1_ties_{cell}.json"
    fp.write_text(json.dumps(out, indent=2))
    t1 = out["ties"]["0.01"]
    print(f"  [{cell}] {out['mode']} ties@1%: " +
          " ".join(f"{k}={t1[k]:.2f}" for k in sorted(t1)) +
          f" | dense-only@1%: {out['dense_only_granularity']['0.01']}"
          f" | {out['runtime_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
