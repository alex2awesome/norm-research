"""E-cert verdict-slice harvest (task #32; prereg frozen 2026-08-12 in
notes/2026-08-05__direction12-battery-plan.md BEFORE mining).

Committed re-derivation of the /tmp post-processors (RESULTS_LEDGER provenance gap):
  - HB138: binarize M_i at its per-metric MEDIAN (fixes degenerate all-YES metrics).
  - achieved = max over mined prompts of agreement(0.5-binarized sig row, median-binarized M_i).
  - HB142 ceiling: achieved + max spacing among the top-10 order statistics of the
    per-prompt agreement distribution (distribution-free one-draw bound).
  - HB145 calibration: calibrated = achieved + 2.0 * (ceil_conservative - achieved).
Readout per frozen prereg: cells A (pooled-BOUNDED) / B (plateaued-everywhere) /
C (rising-everywhere); P1 A+B exhaust (low calibrated-achieved gap) vs C; P2 C shows
larger unexhausted mass; P3 A exhausts at lower absolute achieved than B.
Instrument note: 185 fresh banks ONLY (canonical 4-family, uniform); the 10 metrics
overlapping the 272-bank are excluded from cells (different instrument: R3, 3xGLM, N~640).
Threshold-free readouts: rank stats + paired-free bootstrap between cells.
Runs on sk3. Output: outputs/ecert_slice_v1/harvest_v1.json (+ printed table).
"""
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np

BANKS = "/lfs/skampere3/0/alexspan/outputs/ecert_slice_v1"
SLICE = "/lfs/skampere3/0/alexspan/outputs/ecert_slice_v1/gilists.json"
CELLJ = "/lfs/skampere3/0/alexspan/outputs/ecert_slice_v1/slice_cells.json"
TASKMAP = {"humor": "humor", "creative-writing": "creative-writing",
           "news-homepages": "news-homepages", "math-stackexchange": "math-stackexchange",
           "peer-review": "peer-review"}


def bank_stats(path):
    z = np.load(path, allow_pickle=True)
    sigs = np.asarray(z["sigs"], float)
    Mi = np.asarray(z["M_i"], float)
    med = np.median(Mi)
    mib = (Mi >= med).astype(int)
    if mib.std() == 0:
        return None
    agree = []
    for row in sigs:
        ok = np.isfinite(row)
        if ok.sum() < 50:
            continue
        rb = (row[ok] >= 0.5).astype(int)
        agree.append(float((rb == mib[ok]).mean()))
    if len(agree) < 20:
        return None
    agree = np.sort(agree)
    achieved = float(agree[-1])
    top = agree[-10:]
    spacing = float(np.max(np.diff(top))) if len(top) >= 2 else 0.0
    ceil_cons = min(1.0, achieved + spacing)
    calibrated = min(1.0, achieved + 2.0 * (ceil_cons - achieved))
    return {"n_prompts": len(agree), "achieved": round(achieved, 4),
            "ceil_conservative": round(ceil_cons, 4), "calibrated": round(calibrated, 4),
            "gap": round(calibrated - achieved, 4), "name": str(z["name"])}


def main():
    cells = json.load(open(CELLJ))          # {task, name, cell} rows (shipped from laptop)
    cell_of = {}
    gil = json.load(open(SLICE))
    gi_name = {}
    for task, rows in gil.items():
        for r in rows:
            gi_name[(task, r["gi"])] = r["name"]
    for r in cells:
        cell_of[r["name"].strip().lower()] = r["cell"]
    rows = []
    for f in sorted(glob.glob(f"{BANKS}/*_R2_metric*_sigs.npz")):
        base = os.path.basename(f)
        task = base.split("_R2_")[0]
        st = bank_stats(f)
        if st is None:
            rows.append({"file": base, "task": task, "status": "DEGENERATE"})
            continue
        cell = cell_of.get(st["name"].strip().lower(), "UNMAPPED")
        rows.append({"file": base, "task": task, "cell": cell, **st, "status": "ok"})
    ok = [r for r in rows if r["status"] == "ok" and r.get("cell", "UNMAPPED") != "UNMAPPED"]
    by = defaultdict(list)
    for r in ok:
        by[r["cell"]].append(r)
    out = {"rows": rows, "cells": {}}
    print(f"{'cell':24s} {'n':>4s} {'med_achieved':>12s} {'med_gap':>8s} {'p90_gap':>8s}")
    for c, rr in sorted(by.items()):
        g = [r["gap"] for r in rr]; a = [r["achieved"] for r in rr]
        out["cells"][c] = {"n": len(rr), "median_achieved": round(float(np.median(a)), 4),
                           "median_gap": round(float(np.median(g)), 4),
                           "p90_gap": round(float(np.percentile(g, 90)), 4)}
        print(f"{c:24s} {len(rr):4d} {np.median(a):12.4f} {np.median(g):8.4f} "
              f"{np.percentile(g, 90):8.4f}")
    # P1/P2: rank test gap C vs A+B (bootstrap on median difference, seeded)
    gA = [r["gap"] for r in by.get("A_pooled_bounded", [])]
    gB = [r["gap"] for r in by.get("B_plateaued_everywhere", [])]
    gC = [r["gap"] for r in by.get("C_rising_everywhere", [])]
    aA = [r["achieved"] for r in by.get("A_pooled_bounded", [])]
    aB = [r["achieved"] for r in by.get("B_plateaued_everywhere", [])]
    rng = random.Random(0)

    def boot_diff(x, y, stat=np.median, B_=20000):
        obs = stat(y) - stat(x)
        lo_hi = []
        for _ in range(B_):
            bx = [x[rng.randrange(len(x))] for _ in x]
            by_ = [y[rng.randrange(len(y))] for _ in y]
            lo_hi.append(stat(by_) - stat(bx))
        lo_hi.sort()
        return obs, lo_hi[int(.025 * B_)], lo_hi[int(.975 * B_)]

    if gC and (gA or gB):
        d, lo, hi = boot_diff(gA + gB, gC)
        out["P1_P2_gapC_minus_gapAB"] = {"diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)]}
        print(f"\nP1/P2 gap(C) - gap(A+B): {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    if aA and aB:
        d, lo, hi = boot_diff(aB, aA)
        out["P3_achievedA_minus_achievedB"] = {"diff": round(d, 4),
                                               "ci": [round(lo, 4), round(hi, 4)]}
        print(f"P3 achieved(A) - achieved(B): {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    n_deg = sum(1 for r in rows if r["status"] == "DEGENERATE")
    n_unm = sum(1 for r in rows if r.get("cell") == "UNMAPPED")
    print(f"\nbanks: {len(rows)} | ok+mapped: {len(ok)} | degenerate: {n_deg} | unmapped: {n_unm}")
    json.dump(out, open(f"{BANKS}/harvest_v1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
