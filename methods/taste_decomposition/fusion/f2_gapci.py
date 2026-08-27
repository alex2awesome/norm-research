#!/usr/bin/env python3
"""Group-bootstrap CI on the FIGURE-3 gap contrast, (d) fused - (a) enriched bank.

Why this exists: f2_deconf.py bootstraps the PRIMARY stacked increment (d)-(c)
[fused vs bank+nuisance] but only reports a point estimate for
fused_must_beat_bank margin = (d)-(a), which is the number Figure 3 plots as the
articulability gap for the certified-frame cells.  The user asked how confident
we are in the Regulatory median (n=2 cells), so we need the interval on the
plotted contrast itself, not on a neighbouring one.

Same rows, same frozen Layer-1 stack, same shared folds, same 2,000-draw
group-paired bootstrap on the cell's grouping unit.  Only arms (a) and (d) are
refit -- no Westfall-Yarkoni band, no Leg-2 matched sweep, no e-value sweep.
Convention inherited from f2_deconf: the bootstrap runs on the SEED-0 OOF
vectors while the ledger quotes the 3-seed mean AUC, so both points are written
out and the seed gap is visible.

CPU only.  Usage: python3 f2_gapci.py --cell nc_responded --cell nc_outcome
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
sys.path.insert(0, str(HERE))

import importlib.util


def _mod(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


F2 = _mod(HERE / "f2_deconf.py", "f2_deconf_mod")
D1 = F2.D1
fit_arm, gboot = F2.fit_arm, F2.gboot


def run(cell, n_boot=2000):
    t0 = time.time()
    meta, ids_E, y, groups, dense, t0col = F2.load_E(cell)
    family = meta["family"]
    a = F2.F2C.ADAPTERS[cell]()
    bank, nuis, join = F2.align(cell, a, ids_E, y, groups)
    bn = np.column_stack([bank, nuis])
    print(f"  [{cell}] n_E={len(y)} groups={len(set(groups))} bank={bank.shape}",
          flush=True)

    r_bank = fit_arm(family, bank, dense, y, groups)     # (a)
    r_bn_T = fit_arm(family, bn, dense, y, groups)       # (d)
    oof_a, oof_d = r_bank["_oof_VA_nl0"], r_bn_T["_oof_VAT_nl0"]

    boot = gboot(y, oof_d, oof_a, groups, n_boot=n_boot)
    out = {
        "cell": cell,
        "contrast": "(d) VAT_dec_trained  minus  (a) VA_enriched_bank  [= Fig-3 gap]",
        "n_E": int(len(y)),
        "n_groups": int(len(set(groups))),
        "group_column": meta.get("group_column"),
        "point_3seed": float(r_bn_T["VAT_nl_mean"] - r_bank["VA_nl_mean"]),
        "point_seed0": float(roc_auc_score(y, oof_d) - roc_auc_score(y, oof_a)),
        "arm_a_3seed": float(r_bank["VA_nl_mean"]),
        "arm_d_3seed": float(r_bn_T["VAT_nl_mean"]),
        "GROUP_BOOTSTRAP_seed0": boot,
        "convention": ("bootstrap on seed-0 OOF vectors, point quoted at the 3-seed "
                       "mean -- identical to f2_deconf.py PRIMARY"),
        "env": {"platform": platform.platform(), "python": platform.python_version()},
        "runtime_sec": time.time() - t0,
    }
    p = RESULTS / f"f2_gapci_{cell}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"  [{cell}] gap_3seed={out['point_3seed']:+.4f} "
          f"seed0={out['point_seed0']:+.4f} "
          f"CI95=[{boot['ci95'][0]:+.4f},{boot['ci95'][1]:+.4f}] "
          f"p>0={boot['p_gt_0']:.3f} | {out['runtime_sec']:.0f}s -> {p}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    for c in args.cell:
        print(f"=== F2 gap-CI {c} ===", flush=True)
        run(c, args.n_boot)
