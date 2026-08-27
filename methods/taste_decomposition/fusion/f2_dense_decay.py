#!/usr/bin/env python3
"""ADDENDUM H — spurious-feature DECAY CURVE for the dense arm.

Mirror of the bank-saturation curve (addF F-a), pointed at the dense readout:
for growing random subsets of k declared nuisance channels, the dense score is
linearly residualized on those channels (cross-fitted, GroupKFold(5) on the
cell's grouping unit) and the OOF residual's AUC is read out on the master-
ledger E rows. Asymptote fit A + B*exp(-k/tau) on the subset means.

Frozen spec: notes/2026-08-21__rung12_design_gap_consequences.md ADDENDUM H.
CAVEAT travels in the JSON: linear partial-out removes all collinear variance,
shared quality variance included — the asymptote LOWER-bounds deconfounded
dense performance; F2 (d)-(c) remains the primary deconfounding readout.

CPU only.  Usage: python3 f2_dense_decay.py --cell cw_community
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


DEC = _mod(HERE / "f2_deconf.py", "f2_deconf_h")
F2C = sys.modules["f2_cells_mod"]

R_SUBSETS = 20
SEED = 20260826
CAVEAT = ("Linear cross-fitted partial-out removes ALL dense variance collinear "
          "with the sampled nuisance channels, genuinely shared quality variance "
          "included; the asymptote is therefore a LOWER bound on deconfounded "
          "dense performance (anti-dense-conservative). The F2 stacked increment "
          "(d)-(c) remains the primary deconfounding readout.")


def residual_auc_oof(dense, N, y, groups, k_cols, rng_folds_seed=0):
    """AUC of dense residualized on nuisance columns k_cols, cross-fitted."""
    X = N[:, k_cols].astype(float)
    res = np.empty(len(dense))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for j in range(X.shape[1]):
            Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
            Xte[~np.isfinite(Xte[:, j]), j] = med[j]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        A = np.column_stack([np.ones(len(tr)), Xtr])
        beta, *_ = np.linalg.lstsq(A, dense[tr], rcond=None)
        res[te] = dense[te] - np.column_stack([np.ones(len(te)), Xte]) @ beta
    return float(roc_auc_score(y, res))


def run_cell(cell):
    t0 = time.time()
    meta, ids, y, groups, dense, _t0 = DEC.load_E(cell)
    a = F2C.ADAPTERS[cell]()
    _bank, N, join = DEC.align(cell, a, ids, y, groups)
    K = N.shape[1]
    auc0 = float(roc_auc_score(y, dense))
    nuis_alone = None  # context only; sourced from f2_deconf artifact if present
    art = RESULTS / f"f2_deconf_{cell}.json"
    if art.exists():
        d = json.loads(art.read_text())
        nuis_alone = d.get("spurious_alone_b") or d.get("arms", {}).get("NUIS")

    grid = sorted({k for k in [1, 2, 4, 8, 16, 32, 48, 64] if k < K} | {K})
    rng = np.random.default_rng(SEED)
    curve = {0: {"mean": auc0, "sd": 0.0, "n_subsets": 1}}
    for k in grid:
        aucs = []
        reps = 1 if k == K else R_SUBSETS
        for _ in range(reps):
            cols = rng.choice(K, size=k, replace=False)
            aucs.append(residual_auc_oof(dense, N, y, groups, cols))
        curve[k] = {"mean": float(np.mean(aucs)), "sd": float(np.std(aucs)),
                    "n_subsets": reps}
        print(f"[{cell}] k={k:3d}  AUC={curve[k]['mean']:.4f} ±{curve[k]['sd']:.4f}",
              flush=True)

    ks = np.array(sorted(curve), float)
    ms = np.array([curve[int(k)]["mean"] for k in ks])
    fit = None
    try:
        (A, B, tau), pcov = curve_fit(
            lambda k, A, B, tau: A + B * np.exp(-k / tau),
            ks, ms, p0=[ms[-1], ms[0] - ms[-1], max(K / 4, 1.0)],
            maxfev=20000)
        fit = {"asymptote_A": float(A), "B": float(B), "tau": float(tau),
               "A_se": float(np.sqrt(pcov[0, 0]))}
    except Exception as e:  # noqa: BLE001
        fit = {"error": str(e)}

    out = {
        "design": "ADDENDUM H spurious-feature decay curve (dense arm)",
        "spec": "notes/2026-08-21__rung12_design_gap_consequences.md ADDENDUM H",
        "cell": cell, "n_E": len(y), "n_nuis_channels": int(K),
        "group_column": meta.get("group_column"),
        "join_assertions": join,
        "dense_auc_k0": auc0,
        "nuis_alone_auc_from_f2": nuis_alone,
        "curve": {str(k): v for k, v in sorted(curve.items())},
        "asymptote_fit": fit,
        "decay_total_k0_minus_A": (None if "error" in fit
                                   else float(auc0 - fit["asymptote_A"])),
        "full_K_point": curve[K]["mean"],
        "caveat": CAVEAT,
        "env": DEC.env_block(), "runtime_sec": time.time() - t0,
    }
    p = RESULTS / f"f2_dense_decay_{cell}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"[{cell}] wrote {p}  decay={out['decay_total_k0_minus_A']}  "
          f"A={fit.get('asymptote_A')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="cw_community")
    args = ap.parse_args()
    if args.cell == "ALL":
        for c in sorted(F2C.ADAPTERS):
            try:
                run_cell(c)
            except Exception as e:  # noqa: BLE001
                print(f"[{c}] FAILED: {e}", flush=True)
        print("H_ALL_DONE", flush=True)
    else:
        run_cell(args.cell)
        print("H_DONE", flush=True)
