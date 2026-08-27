#!/usr/bin/env python3
"""ADDENDUM F probes a/b/c — capacity-null battery (frozen 2026-08-25).

F-a scaling curve: AUC(k) over criteria subsets, frozen recipe, grouped OOF.
F-b distillation: grouped-OOF regression of the DENSE SCORE on bank features;
    + label-relevance of the residual (does y-signal live outside the span?).
F-c head swap: MLP vs GBM on identical features.

ONE CELL PER PROCESS. CPU. Usage: python3 addF_capacity_probes.py --cell cw_community
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


F2 = _mod(HERE / "f2_deconf.py", "f2_deconf_addF")
fit_arm = F2.fit_arm

KGRID = (8, 16, 32, 64, 96, 128)
NSUB = 3


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
    dcol = np.asarray(dense, float)
    P = bank.shape[1]
    out = {"cell": cell, "n_E": int(len(y)), "n_features": int(P),
           "design": "ADDENDUM F, notes/2026-08-21__rung12_design_gap_consequences.md"}

    # ---- F-a scaling curve --------------------------------------------------
    curve = []
    for k in [kk for kk in KGRID if kk < P]:
        for rep in range(NSUB):
            rng = np.random.default_rng(1000 * k + rep)
            cols = rng.choice(P, size=k, replace=False)
            r = fit_arm(meta["family"], bank[:, cols], dense, y, groups)
            curve.append({"k": k, "rep": rep, "auc": float(r["VA_nl_mean"])})
            print(f"  [{cell}] k={k} rep={rep} auc={r['VA_nl_mean']:.4f}", flush=True)
    r_full = fit_arm(meta["family"], bank, dense, y, groups)
    full_auc = float(r_full["VA_nl_mean"])
    curve.append({"k": P, "rep": 0, "auc": full_auc})
    # saturation fit: auc(k) = A - B * exp(-k / tau)
    from scipy.optimize import curve_fit
    ks = np.array([c["k"] for c in curve], float)
    au = np.array([c["auc"] for c in curve], float)
    try:
        popt, pcov = curve_fit(lambda k, A, B, tau: A - B * np.exp(-k / tau),
                               ks, au, p0=[au.max() + .01, .2, 30.0],
                               maxfev=20000)
        A, B, tau = (float(v) for v in popt)
        A_sd = float(np.sqrt(pcov[0, 0]))
    except Exception as e:
        A = B = tau = A_sd = None
        print("  asymptote fit failed:", e, flush=True)
    out["Fa_curve"] = curve
    out["Fa_asymptote"] = {"A": A, "A_sd": A_sd, "tau": tau,
                           "full_bank_auc": full_auc,
                           "dense_T_auc": float(roc_auc_score(y, dcol)),
                           "note": "A = fitted articulated plateau; compare A+2sd vs dense"}

    # ---- F-b distillation probe --------------------------------------------
    # grouped-OOF regression of the (rank-transformed) dense score on bank
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold
    dr = np.argsort(np.argsort(dcol)) / (len(dcol) - 1)
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(bank, dr, np.asarray(groups)):
        imp = SimpleImputer(strategy="median", add_indicator=True)
        Xtr, Xte = imp.fit_transform(bank[tr]), imp.transform(bank[te])
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=.06,
                                          max_leaf_nodes=31, random_state=0)
        m.fit(Xtr, dr[tr])
        oof[te] = m.predict(Xte)
    rho_span = float(spearmanr(oof, dr).statistic)
    resid = dr - oof
    # label-relevance of the residual: AUC(y, resid) and AUC(y, resid | bank OOF)
    auc_resid = float(roc_auc_score(y, resid))
    b_oof = r_full["_oof_VA_nl0"]
    # increment: does resid add to the bank's own label prediction? stack simply
    from sklearn.linear_model import LogisticRegression
    Z = np.column_stack([np.argsort(np.argsort(b_oof)) / (len(y) - 1), resid])
    oof2 = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(Z, y, np.asarray(groups)):
        lr = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
        oof2[te] = lr.predict_proba(Z[te])[:, 1]
    auc_bank_only = float(roc_auc_score(y, b_oof))
    auc_bank_plus_resid = float(roc_auc_score(y, oof2))
    out["Fb_distillation"] = {
        "rho_bank_predicts_dense_score": rho_span,
        "auc_residual_alone_vs_y": auc_resid,
        "auc_bank_only": auc_bank_only,
        "auc_bank_plus_residual": auc_bank_plus_resid,
        "residual_increment": auc_bank_plus_resid - auc_bank_only,
        "note": ("rho high = criteria span dense signal (estimation gap); "
                 "residual_increment > 0 = label-relevant signal OUTSIDE the "
                 "named span = operational tacit component (user framing)")}

    # ---- F-c head swap ------------------------------------------------------
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    heads = {}
    for name, hidden in (("mlp_64", (64,)), ("mlp_256_64", (256, 64))):
        oofh = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(bank, y, np.asarray(groups)):
            imp = SimpleImputer(strategy="median", add_indicator=True)
            Xtr, Xte = imp.fit_transform(bank[tr]), imp.transform(bank[te])
            clf = make_pipeline(StandardScaler(),
                                MLPClassifier(hidden_layer_sizes=hidden,
                                              max_iter=400, random_state=0,
                                              early_stopping=True))
            clf.fit(Xtr, y[tr])
            oofh[te] = clf.predict_proba(Xte)[:, 1]
        heads[name] = float(roc_auc_score(y, oofh))
        print(f"  [{cell}] head {name}: {heads[name]:.4f}", flush=True)
    heads["gbm_frozen"] = auc_bank_only
    out["Fc_head_swap"] = heads

    out["runtime_sec"] = time.time() - t0
    fp = RESULTS / f"addF_capacity_{cell}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"ADDF_DONE {cell} | asymptote A={A} vs dense "
          f"{out['Fa_asymptote']['dense_T_auc']:.4f} | span rho={rho_span:.3f} "
          f"resid_incr={out['Fb_distillation']['residual_increment']:+.4f} | "
          f"{out['runtime_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
