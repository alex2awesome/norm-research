#!/usr/bin/env python3
"""Direction 1b (secondary readout): two-stage V+A+T fusion that avoids the
E-refit data-starvation handicap without leaking.

Stage 1: the FULL-population Layer-1 OOF prediction of the VA stack (linear and
GBM seeds 0-2) — honest out-of-sample for every row by grouped OOF construction.
Stage 2, on E only (rows outside the dense train split, where the dense column
is out-of-sample):
  (i)  rank-average ensemble of [VA_nl_oof, dense_prob]  — no fitting at all;
  (ii) 2-column logistic combiner [VA_oof, dense_prob], GroupKFold(5) OOF on E;
  (iii) same combiner on [VA_oof] alone (calibration control).

This answers "does adding the dense scalar to the bank's substance help?" with
the bank at FULL training strength — complementary to direction1_stack.py's
matched-footing E-refit (where VA and VAT share the small-train handicap).

CPU only. Usage: python3 direction1b_twostage.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
TD = HERE.parent


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


L1 = load_module(TD / "layer1_gemma_cells.py", "l1_fusion_1b")
D1 = load_module(HERE / "direction1_stack.py", "d1_fusion_1b")


def combiner_oof(cols, y, groups, seed=0):
    X = np.column_stack(cols)
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def fullpop_oof(cell):
    d = L1.CELLS[cell]["loader"]()
    mats, y, groups, family = d["mats"], d["y"], d["groups"], d["family"]
    folds = L1.outer_folds(len(y), groups, n_splits=5)
    linfn = L1.linear_oof_family1 if family == "family1" else L1.linear_oof_family2
    gbmfn = L1.gbm_oof_family1 if family == "family1" else L1.gbm_oof_raw
    _, lin_oof = linfn(mats["VA"], y, groups, folds)
    nl_oofs = [gbmfn(mats["VA"], y, groups, folds, s)["oof"] for s in (0, 1, 2)]
    return d, y, groups, lin_oof, np.mean(nl_oofs, axis=0)


def run_cell(cell):
    t0 = time.time()
    if cell == "style_inv_toptier":
        probs_by_seed, _ = D1.si_dense()
        d, y_all, groups_all, lin_oof, nl_oof = fullpop_oof(cell)
        meta = L1.rvg.load_bank("style_invitational")[0]
        ids = list(meta["item_ids"])
        in_E = np.array([i in probs_by_seed["42"] for i in ids])
        dense_by_seed = {s: np.array([probs_by_seed[s][ids[i]] for i in np.flatnonzero(in_E)])
                         for s in ("42", "1", "2")}
        dense_cols = {"per_seed": dense_by_seed,
                      "mean": np.mean([dense_by_seed[s] for s in dense_by_seed], axis=0)}
    else:
        dense = D1.caption_dense(cell)
        d, y_all, groups_all, lin_oof, nl_oof = fullpop_oof(cell)
        c = L1._caption_pools()
        id_key = "crowd_ids" if cell == "cap_crowd" else "hardneg_ids"
        ids = sorted(x for x in c[id_key] if x in c["X_by_id"])
        in_E = np.array([not dense[x][2] for x in ids])
        dense_cols = {"mean": np.array([dense[x][0] for x in ids if not dense[x][2]])}

    y = y_all[in_E]
    g = groups_all[in_E]
    va_nl_E = nl_oof[in_E]
    va_lin_E = lin_oof[in_E]

    def block(dcol):
        r = {}
        r["T_E"] = float(roc_auc_score(y, dcol))
        r["VA_nl_fullfit_at_E"] = float(roc_auc_score(y, va_nl_E))
        rank_ens = rankdata(va_nl_E) / len(y) + rankdata(dcol) / len(y)
        r["rankavg_VAnl_plus_T"] = float(roc_auc_score(y, rank_ens))
        r["combiner_VAnl_only"], _ = combiner_oof([va_nl_E], y, g)
        r["combiner_VAnl_plus_T"], oof2 = combiner_oof([va_nl_E, dcol], y, g)
        r["combiner_VAlin_VAnl_plus_T"], _ = combiner_oof([va_lin_E, va_nl_E, dcol], y, g)
        r["boot_combiner2_minus_VAnl"] = D1.paired_boot(y, oof2, va_nl_E)
        return r

    out = {"cell": cell, "direction": "1b (two-stage, secondary)",
           "n_E": int(len(y)), "n_groups_E": int(len(np.unique(g)))}
    if cell == "style_inv_toptier":
        out["per_dense_seed"] = {s: block(dense_cols["per_seed"][s]) for s in ("42", "1", "2")}
        out["ensemble_meanprob"] = block(dense_cols["mean"])
    else:
        out.update(block(dense_cols["mean"]))
    out["runtime_sec"] = time.time() - t0
    p = HERE / f"{cell}_direction1b.json"
    p.write_text(json.dumps(out, indent=2))
    print(cell, json.dumps({k: v for k, v in out.items()
                            if isinstance(v, (int, float)) and k != "runtime_sec"}, indent=1))
    print("wrote", p, flush=True)


if __name__ == "__main__":
    for cell in ("cap_crowd", "cap_finalist", "style_inv_toptier"):
        run_cell(cell)
