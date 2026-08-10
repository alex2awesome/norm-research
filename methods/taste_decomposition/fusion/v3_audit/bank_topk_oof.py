#!/usr/bin/env python3
"""Mechanism control for the V3 audit: how much AUC do the top-k criterion
columns carry ALONE, under the bank's own aggregator?

Protocol = "VA_nl fullfit@E" from the fusion note: full-population grouped OOF
(GroupKFold(5), frozen HistGB grid picked by inner GroupKFold(3), seeds
{0,1,2} mean), restricted to the top-k importance-ranked columns, AUC read on
the SAME evaluation-valid rows E (dense eval+test). Also the linear stack
(StandardScaler + LogisticRegression C=1) on the same columns, and the CW cell
(FIT+MINE-rank columns, AUC on MONITOR/TEST via the same full-pop OOF).

CPU only. Usage: python3 bank_topk_oof.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FUS = HERE.parent
TD = FUS.parent
REPO = TD.parents[1]
CAP = REPO / "datasets/humor/caption_multiy"
CW = TD / "closure" / "cw_community"


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


L1 = load_module(TD / "layer1_gemma_cells.py", "l1_topk_oof")
capagg = sys.modules["capagg_taste_decomp"]


def gbm_oof_seedmean(X, y, groups):
    folds = L1.outer_folds(len(y), groups)
    oofs = []
    for seed in L1.GBM_SEEDS:
        r = L1.gbm_oof_raw(X, y, groups, folds, seed)
        oofs.append(r["oof"])
    return np.mean(oofs, axis=0)


def lin_oof(X, y, groups):
    folds = L1.outer_folds(len(y), groups)
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def main():
    out = {}

    for cell, dl_name, id_key, y_key in (
            ("cap_crowd", "crowd", "crowd_ids", "y_crowd"),
            ("cap_finalist", "finalist", "hardneg_ids", "y_fin")):
        c = L1._caption_pools()
        meta = capagg.load_pool()
        rank = json.loads((HERE / f"importance_full_{cell}.json").read_text())["ranking"]
        ids_all = sorted(d for d in c[id_key] if d in c["X_by_id"])
        A = np.array([c["X_by_id"][d] for d in ids_all], dtype=float)
        V = np.array([c["V_by_id"][d] for d in ids_all], dtype=float)
        Ac, a_keep = capagg.clean_cols(A)
        Vc, v_keep = capagg.clean_cols(V)
        VA = np.column_stack([Vc, Ac])
        y = np.array([c[y_key][d] for d in ids_all])
        groups = np.array([str(c["contest_by_id"][d]) for d in ids_all])

        # E mask: rows of the dense eval+test splits
        dl = CAP / "dense_llama" / dl_name
        by_ct = {}
        for did in c[id_key]:
            if did in c["X_by_id"]:
                by_ct[(str(c["contest_by_id"][did]), meta[did]["text"])] = did
        e_dids = set()
        for s in ("eval", "test"):
            sf = pd.read_csv(dl / "split" / f"{s}.csv")
            for t, g in zip(sf.text, sf.group):
                e_dids.add(by_ct[(str(g), t)])
        emask = np.array([d in e_dids for d in ids_all])

        res = {}
        for k in (10, 20, 40, VA.shape[1]):
            cols = [r["col"] for r in rank[:k]] if k < VA.shape[1] else list(range(VA.shape[1]))
            Xk = VA[:, cols]
            nl = gbm_oof_seedmean(Xk, y, groups)
            li = lin_oof(Xk, y, groups)
            res[f"k{k}"] = {
                "nl_fullfit_at_E": float(roc_auc_score(y[emask], nl[emask])),
                "lin_fullfit_at_E": float(roc_auc_score(y[emask], li[emask])),
            }
            print(cell, f"k={k}", json.dumps({a: round(b, 4) for a, b in res[f'k{k}'].items()}))
        out[cell] = res

    # ---- CW: top-k columns under the bank aggregator, full-pop OOF
    d = np.load(CW / "round7_state.npz", allow_pickle=True)
    VA, y = d["VA"], d["y"].astype(int)
    groups = d["groups"].astype(str)
    split = pd.Series(d["split"]).astype(str).values
    col_med = np.nanmedian(VA[split == "fit_mine"], axis=0)
    VAi = np.where(np.isnan(VA), col_med[None, :], VA)
    rank = json.loads((HERE / "importance_full_cw_community.json").read_text())["ranking"]
    res = {}
    for k in (20, 40, VA.shape[1]):
        cols = [r["col"] for r in rank[:k]] if k < VA.shape[1] else list(range(VA.shape[1]))
        Xk = VAi[:, cols]
        nl = gbm_oof_seedmean(Xk, y, groups)
        li = lin_oof(Xk, y, groups)
        res[f"k{k}"] = {}
        for s, nm in (("monitor", "monitor"), ("test", "test")):
            m = split == s
            res[f"k{k}"][f"nl_oof_{nm}"] = float(roc_auc_score(y[m], nl[m]))
            res[f"k{k}"][f"lin_oof_{nm}"] = float(roc_auc_score(y[m], li[m]))
        print("cw_community", f"k={k}", json.dumps({a: round(b, 4) for a, b in res[f'k{k}'].items()}))
    out["cw_community"] = res

    (HERE / "bank_topk_oof.json").write_text(json.dumps(out, indent=2))
    print("BANK_TOPK_OOF_DONE")


if __name__ == "__main__":
    main()
