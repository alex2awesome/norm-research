#!/usr/bin/env python3
"""V-only leg for the code_v3 (PR merge) cell, WITHIN-REPO frame — fills the VERDICT
ladder's V column (user request 2026-08-12; the within-repo frame never had a V-only
run: the ledger records bank/dense/fused only).

Instrument identity: the frozen Layer-1 nonlinear stack imported VERBATIM from
readout_code_v3.py (HistGB grid {15,31} leaf nodes, GroupKFold(5) by repo, nested
GroupKFold(3) grid selection, seeds 0/1/2 OOF-averaged) on the 36 V columns
(17 execution-derived + 19 text-recomputed) of V_matrix_v3.parquet. The within-repo
readout is the within_repo.py estimator verbatim: repos with >=20 rows and both
classes, per-repo AUC, n-weighted mean. VA_nl is recomputed from the saved OOF
vectors on the identical repo subset so V vs VA is same-frame.

CPU only.  Usage:  OMP_NUM_THREADS=6 python3 v_only_within_repo.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from readout_code_v3 import clean_cols, folds_for, gbm_oof

HERE = Path(__file__).resolve().parent
V = pd.read_parquet(HERE / "V_matrix_v3.parquet")
v_cols = [c for c in V.columns if c.startswith(("ve_", "vt_"))] + ["n_review_comments"]
v_cols = [c for c in v_cols if c in V.columns]

out = {"v_cols_n": len(v_cols), "seeds": [0, 1, 2], "splits": {}}
for sp in ("eval", "test"):
    d = V[V["split"] == sp].reset_index(drop=True)
    y = d["judgement"].astype(int).values
    groups = d["repo"].astype(str).values
    X_raw = d[v_cols].astype(float).values
    X, kept = clean_cols(X_raw)
    folds = folds_for(groups)

    oofs, pooled = [], []
    for s in (0, 1, 2):
        g = gbm_oof(X, y, groups, folds, seed=s)
        oofs.append(g["oof"])
        pooled.append(g["auc"])
        print(f"[{sp}] seed{s} pooled V_nl {g['auc']:.4f}", flush=True)
    v_oof = np.mean(oofs, axis=0)
    np.save(HERE / f"code_v3_{sp}_v_nl_oof_meanseeds.npy", v_oof)

    va_oof = np.mean([np.load(HERE / f"code_v3_{sp}_va_nl_oof_seed{s}.npy")
                      for s in (0, 1, 2)], axis=0)
    assert len(va_oof) == len(d), "VA OOF row-count mismatch vs V matrix split"

    rows, nw_v, nw_va, tot = [], 0.0, 0.0, 0
    d = d.assign(v_nl=v_oof, va_nl=va_oof)
    for repo, gdf in d.groupby("repo"):
        if gdf["judgement"].nunique() < 2 or len(gdf) < 20:
            continue
        av = roc_auc_score(gdf["judgement"], gdf["v_nl"])
        ava = roc_auc_score(gdf["judgement"], gdf["va_nl"])
        rows.append({"repo": repo, "n": len(gdf), "v_nl": av, "va_nl": ava})
        nw_v += av * len(gdf); nw_va += ava * len(gdf); tot += len(gdf)
    t = pd.DataFrame(rows)
    out["splits"][sp] = {
        "n_rows_split": int(len(d)), "n_repos_scored": int(len(t)), "n_rows_scored": int(tot),
        "v_cols_kept": int(X.shape[1]),
        "V_nl_pooled_per_seed": [float(a) for a in pooled],
        "V_nl_within_nwtd": nw_v / tot, "V_nl_within_median": float(t["v_nl"].median()),
        "VA_nl_within_nwtd_same_subset": nw_va / tot,
        "V_beats_VA_repos": int((t["v_nl"] > t["va_nl"]).sum()),
    }
    print(sp, json.dumps(out["splits"][sp], indent=1), flush=True)

json.dump(out, open(HERE / "v_only_within_repo.json", "w"), indent=1)
print("V_ONLY_DONE wrote v_only_within_repo.json", flush=True)
