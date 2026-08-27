"""CF rebuilt-cell bank AUC — headroom_audit recipe, exact replication.

Inputs (laptop paths; copy cf2_bank_scores.parquet from sk3 first):
  outputs/v2_analysis/cf_rebuild_cell/cf2_bank_scores.parquet   (aNNN_score/_applied)
  outputs/v2_analysis/cf_rebuild_cell/cf2_bank_input.parquet    (canonical_pid join)
  outputs/v2_analysis/cf_rebuild_cell/results/shard_cf2_*_results.jsonl (L1 labels)

Recipe (identical to outputs/v2_analysis/headroom_audit/run_headroom.py):
  - features: aNNN_score + aNNN_applied, EXCLUDING a6xx (C++ superset metrics)
  - StratifiedGroupKFold(5) grouped by canonical_pid (GroupKFold fallback), 5 seeds
  - LR (median impute + standardize + liblinear C=1)
  - RF lightly tuned over 4 configs on seed 17, then 5-seed run of best
  - rank-average ensemble of LR + RF OOF predictions

Output: outputs/v2_analysis/cf_rebuild_cell/eval_results.json + printed table.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

D = Path("/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/cf_rebuild_cell")
N_FOLDS = 5
SEEDS = [17, 23, 29, 31, 37]


def make_lr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
    ])


def make_rf(seed=17, n_estimators=500, max_depth=None, min_samples_leaf=2):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, n_jobs=-1,
            class_weight="balanced", random_state=seed)),
    ])


def safe_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def grouped_splits(X, y, groups, seed):
    try:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        s = list(sgkf.split(X, y, groups))
        if all(len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2 for tr, te in s):
            return s
    except Exception:
        pass
    return list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))


def cv_run(model_factory, X, y, groups, seeds):
    per_seed_auc, per_seed_oof = [], []
    for sd in seeds:
        oof = np.full(len(y), np.nan)
        for tr, te in grouped_splits(X, y, groups, sd):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = model_factory(sd)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        mask = ~np.isnan(oof)
        per_seed_auc.append(safe_auc(y[mask], oof[mask]))
        per_seed_oof.append(oof)
    arr = np.array(per_seed_auc)
    oof_mean = np.nanmean(np.vstack(per_seed_oof), axis=0)
    return {"auc_mean": float(np.nanmean(arr)), "auc_std": float(np.nanstd(arr)),
            "per_seed": [float(x) for x in arr]}, oof_mean


def rank_avg(a, b):
    ra = pd.Series(a).rank(pct=True, na_option="keep").values
    rb = pd.Series(b).rank(pct=True, na_option="keep").values
    return (ra + rb) / 2.0


def main():
    bank = pd.read_parquet(D / "cf2_bank_scores.parquet")
    inp = pd.read_parquet(D / "cf2_bank_input.parquet")[["pair_id", "canonical_pid"]]
    labels = {}
    for p in sorted(glob.glob(str(D / "results" / "shard_cf2_*_results.jsonl"))):
        for ln in open(p):
            r = json.loads(ln)
            if r.get("l1") in (0, 1) and r["pair_id"] not in labels:
                labels[r["pair_id"]] = int(r["l1"])  # first occurrence wins (dups)
    print(f"labels: {len(labels)}")
    df = bank.merge(inp, on="pair_id", how="left")
    if "canonical_pid_x" in df.columns:
        df["canonical_pid"] = df["canonical_pid_x"].fillna(df["canonical_pid_y"])
    df["label"] = df["pair_id"].map(labels)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    print(f"labeled rows with bank scores: {len(df)}; "
          f"problems: {df['canonical_pid'].nunique()}; pos rate: {df['label'].mean():.3f}")

    score = sorted(c for c in df.columns if c.endswith("_score") and not c.startswith("a6"))
    applied = sorted(c for c in df.columns if c.endswith("_applied") and not c.startswith("a6"))
    X = df[score + applied].astype(float).values
    y = df["label"].values
    groups = df["canonical_pid"].astype(str).values
    print(f"features: {len(score)} score + {len(applied)} applied")

    r_lr, oof_lr = cv_run(lambda sd: make_lr(), X, y, groups, SEEDS)
    print("LR:", r_lr)

    rf_configs = [
        dict(n_estimators=500, max_depth=None, min_samples_leaf=2),
        dict(n_estimators=500, max_depth=8, min_samples_leaf=4),
        dict(n_estimators=500, max_depth=None, min_samples_leaf=5),
        dict(n_estimators=800, max_depth=12, min_samples_leaf=2),
    ]
    best_cfg, best_auc = None, -1
    for cfg in rf_configs:
        oof = np.full(len(y), np.nan)
        for tr, te in grouped_splits(X, y, groups, 17):
            m = make_rf(seed=17, **cfg)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        mask = ~np.isnan(oof)
        auc = safe_auc(y[mask], oof[mask])
        print(f"  rf cfg {cfg}: {auc:.4f}")
        if auc > best_auc:
            best_auc, best_cfg = auc, cfg
    r_rf, oof_rf = cv_run(lambda sd, c=best_cfg: make_rf(seed=sd, **c), X, y, groups, SEEDS)
    print("RF:", r_rf, best_cfg)

    ens = rank_avg(oof_lr, oof_rf)
    mask = ~np.isnan(ens)
    ens_auc = safe_auc(y[mask], ens[mask])
    print(f"Ens(LR+RF rank-avg): {ens_auc:.4f}")

    out = {
        "n": int(len(df)), "n_problems": int(df["canonical_pid"].nunique()),
        "pos_rate": float(y.mean()),
        "n_features": len(score) + len(applied),
        "lr": r_lr, "rf": {**r_rf, "config": best_cfg},
        "ensemble_lr_rf_rankavg": ens_auc,
    }
    (D / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
