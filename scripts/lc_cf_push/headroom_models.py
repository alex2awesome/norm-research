#!/usr/bin/env python3
"""Item 2 of lc_cf_push: model/eval headroom on the EXISTING CF and LC cells.

Headline stays candidate-only. Variants tried (all grouped CV, 5 seeds, identical
splits to the frozen recipes):
  - frozen LR / RF / LR+RF rank-avg (replication baseline)
  - LR with class_weight=balanced (CF frozen LR had none)
  - HistGradientBoosting (sklearn) as a third member
  - 3-way rank-avg LR+RF+HGB
  - feature selection by applied-rate (metrics with applied-rate >= 5% / >= 20%)
  - NOTE: probability calibration (sigmoid/isotonic) is monotone per-model and
    cannot change AUC; not run.

NOTHING here uses problem identity, cosine, or editorial-side features.

Output: outputs/v2_analysis/lc_cf_push/headroom_results.json + printed table.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
CF = ROOT / "outputs/v2_analysis/cf_rebuild_cell"
LC = ROOT / "outputs/v2_analysis/lc_organic_relabel"
OUT = ROOT / "outputs/v2_analysis/lc_cf_push"
N_FOLDS = 5
SEEDS = [17, 23, 29, 31, 37]


def make_lr(balanced=False):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                                  class_weight="balanced" if balanced else None)),
    ])


def make_rf(seed=17, **cfg):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_jobs=-1, class_weight="balanced",
                                      random_state=seed, **cfg)),
    ])


def make_hgb(seed=17):
    # HGB handles NaN natively; no imputer needed
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed)


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


def cv_run(factory, X, y, groups):
    per_seed_auc, per_seed_oof = [], []
    for sd in SEEDS:
        oof = np.full(len(y), np.nan)
        for tr, te in grouped_splits(X, y, groups, sd):
            m = factory(sd)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        mask = ~np.isnan(oof)
        per_seed_auc.append(safe_auc(y[mask], oof[mask]))
        per_seed_oof.append(oof)
    arr = np.array(per_seed_auc)
    return {"auc_mean": float(np.nanmean(arr)), "auc_std": float(np.nanstd(arr))}, \
        np.nanmean(np.vstack(per_seed_oof), axis=0)


def rank_avg(*preds):
    rs = [pd.Series(p).rank(pct=True, na_option="keep").values for p in preds]
    return np.nanmean(np.vstack(rs), axis=0)


def load_cf():
    bank = pd.read_parquet(CF / "cf2_bank_scores.parquet")
    inp = pd.read_parquet(CF / "cf2_bank_input.parquet")[["pair_id", "canonical_pid"]]
    labels = {}
    for p in sorted(glob.glob(str(CF / "results" / "shard_cf2_*_results.jsonl"))):
        for ln in open(p):
            r = json.loads(ln)
            if r.get("l1") in (0, 1) and r["pair_id"] not in labels:
                labels[r["pair_id"]] = int(r["l1"])
    df = bank.merge(inp, on="pair_id", how="left")
    if "canonical_pid_x" in df.columns:
        df["canonical_pid"] = df["canonical_pid_x"].fillna(df["canonical_pid_y"])
    df["label"] = df["pair_id"].map(labels)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    score = sorted(c for c in df.columns if c.endswith("_score") and not c.startswith("a6"))
    applied = sorted(c for c in df.columns if c.endswith("_applied") and not c.startswith("a6"))
    return df, score, applied, "canonical_pid", dict(n_estimators=500, max_depth=None, min_samples_leaf=5)


def load_lc():
    labels = pd.read_parquet(LC / "merged_labels.parquet")
    bank = pd.read_parquet(LC / "bank_scores.parquet")
    par = pd.read_parquet(LC / "paradigm_scores.parquet")
    df = labels.merge(bank, on="candidate_id", how="left").merge(par, on="candidate_id", how="left")
    df = df[df["l1"].isin([0, 1])].copy()
    df["label"] = df["l1"].astype(int)
    df["canonical_pid"] = df["problem_id"].astype(str)
    score = sorted(c for c in df.columns if c.endswith("_score"))
    applied = sorted(c for c in df.columns if c.endswith("_applied"))
    return df, score, applied, "canonical_pid", dict(n_estimators=500, min_samples_leaf=2)


def run_cell(name, loader):
    df, score, applied, gcol, rf_cfg = loader()
    y = df["label"].values
    groups = df[gcol].astype(str).values
    print(f"\n=== {name}: n={len(df)} pos={y.mean():.3f} groups={df[gcol].nunique()} "
          f"features={len(score)+len(applied)}")
    res = {"n": int(len(df)), "pos_rate": float(y.mean())}

    feat_sets = {"full": score + applied}
    # applied-rate selection
    app_rate = df[applied].mean()
    for thr in (0.05, 0.20):
        keep = [a[:-8] for a in applied if app_rate[a] >= thr]
        feat_sets[f"applied>={int(thr*100)}%"] = \
            [f"{k}_score" for k in keep] + [f"{k}_applied" for k in keep]

    for fname, cols in feat_sets.items():
        X = df[cols].astype(float).values
        r_lr, oof_lr = cv_run(lambda sd: make_lr(False), X, y, groups)
        r_lrb, oof_lrb = cv_run(lambda sd: make_lr(True), X, y, groups)
        r_rf, oof_rf = cv_run(lambda sd, c=rf_cfg: make_rf(seed=sd, **c), X, y, groups)
        r_hgb, oof_hgb = cv_run(lambda sd: make_hgb(sd), X, y, groups)
        ens2 = safe_auc(y, rank_avg(oof_lr, oof_rf))
        ens3 = safe_auc(y, rank_avg(oof_lr, oof_rf, oof_hgb))
        ens3b = safe_auc(y, rank_avg(oof_lrb, oof_rf, oof_hgb))
        res[fname] = {
            "n_features": len(cols),
            "lr": r_lr, "lr_balanced": r_lrb, "rf": r_rf, "hgb": r_hgb,
            "ens_lr_rf": ens2, "ens_lr_rf_hgb": ens3, "ens_lrb_rf_hgb": ens3b,
        }
        print(f"  [{fname:>12}] LR {r_lr['auc_mean']:.3f} | LRbal {r_lrb['auc_mean']:.3f} | "
              f"RF {r_rf['auc_mean']:.3f} | HGB {r_hgb['auc_mean']:.3f} | "
              f"ens2 {ens2:.3f} | ens3 {ens3:.3f} | ens3bal {ens3b:.3f}", flush=True)
    return res


def main():
    out = {"cf": run_cell("CF", load_cf), "lc": run_cell("LC", load_lc)}
    (OUT / "headroom_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT/'headroom_results.json'}")


if __name__ == "__main__":
    main()
