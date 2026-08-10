#!/usr/bin/env python3
"""Item 1 eval: frozen-recipe AUC on (a) ORIGINAL cells (replication, for
comparability) and (b) EXTENDED cells (original + negative-enriched extension).

Per-cell frozen recipes (unchanged from the published numbers):
  CF: LR(no class_weight, liblinear C=1) / RF(500, best-cfg leaf=5, balanced) /
      LR+RF rank-avg; StratifiedGroupKFold(5) by canonical_pid; seeds 17,23,29,31,37.
  LC: LR(balanced, with_mean=False scaler) / RF(500, leaf=2, balanced) / per-fold
      rank-avg; StratifiedGroupKFold(5) by problem_id; seeds 13,21,34,55,89;
      features = bank + paradigm.

CIs: cluster bootstrap over problems (1,000 reps) on pooled OOF ensemble scores.
Tolerates partial extension labels (reports how many).

Output: outputs/v2_analysis/lc_cf_push/extended_eval_results.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
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
PUSH = ROOT / "outputs/v2_analysis/lc_cf_push"
N_FOLDS = 5


def safe_auc(y, p):
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


def cluster_bootstrap_ci(y, p, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "p": p, "g": groups})
    by_g = {g: sub for g, sub in df.groupby("g")}
    gs = list(by_g)
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(len(gs), size=len(gs), replace=True)
        sub = pd.concat([by_g[gs[i]] for i in pick])
        if sub["y"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(sub["y"], sub["p"]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


# ---------------- CF recipe ----------------

def cf_models(seed):
    lr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
    ])
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=500, max_depth=None,
                                      min_samples_leaf=5, n_jobs=-1,
                                      class_weight="balanced", random_state=seed)),
    ])
    return lr, rf


def run_cf_recipe(X, y, groups):
    seeds = [17, 23, 29, 31, 37]
    aucs = {"lr": [], "rf": []}
    oof_lr_all, oof_rf_all = [], []
    for sd in seeds:
        oof_lr = np.full(len(y), np.nan)
        oof_rf = np.full(len(y), np.nan)
        for tr, te in grouped_splits(X, y, groups, sd):
            lr, rf = cf_models(sd)
            lr.fit(X[tr], y[tr]); oof_lr[te] = lr.predict_proba(X[te])[:, 1]
            rf.fit(X[tr], y[tr]); oof_rf[te] = rf.predict_proba(X[te])[:, 1]
        aucs["lr"].append(safe_auc(y, oof_lr))
        aucs["rf"].append(safe_auc(y, oof_rf))
        oof_lr_all.append(oof_lr); oof_rf_all.append(oof_rf)
    m_lr = np.nanmean(np.vstack(oof_lr_all), axis=0)
    m_rf = np.nanmean(np.vstack(oof_rf_all), axis=0)
    ens = (pd.Series(m_lr).rank(pct=True).values + pd.Series(m_rf).rank(pct=True).values) / 2
    return {
        "lr": {"mean": float(np.mean(aucs["lr"])), "std": float(np.std(aucs["lr"]))},
        "rf": {"mean": float(np.mean(aucs["rf"])), "std": float(np.std(aucs["rf"]))},
        "ens": safe_auc(y, ens),
        "ens_ci95_cluster_boot": cluster_bootstrap_ci(y, ens, groups),
    }


# ---------------- LC recipe (eval_auc.py replication) ----------------

def run_lc_recipe(X, y, groups):
    seeds = [13, 21, 34, 55, 89]
    per_seed = {"lr": [], "rf": [], "ens": []}
    oof_lr_all, oof_rf_all = [], []
    for sd in seeds:
        oof_lr = np.full(len(y), np.nan)
        oof_rf = np.full(len(y), np.nan)
        fold_ens = []
        for tr, te in grouped_splits(X, y, groups, sd):
            imp = SimpleImputer(strategy="median")
            Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
            sc = StandardScaler(with_mean=False)
            Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
            lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                    random_state=sd)
            lr.fit(Xtr_s, y[tr]); p_lr = lr.predict_proba(Xte_s)[:, 1]
            rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                        n_jobs=-1, class_weight="balanced",
                                        random_state=sd)
            rf.fit(Xtr, y[tr]); p_rf = rf.predict_proba(Xte)[:, 1]
            oof_lr[te], oof_rf[te] = p_lr, p_rf
            fold_ens.append(safe_auc(y[te], (rankdata(p_lr) + rankdata(p_rf)) / 2))
        per_seed["lr"].append(safe_auc(y, oof_lr))
        per_seed["rf"].append(safe_auc(y, oof_rf))
        per_seed["ens"].append(float(np.nanmean(fold_ens)))
        oof_lr_all.append(oof_lr); oof_rf_all.append(oof_rf)
    m_lr = np.nanmean(np.vstack(oof_lr_all), axis=0)
    m_rf = np.nanmean(np.vstack(oof_rf_all), axis=0)
    ens_pool = (pd.Series(m_lr).rank(pct=True).values + pd.Series(m_rf).rank(pct=True).values) / 2
    return {
        "lr": {"mean": float(np.mean(per_seed["lr"])), "std": float(np.std(per_seed["lr"]))},
        "rf": {"mean": float(np.mean(per_seed["rf"])), "std": float(np.std(per_seed["rf"]))},
        "ens": {"mean": float(np.mean(per_seed["ens"])), "std": float(np.std(per_seed["ens"]))},
        "ens_pooled_oof": safe_auc(y, ens_pool),
        "ens_ci95_cluster_boot": cluster_bootstrap_ci(y, ens_pool, groups),
    }


# ---------------- data ----------------

def read_labels(pattern):
    labels = {}
    for p in sorted(glob.glob(pattern)):
        for ln in open(p):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("l1") in (0, 1) and r["pair_id"] not in labels:
                labels[r["pair_id"]] = int(r["l1"])
    return labels


def load_cf_frames():
    bank = pd.read_parquet(CF / "cf2_bank_scores.parquet")
    inp = pd.read_parquet(CF / "cf2_bank_input.parquet")[["pair_id", "canonical_pid"]]
    orig = bank.merge(inp, on="pair_id", how="left")
    if "canonical_pid_x" in orig.columns:
        orig["canonical_pid"] = orig["canonical_pid_x"].fillna(orig["canonical_pid_y"])
    lab_o = read_labels(str(CF / "results" / "shard_cf2_*_results.jsonl"))
    orig["label"] = orig["pair_id"].map(lab_o)

    ext_bank = pd.read_parquet(PUSH / "cfx_bank_scores.parquet")
    lab_e = read_labels(str(PUSH / "cf_ext" / "results" / "shard_cfx_*_results.jsonl"))
    ext_bank["label"] = ext_bank["pair_id"].map(lab_e)
    feat = sorted(c for c in bank.columns
                  if (c.endswith("_score") or c.endswith("_applied")) and not c.startswith("a6"))
    return orig, ext_bank, feat


def load_lc_frames():
    labels = pd.read_parquet(LC / "merged_labels.parquet")
    bank = pd.read_parquet(LC / "bank_scores.parquet")
    par = pd.read_parquet(LC / "paradigm_scores.parquet")
    orig = labels.merge(bank, on="candidate_id", how="left").merge(par, on="candidate_id", how="left")
    orig = orig[orig["l1"].isin([0, 1])].copy()
    orig["label"] = orig["l1"].astype(int)
    orig["canonical_pid"] = orig["problem_id"].astype(str)

    # extension: candidate_id keyed scores; map via shards
    import json as _json
    rows = []
    for f in sorted((PUSH / "lc_ext" / "shards").glob("shard_lcx_*.jsonl")):
        for ln in open(f):
            d = _json.loads(ln)
            rows.append({"pair_id": d["pair_id"], "candidate_id": str(d["candidate_id"]),
                         "canonical_pid": d["problem_id"]})
    emap = pd.DataFrame(rows)
    eb = pd.read_parquet(PUSH / "lc_ext" / "bank_scores.parquet")
    ep = pd.read_parquet(PUSH / "lc_ext" / "paradigm_scores.parquet")
    ext = emap.merge(eb, on="candidate_id", how="left").merge(ep, on="candidate_id", how="left")
    lab_e = read_labels(str(PUSH / "lc_ext" / "results" / "shard_lcx_*_results.jsonl"))
    ext["label"] = ext["pair_id"].map(lab_e)
    feat = sorted(c for c in pd.concat([bank.head(0), par.head(0)], axis=1).columns
                  if c.endswith("_score") or c.endswith("_applied"))
    return orig, ext, feat


def eval_cell(name, orig, ext, feat, runner):
    out = {}
    o = orig[orig["label"].notna()].copy()
    o["label"] = o["label"].astype(int)
    print(f"\n=== {name} ORIGINAL: n={len(o)} pos={o.label.mean():.3f} "
          f"neg={int((1-o.label).sum())}")
    out["original"] = {
        "n": int(len(o)), "pos_rate": float(o.label.mean()),
        "n_neg": int((1 - o.label).sum()),
        "n_problems": int(o.canonical_pid.nunique()),
        **runner(o[feat].astype(float).values, o.label.values,
                 o.canonical_pid.astype(str).values),
    }
    print(json.dumps(out["original"], indent=1)[:400])

    e = ext[ext["label"].notna()].copy()
    out["ext_labeled_so_far"] = int(len(e))
    if len(e) >= 200:
        e["label"] = e["label"].astype(int)
        both = pd.concat([o[["canonical_pid", "label"] + feat],
                          e[["canonical_pid", "label"] + feat]], ignore_index=True)
        print(f"=== {name} EXTENDED: n={len(both)} pos={both.label.mean():.3f} "
              f"neg={int((1-both.label).sum())} (ext labeled: {len(e)}, "
              f"ext pos={e.label.mean():.3f})")
        out["extension_only"] = {"n": int(len(e)), "pos_rate": float(e.label.mean()),
                                 "n_neg": int((1 - e.label).sum())}
        out["extended"] = {
            "n": int(len(both)), "pos_rate": float(both.label.mean()),
            "n_neg": int((1 - both.label).sum()),
            "n_problems": int(both.canonical_pid.nunique()),
            **runner(both[feat].astype(float).values, both.label.values,
                     both.canonical_pid.astype(str).values),
        }
        print(json.dumps(out["extended"], indent=1)[:400])
    else:
        print(f"=== {name}: only {len(e)} extension labels so far — skipping extended eval")
    return out


def main():
    res = {}
    orig, ext, feat = load_cf_frames()
    res["cf"] = eval_cell("CF", orig, ext, feat, run_cf_recipe)
    orig, ext, feat = load_lc_frames()
    res["lc"] = eval_cell("LC", orig, ext, feat, run_lc_recipe)
    (PUSH / "extended_eval_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {PUSH/'extended_eval_results.json'}")


if __name__ == "__main__":
    main()
