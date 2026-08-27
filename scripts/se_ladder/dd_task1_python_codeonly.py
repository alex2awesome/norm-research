"""Deep dive task 1b: python-slice code-filter heterogeneity (2026-06-12).

SO[python] v2 has NO code-presence filter (js/sql do). Quantify:
  - fraction of the balanced set lacking a fenced code block (overall,
    per label, per stratum, per split);
  - bank ENS + TF-IDF(answer body) + bge-probe AUC re-run on the
    code-only subset (train on code-only train, test on code-only test)
    to see whether the filter delta changes any ladder conclusion.

CPU only. Run on sk3 from /lfs/skampere3/0/alexspan/norm-research.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"
SLICE = "so_python"


def bank_eval(df, mask, tag):
    """Re-fit LR/RF/ENS on the masked subset (train→test)."""
    sub = df[mask]
    tr, te = sub[sub.split == "train"], sub[sub.split == "test"]
    cand = [c for c in df.columns if c.endswith("_score") or c.endswith("_applied")]
    feats = []
    for c in cand:
        v = tr[c]
        if v.notna().mean() < 0.05:
            continue
        nz = v[v.notna()]
        if c.endswith("_applied"):
            if nz.nunique() < 2:
                continue
        elif nz.nunique() < 3:
            continue
        feats.append(c)
    Xtr, ytr = tr[feats].values.astype(float), tr.label.values
    Xte, yte = te[feats].values.astype(float), te.label.values
    lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                              solver="liblinear"))])
    rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("clf", RandomForestClassifier(n_estimators=500,
                                                  min_samples_leaf=3,
                                                  n_jobs=16, random_state=0))])
    lr.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    p_rf = rf.predict_proba(Xte)[:, 1]
    p_en = (rankdata(p_lr) + rankdata(p_rf)) / (2 * len(p_lr))
    res = {"n_train": int(len(tr)), "n_test": int(len(te)),
           "n_features": len(feats),
           "LR": float(roc_auc_score(yte, p_lr)),
           "RF": float(roc_auc_score(yte, p_rf)),
           "ENS": float(roc_auc_score(yte, p_en))}
    strata = {}
    te2 = te.copy()
    te2["p"] = p_en
    for s, g in te2.groupby("stratum"):
        if len(g) >= 200 and g.label.nunique() == 2:
            strata[s] = {"n": int(len(g)),
                         "ENS": float(roc_auc_score(g.label, g.p))}
    res["per_stratum_ENS"] = strata
    print(tag, json.dumps(res, indent=2), flush=True)
    return res


def tfidf_eval(df, mask, tag):
    sub = df[mask]
    tr, te = sub[sub.split == "train"], sub[sub.split == "test"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000)
    Xtr = vec.fit_transform(tr.body.fillna(""))
    Xte = vec.transform(te.body.fillna(""))
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr.fit(Xtr, tr.label.values)
    p = lr.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(te.label.values, p))
    te2 = te.copy(); te2["p"] = p
    strata = {s: float(roc_auc_score(g.label, g.p))
              for s, g in te2.groupby("stratum")
              if len(g) >= 200 and g.label.nunique() == 2}
    print(f"{tag} tfidf AUC={auc:.4f} strata={strata}", flush=True)
    return {"AUC": auc, "per_stratum": strata,
            "n_train": int(len(tr)), "n_test": int(len(te))}


def bge_eval(df, emb, mask, tag):
    sub_idx = np.where(mask.values)[0]
    sub = df.iloc[sub_idx]
    tr_m = (sub.split == "train").values
    te_m = (sub.split == "test").values
    E = emb[sub_idx]
    lr = LogisticRegression(max_iter=3000, C=1.0)
    lr.fit(E[tr_m], sub.label.values[tr_m])
    p = lr.predict_proba(E[te_m])[:, 1]
    auc = float(roc_auc_score(sub.label.values[te_m], p))
    print(f"{tag} bge probe AUC={auc:.4f}", flush=True)
    return auc


def main():
    inp = pd.read_parquet(OUT_DIR / f"{SLICE}_input.parquet")
    inp["has_code"] = inp.code.fillna("").str.len() > 0
    res = {"n": int(len(inp)),
           "no_code_frac_overall": float(1 - inp.has_code.mean()),
           "no_code_frac_by_label": {int(k): float(1 - v) for k, v in
                                     inp.groupby("label").has_code.mean().items()},
           "no_code_frac_by_stratum": {k: float(1 - v) for k, v in
                                       inp.groupby("stratum").has_code.mean().items()},
           "no_code_frac_by_split": {k: float(1 - v) for k, v in
                                     inp.groupby("split").has_code.mean().items()}}
    print(json.dumps(res, indent=2), flush=True)

    # bank features
    shard_dir = OUT_DIR / "shards" / SLICE
    scored = pd.concat([pd.read_parquet(p) for p in
                        sorted(shard_dir.glob("shard_*.parquet"))],
                       ignore_index=True)
    df = inp.merge(scored, on="row_id", how="inner")
    assert len(df) == len(inp)

    res["bank_codeonly"] = bank_eval(df, df.has_code, "[bank code-only]")
    res["bank_all_ref"] = bank_eval(df, pd.Series(True, index=df.index),
                                    "[bank all ref]")
    res["tfidf_codeonly"] = tfidf_eval(df, df.has_code, "[tfidf code-only]")
    res["tfidf_all_ref"] = tfidf_eval(df, pd.Series(True, index=df.index),
                                      "[tfidf all ref]")
    emb = np.load(OUT_DIR / f"{SLICE}_bge.npy")
    assert len(emb) == len(inp), (emb.shape, len(inp))
    res["bge_codeonly_auc"] = bge_eval(inp, emb, inp.has_code, "[bge code-only]")

    out = OUT_DIR / "dd_python_codeonly.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
