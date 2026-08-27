"""Method A: TF-IDF char_wb 3-5gram LR on candidate_text ALONE.

LEAKAGE GUARD: TF-IDF is fit on candidate_text only. The editorial_text is
never used as a feature. No cosine, no embedding similarity, no pairwise term.

Per-fold pipeline:
  TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=50_000)
  LogisticRegression(C=1.0, solver='liblinear', max_iter=2000)
  StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

Reports AUC for: pooled, lc-only, luogu-only, and per-decile within each platform.
"""
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
OUT = f"{ROOT}/outputs/v2_analysis"

SUB = f"{OUT}/comp_qwen_phase1_stratified_subsample.parquet"
OUT_JSON = f"{OUT}/comp_qwen_phase1_tfidf_auc.json"

RNG = 42
N_SPLITS = 5
DECILE_LABELS = ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]


def cv_auc(texts, y, n_splits=N_SPLITS, seed=RNG):
    """5-fold CV. Fit TF-IDF inside each fold (avoid vocab leakage)."""
    texts = np.asarray(texts, dtype=object)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
        return None, None, None
    n_splits = min(n_splits, int(min(np.bincount(y))))
    if n_splits < 2:
        return None, None, None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(texts, y):
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=50_000,
            lowercase=False,
            min_df=2,
        )
        Xtr = vec.fit_transform(texts[tr])
        Xte = vec.transform(texts[te])
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs)), int(n_splits)


def main():
    df = pd.read_parquet(SUB)
    print(f"loaded {len(df)} rows", flush=True)

    # PRE-FLIGHT LEAKAGE CHECK
    print("--- pre-flight leakage check ---", flush=True)
    print(f"label col: qwen_label", flush=True)
    print(f"feature: TF-IDF over candidate_text ONLY (editorial_text not used)", flush=True)
    assert "cosine" not in df.columns, "subsample must not carry cosine column"
    bad = [c for c in df.columns if any(t in c.lower() for t in ["cos", "sim", "embed"])]
    assert not bad, f"LEAKAGE GUARD: blocked subsample cols {bad}"
    print("OK: no cosine/sim/embed column on input", flush=True)

    out = {
        "method": "tfidf_charwb_3_5_LR",
        "features": "TF-IDF char_wb ngram_range=(3,5) max_features=50000 on candidate_text ONLY",
        "leakage_note": "editorial_text NOT used; no cosine/embedding/pairwise feature",
        "label_col": "qwen_label",
        "n_splits": N_SPLITS,
        "seed": RNG,
        "cells": {},
        "deciles": {},
    }

    # Pooled / per-platform
    for tag, mask_expr in [
        ("pooled", np.ones(len(df), bool)),
        ("lc", (df["platform"] == "lc").to_numpy()),
        ("luogu", (df["platform"] == "luogu").to_numpy()),
    ]:
        mask = mask_expr
        texts = df.loc[mask, "candidate_text"].astype(str).tolist()
        y = df.loc[mask, "qwen_label"].to_numpy(dtype=int)
        auc, sd, used_k = cv_auc(texts, y)
        pos = float(y.mean()) if len(y) else None
        out["cells"][tag] = {"n": int(mask.sum()), "pos_rate": pos, "auc": auc, "auc_sd": sd, "n_splits": used_k}
        print(f"[A] {tag}: n={int(mask.sum())} pos={pos:.3f} auc={auc} sd={sd}", flush=True)

    # Per-decile within platform
    for plat in ["lc", "luogu"]:
        out["deciles"][plat] = {}
        for d_idx, d_label in enumerate(DECILE_LABELS):
            mask = ((df["platform"] == plat) & (df["decile"] == d_idx)).to_numpy()
            texts = df.loc[mask, "candidate_text"].astype(str).tolist()
            y = df.loc[mask, "qwen_label"].to_numpy(dtype=int)
            if len(y) < 10:
                out["deciles"][plat][d_label] = {"n": int(mask.sum()), "pos_rate": None, "auc": None, "auc_sd": None}
                continue
            auc, sd, used_k = cv_auc(texts, y)
            pos = float(y.mean())
            out["deciles"][plat][d_label] = {"n": int(mask.sum()), "pos_rate": pos, "auc": auc, "auc_sd": sd}
            print(f"[B] {plat} {d_label}: n={int(mask.sum())} pos={pos:.3f} auc={auc}", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
