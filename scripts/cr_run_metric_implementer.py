"""Score all metric_implementer metrics on the 4978 v2 code_review datapoints.

For each datapoint we feed the DENSE diff text (not the v2 comment-thread
artifact), so tool-grounded metrics see actual code.

Joins to the existing cr_tier12 + cr_tier34 features and reports the
combined RF AUC ladder.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
SEED = 42

# Make miniconda binaries reachable so subprocess-based metrics work
os.environ["PATH"] = (f"/lfs/skampere3/0/alexspan/miniconda3/bin:"
                      + os.environ.get("PATH", ""))

sys.path.insert(0, str(REPO))


def score_chunk(payload):
    rows, return_cols = payload
    # Re-import inside child process to avoid sharing state
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    out = []
    for dp_id, text, y in rows:
        row = {"datapoint_id": dp_id, "y": y}
        for m in metrics:
            try:
                a = m.applies(text)
            except Exception:
                a = False
            applied = int(bool(a))
            score = float("nan")
            if applied:
                try:
                    s = m.score(text)
                    if s is not None and not (isinstance(s, float)
                                              and math.isnan(s)):
                        score = float(s)
                except Exception:
                    pass
            row[f"{m.ASPECT_ID}_score"] = score
            row[f"{m.ASPECT_ID}_applied"] = applied
        out.append(row)
    return out


def main():
    print("Loading v2 datapoints + dense diff text...")
    dps_file = REPO / "runs/validity_full/v2/code_review/datapoints.json"
    dense_file = REPO / "datasets/code-review/code_review_dense_4096tok.csv.gz"
    dps = json.loads(dps_file.read_text())
    dps = [d for d in dps if d.get("judgement") is not None and d.get("text")]

    v2 = pd.DataFrame([{
        "datapoint_id": d["datapoint_id"],
        "y": int(d["judgement"]),
        "title": (re.match(r"PR TITLE: ([^\n]+)", d["text"]) or [None, None])[1],
    } for d in dps]).dropna(subset=["title"])

    dense = pd.read_csv(dense_file, usecols=["text"])
    dense["title"] = dense["text"].str.extract(
        r"## PR Title\s*(.+?)(?:\n|$)", expand=False)
    j = v2.merge(dense.drop_duplicates("title", keep="first"),
                 on="title", how="left").dropna(subset=["text"])
    print(f"joined {len(j)} datapoints to diff text")

    rows = [(r.datapoint_id, r.text, r.y) for r in j.itertuples()]

    # Sanity-load metrics in parent process
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    print(f"loaded {len(metrics)} metrics in parent")

    n_workers = 4
    chunk_size = max(1, len(rows) // (n_workers * 4))
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    print(f"workers={n_workers}, chunks={len(chunks)}, "
          f"chunk_size={chunk_size}")

    feat_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(score_chunk, (c, None)) for c in chunks]
        for k, fut in enumerate(as_completed(futs)):
            feat_rows.extend(fut.result())
            elapsed = time.time() - t0
            done = len(feat_rows)
            eta = elapsed / max(done, 1) * (len(rows) - done) / 60
            print(f"  chunk {k+1}/{len(chunks)} done — total scored={done}, "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.1f}min")

    feat = pd.DataFrame(feat_rows)
    out_p = REPO / "outputs/v2_analysis/cr_metric_implementer_scores.parquet"
    feat.to_parquet(out_p)
    print(f"\nwrote {out_p}, shape={feat.shape}")

    # === Combined RF ladder ===
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n=== Building combined feature matrix ===")
    t12 = pd.read_parquet(REPO / "outputs/v2_analysis/cr_tier12_features.parquet")
    t34 = pd.read_parquet(REPO / "outputs/v2_analysis/cr_tier34_features.parquet"
                          ).drop(columns=["y"], errors="ignore")
    base = t12.merge(t34, on="datapoint_id")

    mi = feat.drop(columns=["y"], errors="ignore")
    combo = base.merge(mi, on="datapoint_id", how="inner")
    y = combo["y"].astype(int).values
    base_cols = [c for c in t12.columns if c not in ("datapoint_id", "y")]
    t3_cols = [c for c in t34.columns if c.startswith("tier3_")]
    mi_score = [c for c in mi.columns if c.endswith("_score")]
    mi_applied = [c for c in mi.columns if c.endswith("_applied")]
    mi_all = mi_score + mi_applied
    print(f"  rows={len(combo)}, base={len(base_cols)}, t3={len(t3_cols)}, "
          f"mi_score={len(mi_score)}, mi_applied={len(mi_applied)}")

    def fit(cols, label):
        Xc = combo[cols].values
        Xtr, Xte, ytr, yte = train_test_split(
            Xc, y, test_size=0.20, stratify=y, random_state=SEED)
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("rf", RandomForestClassifier(
                             n_estimators=500, min_samples_leaf=2,
                             class_weight="balanced",
                             n_jobs=-1, random_state=SEED))])
        pipe.fit(Xtr, ytr)
        auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
        pipe2 = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler(with_mean=False)),
                          ("lr", LogisticRegression(
                              penalty="l2", C=0.5, class_weight="balanced",
                              max_iter=3000, solver="lbfgs"))])
        pipe2.fit(Xtr, ytr)
        auc_lr = roc_auc_score(yte, pipe2.predict_proba(Xte)[:, 1])
        print(f"  {label:<45} feats={len(cols):4d}  RF={auc:.3f}  LR={auc_lr:.3f}")
        return pipe

    print()
    print("=" * 78)
    print("CODE_REVIEW LADDER WITH METRIC_IMPLEMENTER")
    print("=" * 78)
    fit(base_cols, "T1+T2 (metadata+diff parse)")
    fit(base_cols + t3_cols, "T1+T2+T3 (+lizard)")
    fit(mi_score, "MI scores only")
    fit(mi_all, "MI scores + applied flags")
    fit(base_cols + t3_cols + mi_score, "T1+T2+T3 + MI scores")
    fit(base_cols + t3_cols + mi_all, "T1+T2+T3 + MI all")


if __name__ == "__main__":
    main()
