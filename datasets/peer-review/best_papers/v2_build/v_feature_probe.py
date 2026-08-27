#!/usr/bin/env python3
"""Deterministic V-feature probe for best_papers_v2.

Reuses the per-aspect Python predict-programs at
runs/validity_full/v2/peer_review/codegen_claude/ (each exposes
score(text)->float). We run every program over each paper's TITLE+ABSTRACT
to build a deterministic feature matrix, then fit logistic regression
(train split) and measure AUC on held-out (eval+test). If AUC > 0.5 the
verifiable layer carries signal here (V > 0).

These programs were authored for peer-review *review text*, not paper
abstracts, so they are a conservative lower bound on V for this task -- any
lift above 0.5 is real deterministic-feature signal.
"""
import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_programs(d):
    progs = []
    for path in sorted(glob.glob(os.path.join(d, "*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "score"):
                progs.append((name, mod.score))
        except Exception:
            continue
    return progs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="best_papers_v2_full.csv.gz")
    ap.add_argument("--progdir",
                    default="/lfs/skampere3/0/alexspan/norm-research/runs/validity_full/v2/peer_review/codegen_claude")
    args = ap.parse_args()
    df = pd.read_csv(args.data).dropna(subset=["text"]).reset_index(drop=True)
    progs = load_programs(args.progdir)
    print(f"loaded {len(progs)} V programs; scoring {len(df)} papers...", flush=True)

    texts = df.text.tolist()
    feats = np.zeros((len(df), len(progs)), dtype=np.float32)
    for j, (name, fn) in enumerate(progs):
        for i, t in enumerate(texts):
            try:
                v = fn(t)
                feats[i, j] = float(v) if v is not None else 0.5
            except Exception:
                feats[i, j] = 0.5
        if (j + 1) % 100 == 0:
            print(f"  scored {j+1}/{len(progs)} programs", flush=True)

    # drop constant columns
    keep = feats.std(axis=0) > 1e-9
    feats = feats[:, keep]
    print(f"non-constant V features: {feats.shape[1]}")

    tr = df.split == "train"
    te = df.split.isin(["eval", "test"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(feats[tr.values], df.label[tr].values)
    p = clf.predict_proba(feats[te.values])[:, 1]
    auc = roc_auc_score(df.label[te].values, p)
    print(f"\nV-FEATURE AUC (LR over {feats.shape[1]} deterministic programs): {auc:.3f}")
    print(f"V > 0: {'YES' if auc > 0.52 else 'marginal' if auc > 0.50 else 'NO'}")

    # within venue-year V-AUC
    dte = df[te].copy(); dte["pred"] = p
    aucs = []
    for (v, y), g in dte.groupby(["venue", "year"]):
        if g.label.nunique() == 2 and len(g) >= 4:
            aucs.append(roc_auc_score(g.label, g.pred))
    if aucs:
        print(f"within-vy V-AUC: mean {np.mean(aucs):.3f} over {len(aucs)} vy")


if __name__ == "__main__":
    main()
