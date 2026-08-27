#!/usr/bin/env python3
"""bge upper-bound for best_papers_v2: embed TITLE+ABSTRACT with bge, fit LR.

Runs on a single GPU (set CUDA_VISIBLE_DEVICES=1 externally). Reports pooled
AUC and within-venue-year AUC as a dense-embedding upper bound on the text
floor. Optional; skipped if torch/sentence-transformers unavailable.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="best_papers_v2_full.csv.gz")
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    args = ap.parse_args()
    df = pd.read_csv(args.data).dropna(subset=["text"]).reset_index(drop=True)

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(args.model, device="cuda")
    emb = m.encode(df.text.tolist(), batch_size=128, show_progress_bar=True,
                   normalize_embeddings=True)
    tr = (df.split == "train").values
    te = df.split.isin(["eval", "test"]).values
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(emb[tr], df.label[tr].values)
    p = clf.predict_proba(emb[te])[:, 1]
    auc = roc_auc_score(df.label[te].values, p)
    print(f"\nbge UPPER BOUND AUC (pooled): {auc:.3f}")
    dte = df[te].copy(); dte["pred"] = p
    aucs = []
    for (v, y), g in dte.groupby(["venue", "year"]):
        if g.label.nunique() == 2 and len(g) >= 4:
            aucs.append(roc_auc_score(g.label, g.pred))
    if aucs:
        print(f"bge within-vy AUC: mean {np.mean(aucs):.3f} over {len(aucs)} vy")


if __name__ == "__main__":
    main()
