#!/usr/bin/env python3
"""Citation deconfound + TF-IDF bounds for best_papers_v2.

Reports honestly:
  (a) citation-only AUC          -- the confound (no text)
  (b) raw text TF-IDF floor      -- text signal, citation-entangled
  (c) citation-decile-matched text TF-IDF floor -- de-confounded text signal
       (match y=1/y=0 on cited_by_count deciles WITHIN venue-year, then
        re-measure the text floor on the matched subset)
  (d) within-venue-year stratified AUC -- awards are budget-bound (1-3/vy)
Plus: top TF-IDF features (leakage watch), title+abstract length confound.

Train on split in {train}, evaluate on {eval,test} pooled (held out by hash).
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def auc_text(tr, te, text_col="text", label_col="label"):
    vec = TfidfVectorizer(min_df=3, max_df=0.6, ngram_range=(1, 2),
                          sublinear_tf=True, strip_accents="unicode",
                          stop_words="english", max_features=60000)
    Xtr = vec.fit_transform(tr[text_col]); Xte = vec.transform(te[text_col])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, tr[label_col])
    p = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(te[label_col], p), vec, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="best_papers_v2_full.csv.gz")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text"])
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce").fillna(0)
    tr = df[df.split == "train"].copy()
    te = df[df.split.isin(["eval", "test"])].copy()
    print(f"rows: {len(df)}  train {len(tr)} (pos {int(tr.label.sum())})  "
          f"test {len(te)} (pos {int(te.label.sum())})")
    print(f"overall positive rate: {df.label.mean():.3f}\n")

    # ---- (a) citation-only AUC (log1p of cited_by_count) ----
    ctr = np.log1p(tr.cited_by_count.values).reshape(-1, 1)
    cte = np.log1p(te.cited_by_count.values).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression as LR
    cl = LR(max_iter=1000, class_weight="balanced").fit(ctr, tr.label)
    a_cit = roc_auc_score(te.label, cl.predict_proba(cte)[:, 1])
    # also raw rank correlation form
    a_cit_raw = roc_auc_score(te.label, np.log1p(te.cited_by_count.values))
    print(f"(a) CITATION-ONLY AUC: {a_cit:.3f}  (raw cite-rank AUC {a_cit_raw:.3f})  <- the confound")

    # ---- (b) raw text TF-IDF floor ----
    a_text, vec, clf = auc_text(tr, te)
    print(f"(b) RAW TEXT TF-IDF AUC: {a_text:.3f}")

    # ---- (c) citation-decile-matched text floor ----
    # within each venue-year, bin cited_by_count into deciles; for matching,
    # keep y=0 papers whose citation decile is occupied by a y=1 paper in that
    # vy (Math.SE v3.3 style), balancing positives vs negatives per decile.
    matched_idx = []
    for (v, y), g in df.groupby(["venue", "year"]):
        pos = g[g.label == 1]; neg = g[g.label == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        # decile edges from the full vy citation distribution
        try:
            edges = np.quantile(g.cited_by_count.values, np.linspace(0, 1, 11))
        except Exception:
            continue
        edges = np.unique(edges)
        if len(edges) < 2:
            # all same citation -> keep all
            matched_idx.extend(pos.index.tolist())
            matched_idx.extend(neg.index.tolist())
            continue
        gb = pd.cut(g.cited_by_count, bins=edges, include_lowest=True, labels=False)
        g = g.assign(dec=gb)
        for d, gg in g.groupby("dec"):
            p = gg[gg.label == 1]; n = gg[gg.label == 0]
            if len(p) == 0 or len(n) == 0:
                continue
            # 1:1 match per decile so the citation distribution is identical
            # across labels within the matched set (kills the citation signal)
            k = min(len(p), len(n))
            p_keep = p.sample(n=k, random_state=42) if len(p) > k else p
            n_keep = n.sample(n=k, random_state=42) if len(n) > k else n
            matched_idx.extend(p_keep.index.tolist())
            matched_idx.extend(n_keep.index.tolist())
    md = df.loc[sorted(set(matched_idx))].copy()
    mtr = md[md.split == "train"]; mte = md[md.split.isin(["eval", "test"])]
    print(f"    matched subset: {len(md)} rows (pos {int(md.label.sum())}, "
          f"neg {int((md.label==0).sum())})")
    # verify citation confound is now weak in matched set
    if len(mte) > 20 and mte.label.nunique() == 2:
        a_cit_m = roc_auc_score(mte.label, np.log1p(mte.cited_by_count.values))
        print(f"    citation-only AUC IN MATCHED set: {a_cit_m:.3f} (should be ~0.5)")
    if len(mtr) > 30 and mtr.label.nunique() == 2 and len(mte) > 20 and mte.label.nunique() == 2:
        a_text_m, _, _ = auc_text(mtr, mte)
        print(f"(c) CITATION-MATCHED TEXT TF-IDF AUC: {a_text_m:.3f}  <- de-confounded text signal")
    else:
        print("(c) matched set too small/degenerate for a stable floor")

    # ---- (d) within-venue-year stratified AUC ----
    # pool predictions but rank within venue-year (awards are 1-3/vy)
    p_all = clf.predict_proba(vec.transform(te.text))[:, 1]
    te = te.assign(pred=p_all)
    aucs = []
    for (v, y), g in te.groupby(["venue", "year"]):
        if g.label.nunique() == 2 and len(g) >= 4:
            aucs.append(roc_auc_score(g.label, g.pred))
    if aucs:
        print(f"(d) WITHIN-VENUE-YEAR text AUC: mean {np.mean(aucs):.3f} "
              f"median {np.median(aucs):.3f} over {len(aucs)} vy")

    # ---- top features (leakage watch) ----
    coef = clf.coef_[0]
    names = np.array(vec.get_feature_names_out())
    top_pos = names[np.argsort(coef)[-25:]][::-1]
    top_neg = names[np.argsort(coef)[:25]]
    print("\nTOP y=1 features:", ", ".join(top_pos))
    print("TOP y=0 features:", ", ".join(top_neg))

    # ---- length confound ----
    df["len_tok"] = df.text.str.split().map(len)
    for col, lab in [("len_tok", "text token length")]:
        c1 = df[df.label == 1][col].median(); c0 = df[df.label == 0][col].median()
        a_len = roc_auc_score(te.label, te.text.str.split().map(len))
        print(f"\nLENGTH confound ({lab}): median y=1 {c1:.0f} vs y=0 {c0:.0f}; "
              f"length-only AUC {a_len:.3f}")


if __name__ == "__main__":
    main()
