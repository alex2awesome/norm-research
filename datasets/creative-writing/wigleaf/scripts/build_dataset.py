#!/usr/bin/env python3
"""Build the bounded, de-confounded Wigleaf cw-B dataset + run all audits.

Inputs:  top50_texts.jsonl, longlist_texts.jsonl  (+ pilot for top-up)
Label:   y=1 Top50, y=0 longlist  (expert finer-cut within Scott Garson's selection)
Output:  built/{train,eval,test}.csv.gz with cols
         text, raw_text, judgement, year, magazine, tier, fetch_source
Split:   stable hash md5(title+author+year)%10 -> 0-7 train / 8 eval / 9 test

Audits (DATASET-FIRST):
  - cross-tier dedup: drop any longlist (title,author casefold) also in Top50
  - presentation-leak: does fetch_source predict y? (LR on fetch_source dummies)
  - TF-IDF/LR floor on own-split + top features + magazine/bio-leak check
  - length correlation with y
  - V-feature probe (reuse creative_writing codegen programs) -> confirm V>0
"""
import gzip
import hashlib
import json
import os
import re
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/wigleaf"
ROOT = "/lfs/skampere3/0/alexspan/norm-research"
sys.path.insert(0, os.path.join(BASE, "scripts"))
from wig_textproc import (strip_cms_boilerplate, strip_trailing_junk,
                          strip_bio_tail, strip_inline_bio, normalize_text,
                          looks_like_junk_page)

# de-confound switch: use Wayback-fetched Top50 (identical fetch path to longlist)
TOP50_FILE = "top50_wayback_texts.jsonl"


def renorm(raw):
    """Re-run the CURRENT presentation pipeline from raw_text so every row (both
    classes, regardless of when fetched) gets identical normalization."""
    if not raw:
        return ""
    c = strip_cms_boilerplate(raw)
    c = strip_trailing_junk(c)
    c = strip_bio_tail(c)
    c = strip_inline_bio(c)
    c = strip_trailing_junk(c)
    c = normalize_text(c)
    return "" if looks_like_junk_page(c) else c


def load_jsonl(p):
    rows = []
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l)
                r["text"] = renorm(r.get("raw_text") or "")  # uniform renorm
                rows.append(r)
            except Exception:
                pass
    return rows


def story_id(title, author, year):
    return hashlib.md5(f"{title}|{author}|{year}".encode()).hexdigest()


def split_of(sid):
    h = int(sid, 16) % 10
    return "train" if h <= 7 else ("eval" if h == 8 else "test")


def casefold_key(title, author):
    t = re.sub(r"[^a-z0-9 ]", "", str(title).lower()).strip()
    a = re.sub(r"[^a-z0-9 ]", "", str(author).lower()).strip()
    # author surname (last token) to be robust to casing/translator suffixes
    asur = a.split()[-1] if a.split() else a
    return (t, asur)


def main():
    t50 = load_jsonl(os.path.join(BASE, TOP50_FILE))
    ll = load_jsonl(os.path.join(BASE, "longlist_texts.jsonl"))
    print(f"raw rows: top50={len(t50)} ({TOP50_FILE}) longlist={len(ll)}")

    # keep only rows with recovered text
    t50 = [r for r in t50 if r.get("text") and len(r["text"]) >= 300]
    ll = [r for r in ll if r.get("text") and len(r["text"]) >= 300]
    print(f"with text >=300: top50={len(t50)} longlist={len(ll)}")

    # ---- cross-tier de-confliction: drop longlist (title,surname) in Top50 ----
    t50_keys = {casefold_key(r["title"], r["author"]) for r in t50}
    removed = [r for r in ll if casefold_key(r["title"], r["author"]) in t50_keys]
    ll = [r for r in ll if casefold_key(r["title"], r["author"]) not in t50_keys]
    print(f"cross-tier dupes removed from longlist (also in Top50 recovered): {len(removed)}")
    for r in removed[:10]:
        print(f"    DROP {r['year']} {r['title'][:40]} / {r['author']}")

    df = pd.DataFrame(t50 + ll)
    df["judgement"] = (df["tier"] == "top50").astype(int)
    df["sid"] = [story_id(t, a, y) for t, a, y in
                 zip(df["title"], df["author"], df["year"])]
    df["split"] = df["sid"].map(split_of)
    # drop exact-duplicate texts (same story recovered twice)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"\nFINAL: {len(df)} rows  | y=1 (top50)={int(df.judgement.sum())}  "
          f"y=0 (longlist)={int((1-df.judgement).sum())}")
    print("split counts:", df.split.value_counts().to_dict())
    print("balance per split:")
    print(df.groupby("split")["judgement"].mean().round(3).to_dict())

    # ---------- AUDIT 1: presentation-leak (fetch_source predicts y?) ----------
    print("\n=== AUDIT 1: presentation-leak (fetch_source vs y) ===")
    print("fetch_source x tier:")
    print(pd.crosstab(df.fetch_source, df.tier))
    fs = pd.get_dummies(df["fetch_source"])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(fs, df.judgement)
    auc_fs = roc_auc_score(df.judgement, lr.predict_proba(fs)[:, 1])
    print(f"AUC(y ~ fetch_source dummies) = {auc_fs:.3f}  "
          f"(0.5 = no leak; >0.6 = source predicts label)")

    # length correlation
    df["len"] = df.text.str.len()
    print("\n=== AUDIT: length correlation ===")
    print("mean len  top50=%.0f  longlist=%.0f" % (
        df[df.judgement == 1].len.mean(), df[df.judgement == 0].len.mean()))
    print("corr(len, y) = %.3f" % np.corrcoef(df.len, df.judgement)[0, 1])
    auc_len = roc_auc_score(df.judgement, df.len)
    print("AUC(y ~ length alone) = %.3f" % max(auc_len, 1 - auc_len))

    # ---------- write built files ----------
    os.makedirs(os.path.join(BASE, "built"), exist_ok=True)
    cols = ["text", "raw_text", "judgement", "year", "magazine", "tier", "fetch_source"]
    for sp in ["train", "eval", "test"]:
        sub = df[df.split == sp][cols]
        out = os.path.join(BASE, "built", f"{sp}.csv.gz")
        sub.to_csv(out, index=False, compression="gzip")
        print(f"wrote {out}  ({len(sub)} rows, y1={int(sub.judgement.sum())})")

    # ---------- AUDIT 2: TF-IDF/LR floor + top features ----------
    print("\n=== AUDIT 2: TF-IDF/LR floor (own split) ===")
    tr = df[df.split == "train"]
    te = df[df.split.isin(["eval", "test"])]
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True, stop_words="english")
    Xtr = vec.fit_transform(tr.text)
    Xte = vec.transform(te.text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Xtr, tr.judgement)
    auc_tf = roc_auc_score(te.judgement, clf.predict_proba(Xte)[:, 1])
    print(f"TF-IDF/LR AUC = {auc_tf:.3f}")
    names = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    top_pos = names[np.argsort(coef)[-25:][::-1]]
    top_neg = names[np.argsort(coef)[:25]]
    print("TOP y=1 (top50) features:", list(top_pos))
    print("TOP y=0 (longlist) features:", list(top_neg))
    # leakage flag: magazine names among top features
    mags = set(w.lower() for m in df.magazine.dropna() for w in re.split(r"\W+", str(m)) if len(w) > 3)
    leaks = [f for f in list(top_pos) + list(top_neg) if any(t in mags for t in f.split())]
    print("POSSIBLE magazine-name leak features:", leaks)

    df.to_parquet(os.path.join(BASE, "built", "all.parquet"))
    print("\nwrote built/all.parquet (full table for V-probe)")


if __name__ == "__main__":
    main()
