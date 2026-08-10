#!/usr/bin/env python3
"""v3.3: balance classes within question-PROPENSITY deciles (Math.SE).

Escalation after v3.2 (topic clusters K=100) failed to move the question-only
floor (0.665 -> 0.640; target <= 0.55). The question-side signal is finer than
topic clusters — register/phrasing ("is it true that..." vs "help me solve").
So balance directly on what the model can see: a CROSS-FITTED question-only
TF-IDF+LR propensity score.

  1. 5-fold group split by question_id; out-of-fold propensity score
     p(y=1 | question text) for every row (cross-fitting prevents the
     propensity model from memorizing the rows it later balances).
  2. Cells = (OOF-score decile x 3-year bin). Within each cell, downsample the
     majority class to exact 50/50. Year stays in the key so v3.1's year
     matching survives.
  3. Priority-tag retention -> --target-rows, same rules as the build.
  4. Validation baked in: fresh question-only TF-IDF+LR (train->test) on input
     and output; manifest reports both. In-sample-family guarantee: a fresh
     model from the same family should land near 0.5 on the output; whatever
     residual remains is fold/regularization noise, reported honestly.

Run on sk3:
  python3.11 propensity_balance_v3_3.py --pool math_se_v3_1_pool.csv.gz \
      --out math_se_v3_3_propensity_balanced.csv.gz --target-rows 100000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

PRIORITY_TAGS = {
    "proof-writing", "proof-verification", "proof-explanation",
    "alternative-proof", "solution-verification", "intuition", "soft-question",
}
ANSWER_SPLIT = re.compile(r"\n\nAnswer:", re.S)


def question_part(text: str) -> str:
    parts = ANSWER_SPLIT.split(text, maxsplit=1)
    return parts[0][len("Question: "):] if parts else ""


def split_of(qid) -> str:
    h = int(hashlib.md5(str(qid).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("eval" if h < 90 else "test")


def fit_lr(Xtr, ytr, seed):
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    lr.fit(Xtr, ytr)
    return lr


def question_only_auc(df, seed):
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000)
    Xtr = vec.fit_transform(tr["question_text"])
    Xte = vec.transform(te["question_text"])
    lr = fit_lr(Xtr, tr["judgement"].values, seed)
    return float(roc_auc_score(te["judgement"].values,
                               lr.predict_proba(Xte)[:, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--target-rows", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"[{datetime.now():%H:%M:%S}] loading pool", flush=True)
    df = pd.read_csv(args.pool)
    df["question_text"] = df["text"].map(question_part)
    df["split"] = df["question_id"].map(split_of)
    manifest = {"args": vars(args), "input_rows": len(df)}

    print(f"[{datetime.now():%H:%M:%S}] input floor...", flush=True)
    manifest["question_only_auc_input"] = question_only_auc(df, args.seed)
    print(f"  input floor = {manifest['question_only_auc_input']:.4f}", flush=True)

    # --- cross-fitted propensity ---
    print(f"[{datetime.now():%H:%M:%S}] OOF propensity ({args.folds} folds)",
          flush=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000)
    X = vec.fit_transform(df["question_text"])
    y = df["judgement"].values
    oof = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=args.folds)
    for i, (tr_i, te_i) in enumerate(
            gkf.split(X, y, groups=df["question_id"])):
        lr = fit_lr(X[tr_i], y[tr_i], args.seed + i)
        oof[te_i] = lr.predict_proba(X[te_i])[:, 1]
        print(f"  fold {i+1}/{args.folds} done", flush=True)
    assert not np.isnan(oof).any()
    df["propensity"] = oof
    manifest["oof_propensity_auc"] = float(roc_auc_score(y, oof))
    print(f"  OOF propensity AUC = {manifest['oof_propensity_auc']:.4f}",
          flush=True)

    # --- balance within (decile x year3) ---
    edges = np.quantile(oof, np.linspace(0, 1, args.deciles + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    df["decile"] = np.searchsorted(edges[1:-1], df["propensity"])
    df["cell"] = list(zip(df["decile"], df["answer_year"] // 3))
    kept = []
    stats = Counter()
    for cell, g in df.groupby("cell"):
        pos, neg = g[g["judgement"] == 1], g[g["judgement"] == 0]
        n = min(len(pos), len(neg))
        if n == 0:
            stats["dropped_cells"] += 1
            continue
        kept += list(rng.choice(pos.index, n, replace=False))
        kept += list(rng.choice(neg.index, n, replace=False))
        stats["kept_cells"] += 1
    bal = df.loc[kept]
    manifest["cells"] = dict(stats)
    manifest["balanced_rows"] = len(bal)
    print(f"[{datetime.now():%H:%M:%S}] balanced: {len(bal)} rows", flush=True)

    # --- sample to target, priority first, per-cell-per-class equal ---
    if len(bal) > args.target_rows:
        is_prio = bal["question_tags"].fillna("").map(
            lambda t: bool(PRIORITY_TAGS & set(str(t).split("|"))))
        prio, rest = bal[is_prio], bal[~is_prio]
        keep = []
        for cell, g in prio.groupby("cell"):
            p, q = g[g["judgement"] == 1], g[g["judgement"] == 0]
            n = min(len(p), len(q))
            keep += list(p.index[:n]) + list(q.index[:n])
        n_rem = args.target_rows - len(keep)
        if n_rem > 0 and len(rest):
            frac = n_rem / len(rest)
            for cell, g in rest.groupby("cell"):
                p, q = g[g["judgement"] == 1], g[g["judgement"] == 0]
                n = min(len(p), len(q))
                take = min(int(round(n * frac)), n)
                keep += list(rng.choice(p.index, take, replace=False))
                keep += list(rng.choice(q.index, take, replace=False))
        bal = bal.loc[keep]
    manifest["final_rows"] = len(bal)
    manifest["final_pos_rate"] = float(bal["judgement"].mean())

    print(f"[{datetime.now():%H:%M:%S}] output floor...", flush=True)
    manifest["question_only_auc_output"] = question_only_auc(bal, args.seed)
    print(f"  output floor = {manifest['question_only_auc_output']:.4f}",
          flush=True)

    yb = bal.groupby("answer_year")["judgement"].agg(["count", "mean"])
    manifest["year_balance_output"] = {int(yy): [int(c), round(float(m), 4)]
                                       for yy, (c, m) in yb.iterrows()}

    drop = ("question_text", "cell")
    bal = bal.sample(frac=1.0, random_state=args.seed)
    bal[[c for c in bal.columns if c not in drop]].to_csv(
        args.out, index=False, compression="gzip")
    mpath = args.out.replace(".csv.gz", ".manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2, default=str)
    print(f"[{datetime.now():%H:%M:%S}] wrote {args.out} ({len(bal)} rows)",
          flush=True)
    print(json.dumps({k: v for k, v in manifest.items()
                      if "year_balance" not in k}, indent=2, default=str),
          flush=True)


if __name__ == "__main__":
    main()
