#!/usr/bin/env python3
"""v3.2: balance classes within question-TOPIC strata (Math.SE).

Motivation (v3_leakage_audit/REPORT.md, 2026-06-10): a question-only TF-IDF+LR
reaches 0.65 test AUC despite question-disjoint splits — question topic/register
("is it true that..." vs "help me solve...") predicts which answers win. This
script removes topic-level label signal the same way news_homepages removed it
(LDA-50 topic balancing, `project_homepage_newsworthiness`):

  1. Cluster questions: TF-IDF (word 1-2) -> TruncatedSVD(128) -> MiniBatchKMeans(K).
  2. Within every (cluster x 3-year-bin) cell, downsample the majority class to
     exact 50/50. Year stays in the cell key so topic balancing cannot undo the
     v3.1 year matching.
  3. Sample down to --target-rows keeping priority-tag rows first (same rule as
     the build), classes sampled equally per cell to preserve cell balance.
  4. Regenerate 80/10/10 train/eval/test split by md5(question_id) (build rule).
  5. VALIDATION baked in: question-only TF-IDF+LR (fit on train, AUC on test)
     computed on the input pool and on the output — the manifest reports both.
     Success = output question-only AUC near 0.5 (target <= 0.55).

Run on sk3:
  python3.11 topic_balance_v3_2.py --pool math_se_v3_1_pool.csv.gz \
      --out math_se_v3_2_topic_balanced.csv.gz --k 100 --target-rows 100000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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


def question_only_auc(df: pd.DataFrame, seed: int) -> float:
    """Question-only TF-IDF+LR floor: fit train, AUC on test."""
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000)
    Xtr = vec.fit_transform(tr["question_text"])
    Xte = vec.transform(te["question_text"])
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    lr.fit(Xtr, tr["judgement"].values)
    return float(roc_auc_score(te["judgement"].values,
                               lr.predict_proba(Xte)[:, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--svd-dims", type=int, default=128)
    ap.add_argument("--target-rows", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"[{datetime.now():%H:%M:%S}] loading pool", flush=True)
    df = pd.read_csv(args.pool)
    df["question_text"] = df["text"].map(question_part)
    df["split"] = df["question_id"].map(split_of)
    manifest = {"args": vars(args), "input_rows": len(df),
                "input_pos_rate": float(df["judgement"].mean())}

    print(f"[{datetime.now():%H:%M:%S}] input question-only AUC...", flush=True)
    manifest["question_only_auc_input"] = question_only_auc(df, args.seed)
    print(f"  input floor = {manifest['question_only_auc_input']:.4f}", flush=True)

    # --- cluster questions (one vector per unique question) ---
    print(f"[{datetime.now():%H:%M:%S}] clustering questions K={args.k}", flush=True)
    qdf = df.drop_duplicates("question_id")[["question_id", "question_text"]]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=100_000)
    X = vec.fit_transform(qdf["question_text"])
    Xs = TruncatedSVD(args.svd_dims, random_state=args.seed).fit_transform(X)
    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed,
                         batch_size=4096, n_init=10)
    qdf = qdf.assign(cluster=km.fit_predict(Xs))
    df = df.merge(qdf[["question_id", "cluster"]], on="question_id")

    # --- balance within (cluster x 3yr bin) ---
    df["cell"] = list(zip(df["cluster"], df["answer_year"] // 3))
    kept_idx = []
    cell_stats = Counter()
    for cell, g in df.groupby("cell"):
        pos = g[g["judgement"] == 1]
        neg = g[g["judgement"] == 0]
        n = min(len(pos), len(neg))
        if n == 0:
            cell_stats["dropped_cells"] += 1
            continue
        kept_idx += list(rng.choice(pos.index, n, replace=False))
        kept_idx += list(rng.choice(neg.index, n, replace=False))
        cell_stats["kept_cells"] += 1
    bal = df.loc[kept_idx]
    manifest["cells"] = dict(cell_stats)
    manifest["balanced_rows"] = len(bal)
    print(f"[{datetime.now():%H:%M:%S}] balanced pool: {len(bal)} rows "
          f"({cell_stats['kept_cells']} cells)", flush=True)

    # --- sample to target, priority tags first, per-cell-per-class equal ---
    if len(bal) > args.target_rows:
        is_prio = bal["question_tags"].fillna("").map(
            lambda t: bool(PRIORITY_TAGS & set(str(t).split("|"))))
        prio, rest = bal[is_prio], bal[~is_prio]
        # keep all priority pairs (balanced within cell already; re-balance
        # priority subset per cell to stay exactly 50/50)
        keep = []
        for sub in (prio,):
            for cell, g in sub.groupby("cell"):
                p, q = g[g["judgement"] == 1], g[g["judgement"] == 0]
                n = min(len(p), len(q))
                keep += list(p.index[:n]) + list(q.index[:n])
        n_remaining = args.target_rows - len(keep)
        if n_remaining > 0 and len(rest):
            frac = n_remaining / len(rest)
            for cell, g in rest.groupby("cell"):
                p, q = g[g["judgement"] == 1], g[g["judgement"] == 0]
                n = min(len(p), len(q))
                take = int(round(n * frac))
                keep += list(rng.choice(p.index, min(take, n), replace=False))
                keep += list(rng.choice(q.index, min(take, n), replace=False))
        bal = bal.loc[keep]
    manifest["final_rows"] = len(bal)
    manifest["final_pos_rate"] = float(bal["judgement"].mean())

    print(f"[{datetime.now():%H:%M:%S}] output question-only AUC...", flush=True)
    manifest["question_only_auc_output"] = question_only_auc(bal, args.seed)
    print(f"  output floor = {manifest['question_only_auc_output']:.4f}", flush=True)

    # year balance after
    yb = bal.groupby("answer_year")["judgement"].agg(["count", "mean"])
    manifest["year_balance_output"] = {int(y): [int(c), round(float(m), 4)]
                                       for y, (c, m) in yb.iterrows()}

    out_cols = [c for c in bal.columns if c not in
                ("question_text", "cluster", "cell")]
    bal = bal.sample(frac=1.0, random_state=args.seed)  # shuffle
    bal[out_cols + ["cluster"]].to_csv(args.out, index=False, compression="gzip")
    mpath = args.out.replace(".csv.gz", ".manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2, default=str)
    print(f"[{datetime.now():%H:%M:%S}] wrote {args.out} ({len(bal)} rows) + "
          f"{mpath}", flush=True)
    print(json.dumps({k: v for k, v in manifest.items()
                      if "year_balance" not in k}, indent=2, default=str),
          flush=True)


if __name__ == "__main__":
    main()
