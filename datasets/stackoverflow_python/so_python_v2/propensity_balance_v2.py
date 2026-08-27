#!/usr/bin/env python3
"""SO[python] v2 propensity balancing with fail-loud floor gate
and PER-STRATUM answer-text margin reporting.

Mirrors CR.SE v2 / Math.SE v3.3; differences:
  - Floor gate: question-only AUC <= --max-floor (default 0.55)
  - Reports answer-text margin OVERALL and per-stratum (stdlib/lib_data/framework)
  - Cells = (OOF-propensity decile x year3); same as Math.SE/CR.SE v2.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from splits import split_of  # noqa: E402


ANSWER_SPLIT = re.compile(r"\n\nAnswer:", re.S)


def question_part(text: str) -> str:
    parts = ANSWER_SPLIT.split(text, maxsplit=1)
    return parts[0][len("Question: "):] if parts else ""


def answer_part(text: str) -> str:
    parts = ANSWER_SPLIT.split(text, maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def fit_lr(Xtr, ytr, seed):
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    lr.fit(Xtr, ytr)
    return lr


def grouped_text_auc(df, text_col, seed, label_col="judgement",
                     groups_col="question_id", n_splits=5):
    vec_proto = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                                        max_features=200_000)
    y = df[label_col].values
    groups = df[groups_col].values
    aucs = []
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(df, y, groups=groups)):
        v = vec_proto()
        Xtr = v.fit_transform(df.iloc[tr][text_col].fillna(""))
        Xte = v.transform(df.iloc[te][text_col].fillna(""))
        lr = fit_lr(Xtr, y[tr], seed + fold)
        try:
            auc = roc_auc_score(y[te], lr.predict_proba(Xte)[:, 1])
        except ValueError:
            auc = float("nan")
        aucs.append(float(auc))
    return float(np.nanmean(aucs)), aucs


def position_floor_auc(df, seed, label_col="judgement",
                       groups_col="question_id", n_splits=5):
    y = df[label_col].values
    groups = df[groups_col].values
    pos = df["answer_position"].clip(upper=3).values.reshape(-1, 1).astype(float)
    aucs = []
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(df, y, groups=groups)):
        lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed + fold)
        lr.fit(pos[tr], y[tr])
        try:
            auc = roc_auc_score(y[te], lr.predict_proba(pos[te])[:, 1])
        except ValueError:
            auc = float("nan")
        aucs.append(float(auc))
    return float(np.nanmean(aucs)), aucs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--target-rows", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-floor", type=float, default=0.55)
    ap.add_argument("--ignore-floor-gate", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"[{datetime.now():%H:%M:%S}] loading pool {args.pool}", flush=True)
    df = pd.read_csv(args.pool)
    df["question_text"] = df["text"].map(question_part)
    df["answer_text_only"] = df["text"].map(answer_part)
    df["split"] = df["question_id"].astype(str).map(split_of)

    manifest: dict = {
        "args": vars(args),
        "input_rows": int(len(df)),
        "input_pos_rate": float(df.judgement.mean()),
    }

    print(f"[{datetime.now():%H:%M:%S}] INPUT diagnostics", flush=True)
    q_in, q_in_folds = grouped_text_auc(df, "question_text", args.seed)
    print(f"  INPUT question-only floor = {q_in:.4f}  folds={q_in_folds}", flush=True)
    a_in, a_in_folds = grouped_text_auc(df, "answer_text_only", args.seed)
    print(f"  INPUT answer-text margin  = {a_in:.4f}", flush=True)
    p_in, p_in_folds = position_floor_auc(df, args.seed)
    print(f"  INPUT position-only floor = {p_in:.4f}  folds={p_in_folds}", flush=True)
    manifest["input_floors"] = {
        "question_only_auc": q_in, "question_only_folds": q_in_folds,
        "answer_text_auc": a_in, "answer_text_folds": a_in_folds,
        "position_only_auc": p_in, "position_only_folds": p_in_folds,
    }

    print(f"[{datetime.now():%H:%M:%S}] OOF propensity ({args.folds} folds)", flush=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000)
    X = vec.fit_transform(df["question_text"])
    y = df["judgement"].values
    oof = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=args.folds)
    for i, (tr_i, te_i) in enumerate(gkf.split(X, y, groups=df["question_id"])):
        lr = fit_lr(X[tr_i], y[tr_i], args.seed + i)
        oof[te_i] = lr.predict_proba(X[te_i])[:, 1]
        print(f"  fold {i+1}/{args.folds} done", flush=True)
    assert not np.isnan(oof).any()
    df["propensity"] = oof
    manifest["oof_propensity_auc"] = float(roc_auc_score(y, oof))

    edges = np.quantile(oof, np.linspace(0, 1, args.deciles + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    df["decile"] = np.searchsorted(edges[1:-1], df["propensity"])
    df["cell"] = list(zip(df["decile"], df["answer_year"] // 3))

    kept: list[int] = []
    stats: Counter = Counter()
    for cell, g in df.groupby("cell"):
        pos_g = g[g["judgement"] == 1]
        neg_g = g[g["judgement"] == 0]
        n = min(len(pos_g), len(neg_g))
        if n == 0:
            stats["dropped_cells"] += 1
            continue
        kept += list(rng.choice(pos_g.index, n, replace=False))
        kept += list(rng.choice(neg_g.index, n, replace=False))
        stats["kept_cells"] += 1
    bal = df.loc[kept].copy()
    manifest["cells"] = dict(stats)
    manifest["balanced_rows_pre_target"] = int(len(bal))

    if len(bal) > args.target_rows:
        keep: list[int] = []
        frac = args.target_rows / max(1, len(bal))
        for cell, g in bal.groupby("cell"):
            p = g[g["judgement"] == 1]
            q = g[g["judgement"] == 0]
            n = min(len(p), len(q))
            take = min(int(round(n * frac)), n)
            if take == 0:
                continue
            keep += list(rng.choice(p.index, take, replace=False))
            keep += list(rng.choice(q.index, take, replace=False))
        bal = bal.loc[keep].copy()

    manifest["final_rows"] = int(len(bal))
    manifest["final_pos_rate"] = float(bal["judgement"].mean())

    print(f"[{datetime.now():%H:%M:%S}] OUTPUT diagnostics", flush=True)
    q_out, q_out_folds = grouped_text_auc(bal, "question_text", args.seed)
    print(f"  OUTPUT question-only floor = {q_out:.4f}  folds={q_out_folds}", flush=True)
    a_out, a_out_folds = grouped_text_auc(bal, "answer_text_only", args.seed)
    print(f"  OUTPUT answer-text margin  = {a_out:.4f}", flush=True)
    p_out, p_out_folds = position_floor_auc(bal, args.seed)
    print(f"  OUTPUT position-only floor = {p_out:.4f}  folds={p_out_folds}", flush=True)
    manifest["output_floors"] = {
        "question_only_auc": q_out, "question_only_folds": q_out_folds,
        "answer_text_auc": a_out, "answer_text_folds": a_out_folds,
        "position_only_auc": p_out, "position_only_folds": p_out_folds,
    }

    # Per-stratum answer-text margin (the #2 design)
    per_stratum = {}
    for s in sorted(bal.stratum.dropna().unique().tolist()):
        sub = bal[bal.stratum == s]
        # need both classes to fit
        if sub.judgement.nunique() < 2:
            per_stratum[s] = {
                "rows": int(len(sub)),
                "pos_rate": float(sub.judgement.mean()) if len(sub) else float("nan"),
                "answer_text_auc": None,
                "note": "single-class subset; cannot compute AUC.",
            }
            continue
        # if stratum has too few unique questions for 5-fold, drop to 3-fold
        n_groups = sub.question_id.nunique()
        n_folds = 5 if n_groups >= 5 else max(2, min(3, n_groups))
        try:
            a_s, a_s_folds = grouped_text_auc(sub, "answer_text_only", args.seed,
                                              n_splits=n_folds)
        except Exception as e:
            a_s, a_s_folds = float("nan"), [str(e)]
        per_stratum[s] = {
            "rows": int(len(sub)),
            "pos_rate": float(sub.judgement.mean()),
            "answer_text_auc": a_s,
            "answer_text_folds": a_s_folds,
            "n_folds": n_folds,
        }
        print(f"  STRATUM {s}: rows={len(sub)}  pos_rate={sub.judgement.mean():.3f}  "
              f"answer_text_auc={a_s:.4f} ({n_folds}f)", flush=True)
    manifest["per_stratum_output"] = per_stratum

    yb = bal.groupby("answer_year")["judgement"].agg(["count", "mean"])
    manifest["year_balance_output"] = {
        int(yy): [int(c), round(float(m), 4)] for yy, (c, m) in yb.iterrows()
    }

    drop = ("question_text", "answer_text_only", "cell")
    bal = bal.sample(frac=1.0, random_state=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bal[[c for c in bal.columns if c not in drop]].to_csv(
        args.out, index=False, compression="gzip")
    mpath = args.out.replace(".csv.gz", ".manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[{datetime.now():%H:%M:%S}] wrote {args.out} ({len(bal)} rows)", flush=True)

    if q_out > args.max_floor:
        msg = (f"\n*** FLOOR GATE FAILED ***\n"
               f"OUTPUT question-only floor = {q_out:.4f} "
               f"> max-floor {args.max_floor:.4f}.\n")
        if args.ignore_floor_gate:
            print("WARNING: " + msg, flush=True)
        else:
            print(msg, flush=True)
            sys.exit(2)
    else:
        print(f"\nFloor gate passed: {q_out:.4f} <= {args.max_floor:.4f}.", flush=True)


if __name__ == "__main__":
    main()
