"""Analyze Math.SE A-metric judge verdicts and run the V-vs-A head-to-head on
identical rows. Computes:
  - per-metric A AUC vs floor (0.461)
  - combined A-layer LR (question-grouped CV); priority/exposition/depth subsets
  - per-answer_type combined AUC
  - deterministic-V combined AUC (lint features) on the SAME rows
  - lexicon V-features (v11 hedging, v13 clearly, v12 direct-open, v14
    connectives) computed from text, for the V/A convergence pairs
    (a04~v13, a10~v11, a11~v12)
Floor (v3.3 question-only): 0.461. V-lint reference: 0.567 eval / 0.569 test.
"""
import argparse
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FLOOR = 0.461
METRIC_IDS = [f"a{n:02d}" for n in range(1, 15)]
METRIC_NAMES = {
    "a01": "motivation", "a02": "audience-calibration", "a03": "generality",
    "a04": "no-gaps", "a05": "words-symbols", "a06": "idea-visible",
    "a07": "elegance", "a08": "rigor", "a09": "scaffolding",
    "a10": "epistemic-honesty", "a11": "directness", "a12": "reusable",
    "a13": "notation", "a14": "profundity",
}
HEDGE = re.compile(r"\b(maybe|i think|not sure|i guess|probably|perhaps|"
                   r"i believe|might be|i'm not sure|seems? to)\b", re.I)
CLEARLY = re.compile(r"\b(clearly|obviously|trivially|evidently|of course|"
                     r"easy to see|easily)\b", re.I)
CONNECT = re.compile(r"\b(hence|thus|therefore|since|it follows|so that|"
                     r"because|consequently)\b", re.I)
DIRECT_OPEN = re.compile(r"^\s*(yes|no|note that|hint|sure|indeed|"
                         r"the answer is)\b", re.I)


def lexicon_v(text):
    body = text.split("Answer:", 1)[-1] if "Answer:" in text else text
    w = max(len(body.split()), 1)
    return {"v11_hedging": len(HEDGE.findall(body)) / w * 100,
            "v13_clearly": len(CLEARLY.findall(body)) / w * 100,
            "v14_connectives": len(CONNECT.findall(body)) / w * 100,
            "v12_direct_open": 1.0 if DIRECT_OPEN.search(body) else 0.0}


def cv_auc(df, cols, label="judgement", group="question_id"):
    sub = df.dropna(subset=cols)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    p = cross_val_predict(clf, sub[cols].values, sub[label].values,
                          cv=GroupKFold(5), groups=sub[group].values,
                          method="predict_proba")[:, 1]
    return roc_auc_score(sub[label].values, p), len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--pool", required=True, help="v3.3 csv.gz for text/join")
    ap.add_argument("--lint", default=None, help="mathse_lint_features.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.verdicts)]
    df = pd.DataFrame([r for r in rows if "judge_failed" not in r])
    fail = sum(1 for r in rows if "judge_failed" in r)
    print(f"verdicts: {len(df)} ok / {fail} failed "
          f"({fail / max(len(rows), 1):.1%}); "
          f"salvaged {df.get('salvaged', pd.Series(dtype=bool)).fillna(False).sum()}")
    df = df.drop_duplicates("answer_id")

    pool = pd.read_csv(args.pool)[["answer_id", "text"]].drop_duplicates("answer_id")
    df = df.merge(pool, on="answer_id", how="left")
    lex = df["text"].fillna("").map(lexicon_v).apply(pd.Series)
    df = pd.concat([df, lex], axis=1)

    for sp in sorted(df["split"].dropna().unique()):
        d = df[df["split"] == sp]
        print(f"\n########## SPLIT: {sp}  (n={len(d)}, "
              f"base={d.judgement.mean():.3f}, q={d.question_id.nunique()}) ##########")
        y = d.judgement.values
        print(f"answer_type: {d.answer_type.value_counts().to_dict()}")

        print("--- per-metric A AUC (direction-free; floor 0.461) ---")
        for mid in METRIC_IDS:
            auc = roc_auc_score(y, d[mid].values)
            mark = " *" if max(auc, 1 - auc) > FLOOR + 0.05 else ""
            print(f"  {mid} {METRIC_NAMES[mid]:20s} {max(auc,1-auc):.3f} "
                  f"({'+' if auc>=.5 else '-'}) mean={d[mid].mean():.2f}{mark}")

        a_all, n = cv_auc(d, METRIC_IDS)
        print(f"--- combined A-layer LR (q-grouped CV) ---")
        print(f"  ALL 14 A-metrics:   AUC = {a_all:.3f}  (margin "
              f"+{a_all - FLOOR:.3f} over floor; V-lint ref 0.567)")
        for name, cols in [("priority a10/a11/a04/a08", ["a10", "a11", "a04", "a08"]),
                           ("exposition a01/a02/a05/a09", ["a01", "a02", "a05", "a09"]),
                           ("depth a03/a06/a07/a14", ["a03", "a06", "a07", "a14"])]:
            au, _ = cv_auc(d, cols)
            print(f"  {name:28s} AUC = {au:.3f}")

        lexcols = ["v11_hedging", "v13_clearly", "v14_connectives", "v12_direct_open"]
        lv, _ = cv_auc(d, lexcols)
        print(f"  lexicon-V (v11/v12/v13/v14) AUC = {lv:.3f}")
        av, _ = cv_auc(d, METRIC_IDS + lexcols)
        print(f"  A + lexicon-V combined      AUC = {av:.3f}")

        # deterministic-V head-to-head on identical rows
        if args.lint and Path(args.lint).exists():
            lint = pd.read_csv(args.lint)
            vcols = [c for c in lint.columns if c not in
                     ("answer_id", "question_id", "judgement", "split",
                      "question_type")]
            j = d.merge(lint[["answer_id"] + vcols], on="answer_id", how="inner")
            j = j.replace([np.inf, -np.inf], np.nan)
            j[vcols] = j[vcols].apply(pd.to_numeric, errors="coerce").fillna(0)
            vauc, vn = cv_auc(j, vcols)
            aauc2, _ = cv_auc(j, METRIC_IDS)
            bothauc, _ = cv_auc(j, METRIC_IDS + vcols)
            print(f"--- V vs A head-to-head (identical {vn} rows) ---")
            print(f"  deterministic-V (lint) AUC = {vauc:.3f}")
            print(f"  A-layer            AUC = {aauc2:.3f}")
            print(f"  V + A              AUC = {bothauc:.3f}  "
                  f"(A lift over V: +{bothauc - vauc:.3f})")

        print("--- V/A convergence pairs (spearman) ---")
        for aid, vcol, desc in [("a04", "v13_clearly", "clearly-count"),
                                ("a10", "v11_hedging", "hedging"),
                                ("a11", "v12_direct_open", "direct-open"),
                                ("a08", "v14_connectives", "connectives")]:
            r = d[[aid, vcol]].corr(method="spearman").iloc[0, 1]
            print(f"  {aid} {METRIC_NAMES[aid]:18s} ~ {vcol:16s} ({desc}): "
                  f"spearman {r:+.3f}")

    if args.out:
        df.to_parquet(args.out)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
