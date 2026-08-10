"""Multivariate aspect-score → label logistic regression per (task, judge).

For each task and judge with sufficient data:
  1. Build feature matrix: for each aspect a, two features:
       feat_score_a       = score if applicable else 0 (mean-imputed in fold)
       feat_applicable_a  = 1 if applicable else 0
     This captures both "is this norm engaged?" and "if engaged, what's the score?"
  2. 5-fold stratified CV with L2 logistic regression (sklearn defaults plus class_weight=balanced).
  3. Report CV AUC, CV accuracy, majority baseline, and top |coef| aspects.

Skips (task, judge) with <100 dp overlap (regression not meaningful at smaller N).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASKS = [
    "peer_review", "math", "notice_and_comment", "press_releases",
    "humor", "news_homepages", "patents", "code_review", "creative_writing",
]
JUDGES = ["qwen_thinking_fp8", "claude"]
MIN_OVERLAP = 100
CV_FOLDS = 5
RANDOM_SEED = 17


def load_task(task: str, judge: str):
    labels_p = REPO / f"runs/validity_full/v2/{task}/datapoints.json"
    if not labels_p.exists():
        return None
    dps = json.loads(labels_p.read_text())
    labels = pd.Series(
        {d["datapoint_id"]: int(d["judgement"])
         for d in dps if d.get("judgement") is not None}
    )
    f = REPO / f"outputs/v2_db/cells_v1/task={task}/judge={judge}/data.parquet"
    if not f.exists():
        return None
    cells = pd.read_parquet(f)
    cells["score_num"] = cells["score"].where(cells["applicable"], np.nan).astype(float)
    score_mat = (cells.groupby(["datapoint_id", "aspect_id"])["score_num"]
                       .mean().unstack("aspect_id"))
    appl_mat = (cells.groupby(["datapoint_id", "aspect_id"])["applicable"]
                      .max().unstack("aspect_id").fillna(False).astype(int))
    common = score_mat.index.intersection(labels.index)
    if len(common) < MIN_OVERLAP:
        return None
    return (score_mat.loc[common], appl_mat.loc[common], labels.loc[common])


def build_features(score_mat, appl_mat):
    aspects = sorted(set(score_mat.columns) & set(appl_mat.columns))
    score = score_mat[aspects].copy()
    appl = appl_mat[aspects].copy()
    # Imputed score: when not applicable, score is NaN. Replace with 0; the
    # applicability indicator distinguishes "not engaged" from "engaged-and-zero".
    score_imp = score.fillna(0.0).values
    appl_arr = appl.values.astype(float)
    X = np.concatenate([score_imp, appl_arr], axis=1)
    feat_names = ([f"score_{a}" for a in aspects]
                  + [f"appl_{a}" for a in aspects])
    return X, feat_names, aspects


def run_cv(X, y, n_folds=CV_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs, accs = [], []
    coefs = []
    for tr, te in skf.split(X, y):
        # standardize on train
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        # train; L2, class-balanced
        clf = LogisticRegression(
            penalty="l2", C=0.5, class_weight="balanced",
            max_iter=2000, solver="lbfgs",
        )
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        yhat = (p > 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], yhat))
        coefs.append(clf.coef_[0])
    return np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs), np.mean(coefs, axis=0)


def main():
    rows = []
    skipped = []
    for task in TASKS:
        for judge in JUDGES:
            data = load_task(task, judge)
            if data is None:
                skipped.append((task, judge, "no data or overlap <100"))
                continue
            score_mat, appl_mat, labels = data
            X, feat_names, aspects = build_features(score_mat, appl_mat)
            y = labels.values
            n = len(y)
            n_pos = int(y.sum())
            n_neg = n - n_pos
            maj = max(n_pos, n_neg) / n
            try:
                auc_m, auc_s, acc_m, acc_s, coefs = run_cv(X, y)
            except Exception as e:
                skipped.append((task, judge, f"cv failed: {e}"))
                continue
            # Top 5 features by |coef|
            top_idx = np.argsort(-np.abs(coefs))[:5]
            top = [(feat_names[i], coefs[i]) for i in top_idx]
            rows.append({
                "task": task, "judge": judge,
                "n": n, "n_pos": n_pos, "n_neg": n_neg,
                "majority": maj, "auc": auc_m, "auc_std": auc_s,
                "acc": acc_m, "acc_std": acc_s,
                "top_feats": top,
            })

    # Print table
    print(f"{'task':<20} {'judge':<22} {'n':>5} {'maj':>6} "
          f"{'CV_AUC':>8} {'(std)':>7} {'CV_acc':>7} {'(std)':>7}")
    print("-" * 90)
    for r in rows:
        print(f"{r['task']:<20} {r['judge']:<22} {r['n']:>5} "
              f"{r['majority']:>6.1%} {r['auc']:>8.3f} {r['auc_std']:>7.3f} "
              f"{r['acc']:>7.1%} {r['acc_std']:>7.1%}")
    if skipped:
        print()
        print("Skipped:")
        for s in skipped:
            print(f"  {s[0]:<20} {s[1]:<22} {s[2]}")

    print()
    print("=" * 90)
    print("TOP 5 FEATURES BY |COEF| PER (task, judge)")
    print("=" * 90)
    for r in rows:
        print(f"\n{r['task']} / {r['judge']}  (CV AUC = {r['auc']:.3f}):")
        for name, c in r["top_feats"]:
            print(f"   {name:<14} coef={c:+.3f}")

    # Persist
    out_p = REPO / "outputs/v2_analysis/norm_logreg_all_tasks.csv"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in rows:
        flat.append({k: v for k, v in r.items() if k != "top_feats"})
    pd.DataFrame(flat).to_csv(out_p, index=False)
    print(f"\nwrote summary -> {out_p}")


if __name__ == "__main__":
    main()
