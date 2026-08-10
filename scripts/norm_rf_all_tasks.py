"""Multivariate aspect-score → label random-forest per task, combining
Qwen + Claude judges with union coverage.

For each (dp, aspect):
  - score_cell  = mean of available judge scores (Qwen and/or Claude); NaN if none
  - appl_cell   = True if any judge marked applicable

Features per datapoint:
  feat_score_a       = score if applicable else 0
  feat_applicable_a  = 1 if applicable else 0
  feat_n_judges_a    = number of judges that scored this cell (0/1/2)

Fixed 80/20 stratified split with random_state=42 — same held-out across
all future experiments (dense ceiling, prompt-upper-bound).

Random forest + bonus L2 logreg for comparison.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASKS = [
    "peer_review", "math", "notice_and_comment", "press_releases",
    "humor", "news_homepages", "patents", "code_review", "creative_writing",
]
JUDGES = ["qwen_thinking_fp8", "claude"]
TEST_FRAC = 0.20
RANDOM_SEED = 42
MIN_N = 200


def load_combined(task: str):
    labels_p = REPO / f"runs/validity_full/v2/{task}/datapoints.json"
    if not labels_p.exists():
        return None
    dps = json.loads(labels_p.read_text())
    labels = pd.Series(
        {d["datapoint_id"]: int(d["judgement"])
         for d in dps if d.get("judgement") is not None}
    )
    score_dfs = []
    appl_dfs = []
    count_dfs = []
    for j in JUDGES:
        f = REPO / f"outputs/v2_db/cells_v1/task={task}/judge={j}/data.parquet"
        if not f.exists():
            continue
        cells = pd.read_parquet(f)
        cells["score_num"] = cells["score"].where(cells["applicable"], np.nan).astype(float)
        sc = (cells.groupby(["datapoint_id", "aspect_id"])["score_num"]
                    .mean().unstack("aspect_id"))
        ap = (cells.groupby(["datapoint_id", "aspect_id"])["applicable"]
                    .max().unstack("aspect_id").fillna(False).astype(int))
        cnt = (cells.groupby(["datapoint_id", "aspect_id"])
                    ["score_num"].apply(lambda s: s.notna().sum())
                    .unstack("aspect_id").fillna(0).astype(int))
        score_dfs.append(sc)
        appl_dfs.append(ap)
        count_dfs.append(cnt)

    if not score_dfs:
        return None

    # Union dp_ids and aspect_ids across judges
    all_dps = sorted(set().union(*(df.index for df in score_dfs)))
    all_asp = sorted(set().union(*(df.columns for df in score_dfs)))

    def reindex(dfs, fill=np.nan):
        return [df.reindex(index=all_dps, columns=all_asp, fill_value=fill)
                for df in dfs]

    sc = reindex(score_dfs)
    ap = reindex(appl_dfs, fill=0)
    cnt = reindex(count_dfs, fill=0)

    # Combine
    score = pd.concat([s.stack(dropna=False) for s in sc], axis=1).mean(
        axis=1, skipna=True).unstack().reindex(index=all_dps, columns=all_asp)
    appl = sum(ap)  # any judge applicable → ≥1
    appl = (appl > 0).astype(int)
    count = sum(cnt)  # number of judges that contributed a non-null score

    common = score.index.intersection(labels.index)
    if len(common) < MIN_N:
        return None

    score = score.loc[common]
    appl = appl.loc[common]
    count = count.loc[common]
    y = labels.loc[common].values

    return score, appl, count, y, all_asp


def build_features(score, appl, count, aspects):
    score_imp = score[aspects].fillna(0.0).values
    appl_arr = appl[aspects].values.astype(float)
    count_arr = count[aspects].values.astype(float)
    X = np.concatenate([score_imp, appl_arr, count_arr], axis=1)
    names = (
        [f"score_{a}" for a in aspects]
        + [f"appl_{a}" for a in aspects]
        + [f"njudges_{a}" for a in aspects]
    )
    return X, names


def run_eval(X, y, feat_names):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_FRAC, stratify=y, random_state=RANDOM_SEED)
    out = {"n_train": len(ytr), "n_test": len(yte)}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
    )
    rf.fit(Xtr, ytr)
    p_rf = rf.predict_proba(Xte)[:, 1]
    out["rf_auc"] = roc_auc_score(yte, p_rf)
    out["rf_acc"] = accuracy_score(yte, (p_rf > 0.5).astype(int))
    # top features by RF importance
    top_idx = np.argsort(-rf.feature_importances_)[:10]
    out["rf_top"] = [(feat_names[i], float(rf.feature_importances_[i])) for i in top_idx]

    # L2 Logreg (with scaling, for comparison)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    lr = LogisticRegression(
        penalty="l2", C=0.5, class_weight="balanced",
        max_iter=2000, solver="lbfgs",
    )
    lr.fit(Xtr_s, ytr)
    p_lr = lr.predict_proba(Xte_s)[:, 1]
    out["lr_auc"] = roc_auc_score(yte, p_lr)
    out["lr_acc"] = accuracy_score(yte, (p_lr > 0.5).astype(int))

    out["majority"] = max(yte.mean(), 1 - yte.mean())
    return out


def main():
    rows = []
    for task in TASKS:
        data = load_combined(task)
        if data is None:
            rows.append({"task": task, "skip": "insufficient data"})
            continue
        score, appl, count, y, aspects = data
        X, names = build_features(score, appl, count, aspects)
        n = len(y)
        n_aspects = len(aspects)
        try:
            res = run_eval(X, y, names)
        except Exception as e:
            rows.append({"task": task, "skip": str(e)})
            continue
        rows.append({"task": task, "n": n, "aspects": n_aspects, **res})

    print(f"{'task':<22} {'n':>5} {'asp':>4} {'maj':>6} "
          f"{'RF_AUC':>7} {'RF_acc':>7} {'LR_AUC':>7} {'LR_acc':>7}")
    print("-" * 84)
    for r in rows:
        if "skip" in r:
            print(f"{r['task']:<22}  skip: {r['skip']}")
            continue
        print(f"{r['task']:<22} {r['n']:>5} {r['aspects']:>4} "
              f"{r['majority']:>6.1%} {r['rf_auc']:>7.3f} {r['rf_acc']:>7.1%} "
              f"{r['lr_auc']:>7.3f} {r['lr_acc']:>7.1%}")

    print()
    print("=" * 78)
    print("TOP 10 FEATURES BY RF IMPORTANCE")
    print("=" * 78)
    for r in rows:
        if "skip" in r:
            continue
        print(f"\n{r['task']}  (RF AUC = {r['rf_auc']:.3f}, n={r['n']}):")
        for name, imp in r["rf_top"]:
            print(f"   {name:<18} importance={imp:.4f}")

    out_p = REPO / "outputs/v2_analysis/norm_rf_all_tasks.csv"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in rows:
        if "skip" in r:
            flat.append(r); continue
        flat.append({k: v for k, v in r.items() if k != "rf_top"})
    pd.DataFrame(flat).to_csv(out_p, index=False)
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
