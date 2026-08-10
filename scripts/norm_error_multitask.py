"""Multi-task error deep dive.

For each task with sufficient N:
  1. Train RF (random_state=42, 80/20 stratified) using union of qwen+claude
     judge cells.
  2. Identify top 3 confident FP and FN on the held-out test set.
  3. For each error, show truncated artifact text and the RF's top global
     features (with their aspect *names* and *descriptions*).
  4. Print a meta-pattern: per task, the names of the top-5 most-important
     aspects per RF feature importance.

For each task we use that task's own `aspects.json` to translate aspect_id
→ {name, description}.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASKS = [
    "peer_review", "math", "notice_and_comment", "press_releases",
    "humor", "news_homepages", "patents", "code_review", "creative_writing",
]
JUDGES = ["qwen_thinking_fp8", "claude"]
RANDOM_SEED = 42
MIN_N = 300


def load_aspects(task):
    p = REPO / f"runs/validity_full/v2/{task}/aspects.json"
    arr = json.loads(p.read_text())
    return {a["aspect_id"]: a for a in arr}


def load_task_data(task):
    labels_p = REPO / f"runs/validity_full/v2/{task}/datapoints.json"
    dps_raw = json.loads(labels_p.read_text())
    dp_text = {d["datapoint_id"]: d.get("text", "") for d in dps_raw}
    labels = pd.Series(
        {d["datapoint_id"]: int(d["judgement"])
         for d in dps_raw if d.get("judgement") is not None}
    )
    score_dfs, appl_dfs = [], []
    for j in JUDGES:
        f = REPO / f"outputs/v2_db/cells_v1/task={task}/judge={j}/data.parquet"
        if not f.exists():
            continue
        c = pd.read_parquet(f)
        c["score_num"] = c["score"].where(c["applicable"], np.nan).astype(float)
        sc = (c.groupby(["datapoint_id", "aspect_id"])["score_num"]
                .mean().unstack("aspect_id"))
        ap = (c.groupby(["datapoint_id", "aspect_id"])["applicable"]
                .max().unstack("aspect_id").fillna(False).astype(int))
        score_dfs.append(sc)
        appl_dfs.append(ap)
    if not score_dfs:
        return None
    all_dps = sorted(set().union(*(df.index for df in score_dfs)))
    all_asp = sorted(set().union(*(df.columns for df in score_dfs)))

    def reidx(dfs, fill=np.nan):
        return [df.reindex(index=all_dps, columns=all_asp, fill_value=fill)
                for df in dfs]

    sc = reidx(score_dfs)
    ap = reidx(appl_dfs, fill=0)
    score = pd.concat([s.stack(dropna=False) for s in sc], axis=1).mean(
        axis=1, skipna=True).unstack().reindex(index=all_dps, columns=all_asp)
    appl = sum(ap)
    appl = (appl > 0).astype(int)
    common = sorted(set(score.index) & set(labels.index))
    if len(common) < MIN_N:
        return None
    return (
        score.loc[common], appl.loc[common], labels.loc[common],
        all_asp, dp_text,
    )


def run_task(task, aspect_meta):
    data = load_task_data(task)
    if data is None:
        return None
    score, appl, labels, aspects, dp_text = data
    X = np.concatenate([
        score[aspects].fillna(0).values,
        appl[aspects].values.astype(float),
    ], axis=1)
    y = labels.values
    dp_ids = list(score.index)
    feat_names = [f"score_{a}" for a in aspects] + [f"appl_{a}" for a in aspects]

    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(dp_ids)),
        test_size=0.20, stratify=y, random_state=RANDOM_SEED,
    )
    test_dp_ids = [dp_ids[i] for i in idx_te]

    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
    )
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)

    test_df = pd.DataFrame({
        "dp_id": test_dp_ids,
        "y": yte,
        "p": p,
        "correct": (p > 0.5).astype(int) == yte,
        "text_words": [len(dp_text.get(d, "").split()) for d in test_dp_ids],
    })

    # Top features
    top_idx = np.argsort(-rf.feature_importances_)[:6]
    top_features = []
    for i in top_idx:
        f = feat_names[i]
        a = f.split("_", 1)[1]
        meta = aspect_meta.get(a, {})
        top_features.append({
            "feature": f, "aspect_id": a,
            "name": meta.get("name", "?"),
            "description": meta.get("description", "?")[:160],
            "importance": float(rf.feature_importances_[i]),
        })

    # Top errors
    fn = test_df[test_df["y"] == 1].nsmallest(3, "p")
    fp = test_df[test_df["y"] == 0].nlargest(3, "p")

    return {
        "task": task, "n_test": len(yte), "auc": auc,
        "top_features": top_features,
        "fn": fn.to_dict("records"),
        "fp": fp.to_dict("records"),
        "dp_text": dp_text,
    }


def main():
    print("=" * 88)
    print("PER-TASK TOP RF FEATURES — NAMED")
    print("=" * 88)
    results = []
    for task in TASKS:
        meta = load_aspects(task)
        r = run_task(task, meta)
        if r is None:
            print(f"\n--- {task} --- (skipped: insufficient n)")
            continue
        results.append(r)
        print(f"\n--- {task} (test AUC = {r['auc']:.3f}, n_test={r['n_test']}) ---")
        for f in r["top_features"]:
            print(f"  {f['feature']:<14} imp={f['importance']:.3f}  "
                  f"[{f['name'][:60]}]")
            print(f"  {'':<14}      {f['description']}")

    print()
    print("=" * 88)
    print("PER-TASK SAMPLE ERRORS (top confident FP and FN)")
    print("=" * 88)
    for r in results:
        print(f"\n##### {r['task']} (AUC {r['auc']:.3f}) #####")
        print("\nWORST FALSE NEGATIVES (y=1, model said no):")
        for row in r["fn"]:
            text = r["dp_text"].get(row["dp_id"], "")[:500].replace("\n", " ")
            print(f"  dp={row['dp_id']}  p={row['p']:.3f}  words={row['text_words']}")
            print(f"  TEXT: {text}...")
        print("\nWORST FALSE POSITIVES (y=0, model said yes):")
        for row in r["fp"]:
            text = r["dp_text"].get(row["dp_id"], "")[:500].replace("\n", " ")
            print(f"  dp={row['dp_id']}  p={row['p']:.3f}  words={row['text_words']}")
            print(f"  TEXT: {text}...")


if __name__ == "__main__":
    main()
