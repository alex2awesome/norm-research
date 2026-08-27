"""Deep dive into errors of the norm-based RF on creative_writing.

For the test split (random_state=42, 80/20 stratified):
  1. Fit RF on train, predict on test
  2. Identify the worst confident errors (|p-y| highest) and near-boundary cases
  3. For each, pull artifact text and aspect scores; show what the model used
  4. Summarize patterns by length/content type
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASK = "creative_writing"
JUDGES = ["qwen_thinking_fp8", "claude"]
RANDOM_SEED = 42


def load_combined(task):
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
    return (
        score.loc[common],
        appl.loc[common],
        labels.loc[common],
        all_asp,
        dp_text,
    )


def main():
    score, appl, labels, aspects, dp_text = load_combined(TASK)
    dp_ids = list(score.index)
    X = np.concatenate([
        score[aspects].fillna(0).values,
        appl[aspects].values.astype(float),
    ], axis=1)
    y = labels.values

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
    yhat = (p > 0.5).astype(int)
    auc = roc_auc_score(yte, p)
    acc = (yhat == yte).mean()
    print(f"test n={len(yte)} pos={yte.sum()} neg={(1-yte).sum()} "
          f"AUC={auc:.3f} acc={acc:.1%}")
    print()

    # Build a dataframe of test results
    test_df = pd.DataFrame({
        "dp_id": test_dp_ids,
        "y": yte,
        "p": p,
        "yhat": yhat,
        "correct": yhat == yte,
        "confidence": np.abs(p - 0.5),
        "err": np.abs(p - yte),
    })
    test_df["text_len"] = test_df["dp_id"].map(lambda d: len(dp_text.get(d, "")))
    test_df["text_words"] = test_df["dp_id"].map(lambda d: len(dp_text.get(d, "").split()))

    # 1. Worst confidently-wrong y=1 (predicted neg, actually pos)
    print("=" * 78)
    print("WORST CONFIDENT FALSE NEGATIVES (y=1, p low → predicted neg)")
    print("=" * 78)
    fn = test_df[(test_df["y"] == 1)].sort_values("p").head(5)
    for _, r in fn.iterrows():
        print(f"\n  dp={r.dp_id}  y=1  p={r.p:.3f}  words={r.text_words}")
        txt = dp_text.get(r.dp_id, "")[:700].replace("\n", " ")
        print(f"  TEXT: {txt}...")

    # 2. Worst confident false positives (y=0, p high → predicted pos)
    print()
    print("=" * 78)
    print("WORST CONFIDENT FALSE POSITIVES (y=0, p high → predicted pos)")
    print("=" * 78)
    fp = test_df[(test_df["y"] == 0)].sort_values("p", ascending=False).head(5)
    for _, r in fp.iterrows():
        print(f"\n  dp={r.dp_id}  y=0  p={r.p:.3f}  words={r.text_words}")
        txt = dp_text.get(r.dp_id, "")[:700].replace("\n", " ")
        print(f"  TEXT: {txt}...")

    # 3. Near-boundary correct cases (model squeaked through)
    print()
    print("=" * 78)
    print("NEAR-BOUNDARY CORRECT (p ~ 0.5 yet got it right) — what tipped it")
    print("=" * 78)
    nb = test_df[(test_df["correct"]) & (test_df["confidence"] < 0.05)].head(3)
    for _, r in nb.iterrows():
        print(f"\n  dp={r.dp_id}  y={r.y}  p={r.p:.3f}")
        txt = dp_text.get(r.dp_id, "")[:400].replace("\n", " ")
        print(f"  TEXT: {txt}...")

    # 4. Confidence-vs-correctness summary
    print()
    print("=" * 78)
    print("ACCURACY BY CONFIDENCE QUANTILE")
    print("=" * 78)
    test_df["conf_bin"] = pd.qcut(test_df["confidence"], 4, labels=["q1_low", "q2", "q3", "q4_high"])
    print(test_df.groupby("conf_bin", observed=True)["correct"].agg(["mean", "count"]))

    print()
    print("=" * 78)
    print("ACCURACY BY TEXT LENGTH QUANTILE")
    print("=" * 78)
    test_df["len_bin"] = pd.qcut(test_df["text_words"], 4, labels=["short", "med-short", "med-long", "long"])
    print(test_df.groupby("len_bin", observed=True).agg(
        n=("dp_id", "count"),
        acc=("correct", "mean"),
        mean_p=("p", "mean"),
        pos_rate=("y", "mean"),
    ))

    print()
    print("=" * 78)
    print("APPLICATION RATE OF NORMS ACROSS ERRORS vs CORRECTS")
    print("=" * 78)
    appl_per_dp = appl[aspects].sum(axis=1)
    test_df["n_applicable_norms"] = test_df["dp_id"].map(appl_per_dp)
    print(test_df.groupby("correct").agg(
        n=("dp_id", "count"),
        mean_norms=("n_applicable_norms", "mean"),
        median_norms=("n_applicable_norms", "median"),
        mean_words=("text_words", "mean"),
    ))

    # Save
    out_p = REPO / "outputs/v2_analysis/cw_error_analysis.parquet"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(out_p)
    print(f"\nwrote test results -> {out_p}")


if __name__ == "__main__":
    main()
