"""Headline comparison: legacy qwen_thinking_fp8 vs new qwen_relaxed_v2_2026_06_01
on creative_writing.

  1. Per-aspect lift on each judge column.
  2. RF AUC on each judge column alone.
  3. RF AUC on union (legacy + relaxed, averaging cells where both exist).
  4. Predictions on the same fixed test set (random_state=42 80/20).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import ttest_ind

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
TASK = "creative_writing"
SEED = 42
TEST_FRAC = 0.20
JUDGE_LEGACY = "qwen_thinking_fp8"
JUDGE_NEW = "qwen_relaxed_v2_2026_06_01"


def load_labels():
    dps_raw = json.loads((REPO / f"runs/validity_full/v2/{TASK}/datapoints.json").read_text())
    return pd.Series({d["datapoint_id"]: int(d["judgement"])
                      for d in dps_raw if d.get("judgement") is not None})


def load_cells(judge):
    f = REPO / f"outputs/v2_db/cells_v1/task={TASK}/judge={judge}/data.parquet"
    df = pd.read_parquet(f)
    df["score_num"] = df["score"].where(df["applicable"], np.nan).astype(float)
    return df


def build_matrix(df):
    """Return (dp x aspect) score matrix and (dp x aspect) applicability matrix."""
    score = (df.groupby(["datapoint_id", "aspect_id"])["score_num"]
                .mean().unstack("aspect_id"))
    appl = (df.groupby(["datapoint_id", "aspect_id"])["applicable"]
              .max().unstack("aspect_id").fillna(False).astype(int))
    return score, appl


def aspect_lift(score, labels):
    """Per-aspect: delta + p + n on the score matrix."""
    common = score.index.intersection(labels.index)
    if len(common) < 10: return pd.DataFrame()
    s = score.loc[common]
    y = labels.loc[common]
    rows = []
    for a in s.columns:
        col = s[a]; mask = col.notna()
        if mask.sum() < 20: continue
        y_a = y[mask]; s_a = col[mask]
        s0 = s_a[y_a == 0]; s1 = s_a[y_a == 1]
        if len(s0) < 5 or len(s1) < 5: continue
        delta = s1.mean() - s0.mean()
        try: _, p = ttest_ind(s1, s0, equal_var=False)
        except: p = np.nan
        rows.append({"aspect_id": a, "delta": delta, "p": p,
                     "n0": len(s0), "n1": len(s1)})
    return pd.DataFrame(rows)


def rf_auc(score, appl, labels, label):
    aspects = sorted(set(score.columns) & set(appl.columns))
    score = score[aspects]; appl = appl[aspects]
    common = score.index.intersection(labels.index)
    score = score.loc[common]; appl = appl.loc[common]
    y = labels.loc[common].values
    X = np.concatenate([score.fillna(0).values, appl.values.astype(float)], axis=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRAC, stratify=y,
                                          random_state=SEED)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                class_weight="balanced", n_jobs=-1,
                                random_state=SEED)
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    acc = accuracy_score(yte, (p > 0.5).astype(int))
    print(f"  [{label}] n={len(y)} aspects={len(aspects)}  test AUC={auc:.3f}  acc={acc:.1%}")
    return auc, acc, len(aspects)


def main():
    labels = load_labels()
    print(f"labels: n={len(labels)}, pos={int(labels.sum())}")

    print("\n=== loading legacy cells ===")
    leg = load_cells(JUDGE_LEGACY)
    print(f"legacy cells: {len(leg):,}, dps: {leg['datapoint_id'].nunique()}, "
          f"aspects: {leg['aspect_id'].nunique()}")
    leg_score, leg_appl = build_matrix(leg)

    print("\n=== loading new (v2relax) cells ===")
    new = load_cells(JUDGE_NEW)
    print(f"new cells: {len(new):,}, dps: {new['datapoint_id'].nunique()}, "
          f"aspects: {new['aspect_id'].nunique()}")
    new_score, new_appl = build_matrix(new)

    # ---- Aspect-lift comparison ----
    print("\n" + "=" * 78)
    print("PER-ASPECT LIFT — top |delta| under each judge")
    print("=" * 78)
    leg_lift = aspect_lift(leg_score, labels)
    new_lift = aspect_lift(new_score, labels)
    print(f"\nlegacy: {len(leg_lift)} testable aspects")
    print(f"new:    {len(new_lift)} testable aspects")
    if len(leg_lift):
        leg_lift["abs"] = leg_lift["delta"].abs()
        print("\nlegacy top 10 by |delta|:")
        print(leg_lift.sort_values("abs", ascending=False).head(10)[["aspect_id","delta","p","n0","n1"]].to_string(index=False))
    if len(new_lift):
        new_lift["abs"] = new_lift["delta"].abs()
        print("\nnew top 10 by |delta|:")
        print(new_lift.sort_values("abs", ascending=False).head(10)[["aspect_id","delta","p","n0","n1"]].to_string(index=False))

    # ---- RF AUC ----
    print("\n" + "=" * 78)
    print("RF AUC — same test split (random_state=42, 80/20 stratified)")
    print("=" * 78)
    auc_leg, acc_leg, _ = rf_auc(leg_score, leg_appl, labels, "legacy")
    auc_new, acc_new, _ = rf_auc(new_score, new_appl, labels, "new (v2relax)")
    print(f"\n  Δ AUC = {auc_new - auc_leg:+.3f}")

    # ---- Union ----
    print("\n=== union (legacy + new, mean where both) ===")
    all_dps = sorted(set(leg_score.index) | set(new_score.index))
    all_asp = sorted(set(leg_score.columns) | set(new_score.columns))
    leg_re = leg_score.reindex(index=all_dps, columns=all_asp)
    new_re = new_score.reindex(index=all_dps, columns=all_asp)
    union_score = pd.concat([leg_re.stack(dropna=False), new_re.stack(dropna=False)],
                            axis=1).mean(axis=1, skipna=True).unstack()
    leg_ap = leg_appl.reindex(index=all_dps, columns=all_asp, fill_value=0)
    new_ap = new_appl.reindex(index=all_dps, columns=all_asp, fill_value=0)
    union_appl = ((leg_ap + new_ap) > 0).astype(int)
    auc_u, acc_u, _ = rf_auc(union_score, union_appl, labels, "union")
    print(f"  Δ AUC (union vs legacy)  = {auc_u - auc_leg:+.3f}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  legacy AUC       : {auc_leg:.3f}")
    print(f"  v2relax AUC      : {auc_new:.3f}  ({auc_new - auc_leg:+.3f})")
    print(f"  union AUC        : {auc_u:.3f}  ({auc_u - auc_leg:+.3f})")
    maj = max(labels.mean(), 1 - labels.mean())
    print(f"  majority baseline: {maj:.3f}")


if __name__ == "__main__":
    main()
