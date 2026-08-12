#!/usr/bin/env python3
"""Direction-3 / V3 feature-augmented dense for CODE COMPETITIONS (label: v3_aug).
User request 2026-08-12 ("LeetCode/Codeforces/CodeContests: Can you run V3 VAT on it?").
The live same-rows cell is the AtCoder strict-L1 population (n=999, 634 canonical_pid
groups, 149 minority negatives) — the only competitions cell with a certified
same-rows ladder (V .7289 / V+A .7383 / T .6967 / §11 fused .7537 wash).

ESTIMAND (binding, inherited from the RR/caption V3 design): v3_aug is a FUSED
V+A+T arm for the max-of-variants column ONLY. Never an honest T, never Δ_beyond.

Spec follows datasets/creative-writing/build_royalroad_v3aug.py:
  prompt = "VA metrics:" block of the TOP-10 fold-internal criteria, then the code.
  arm a only ("<name>: <score>"); arm b gated on arm a moving.
  importance = per-fold, GroupKFold(3) inside that fold's TRAIN rows, frozen HistGB
  (leaves=31, lr=.06, 400 iter), permutation_importance roc_auc n_repeats=5,
  mean over inner folds. Held-out fold never touches its own ranking.

FOLDS: StratifiedGroupKFold(5, shuffle, random_state=0) by canonical_pid — the
ladder's protocol (stratification MANDATORY at 149 minority; unstratified = .02
fold-composition artifact). Fold assignment saved to manifest for the harvest.

  python3 build_code_competitions_v3aug.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

R = Path(os.environ.get("NR_REPO", Path(__file__).resolve().parents[3]))
HERE = Path(__file__).resolve().parent
OUT = HERE / "dense_crossfit_v3aug"
TOP_K = 10
MAXLEN = 4096          # tokens; assert char budget below


def fmt(v):
    if v != v:
        return "NA"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


b = pd.read_parquet(R / "outputs/v2_analysis/comp_fourplatform_cells/ac_bank_scores.parquet")
c = pd.read_parquet(R / "outputs/v2_analysis/dense_ceiling/cell_ac_l1.parquet")
d = c.merge(b, on="pair_id", how="inner", suffixes=("", "_b")).reset_index(drop=True)
assert len(d) == 999 and d.canonical_pid.nunique() == 634, (len(d), d.canonical_pid.nunique())
y = d["label"].astype(int).values
g = d["canonical_pid"].astype(str).values
assert (int(y.sum()), int((1 - y).sum())) == (850, 149)

V = pd.read_parquet(HERE / "ac_v_features.parquet")
assert len(V) == len(d), "V feature parquet no longer row-aligned with the join"
a_cols = [k for k in b.columns if k.endswith("_score")]
A = d[a_cols].astype(float).values
VA = np.column_stack([V.values.astype(float), A])
names = list(V.columns) + [k[:-len("_score")] for k in a_cols]
texts = d["candidate_code"].astype(str).values
row_id = d["pair_id"].astype(str).values

lens = np.array([len(t) for t in texts])
print(f"[pop] n={len(d)} code chars median {int(np.median(lens))} "
      f"p99 {int(np.percentile(lens, 99))} max {lens.max()}")
n_over = int((lens > (MAXLEN - 400) * 3).sum())   # ~3 chars/token for code
print(f"[pop] rows possibly exceeding {MAXLEN} tokens incl. block: ~{n_over}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
folds = list(sgkf.split(np.zeros(len(y)), y, g))

man = {"design_id": "code_competitions_v3_aug", "direction": 3,
       "cell": "code_competitions AtCoder strict-L1 same-rows (n=999)",
       "estimand": "FUSED V+A+T arm, max-of-variants column ONLY; never an honest T, "
                   "never in Delta_beyond",
       "folds_protocol": "StratifiedGroupKFold(5, shuffle, random_state=0) by canonical_pid",
       "importance_protocol": "PER FOLD: GroupKFold(3) inside that fold's TRAIN rows, "
                              "frozen HistGB (leaves=31, lr=.06, 400 iter), "
                              "permutation_importance roc_auc n_repeats=5, mean over folds",
       "prompt_order": "VA metrics block FIRST, then the complete submission code",
       "max_len": MAXLEN, "arm": "a only (arm b gated on arm a moving)",
       "code_chars": {"median": int(np.median(lens)), "p99": int(np.percentile(lens, 99)),
                      "max": int(lens.max()), "possibly_truncated": n_over},
       "folds": {}}

rng = np.random.default_rng(20260812)
for k, (tr, te) in enumerate(folds):
    Xtr, ytr, gtr = VA[tr], y[tr], g[tr]
    Xi = np.where(np.isnan(Xtr), np.nanmedian(Xtr, axis=0), Xtr)
    imps, nf = np.zeros(VA.shape[1]), 0
    for itr, ite in GroupKFold(n_splits=3).split(Xi, ytr, gtr):
        m = HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.06,
                                           max_iter=400, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=20,
                                           random_state=0)
        m.fit(Xi[itr], ytr[itr])
        r = permutation_importance(m, Xi[ite], ytr[ite], scoring="roc_auc",
                                   n_repeats=5, random_state=0, n_jobs=-1)
        imps += r.importances_mean
        nf += 1
    imps /= nf
    order = np.argsort(-imps)[:TOP_K]
    top = [{"name": names[j], "col": int(j), "importance": float(imps[j])} for j in order]
    man["folds"][f"fold{k}"] = {"top_criteria": top,
                                "n_train": int(len(tr)), "n_test": int(len(te))}

    def block(i):
        out = ["VA metrics:"]
        for t in top:
            out.append(f"    {t['name']}: {fmt(float(VA[i, t['col']]))}")
        return "\n".join(out)

    # grouped eval slice (~15% of train groups) for checkpoint selection
    tr_groups = np.unique(gtr)
    ev_groups = set(rng.choice(tr_groups, size=max(1, int(len(tr_groups) * 0.15)),
                               replace=False))
    is_ev = np.array([gg in ev_groups for gg in gtr])

    dd = OUT / "arm_a" / f"fold{k}"
    (dd / "split").mkdir(parents=True, exist_ok=True)
    def rows_df(idx_global):
        return pd.DataFrame({
            "text": [block(i) + "\n\nSUBMISSION CODE:\n" + texts[i] for i in idx_global],
            "judgement": y[idx_global], "group": g[idx_global], "row_id": row_id[idx_global]})
    df_tr = rows_df(tr[~is_ev]); df_ev = rows_df(tr[is_ev]); df_te = rows_df(te)
    pd.concat([df_tr, df_ev, df_te]).to_csv(dd / "data.csv", index=False)
    df_tr.to_csv(dd / "split/train.csv", index=False)
    df_ev.to_csv(dd / "split/eval.csv", index=False)
    df_te.to_csv(dd / "split/test.csv", index=False)
    print(f"[fold{k}] train {len(df_tr)} eval {len(df_ev)} test {len(df_te)} "
          f"top1={top[0]['name']}", flush=True)

(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print("BUILD_DONE", OUT, flush=True)
