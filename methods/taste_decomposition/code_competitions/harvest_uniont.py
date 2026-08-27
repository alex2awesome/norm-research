#!/usr/bin/env python3
"""Harvest the UNION plain-text Llama dense (code_uniont) — the live starvation test
for coding curation (user 2026-08-13: "make sure you are using ALL the data").

Arm: Llama-3.1-8B LoRA on plain candidate text (NO feature block), trained on the
full 6,353-row four-platform union with the v3max StratifiedGroupKFold(5) folds
(platform-prefixed canonical_pid groups), max_len 4096, seed 42.  OOF = each fold's
test-split preds (the held-out fifth; the eval slice sits INSIDE train groups, selection-only).

READOUTS (composition rule: pooled-across-platform NEVER quoted):
  1. per-platform AUC + n-weighted within-platform mean;
  2. AC-999 same-rows restriction (the v3aug bank-scored intersection) vs the
     same-rows references: AC-fit VA_nl .7535 / AC-fit dense .7241 / union-fit VA
     (recomputed here from union_va_oof.npz on the identical rows).
CPU only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
R = HERE.parents[2]
U = HERE / "dense_crossfit_uniont" / "arm_t"
V3AUG = HERE / "dense_crossfit_v3aug" / "arm_a"

# ---- 1. assemble the union OOF from the 5 folds' eval preds ------------------
rows = []
for k in range(5):
    sp = pd.read_csv(U / f"fold{k}" / "split" / "test.csv")
    pr = pd.read_csv(U / f"fold{k}" / "rm_out_seed42" / "preds_test.csv")
    assert len(sp) == len(pr), f"fold{k}: split {len(sp)} != preds {len(pr)}"
    assert (sp.judgement.values == pr.judgement.values).all(), f"fold{k}: order-join fail (judgement)"
    if "group" in pr.columns:
        assert (sp.group.astype(str).values == pr.group.astype(str).values).all(), \
            f"fold{k}: order-join fail (group)"
    for rid, y_, g_, p_ in zip(sp.row_id.astype(str), sp.judgement.astype(int),
                               sp.group.astype(str), pr.prob.astype(float)):
        rows.append((rid, y_, g_, p_, k))
df = pd.DataFrame(rows, columns=["row_id", "y", "group", "prob", "fold"])
assert df.row_id.is_unique, "a row appears in more than one fold's test split"
print(f"[uniont] OOF rows {len(df)} across 5 folds", flush=True)

# ---- 2. platform join --------------------------------------------------------
u = pd.read_parquet(R / "outputs/v2_analysis/comp_fourplatform_cells/union_bank_scores.parquet")
pm = dict(zip(u.pair_id.astype(str), u.platform))
ym = dict(zip(u.pair_id.astype(str), u.label.astype(int)))
df["platform"] = df.row_id.map(pm)
assert df.platform.notna().all(), f"{df.platform.isna().sum()} rows missing platform"
chk = df.row_id.map(ym)
assert (chk.values == df.y.values).all(), "y disagrees with the union parquet labels"

res = {"cell": "code_competitions", "arm": "uniont (plain-text union dense, no block)",
       "n_oof": int(len(df)),
       "oof_note": "test = the held-out fifth; selection used the eval slice INSIDE train groups, so the test OOF is honest (no select-on-heldout)"}
per, nw, tot = {}, 0.0, 0
for p in ("ac", "cf", "lc", "cc"):
    m = df.platform == p
    a = float(roc_auc_score(df.y[m], df.prob[m]))
    per[p] = {"n": int(m.sum()), "auc": a}
    nw += a * m.sum()
    tot += int(m.sum())
res["pooled_NEVER_QUOTE"] = float(roc_auc_score(df.y, df.prob))
res["per_platform"] = per
res["within_platform_nwtd"] = nw / tot
print("[uniont] within-platform "
      f"{res['within_platform_nwtd']:.4f} | " + " ".join(f"{p}:{per[p]['auc']:.3f}" for p in per),
      flush=True)

# ---- 3. AC-999 same-rows restriction ----------------------------------------
ac999 = set()
for f in (V3AUG / "fold0" / "split").glob("*.csv"):
    ac999 |= set(pd.read_csv(f).row_id.astype(str))
print(f"[uniont] AC-999 id set: {len(ac999)}", flush=True)
m999 = df.row_id.isin(ac999)
res["ac999"] = {"n_found": int(m999.sum()),
                "auc_uniont": float(roc_auc_score(df.y[m999], df.prob[m999]))}

z = np.load(HERE / "union_va_oof.npz", allow_pickle=True)
zp = [str(x) for x in z["pair_id"]]
zmask = np.isin(np.array(zp, dtype=object), list(ac999))
res["ac999"]["auc_union_fit_VA_same_rows"] = float(
    roc_auc_score(z["y"][zmask], z["VA_oof"][zmask]))
res["ac999"]["n_va_rows"] = int(zmask.sum())
res["refs_same_rows"] = {"VA_nl_ac_fit": 0.7535, "dense_ac_fit": 0.7241,
                         "modernbert_old": 0.697, "v3max_prompt_arm": 0.6846}
print(f"[uniont] AC-999: uniont {res['ac999']['auc_uniont']:.4f} vs "
      f"union-fit VA {res['ac999']['auc_union_fit_VA_same_rows']:.4f} vs "
      f"AC-fit VA_nl .7535 / AC-fit dense .7241", flush=True)

json.dump(res, open(HERE / "uniont_harvest.json", "w"), indent=1)
print("UNIONT_HARVEST_DONE", flush=True)
