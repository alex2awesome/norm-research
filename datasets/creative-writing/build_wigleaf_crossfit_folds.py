#!/usr/bin/env python3
"""5-fold cross-fitted dense splits for cw_wigleaf_curation.

Fixes TWO things at once:
  (1) the item-level T .6054 rests on a 170-row eval that was ALSO the
      checkpoint-selection split -- the identical flaw the RoyalRoad cross-fit
      just corrected;
  (2) T_pair for the P2 pairwise Layer-1 is uncomputable from the current arm,
      which has out-of-sample predictions for only 322 of 1,568 rows (so only
      ~25 of 600 pairs have BOTH pieces scored out-of-sample).

DESIGN (same SO template as RoyalRoad):
  bucket t = md5("wigleaf-cf::" + row_id) % 1000 // 100     -> t in 0..9
  fold k:  test tenth = 2k, select tenth = 2k+1, train = the other 8 tenths
  honest set = union of the 5 test tenths ~= half of 1,568.

NOTE ON THE PARTITION. Unlike RoyalRoad -- where the shipped split rule was itself
an id hash, so the cross-fit tenths nested inside it exactly -- Wigleaf's shipped
split keys on md5(title|author|year), and the population carries only the coarse
3-way label, not a tenth index. So the cross-fit uses its OWN 10-way stable hash
over row_id. That is sound here because every Wigleaf row is its own group (one
story per row), so any per-id hash partition is leak-free by construction; it is
simply a different partition from the shipped one, and is labelled as such rather
than presented as nesting inside it.

Class weighting is REQUIRED downstream (404 absolute positives).

  python datasets/creative-writing/build_wigleaf_crossfit_folds.py
"""
import hashlib, json, os
from pathlib import Path
import pandas as pd

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CELL = REPO / "datasets/creative-writing/wigleaf"
OUT = CELL / "dense_crossfit"
COLS = ["text", "judgement", "group", "row_id"]


def bucket(rid):
    return int(hashlib.md5(("wigleaf-cf::" + str(rid)).encode()).hexdigest(), 16) % 1000 // 100


df = pd.read_csv(CELL / "va/population.csv.gz")
df["row_id"] = df["row_id"].astype(str)
df["bucket"] = df["row_id"].map(bucket)
print("[tenths]", df["bucket"].value_counts().sort_index().to_dict())

man = {"cell": "cw_wigleaf_curation", "design": "5-fold cross-fit",
       "bucket_rule": 'md5("wigleaf-cf::"+row_id)%1000//100',
       "partition_note": "own 10-way stable hash; does NOT nest inside the shipped "
                         "md5(title|author|year)%10 split (the population carries only "
                         "the coarse 3-way label). Leak-free because every row is its "
                         "own group.",
       "class_weighting": "REQUIRED downstream (404 absolute positives)",
       "n_total": int(len(df)), "folds": {}}
honest, hpos = [], 0
for k in range(5):
    te_t, ev_t = 2 * k, 2 * k + 1
    te, ev = df[df.bucket == te_t], df[df.bucket == ev_t]
    tr = df[~df.bucket.isin([te_t, ev_t])]
    d = OUT / f"fold{k}"; (d / "split").mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(d / "data.csv", index=False)
    tr[COLS].to_csv(d / "split/train.csv", index=False)
    ev[COLS].to_csv(d / "split/eval.csv", index=False)
    te[COLS].to_csv(d / "split/test.csv", index=False)
    honest += te["row_id"].tolist(); hpos += int(te["judgement"].sum())
    man["folds"][f"fold{k}"] = {
        "test_tenth": te_t, "select_tenth": ev_t,
        "n_train": int(len(tr)), "n_eval": int(len(ev)), "n_test": int(len(te)),
        "pos_train": int(tr.judgement.sum()), "pos_eval": int(ev.judgement.sum()),
        "pos_test": int(te.judgement.sum())}
    print(f"[fold{k}] train {len(tr)} ({int(tr.judgement.sum())} pos) | "
          f"select {len(ev)} ({int(ev.judgement.sum())}) | TEST {len(te)} "
          f"({int(te.judgement.sum())})")
assert len(honest) == len(set(honest))
man["honest_set"] = {"n": len(honest), "n_pos": hpos,
                     "pos_rate": round(hpos / len(honest), 4),
                     "selection_free": True, "ids": honest,
                     "vs_old_eval": f"n={len(honest)} vs the old 170-row eval "
                                    f"({len(honest)/170:.1f}x), which was also the "
                                    f"selection split"}
(OUT / "manifest.json").write_text(json.dumps(man, indent=2))
print(f"[honest set] n={len(honest)} pos={hpos} ({hpos/len(honest):.4f})")
print("BUILD_WIGLEAF_CROSSFIT_FOLDS_DONE")
