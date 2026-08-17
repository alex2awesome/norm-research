#!/usr/bin/env python3
"""U4b — so_accepted VERDICT cell: the free cell on V6's rows.

The V6 so_votes population was already restricted to questions carrying BOTH
signals (a defined vote y AND an accepted answer) and carries y_accepted as a
separate never-merged column.  This build changes exactly ONE thing — judgement
becomes y_accepted — and reuses rows, text, groups, and split assignment
VERBATIM (same-rows cross-y contrast is the design goal; identical instruments).
"""
import json
from pathlib import Path

import pandas as pd

NR = Path("/lfs/skampere3/0/alexspan/norm-research")
VA = NR / "datasets/stackoverflow-votes/va"
OUT = NR / "datasets/stackoverflow-votes/so_accepted"
OUT.mkdir(parents=True, exist_ok=True)

pop = pd.read_csv(VA / "population.csv.gz")
assert {"row_id", "group", "split", "text", "y_accepted"} <= set(pop.columns)
assert pop.y_accepted.isin([0, 1]).all(), "y_accepted not binary"
acc_per_q = pop.groupby("group").y_accepted.sum()
assert (acc_per_q >= 1).all(), "V6 both-signals restriction violated (no accepted answer)"
assert (acc_per_q <= 1).all(), "question with >1 accepted answer"

df = pop.rename(columns={"y_accepted": "judgement"}).copy()
df["judgement"] = df.judgement.astype(int)
for s in ("train", "eval", "test"):
    sub = df[df.split == s]
    print(f"  {s}: n={len(sub)} pos={sub.judgement.mean():.4f} groups={sub.group.nunique()}")

DS = OUT / "dense_standard_so_accepted"
(DS / "split").mkdir(parents=True, exist_ok=True)
cols = ["text", "judgement", "group", "row_id"]
df[cols].to_csv(DS / "data.csv", index=False)
for s in ("train", "eval", "test"):
    df[df.split == s][cols].to_csv(DS / "split" / f"{s}.csv", index=False)
man = {
    "cell": "so_accepted (VERDICT: asker accepted this answer, python corpus)",
    "provenance": ("V6 so_votes population/splits VERBATIM (rows, text, groups, split "
                   "assignment untouched); judgement = the y_accepted column V6 "
                   "carried separately by design — same-rows cross-y contrast"),
    "n": int(len(df)), "n_questions": int(df.group.nunique()),
    "pos_rate": float(df.judgement.mean()),
    "splits": {s: int((df.split == s).sum()) for s in ("train", "eval", "test")},
}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print("SO_ACCEPTED_CELL_DONE")
