#!/usr/bin/env python3
"""BBC most-read: emit the dense-standard training bundle from the frozen population.

Same-rows by construction (FREEZE CHANGE 2): the dense arm trains on the
IDENTICAL population and the IDENTICAL day-grouped split the V/A stack uses, so
T's held-out rows are a subset of the A/V-scored rows and no separate rescore is
needed. The dense arm sees the SAME `text` column the A bank is judged on and
the same headline the V features come from -- byte-identical input across arms.

CLASS WEIGHTING (design note S12). Pos rate is .4405, train minority 17,889 rows
-- mild and very far from the regime where the frozen recipe needs adjusting, so
it runs unmodified with NO class_weight_auto, matching V9 and V6. The absolute
minority count is recorded here because the pre-kill checklist requires it on
the record before any dead/terminal verdict can be read off a dense arm.

Format mirrors datasets/journalism-tweets/va/dense_standard_journalism_tweets
exactly: data.csv + split/{train,eval,test}.csv with text,judgement,group,row_id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="/lfs/skampere3/0/alexspan/norm-research/"
                                     "datasets/bbc-mostread/va/population.csv.gz")
    ap.add_argument("--out", default="/lfs/skampere3/0/alexspan/norm-research/"
                                     "datasets/bbc-mostread/va/"
                                     "dense_standard_bbc_mostread")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "split").mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(a.pop)
    v["judgement"] = v["judgement"].astype(int)
    cols = ["text", "judgement", "group", "row_id"]
    v[cols].to_csv(out / "data.csv", index=False)
    man = {"cell": "bbc_mostread", "n": int(len(v)),
           "pos_rate": float(v.judgement.mean()),
           "n_groups": int(v.group.nunique()),
           "group_column": "capture_day",
           "y_definition": "1 = headline appeared in the BBC News home page "
                           "ranked MOST READ top-10 on that capture; 0 = "
                           "appeared elsewhere on the SAME capture",
           "text_definition": "'HEADLINE: ' + the BBC home page link headline",
           "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, "
                     "2 epochs, gradient-checkpointing, select-on-eval "
                     "(dense-standard, no deviation; NO class_weight_auto -- "
                     "pos rate .4405, train minority 17,889)",
           "split_row_counts": {}, "split_group_counts": {}, "split_pos_rates": {}}
    for s in ["train", "eval", "test"]:
        d = v[v.split == s]
        d[cols].to_csv(out / "split" / f"{s}.csv", index=False)
        man["split_row_counts"][s] = int(len(d))
        man["split_group_counts"][s] = int(d.group.nunique())
        man["split_pos_rates"][s] = float(d.judgement.mean())
    man["train_minority_count"] = int(min(
        (v[v.split == "train"].judgement == 0).sum(),
        (v[v.split == "train"].judgement == 1).sum()))
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
