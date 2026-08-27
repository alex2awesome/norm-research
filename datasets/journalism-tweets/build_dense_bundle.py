#!/usr/bin/env python3
"""V9 journalism-tweets: emit the dense-standard training bundle from the frozen
population.

Same-rows by construction (FREEZE CHANGE 2): the dense arm trains on the
IDENTICAL engagement-defined population and the IDENTICAL outlet-day-grouped
split that the V/A stack uses, so T's held-out rows are a subset of the
A/V-scored rows and no separate rescore is needed.

The dense arm sees the SAME `text` column the A bank is judged on and the same
headline the V features are computed from -- byte-identical input across all
three arms, which is what the program's apples-to-apples rule requires.

Format mirrors datasets/stackoverflow-votes/va/dense_standard_so_votes exactly:
data.csv + split/{train,eval,test}.csv with columns text,judgement,group,row_id.

CLASS WEIGHTING (design note S12): this cell is balanced BY CONSTRUCTION -- y is
a within-group median split, so the pos rate is .5000 to four places and the
train minority class is ~12,450 rows. The frozen dense-standard recipe therefore
runs unmodified with NO class_weight_auto. Recorded because the pre-kill
checklist requires the absolute minority count on the record before any
dead/terminal verdict can be read off a dense arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="/lfs/skampere3/0/alexspan/norm-research/"
                                     "datasets/journalism-tweets/va/population.csv.gz")
    ap.add_argument("--out", default="/lfs/skampere3/0/alexspan/norm-research/"
                                     "datasets/journalism-tweets/va/"
                                     "dense_standard_journalism_tweets")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "split").mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(a.pop)
    v["judgement"] = v["judgement"].astype(int)
    cols = ["text", "judgement", "group", "row_id"]
    v[cols].to_csv(out / "data.csv", index=False)
    man = {"cell": "journalism_tweets", "n": int(len(v)),
           "pos_rate": float(v.judgement.mean()),
           "n_groups": int(v.group.nunique()),
           "group_column": "outlet_day",
           "y_definition": "1 = sum_likes strictly above the median sum_likes of "
                           "its own (outlet, first_day) homepage group, 0 = "
                           "strictly below; ties at the median dropped",
           "text_definition": "'HEADLINE: ' + the homepage anchor text. Article "
                              "bodies are deliberately NOT spliced in: they exist "
                              "for only part of the population (latimes/cnn/"
                              "guardian ~92-100%, nytimes 20%, reuters 9%, wapo "
                              "6% -- paywalls), so including them would make the "
                              "dense arm's evidence base differ by outlet and "
                              "break byte-identity with the A bank's input",
           "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, "
                     "2 epochs, gradient-checkpointing, select-on-eval "
                     "(dense-standard, no deviation; NO class_weight_auto -- the "
                     "cell is balanced by construction)",
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
