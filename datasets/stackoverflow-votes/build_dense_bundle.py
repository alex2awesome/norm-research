#!/usr/bin/env python3
"""V6 SO-votes: emit the dense-standard training bundle from the frozen population.

Same-rows by construction (FREEZE CHANGE 2): the dense arm trains on the
IDENTICAL vote-defined population and the IDENTICAL question-grouped split that
the V/A stack uses, so T's held-out rows are a subset of the A/V-scored rows and
no separate rescore is needed.

Format mirrors datasets/math-stackexchange/v2_va/dense_standard_mathse_vote_score
exactly: data.csv + split/{train,eval,test}.csv with columns
text,judgement,group,row_id.

CLASS WEIGHTING (design note S12: "class-weighted where imbalanced"): this cell
is NOT imbalanced -- pos rate .5238, train minority class 4,649 rows -- so the
frozen dense-standard recipe runs unmodified with NO class_weight_auto. Recorded
here because S14 item (1) requires the absolute minority count on the record
before any dead/terminal verdict can be read off a dense arm.

ITEM VIEW (--view). Two views are emitted, and the difference between them is
measured rather than argued about:

  `title`  (DEFAULT, the headline arm) = "QUESTION: {title}\n\nANSWER:\n{body}".
     Identical in shape to the math.SE sibling's dense text, so the two SE vote
     cells' T's are comparable.

  `abank`  (SENSITIVITY arm) = the A judge's exact context: question title +
     tags + truncated question body + answer.

Why both, instead of just matching the judge: the A judge sees title + tags +
question body + answer (~634 median / 1,944 p99 tokens), while the dense recipe
truncates at **1024 tokens**. Prepending the question body therefore does not
simply "give the dense model the same information" -- for a sizeable minority of
items it PUSHES THE ANSWER, the thing actually being judged, out of the dense
window. So the A-matched view is not automatically the fairer one. The headline
T uses the `title` view; the `abank` view is run for one seed as an item-view
sensitivity arm, and both numbers are reported.

The asymmetry that remains on the headline arm -- T sees less CONTEXT than A --
is stated in the note with its direction: it handicaps T, which biases
Delta_beyond = T - VA_nl DOWNWARD, i.e. toward over-stating articulability. That
is the dangerous direction for an "articulable" verdict, which is exactly why
the sensitivity arm is run rather than waved off.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="/lfs/skampere3/0/alexspan/norm-research/"
                                     "datasets/stackoverflow-votes/va/population.csv.gz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--view", choices=["title", "abank"], default="title")
    a = ap.parse_args()

    default_out = ("/lfs/skampere3/0/alexspan/norm-research/datasets/"
                   "stackoverflow-votes/va/dense_standard_so_votes"
                   + ("" if a.view == "title" else "_abankview"))
    out = Path(a.out or default_out)
    (out / "split").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.pop)
    v = df[df.y_vote.notna()].copy()
    v["judgement"] = v.y_vote.astype(int)

    if a.view == "abank":
        # byte-identical to score_so_votes_bank.build_so_votes().ctx
        MARK = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"

        def trunc(s, src, head, tail):
            s = (s or "").strip()
            return s if len(s) <= src else s[:head] + MARK + s[-tail:]

        v["text"] = [
            f"QUESTION TITLE: {str(t)[:400]}\n"
            f"QUESTION TAGS: {str(g)[:200]}\n\n"
            f"QUESTION BODY:\n{trunc(str(qb), 2400, 1600, 800)}\n\n"
            f"ANSWER:\n{trunc(str(b), 5000, 3000, 2000)}"
            for t, g, qb, b in zip(v.q_title, v.tags, v.q_body, v.body)]
    cols = ["text", "judgement", "group", "row_id"]
    v[cols].to_csv(out / "data.csv", index=False)
    man = {"cell": "so_votes" if a.view == "title" else "so_votes_abankview",
           "item_view": a.view, "n": int(len(v)),
           "pos_rate": float(v.judgement.mean()),
           "n_groups": int(v.group.nunique()), "group_column": "question_id",
           "y_definition": "1 = raw net vote Score strictly above the median answer "
                           "Score on its own question, 0 = strictly below; ties at "
                           "the median dropped",
           "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, "
                     "2 epochs, gradient-checkpointing, select-on-eval "
                     "(dense-standard, no deviation; NO class_weight_auto -- the "
                     "cell is balanced, train minority 4,649)",
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
