#!/usr/bin/env python3
"""Freeze the math.SE V2 multi-y A/V population and build the two dense-standard
arms on exactly that population (task V2 / #44).

Upstream: build_multiy_v2.py, which re-parses the raw Posts.xml dump and emits
two UN-BINARIZED, question-grouped populations with no score censoring:
  mathse_v2_accepted_verdict.csv.gz  857,709 rows / 329,558 questions, pos .384
  mathse_v2_vote_score.csv.gz        859,421 rows / 407,240 questions, pos .517

This script takes the INTERSECTION at the question level -- questions that carry
BOTH signals (a recorded accepted answer, and at least one answer strictly above
and one strictly below the question's median answer score) -- so a single Gemma
A-bank scoring pass serves both y's, exactly the way the Style Invitational bank
serves its two y's. The two y's are always reported separately and never merged.

  y_accepted : 1 = this answer is the one the ASKER accepted (verdict signal)
  y_vote     : 1 = this answer's raw vote score is strictly above the median
               score of the answers on its own question (CROWD signal);
               answers exactly at the median carry NaN and are dropped from the
               vote readout only.

Sampling: whole QUESTIONS in stable-hash order sha256("mathse-v2-va|" + qid),
taken until the row target is met. Whole-group draws are correct here because a
question genuinely is a container (both y's are within-question contrasts).

Dense standard: one arm per y, both on this population, question-grouped 80/10/10
via the frozen stable_hash_bucket_map bin-packer (row-count AND pos-rate
balanced) ported from datasets/humor/hashtagwars/build_dense_standard.py.

Usage (CPU only, run on sk3 where the upstream csvs live):
  python3 build_v2_va_population.py --indir <v2_multiy dir> --outdir <va dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SALT = "mathse-v2-va|"
N_TARGET = 13000


def sha256(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def stable_hash_bucket_map(y_by_group: dict, targets=None, lam: float = 2.5) -> dict:
    """Verbatim from datasets/humor/hashtagwars/build_dense_standard.py."""
    targets = targets or {"train": .8, "eval": .1, "test": .1}
    sizes = {g: len(v) for g, v in y_by_group.items()}
    pos = {g: sum(v) for g, v in y_by_group.items()}
    total = sum(sizes.values())
    overall_rate = sum(pos.values()) / total
    order = sorted(sizes, key=lambda g: (-sizes[g], sha1(str(g))))
    filled = {b: 0 for b in targets}
    filled_pos = {b: 0 for b in targets}
    bmap = {}

    def obj():
        o = sum((filled[b] / total - targets[b]) ** 2 for b in targets)
        o += lam * sum(((filled_pos[b] / max(filled[b], 1)) - overall_rate) ** 2
                       for b in targets)
        return o

    for g in order:
        best_b, best_o = None, None
        for b in targets:
            filled[b] += sizes[g]; filled_pos[b] += pos[g]
            o = obj()
            if best_o is None or o < best_o:
                best_o, best_b = o, b
            filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
        bmap[g] = best_b
        filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]

    improved, n_iter = True, 0
    while improved and n_iter < 20:
        improved = False
        n_iter += 1
        for g in order:
            cur = bmap[g]
            best_b, best_o = cur, obj()
            for b in targets:
                if b == cur:
                    continue
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[b] += sizes[g]; filled_pos[b] += pos[g]
                o = obj()
                if o < best_o - 1e-12:
                    best_b, best_o = b, o
                filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
                filled[cur] += sizes[g]; filled_pos[cur] += pos[g]
            if best_b != cur:
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]
                bmap[g] = best_b
                improved = True
    return bmap


def write_dense(rows, outdir: Path, ycol: str, cell: str, meta_extra: dict):
    rows = [r for r in rows if r[ycol] == r[ycol] and r[ycol] is not None]  # drop NaN
    d = outdir / f"dense_standard_{cell}"
    (d / "split").mkdir(parents=True, exist_ok=True)
    out_rows = [{"text": r["text"], "judgement": int(r[ycol]), "group": r["group"],
                 "row_id": r["row_id"]} for r in rows]
    y_by_group = defaultdict(list)
    for r in out_rows:
        y_by_group[r["group"]].append(r["judgement"])
    bmap = stable_hash_bucket_map(y_by_group)
    by_split = {"train": [], "eval": [], "test": []}
    for r in out_rows:
        by_split[bmap[r["group"]]].append(r)
    cols = ["text", "judgement", "group", "row_id"]
    with open(d / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
    for s in ("train", "eval", "test"):
        with open(d / "split" / f"{s}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(by_split[s])
    n = len(out_rows)
    man = {"cell": cell, "n": n,
           "pos_rate": sum(r["judgement"] for r in out_rows) / n,
           "n_groups": len(set(r["group"] for r in out_rows)),
           "group_column": "question_id",
           "split_row_counts": {s: len(by_split[s]) for s in by_split},
           "split_group_counts": {s: len(set(r["group"] for r in by_split[s])) for s in by_split},
           "split_pos_rates": {s: sum(r["judgement"] for r in by_split[s]) / max(len(by_split[s]), 1)
                               for s in by_split},
           "split_fractions": {s: len(by_split[s]) / n for s in by_split},
           "recipe": ("Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                      "gradient-checkpointing, select-on-eval (dense-standard, no deviation)"),
           **meta_extra}
    (d / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-target", type=int, default=N_TARGET)
    a = ap.parse_args()
    ind, out = Path(a.indir), Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    acc = pd.read_csv(ind / "mathse_v2_accepted_verdict.csv.gz")
    vote = pd.read_csv(ind / "mathse_v2_vote_score.csv.gz",
                       usecols=["row_id", "judgement"]).rename(
                           columns={"judgement": "y_vote"})
    acc = acc.rename(columns={"judgement": "y_accepted"})
    df = acc.merge(vote, on="row_id", how="left")
    df["question_id"] = df["question_id"].astype(str)
    df["group"] = df["question_id"]

    # questions carrying BOTH signals
    g = df.groupby("question_id")
    ok = g.apply(lambda d: (d["y_accepted"].sum() >= 1)
                 and (d["y_vote"] == 1).any() and (d["y_vote"] == 0).any(),
                 include_groups=False)
    qs = ok[ok].index.to_numpy()
    print(f"questions with BOTH signals: {len(qs):,} of {df['question_id'].nunique():,}")

    order = sorted(qs, key=lambda q: sha256(SALT + str(q)))
    sizes = df.groupby("question_id").size().to_dict()
    take, n = [], 0
    for q in order:
        take.append(q)
        n += sizes[q]
        if n >= a.n_target:
            break
    sel = set(take)
    pop = df[df["question_id"].isin(sel)].copy()
    pop = pop.sort_values(["question_id", "answer_position"], kind="mergesort").reset_index(drop=True)
    print(f"population: n={len(pop):,} questions={pop['question_id'].nunique():,} "
          f"y_accepted pos={pop['y_accepted'].mean():.4f} "
          f"y_vote pos={pop['y_vote'].mean(skipna=True):.4f} "
          f"(vote-defined rows {int(pop['y_vote'].notna().sum()):,})")
    print(f"text chars: median={pop['text'].str.len().median():.0f} "
          f"p95={pop['text'].str.len().quantile(.95):.0f} max={pop['text'].str.len().max()}")

    keep = ["row_id", "question_id", "answer_id", "group", "text", "y_accepted",
            "y_vote", "score", "accepted", "answer_position", "n_answers",
            "answer_year", "primary_tag"]
    pop[keep].to_csv(out / "population.csv.gz", index=False)

    rows = pop[keep].to_dict("records")
    for r in rows:
        if not np.isfinite(r["y_vote"] if r["y_vote"] is not None else np.nan):
            r["y_vote"] = None
    m_acc = write_dense(rows, out, "y_accepted", "mathse_accepted_verdict",
                        {"y_definition": "1 = the asker's accepted answer, 0 = a "
                                         "non-accepted answer on the same question"})
    m_vote = write_dense(rows, out, "y_vote", "mathse_vote_score",
                         {"y_definition": "1 = raw vote score strictly above the median "
                                          "answer score on its own question, 0 = strictly "
                                          "below; ties at the median dropped"})
    (out / "population_manifest.json").write_text(json.dumps({
        "source_dir": str(ind), "salt": SALT, "n_target": a.n_target,
        "n": int(len(pop)), "n_questions": int(pop["question_id"].nunique()),
        "sampling": f'whole questions in sha256("{SALT}" + question_id) order until '
                    f'>= {a.n_target} rows, restricted to questions carrying BOTH signals',
        "group_column": "question_id",
        "y_accepted_pos_rate": float(pop["y_accepted"].mean()),
        "y_vote_pos_rate": float(pop["y_vote"].mean(skipna=True)),
        "n_vote_defined": int(pop["y_vote"].notna().sum()),
        "dense_accepted": m_acc, "dense_vote": m_vote,
    }, indent=2))
    print("MATHSE_V2_VA_POPULATION_DONE")


if __name__ == "__main__":
    main()
