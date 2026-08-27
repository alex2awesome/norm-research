#!/usr/bin/env python3
"""Freeze the AoPS curation (same-approach) A/V population and build the
dense-standard arm on exactly that population.

Cell: math x curation. The registry carries V .706 (lexicons) and a grouped
Llama dense .769, but **no A-bank stack has ever existed for this cell** — this
build supplies the first one.

REUSE-FIRST (coordinator directive 2026-08-07): the AoPS cell already HAS a
grouped Llama-3.1-8B dense arm with row-level predictions
(`runs/aops_same_approach_dense_llama8b/preds_{eval,test}.csv`, verified to align
row-for-row with `split_full/{eval,test}.csv`: eval AUC .7768 / test .7686 /
pooled .7721 — the registry's .769). So this build trains NOTHING dense. The
A/V population is set to exactly those DENSE-HELD-OUT rows, which makes T
same-rows by construction and costs zero GPU.

Canonical source (the population behind the published dense .769):
  runs/aops_same_approach_dense_llama8b/data_full.csv.gz
  28,415 rows, columns text / judgement / problem, y-rate .6819, 3,027 problems.
  `text` = "Problem: <statement>\\n\\nSolution: <forum post body>" — the
  editorial/wiki solution is NEVER in the text (build_provenance.json).
  Grouping unit = `problem` (e.g. "1959_IMO#3").

Population: the union of the existing dense run's eval and test splits (5,690
rows / 606 problems), NOT a fresh stable-hash draw — reusing the frozen split is
what buys the same-rows T. The split itself was problem-grouped upstream.

V channel: datasets/math/aops/va/v_features.py, computed on the SOLUTION BODY
only (statement stripped) so statement length/LaTeX cannot leak into a
"solution style" feature.

Usage (CPU only, run on sk3):
  python3 build_va_population.py --run-dir <aops_same_approach_dense_llama8b> \
      --outdir <va dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SALT = "aops-va-v1|"
N_TARGET = 13000
SPLIT_MARKER = "\n\nSolution: "


def sha256(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def split_text(t: str):
    """-> (problem statement, solution body). Deterministic, maxsplit=1."""
    s = str(t)
    if SPLIT_MARKER in s:
        stmt, body = s.split(SPLIT_MARKER, 1)
        return stmt.removeprefix("Problem: ").strip(), body.strip()
    return "", s.removeprefix("Problem: ").strip()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="runs/aops_same_approach_dense_llama8b (holds split_full/ and preds_*)")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ("eval", "test"):
        d = pd.read_csv(run / "split_full" / f"{split}.csv")
        p = pd.read_csv(run / f"preds_{split}.csv")
        assert len(d) == len(p), f"{split}: split {len(d)} vs preds {len(p)}"
        assert (d["judgement"].values == p["judgement"].values).all(), \
            f"{split}: preds are not row-aligned with the split (judgement mismatch)"
        assert (d["problem"].values == p["problem"].values).all(), \
            f"{split}: preds are not row-aligned with the split (problem mismatch)"
        d = d.copy()
        d["dense_prob"] = p["dense_prob"].values
        d["dense_split"] = split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["problem"] = df["problem"].astype(str)
    parsed = [split_text(t) for t in df["text"]]
    df["statement"] = [q[0] for q in parsed]
    df["body"] = [q[1] for q in parsed]
    n_nomarker = int((df["statement"] == "").sum())
    df["row_id"] = [sha1(f"{p_}|{b}")[:20] for p_, b in zip(df["problem"], df["body"])]
    dup = int(df["row_id"].duplicated().sum())
    if dup:
        print(f"WARNING: {dup} duplicate row_ids (identical body inside one problem) -- "
              f"deduplicated, first occurrence kept")
        df = df[~df["row_id"].duplicated()].reset_index(drop=True)
    df["group"] = df["problem"]
    df = df.sort_values(["problem", "row_id"], kind="mergesort").reset_index(drop=True)

    from sklearn.metrics import roc_auc_score
    T_pool = float(roc_auc_score(df["judgement"], df["dense_prob"]))
    T_by_split = {s: float(roc_auc_score(g["judgement"], g["dense_prob"]))
                  for s, g in df.groupby("dense_split")}
    print(f"population n={len(df):,} problems={df['problem'].nunique():,} "
          f"pos={df['judgement'].mean():.4f} rows without the Solution marker: {n_nomarker}")
    print(f"body chars median={df['body'].str.len().median():.0f} "
          f"p95={df['body'].str.len().quantile(.95):.0f} max={df['body'].str.len().max()}")
    print(f"REUSED dense T: pooled {T_pool:.4f} | " +
          " ".join(f"{s} {v:.4f}" for s, v in sorted(T_by_split.items())))

    cols = ["row_id", "problem", "group", "statement", "body", "judgement",
            "dense_prob", "dense_split"]
    df[cols].to_csv(out / "population.csv.gz", index=False)

    man = {
        "cell": "aops_curation",
        "title": "AoPS curation (same-approach-as-editorial)",
        "source": str(run / "split_full") + " (eval + test = the dense-held-out rows)",
        "reuse": ("dense arm REUSED, not retrained: runs/aops_same_approach_dense_llama8b "
                  "preds_{eval,test}.csv, verified row-aligned with split_full "
                  "(judgement and problem match elementwise)"),
        "n": int(len(df)), "pos_rate": float(df["judgement"].mean()),
        "n_groups": int(df["problem"].nunique()),
        "group_column": "problem",
        "y_definition": ("1 = the forum solution takes substantially the same approach as "
                         "an editorial/wiki solution for that problem (LLM-judge match "
                         "label, not a preference); text never contains the editorial "
                         "solution"),
        "T_dense_pooled": T_pool, "T_dense_by_split": T_by_split,
        "T_provenance": ("grouped Llama-3.1-8B LoRA arm trained upstream on split_full/"
                         "train.csv; these are its held-out rows, so T is same-rows by "
                         "construction"),
        "duplicate_bodies_dropped": dup,
    }
    (out / "population_manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))
    print("AOPS_VA_POPULATION_DONE")


if __name__ == "__main__":
    main()
