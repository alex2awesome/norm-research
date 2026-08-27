#!/usr/bin/env python3
"""STAGE 1 (per round): the disagreement slice the sealed fleet reads.

Frozen rule (prereg + FREEZE DECLARATION):
  * mining slice M = FIT+MINE rows that are ALSO dense-held-out.  For this cell
    EVERY population row is dense-held-out by construction, so M = FIT+MINE.
  * slice = top |dense percentile - VA_nl percentile| rows within M, balanced
    across the two directions.
  * MONITOR and TEST are never read.  LABEL-BLIND: the emitted file carries the
    writing prompt, the story and the two percentile ranks -- never `judgement`.

Slice size: 40 rows (20 per direction).  The prereg caps the read at 60; CW
stories are 1-2 orders of magnitude longer than peer-review abstracts, so 40
truncated stories already make a ~150 KB sealed prompt.  Recorded, not silent.

Usage: python stage1_slice.py --round 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import closure_lib_cw as C

HERE = Path(__file__).resolve().parent
N_PER_DIR = 20
STORY_CHARS_HEAD = 2400
STORY_CHARS_TAIL = 1200
MARK = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"


def trunc(story: str) -> str:
    s = str(story).strip()
    if len(s) <= STORY_CHARS_HEAD + STORY_CHARS_TAIL:
        return s
    return s[:STORY_CHARS_HEAD] + MARK + s[-STORY_CHARS_TAIL:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--state", default=None,
                    help="state npz to mine from (default round{r-1}_state.npz)")
    ap.add_argument("--pop", default="cw_population_with_splits.csv")
    ap.add_argument("--tag", default=None, help="output prefix (default round{r})")
    a = ap.parse_args()
    state = a.state or f"round{a.round-1}_state.npz"
    tag = a.tag or f"round{a.round}"
    z = np.load(HERE / state, allow_pickle=True)
    VA, y = z["VA"], z["y"]
    groups = np.array([str(g) for g in z["groups"]])
    split = np.array([str(s) for s in z["split"]])
    ids = np.array([str(s) for s in z["ids"]])
    T = z["T"]

    fm = split == "fit_mine"
    print(f"[slice r{a.round}] state={state} VA={VA.shape} |M|={fm.sum()}")

    oof_seeds = []
    for s in C.SEEDS:
        _, nl, _ = C.oof_within(VA[fm], y[fm], groups[fm], seed=s)
        oof_seeds.append(nl)
        print(f"  VA_nl seed {s}: OOF AUC(fit+mine) {C.auc(y[fm], nl):.4f}")
    oof = np.mean(oof_seeds, axis=0)
    print(f"  VA_nl seed-mean OOF AUC(fit+mine) {C.auc(y[fm], oof):.4f}")

    d_rank = pd.Series(T[fm]).rank(pct=True).values
    v_rank = pd.Series(oof).rank(pct=True).values
    gap = d_rank - v_rank
    mi = np.where(fm)[0]
    hi = np.argsort(-gap)[:N_PER_DIR]
    lo = np.argsort(gap)[:N_PER_DIR]

    pop = pd.read_csv(HERE / a.pop).set_index("id")
    rows = []
    for k in list(hi) + list(lo):
        i = int(mi[k])
        r = pop.loc[ids[i]]
        rows.append({
            "i": i, "id": ids[i],
            "direction": "dense_high_va_low" if gap[k] > 0 else "dense_low_va_high",
            "dense_pct": float(d_rank[k]), "va_nl_pct": float(v_rank[k]),
            "rank_gap": float(gap[k]),
            "prompt": str(r["prompt"])[:1200],
            "story": trunc(r["story"]),
        })
    out = HERE / f"{tag}_slice.json"
    out.write_text(json.dumps(rows, indent=1))
    np.savez_compressed(HERE / f"{tag}_oof_fitmine.npz",
                        idx=mi, oof=oof, oof_seeds=np.array(oof_seeds))
    print(f"  wrote {out.name}: {len(rows)} rows, "
          f"median |gap| {np.median([abs(r['rank_gap']) for r in rows]):.3f}, "
          f"total chars {sum(len(r['story']) for r in rows)//1024} KB")


if __name__ == "__main__":
    main()
