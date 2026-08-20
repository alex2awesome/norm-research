#!/usr/bin/env python3
"""VAT V3 (fused arm) for the unified-X cells — §11-style grouped-OOF logistic
stack of [VA_nl OOF, dense T seed-mean] on the dense arm's held-out rows
(eval+test), exactly the homepage_fused_stack.py pattern.  VAT column =
max-of-variants (here: the stack; a V3 criteria-in-prompt arm is NOT built —
the design was answered cross-cell as never-beats-stacking).

Per cell: order-join asserted per split/seed; parents (VA_nl, T) reported on the
same held-out rows; fused = 5-fold GroupKFold logistic on [VA, T].
Usage (on the box holding the dense dirs): python3 unified_fused_stack.py --cell so_bounty
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
NR = HERE.parents[1]
RESULTS = HERE / "results"

CELLS = {
    "mathse_bounty": dict(
        dense=NR / "datasets/math-se/mathse_bounty/dense_standard_mathse_bounty",
        oof=RESULTS / "mathse_bounty_va_oof.npz"),
    "so_bounty": dict(
        dense=NR / "datasets/stackoverflow-votes/so_bounty/dense_standard_so_bounty",
        oof=RESULTS / "so_bounty_va_oof.npz"),
    "so_accepted": dict(
        dense=NR / "datasets/stackoverflow-votes/so_accepted/dense_standard_so_accepted_qtrunc",
        oof=RESULTS / "so_accepted_va_oof.npz"),
    "pr_transition": dict(
        dense=NR / "datasets/code-review/pr_test_execution/dense_standard_pr_transition",
        oof=RESULTS / "pr_transition_va_oof.npz"),
    "kindle_scout": dict(
        dense=NR / "datasets/creative-writing/kindle_scout_cell/dense_standard_kindle_scout",
        oof=RESULTS / "kindle_scout_va_oof.npz"),
    "jokes_removal_v2": dict(
        dense=NR / "datasets/humor/reddit_jokes/dense_standard_jokes_removal_v2",
        oof=RESULTS / "jokes_removal_v2_va_oof.npz"),
}
SEEDS = (42, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    a = ap.parse_args()
    cfg = CELLS[a.cell]

    z = np.load(cfg["oof"], allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    pos = {r: i for i, r in enumerate(ids)}
    assert len(pos) == len(ids), "duplicate ids in OOF matrix"
    va = z["VA_nl"].astype(float)
    y_all = z["y"].astype(int)
    grp_all = np.array([str(g) for g in z["groups"]], dtype=object)

    rows = []
    for leg in ("eval", "test"):
        sp = pd.read_csv(cfg["dense"] / "split" / f"{leg}.csv")
        per_seed = []
        for s in SEEDS:
            p = pd.read_csv(cfg["dense"] / f"rm_out_seed{s}" / f"preds_{leg}.csv")
            assert len(p) == len(sp) and (p["judgement"].values == sp["judgement"].values).all(), \
                f"order-join fail {leg} seed{s}"
            per_seed.append(p["prob"].values.astype(float))
        dm = np.mean(per_seed, axis=0)
        for rid, dp, yy in zip(sp["row_id"].astype(str), dm, sp["judgement"].astype(int)):
            rows.append((rid, dp, yy, leg))

    hit = [(r, d_, y_, l_) for r, d_, y_, l_ in rows if r in pos]
    print(f"[{a.cell}] dense-held-out rows joined to OOF: {len(hit)}/{len(rows)}")
    idx = np.array([pos[r] for r, _, _, _ in hit])
    dense = np.array([d_ for _, d_, _, _ in hit])
    y = y_all[idx]
    assert (y == np.array([y_ for _, _, y_, _ in hit])).all(), "y mismatch on join"
    g = grp_all[idx]
    leg = np.array([l_ for _, _, _, l_ in hit])

    S = np.column_stack([va[idx], dense])
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(S, groups=g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(S[tr], y[tr])
        oof[te] = clf.predict_proba(S[te])[:, 1]

    out = {
        "cell": a.cell, "n_heldout": int(len(y)),
        "fused_stack_VA_T": float(roc_auc_score(y, oof)),
        "fused_eval": float(roc_auc_score(y[leg == "eval"], oof[leg == "eval"])),
        "fused_test": float(roc_auc_score(y[leg == "test"], oof[leg == "test"])),
        "VA_nl_same_rows": float(roc_auc_score(y, va[idx])),
        "dense_T_same_rows": float(roc_auc_score(y, dense)),
        "note": "grouped-OOF logistic stack [VA_nl OOF, dense seed-mean] on the dense "
                "arm's held-out rows; VAT column = this stack (max-of-variants; no "
                "criteria-in-prompt arm by the cross-cell V3 ruling)",
    }
    (RESULTS / f"{a.cell}_fused_stack.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"{a.cell.upper()}_FUSED_DONE")


if __name__ == "__main__":
    main()
