#!/usr/bin/env python3
"""U3/U4 — Layer-1 ladders for the two BOUNTY (curated) cells.

Same frozen estimators as every scaleupC cell (imported from layer1_gemma_cells /
scaleupC_layer1 / so_votes_layer1 helpers).  Loads the cell's own scored shards
(outputs/va_gemma_banks_<cell>/).  Collapse gate per standing ruling.

Curated-cell covariate line (replaces the vote cells' position line):
  * answer_score alone, pooled AND within-question — the community-overlap
    channel (the award-mode audit says manual winners are top-scored 64%; this
    line quantifies what a votes-only instrument buys on the curated y).
  * charlen alone (the length family).
T attaches later (sk1 chains).  Usage: python3 bounty_layer1.py --cell mathse_bounty
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
import layer1_gemma_cells as L  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402
import so_votes_layer1 as SV  # noqa: E402

REPO = SC.REPO
RESULTS = SC.RESULTS_DIR
POPS = {"mathse_bounty": REPO / "datasets/math-se/mathse_bounty/population.csv.gz",
        "so_bounty": REPO / "datasets/stackoverflow-votes/so_bounty/population.csv.gz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(POPS))
    a = ap.parse_args()
    cell = a.cell
    out = REPO / f"outputs/va_gemma_banks_{cell}"

    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(cell, out=out)
    y = np.array(meta["ys"]["bounty"], dtype=float)
    assert np.isfinite(y).all()
    y = y.astype(int)
    a_names = list(meta["a_names"])

    shares = np.array([SV.modal_share(A[:, c]) for c in range(A.shape[1])])
    keep_c = shares <= SV.COLLAPSE_MODAL_MAX
    dropped = [nm for nm, k in zip(a_names, keep_c) if not k]
    A = A[:, keep_c]
    a_names = [nm for nm, k in zip(a_names, keep_c) if k]
    print(f"[{cell}] collapse gate dropped {len(dropped)} -> A {A.shape}", flush=True)

    pop = pd.read_csv(POPS[cell]).set_index("row_id")
    pop.index = pop.index.astype(str)
    order = [str(i) for i in ids]
    ascore = pop.loc[order, "answer_score"].values.astype(float)
    charlen = pop.loc[order, "text"].astype(str).str.len().values.astype(float)

    folds = L.outer_folds(len(y), groups, n_splits=5)
    res = {"cell": f"{cell} (CURATED: manual bounty award, within-question)",
           "n": int(len(y)), "n_questions": int(len(set(groups))),
           "pos_rate": float(y.mean()),
           "collapse_gate": {"dropped": dropped, "kept": int(A.shape[1])}}

    res["community_overlap_line"] = {
        "answer_score_alone_pooled": float(roc_auc_score(y, ascore)),
        "answer_score_within_question": SV.within_group_auc(y, groups, ascore)}
    res["charlen_alone_auc"] = float(roc_auc_score(y, charlen))
    res["charlen_within_question"] = SV.within_group_auc(y, groups, charlen)
    qmean = pd.Series(y).groupby(pd.Series(groups)).transform("mean").values
    res["question_identity_alone_auc"] = float(roc_auc_score(y, qmean))

    mats = {"V": V, "A": A, "VA": np.column_stack([V, A])}
    table, preds = {}, {}
    for k, M in mats.items():
        lin_auc, lin_oof = L.linear_oof_family1(M, y, groups, folds)
        gbm_oofs = [L.gbm_oof_family1(M, y, groups, folds, seed=s)[1]
                    for s in L.GBM_SEEDS]
        nl = np.mean(gbm_oofs, axis=0)
        table[k] = {"linear": float(lin_auc),
                    "nl_mean3": float(roc_auc_score(y, nl)),
                    "nl_within_question": SV.within_group_auc(y, groups, nl)}
        preds[k] = nl
        print(f"[{cell}:{k}] linear {lin_auc:.4f} | nl {table[k]['nl_mean3']:.4f} "
              f"| within-q {table[k]['nl_within_question']['pair_weighted']:.4f}",
              flush=True)
    res["ladder"] = table

    np.savez_compressed(RESULTS / f"{cell}_va_oof.npz",
                        ids=np.array(order, dtype=object), y=y,
                        groups=np.array([str(g) for g in groups], dtype=object),
                        V_nl=preds["V"], A_nl=preds["A"], VA_nl=preds["VA"])
    (RESULTS / f"{cell}_ledger.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{cell.upper()}_L1_DONE", flush=True)


if __name__ == "__main__":
    main()
