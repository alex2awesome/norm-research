#!/usr/bin/env python3
"""DECLARED-CHANNEL deconfounded readout for the unified-X cells (F2 shape,
patents precedent: channels are DECLARED, not yet fleet-mined — the mined
Track-B block arrives with the closure campaigns and SUPERSEDES this).

Arms on the dense-held-out rows, grouped 5-fold OOF (logistic + HistGB mean —
the fused-stack convention extended):
  (b) NUIS alone      [declared channels]
  (c) VA + NUIS
  (d) VA + NUIS + T
PRIMARY = (d)-(c), grouped bootstrap (2,000), the deconfounded dense residual.
Also (c)-(b): what the text instruments add BEYOND the declared channels — on
the curated cells this is the "text beyond the vote score" number.

Declared channels: answer_score (community signal; curated cells' dominant
covariate), char length, n_answers_on_q (so_accepted: position + n_answers_q
from the V6 population; score from the population Score column).
CPU. Usage: python3 unified_deconf_declared.py --cell so_bounty
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
NR = HERE.parents[1]
RESULTS = HERE / "results"


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


F2 = _mod(HERE / "fusion/f2_deconf.py", "udc_f2")   # for gboot only

CELLS = {
    "mathse_bounty": dict(
        dense=NR / "datasets/math-se/mathse_bounty/dense_standard_mathse_bounty",
        oof=RESULTS / "mathse_bounty_va_oof.npz",
        pop=NR / "datasets/math-se/mathse_bounty/population.csv.gz",
        chans=["answer_score", "n_answers_on_q"], text_col="text"),
    "so_bounty": dict(
        dense=NR / "datasets/stackoverflow-votes/so_bounty/dense_standard_so_bounty",
        oof=RESULTS / "so_bounty_va_oof.npz",
        pop=NR / "datasets/stackoverflow-votes/so_bounty/population.csv.gz",
        chans=["answer_score", "n_answers_on_q"], text_col="text"),
    "so_accepted": dict(
        dense=NR / "datasets/stackoverflow-votes/so_accepted/dense_standard_so_accepted_qtrunc",
        oof=RESULTS / "so_accepted_va_oof.npz",
        pop=NR / "datasets/stackoverflow-votes/va/population.csv.gz",
        chans=["Score", "position", "n_answers_q"], text_col="body"),
}
SEEDS = (42, 1, 2)


def grouped_oof(X, y, g):
    """Mean of logistic + 3-seed HistGB, grouped 5-fold OOF (fused-stack convention
    extended with the nonlinear leg so channel interactions can be absorbed)."""
    oofs = []
    lo = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, groups=g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        lo[te] = clf.predict_proba(X[te])[:, 1]
    oofs.append(lo)
    for s in (0, 1, 2):
        go = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, groups=g):
            clf = HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.06,
                                                 random_state=s)
            clf.fit(X[tr], y[tr])
            go[te] = clf.predict_proba(X[te])[:, 1]
        oofs.append(go)
    return np.mean(oofs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    a = ap.parse_args()
    cfg = CELLS[a.cell]

    z = np.load(cfg["oof"], allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    pos = {r: i for i, r in enumerate(ids)}
    va = z["VA_nl"].astype(float)
    y_all = z["y"].astype(int)
    grp_all = np.array([str(g) for g in z["groups"]], dtype=object)

    popdf = pd.read_csv(cfg["pop"]).set_index("row_id")
    popdf.index = popdf.index.astype(str)

    rows = []
    for leg in ("eval", "test"):
        sp = pd.read_csv(cfg["dense"] / "split" / f"{leg}.csv")
        per_seed = []
        for s in SEEDS:
            p = pd.read_csv(cfg["dense"] / f"rm_out_seed{s}" / f"preds_{leg}.csv")
            assert (p["judgement"].values == sp["judgement"].values).all()
            per_seed.append(p["prob"].values.astype(float))
        dm = np.mean(per_seed, axis=0)
        for rid, dp in zip(sp["row_id"].astype(str), dm):
            rows.append((rid, dp))
    hit = [(r, d) for r, d in rows if r in pos]
    idx = np.array([pos[r] for r, _ in hit])
    dense = np.array([d for _, d in hit])
    y, g = y_all[idx], grp_all[idx]
    rids = [r for r, _ in hit]

    chan_cols = []
    for c in cfg["chans"]:
        chan_cols.append(popdf.loc[rids, c].values.astype(float))
    charlen = popdf.loc[rids, cfg["text_col"]].astype(str).str.len().values.astype(float)
    NUIS = np.column_stack(chan_cols + [charlen])
    chan_names = cfg["chans"] + ["char_len"]

    print(f"[{a.cell}] n_heldout={len(y)} channels={chan_names}", flush=True)
    ob = grouped_oof(NUIS, y, g)
    oc = grouped_oof(np.column_stack([va[idx].reshape(-1, 1), NUIS]), y, g)
    od = grouped_oof(np.column_stack([va[idx].reshape(-1, 1), NUIS, dense.reshape(-1, 1)]), y, g)

    prim = F2.gboot(y, od, oc, g, n_boot=2000)
    text_beyond = F2.gboot(y, oc, ob, g, n_boot=2000)
    out = {
        "cell": a.cell, "n_heldout": int(len(y)),
        "declared_channels": chan_names,
        "status": "DECLARED-CHANNEL deconf (patents precedent) — mined Track-B "
                  "block pending the closure campaigns; supersedes nothing, is "
                  "superseded by the fleet-mined block when it exists",
        "b_NUIS_alone": float(roc_auc_score(y, ob)),
        "c_VA_plus_NUIS": float(roc_auc_score(y, oc)),
        "d_plus_T": float(roc_auc_score(y, od)),
        "PRIMARY_d_minus_c": prim,
        "SECONDARY_text_beyond_channels_c_minus_b": text_beyond,
    }
    (RESULTS / f"{a.cell}_deconf_declared.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "status"}, indent=1, default=float))
    print(f"{a.cell.upper()}_DECONF_DONE", flush=True)


if __name__ == "__main__":
    main()
