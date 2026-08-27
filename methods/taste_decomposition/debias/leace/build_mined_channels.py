#!/usr/bin/env python3
"""Assemble the mined N&C Track-B nuisance map (round-5 cumulative, 45 channels
after FREEZE-ADDENDUM-3 parent retirement) over the full 9,521-row population,
aligned to the debias pilot's doc_id order, for the LEACE utility-distortion
frontier.

Column semantics: each channel is a B-routed, non-collapsed LLM-judged score
column from methods/taste_decomposition/closure/nc_responded/round{r}_scores.npz
(rounds 1..5), with retired MIXED parents dropped -- an exact local replica of
track_b_discount.load_b_blocks(upto=5).

Also records a per-channel TRAIN-rows-only AUC vs y, used ONLY to order the
nested frontier sets (top-1 / top-11 / top-22 / all-45); the ordering is
declared in the output and never feeds any quoted number.

Output: leace/build/mined_channels_nc45.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
CLOS = HERE.parents[1] / "closure/nc_responded"
NUIS = HERE.parents[0] / "build/nuisance.npz"
OUT = HERE / "build"
OUT.mkdir(exist_ok=True)


def load_b_blocks(upto=5):
    cols, meta = [], []
    for r in range(1, upto + 1):
        p = CLOS / f"round{r}_scores.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        routed = json.loads((CLOS / f"round{r}_routing_final.json").read_text())
        gate = json.loads((CLOS / f"round{r}_score_report.json").read_text())
        bmap = {c["id"]: c for c in routed["B"]}
        cids = [str(s) for s in z["crit_ids"]]
        cnames = [str(s) for s in z["crit_names"]]
        doc = np.array([str(s) for s in z["doc_id"]])
        for k, cid in enumerate(cids):
            if cid not in bmap or gate["per_criterion"][cid]["collapsed"]:
                continue
            cols.append(z["X"][:, k])
            meta.append({"round": r, "id": cid, "name": cnames[k],
                         "mixed": bool(bmap[cid].get("mixed"))})
    ret = CLOS / "retired_channels.json"
    dead = {x["uid"] for x in json.loads(ret.read_text())["retired"]} if ret.exists() else set()
    keep = [i for i, m in enumerate(meta) if f"r{m['round']}:{m['id']}" not in dead]
    cols = [cols[i] for i in keep]
    meta = [meta[i] for i in keep]
    return np.column_stack(cols), meta, doc


def main():
    X, meta, doc = load_b_blocks(5)
    nz = np.load(NUIS, allow_pickle=True)
    nz_ids = np.array([str(s) for s in nz["doc_id"]])
    order = {d: i for i, d in enumerate(doc)}
    pos = np.array([order[d] for d in nz_ids])          # closure rows -> pilot order
    X = X[pos]
    y = nz["y"].astype(int)
    split = np.array([str(s) for s in nz["split"]])
    tr = split == "train"

    # median-impute NaNs with TRAIN-rows medians (mirrors nc_closure_lib clean_fit/
    # clean_apply discipline: impute stats never read held-out rows)
    n_nan = int(np.isnan(X).sum())
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    ii = np.where(np.isnan(X))
    X[ii] = med[ii[1]]
    print(f"imputed {n_nan} NaN cells ({n_nan/X.size:.2%}) with train medians")

    auc_tr = np.array([roc_auc_score(y[tr], X[tr, j]) for j in range(X.shape[1])])
    dev = np.abs(auc_tr - 0.5)
    rank = np.argsort(-dev)                              # strongest first

    print(f"channels: {X.shape[1]} (expect 45), rows: {X.shape[0]}")
    for j in rank[:8]:
        print(f"  r{meta[j]['round']}:{meta[j]['id']} auc_tr={auc_tr[j]:.3f} {meta[j]['name'][:60]}")

    np.savez_compressed(
        OUT / "mined_channels_nc45.npz",
        doc_id=nz_ids, X=X.astype(np.float32),
        names=np.array([f"r{m['round']}:{m['id']}:{m['name'][:80]}" for m in meta], dtype=object),
        mixed=np.array([m["mixed"] for m in meta]),
        auc_train=auc_tr, rank_by_train_dev=rank,
        note="round-5 cumulative Track-B map, retirement applied; rank = |train AUC - .5| desc",
    )
    print(f"wrote {OUT/'mined_channels_nc45.npz'}")
    assert X.shape[1] == 45, f"expected 45 channels, got {X.shape[1]}"


if __name__ == "__main__":
    main()
