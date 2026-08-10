#!/usr/bin/env python3
"""Recovery-audit Q2(a): per-concept SINGLE-concept depletion refits.

For each of the 24 held-out concepts (3 replicates x 8), drop ONLY that concept's
column footprint from the round-0 A bank, refit VA_nl under the frozen spec, and
measure how much the DISAGREEMENT SLICE the proposers would read actually moves:

  * honest 1,244-row AUC drop vs the full round-0 baseline,
  * slice churn: of the 60 rows in the baseline round-0 slice, how many are
    replaced when this one concept is removed (pilot slice rule, 30/direction),
  * whether the rows that ENTER the slice are rows where the removed concept's
    column was non-null / informative (i.e., does the gap SURFACE in what the
    proposer reads).

CPU only, ~24 x 85 s.  Baseline slice comes from the stored round-0 OOF preds in
m3_depleted_preds.npz (no baseline refit needed; reproduction gate already passed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RMM = HERE.parent
CLOSURE = RMM.parent
sys.path.insert(0, str(RMM))
sys.path.insert(0, str(CLOSURE))
import closure_lib as L  # noqa: E402
from stage4_readout import build_blocks, fit_block  # noqa: E402

N_SLICE = 60


def slice_ids(oof_fitmine, dense, fitm, mining, n_slice=N_SLICE):
    """Pilot rule (stage1_disagreement.py): within-M percentile-rank gap."""
    n = len(dense)
    oof_full = np.full(n, np.nan)
    oof_full[np.where(fitm)[0]] = oof_fitmine
    mi = np.where(mining.astype(bool))[0]
    d_rank = pd.Series(dense[mi]).rank(pct=True).values
    v_rank = pd.Series(oof_full[mi]).rank(pct=True).values
    gap = d_rank - v_rank
    half = n_slice // 2
    picked = list(mi[np.argsort(-gap)][:half]) + list(mi[np.argsort(gap)][:half])
    return [int(i) for i in picked]


def main():
    t0 = time.time()
    cfg = json.loads((RMM / "m3_concepts.json").read_text())
    pop, split, dsplit, XA1, XB1, a1_ids, b1_ids, summary = build_blocks()
    y, nt, A, V = pop["y"], pop["ntitle"], pop["A"], pop["V"]
    _, _, _, mining = L.load_splits()
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])
    dense = pd.read_csv(CLOSURE / "peer_verdict_dense_preds.csv").set_index("i").loc[
        np.arange(len(y)), "dense_prob"].values

    z = np.load(RMM / "m3_depleted_preds.npz", allow_pickle=True)
    base_oof = z["oof_fitmine_baseline_round0"]
    base_slice = slice_ids(base_oof, dense, fitm, mining)
    base_set = set(base_slice)
    base_honest = float(z["nl_mon_baseline_round0"].shape and 0) or None  # unused
    # baseline honest AUC from stored preds
    va0 = np.full(len(y), np.nan)
    va0[fitm] = base_oof
    va0[monm] = z["nl_mon_baseline_round0"]
    base_honest = L.auc(y[held], va0[held])
    print(f"baseline honest AUC (stored preds) = {base_honest:.6f}", flush=True)
    print(f"baseline slice: {len(base_slice)} rows", flush=True)

    # collect the 24 held-out concepts with replicate provenance
    targets = []
    for rep, det in cfg["replicate_detail"].items():
        for c in det:
            targets.append({**c, "rep": rep})

    out = {"baseline_honest_auc": base_honest,
           "baseline_slice_ids": base_slice,
           "concepts": []}

    for t_i, c in enumerate(targets):
        name = c["concept"]
        drop = sorted(c["footprint_columns"])
        A_dep = np.delete(A, drop, axis=1)
        t = time.time()
        rd = fit_block([V, A_dep], fitm, monm, y, nt)
        vad = np.full(len(y), np.nan)
        vad[fitm] = rd["oof_nl_fitmine"]
        vad[monm] = rd["nl_mon"]
        honest = L.auc(y[held], vad[held])
        sl = slice_ids(rd["oof_nl_fitmine"], dense, fitm, mining)
        sl_set = set(sl)
        entered = sorted(sl_set - base_set)
        left = sorted(base_set - sl_set)

        # does the removed concept's column light up on the rows that entered?
        col = A[:, drop]                      # n x f footprint
        nonnull = ~np.isnan(col)
        any_nonnull = nonnull.any(axis=1)
        ent = np.array(entered, dtype=int)
        rec = {
            "rep": c["rep"], "concept": name, "stratum": c["stratum"],
            "alone_auc_fitmine": c["alone_auc_fitmine"],
            "n_cols": len(drop),
            "honest_auc_depleted": honest,
            "drop_honest_1244": base_honest - honest,
            "slice_overlap_with_baseline": len(base_set & sl_set),
            "n_rows_entered": len(entered),
            "n_rows_left": len(left),
            "rows_entered": entered,
            "concept_nonnull_frac_overall": float(any_nonnull.mean()),
            "concept_nonnull_frac_entered_rows": float(any_nonnull[ent].mean()) if len(ent) else None,
            "concept_nonnull_frac_baseline_slice": float(any_nonnull[np.array(base_slice)].mean()),
            "fit_seconds": round(time.time() - t, 1),
        }
        out["concepts"].append(rec)
        print(f"[{t_i+1:02d}/24] {name[:55]:55s} drop={rec['drop_honest_1244']:+.4f} "
              f"churn={rec['n_rows_entered']:2d}/60 ({rec['fit_seconds']:.0f}s)", flush=True)
        (HERE / "q2a_perconcept_depletion.json").write_text(json.dumps(out, indent=1))

    print(f"done in {time.time()-t0:.0f}s -> q2a_perconcept_depletion.json", flush=True)


if __name__ == "__main__":
    main()
