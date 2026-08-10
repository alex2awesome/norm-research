#!/usr/bin/env python3
"""M3 step 2 -- depleted-bank refits + regenerated disagreement slices.

For each of the 3 holdout replicates (8 concepts each, from m3_concepts.json):
  * drop the concepts' ENTIRE column footprint from A (name matches + any
    bit-identical duplicate column),
  * recompute VA_nl under the frozen Layer-1 spec (HistGB grid, grouped OOF inside
    FIT+MINE, seeds {0,1,2} mean; refit-and-predict on MONITOR),
  * record the AUC drop vs the full-bank round-0 baseline on MONITOR (all rows and
    the honest dense-held-out population), with group-level paired bootstrap CIs,
  * regenerate the disagreement slice against the DEPLETED stack using the pilot's
    rule (top |dense rank - VA_nl rank| inside the mining slice, 30 per direction).

Also emits a 4th slice: the FULL round-4 bank (all 4 mined rounds admitted), i.e.
the slice a 5th pilot round would have read.  That one is not a depletion test --
it is the live "is there anything left" probe for the fleet.

LABEL-BLIND: emitted slices carry text, dense prob and VA_nl prediction only.
CPU only.  Usage: python m3_deplete.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
sys.path.insert(0, str(CLOSURE))
import closure_lib as L  # noqa: E402
from stage4_readout import build_blocks, fit_block, group_boot_ci  # noqa: E402
from stage4_round4 import load_round_blocks  # noqa: E402

N_SLICE = 60


def slice_from(oof_fitmine, dense, fitm, mining, texts, n_slice=N_SLICE):
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
    rows = []
    for i in picked:
        k = int(np.where(mi == i)[0][0])
        rows.append({
            "i": int(i),
            "direction": "dense_high_va_low" if gap[k] > 0 else "dense_low_va_high",
            "dense_prob": float(dense[i]), "dense_pct": float(d_rank[k]),
            "va_nl_oof": float(oof_full[i]), "va_nl_pct": float(v_rank[k]),
            "rank_gap": float(gap[k]), "text": texts[i],
        })
    return rows


def main():
    t0 = time.time()
    cfg = json.loads((HERE / "m3_concepts.json").read_text())
    pop, split, dsplit, XA1, XB1, a1_ids, b1_ids, summary = build_blocks()
    XA2, _, _, _, _ = load_round_blocks(2)
    XA3, _, _, _, _ = load_round_blocks(3)
    XA4, _, _, _, _ = load_round_blocks(4)

    y, nt, A, V = pop["y"], pop["ntitle"], pop["A"], pop["V"]
    _, _, _, mining = L.load_splits()
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])
    dense = pd.read_csv(CLOSURE / "peer_verdict_dense_preds.csv").set_index("i").loc[
        np.arange(len(y)), "dense_prob"].values

    ymon, gmon = y[monm], nt[monm]
    same_in_mon = held[monm]

    def readout(r):
        va = np.full(len(y), np.nan)
        va[fitm] = r["oof_nl_fitmine"]
        va[monm] = r["nl_mon"]
        return {
            "n_features": r["n_features"],
            "VA_nl_MONITOR_all": L.auc(ymon, r["nl_mon"]),
            "VA_lin_MONITOR_all": L.auc(ymon, r["lin_mon"]),
            "VA_nl_MONITOR_samerows": L.auc(ymon[same_in_mon], r["nl_mon"][same_in_mon]),
            "VA_nl_honest_level_heldout1244": L.auc(y[held], va[held]),
            "VA_nl_OOF_fitmine": L.auc(y[fitm], r["oof_nl_fitmine"]),
            "VA_nl_MONITOR_all_per_seed": [L.auc(ymon, p) for p in r["nl_mon_seeds"]],
        }, va

    results, fits = {}, {}

    print("=== baseline: full round-0 bank (V + A) ===", flush=True)
    t = time.time()
    r0 = fit_block([V, A], fitm, monm, y, nt)
    rep0, va0 = readout(r0)
    results["baseline_round0"] = rep0
    fits["baseline_round0"] = r0
    print(json.dumps(rep0), f"({time.time()-t:.0f}s)", flush=True)

    print("\n=== full round-4 bank (V + A + all 4 mined rounds) ===", flush=True)
    t = time.time()
    r4 = fit_block([V, A, XA1, XA2, XA3, XA4], fitm, monm, y, nt)
    rep4, va4 = readout(r4)
    results["full_round4"] = rep4
    fits["full_round4"] = r4
    print(json.dumps(rep4), f"({time.time()-t:.0f}s)", flush=True)

    # slice a 5th pilot round would read
    s5 = slice_from(r4["oof_nl_fitmine"], dense, fitm, mining, pop["texts"])
    (HERE / "slice_round5_fullbank.json").write_text(json.dumps(s5, indent=1))

    # ------------------------------------------------------------- replicates --
    footprints = cfg["concept_footprints"]
    for rep, concepts in cfg["replicates"].items():
        print(f"\n=== {rep}: depleting {len(concepts)} concepts ===", flush=True)
        drop = sorted({j for c in concepts for j in footprints[c]})
        A_dep = np.delete(A, drop, axis=1)
        print(f"  dropped {len(drop)} A columns -> A_dep {A_dep.shape[1]} cols", flush=True)
        t = time.time()
        rd = fit_block([V, A_dep], fitm, monm, y, nt)
        repd, vad = readout(rd)
        repd["n_A_columns_dropped"] = len(drop)
        repd["concepts_held_out"] = concepts
        repd["drop_MONITOR_all"] = rep0["VA_nl_MONITOR_all"] - repd["VA_nl_MONITOR_all"]
        repd["drop_honest_1244"] = rep0["VA_nl_honest_level_heldout1244"] - repd["VA_nl_honest_level_heldout1244"]
        repd["drop_MONITOR_all_ci"] = group_boot_ci(ymon, r0["nl_mon"], rd["nl_mon"], gmon)
        repd["drop_honest_1244_ci"] = group_boot_ci(y[held], va0[held], vad[held], nt[held])
        results[rep] = repd
        fits[rep] = rd
        print(json.dumps({k: v for k, v in repd.items() if k != "concepts_held_out"}),
              f"({time.time()-t:.0f}s)", flush=True)

        sl = slice_from(rd["oof_nl_fitmine"], dense, fitm, mining, pop["texts"])
        (HERE / f"slice_{rep}.json").write_text(json.dumps(sl, indent=1))
        base_ids = {r["i"] for r in json.loads((CLOSURE / "round1_disagreement_slice.json").read_text())}
        repd["slice_overlap_with_pilot_round1"] = len(base_ids & {r["i"] for r in sl})

    np.savez_compressed(
        HERE / "m3_depleted_preds.npz",
        **{f"va_{k}": (lambda r: np.concatenate([[np.nan]]))(v) for k, v in {}.items()},
        y=y, ntitle=nt.astype(str), dense=dense, held=held, monitor=monm, fit_mine=fitm,
        **{f"nl_mon_{k}": v["nl_mon"] for k, v in fits.items()},
        **{f"oof_fitmine_{k}": v["oof_nl_fitmine"] for k, v in fits.items()},
    )
    (HERE / "m3_depletion.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote m3_depletion.json ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
