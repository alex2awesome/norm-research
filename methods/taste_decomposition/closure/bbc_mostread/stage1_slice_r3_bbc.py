#!/usr/bin/env python3
"""BBC most-read closure — STAGE 1 for ROUND 3 (2026-08-13): identical to
stage1_slice_bbc.py except (a) the articulated side is the ROUND-2 bank
[V, A, A1(16), A2(14)] — mining must target what the CURRENT instrument misses —
and (b) rows already shown in slice_r1.json OR slice_r2.json are BANNED (prior-round ban,
mirrors the audit probe discipline).

Original docstring follows.


Prereg step 1: "Disagreement slice: top-|dense_prob - VA_nl_OOF| rows within M
(up to 60 read)." M = FIT+MINE and dense-held-out, so the dense score on every
slice row is honest (never an in-sample train prediction).

LABEL BLINDNESS IS THE POINT OF THIS FILE. The rendered cards carry the headline,
the dense score and the articulated instrument's prediction, and NOTHING ELSE.
They never carry y, the most-read rank, the capture day, the bank's per-criterion
scores, or any provenance a proposer could use to reverse-engineer the outcome.
The renderer asserts this before writing.

Ranking is on the PERCENTILE difference, not the raw probability difference: the
dense arm and the articulated instrument are on different scales, and a raw-scale
difference would just rank by dense confidence.

  python3 stage1_slice_bbc.py --round 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "methods/taste_decomposition"))
import layer1_gemma_cells as L  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402

CELL = "bbc_mostread"
VA_DIR = REPO / "datasets/bbc-mostread/va"
BANK_OUT = REPO / "outputs/va_gemma_banks_bbc_mostread"
DENSE = VA_DIR / "dense_standard_bbc_mostread"
SEEDS = (42, 1, 2)
N_SLICE = 60

FORBIDDEN = ("judgement", "most_read", "rank", "y=", "label", "positive", "negative")


def load_dense(pop):
    """Order-join, re-proven here (preds carry no row_id) — see round0_bbc.py."""
    ids, per_seed = [], {s: [] for s in SEEDS}
    for leg in ("eval", "test"):
        sp = pd.read_csv(DENSE / "split" / f"{leg}.csv")
        for s in SEEDS:
            p = pd.read_csv(DENSE / f"rm_out_seed{s}" / f"preds_{leg}.csv")
            assert len(p) == len(sp)
            assert (p["judgement"].values == sp["judgement"].values).all()
            assert (p["group"].astype(str).values == sp["group"].astype(str).values).all()
            per_seed[s].append(p["prob"].values.astype(float))
        ids += sp["row_id"].astype(str).tolist()
    return ids, {s: np.concatenate(v) for s, v in per_seed.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n", type=int, default=N_SLICE)
    a = ap.parse_args()

    pop = pd.read_csv(VA_DIR / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    splits = pd.read_csv(HERE / "splits.csv.gz")
    splits["row_id"] = splits.row_id.astype(str)
    pop = pop.merge(splits[["row_id", "split3", "is_M"]], on="row_id", how="left")

    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(CELL, out=BANK_OUT)
    idl = [str(i) for i in ids]
    idx = {r: i for i, r in enumerate(idl)}
    byid = pop.set_index("row_id")
    y = byid.loc[idl, "judgement"].values.astype(int)
    g = byid.loc[idl, "group"].astype(str).values
    s3 = byid.loc[idl, "split3"].values
    isM = byid.loc[idl, "is_M"].values.astype(bool)

    fm = s3 == "FITMINE"
    # ROUND-1 BANK STATE: append the 16 A-routed mined criteria
    import json as _json
    blocks = [V, A]
    expected = {1: 16, 2: 14}
    for rr in (1, 2):
        zz = np.load(HERE / f"bbc_mostread_r{rr}_scores.npz", allow_pickle=True)
        assert [str(x) for x in zz["row_id"]] == idl, f"r{rr} scores misaligned"
        routing = _json.loads((HERE / f"bbc_mostread_r{rr}_routing_final.json").read_text())
        track = {c["blind_id"]: c["final_route"] for c in routing["final"]}  # ARBITER-FINAL
        cidsr = [str(c) for c in zz["crit_ids"]]
        iA = [i for i, c in enumerate(cidsr) if track.get(c) == "A"]
        assert len(iA) == expected[rr], f"r{rr}: expected {expected[rr]} A-routed, got {len(iA)}"
        blocks.append(zz["X"][:, iA])
    X = np.column_stack(blocks)
    # PRIOR-ROUND BAN (r1 + r2)
    banned = set()
    for rr in (1, 2):
        banned |= {r["row_id"] for r in _json.loads((HERE / f"slice_r{rr}.json").read_text())["rows"]}

    # VA_nl OOF *within FIT+MINE* — the honest articulated prediction on mining rows
    folds = L.outer_folds(int(fm.sum()), g[fm], n_splits=5)
    oofs = [L.gbm_oof_family1(X[fm], y[fm], g[fm], folds, s)["oof"] for s in (0, 1, 2)]
    va_fm = np.mean(oofs, axis=0)
    va = np.full(len(idl), np.nan)
    va[fm] = va_fm

    d_ids, d_per_seed = load_dense(pop)
    dpos = {r: i for i, r in enumerate(d_ids)}
    dense = np.full(len(idl), np.nan)
    dmean = np.mean([d_per_seed[s] for s in SEEDS], axis=0)
    for i, r in enumerate(idl):
        if r in dpos:
            dense[i] = dmean[dpos[r]]

    not_banned = np.array([r not in banned for r in idl])
    sel = isM & np.isfinite(va) & np.isfinite(dense) & not_banned
    print(f"[stage1] M rows with both scores: {int(sel.sum())}")

    # percentile scale on the mining slice itself
    dp = rankdata(dense[sel]) / sel.sum()
    vp = rankdata(va[sel]) / sel.sum()
    gap = np.abs(dp - vp)
    order = np.argsort(-gap)[: a.n]
    midx = np.flatnonzero(sel)[order]

    cards, rows = [], []
    for rank_i, i in enumerate(midx, 1):
        text = str(byid.loc[idl[i], "text"])
        headline = re.sub(r"^HEADLINE:\s*", "", text).strip()
        j = int(np.flatnonzero(np.flatnonzero(sel) == i)[0])
        card = (f"[{rank_i:02d}] HEADLINE: {headline}\n"
                f"     dense percentile {dp[j]:.3f} | articulated percentile {vp[j]:.3f}"
                f" | disagreement {gap[j]:.3f}")
        low = card.lower()
        for bad in FORBIDDEN:
            assert bad not in low.replace("disagreement", ""), \
                f"label leak in card: {bad!r}"
        cards.append(card)
        rows.append({"row_id": idl[i], "dense_pct": float(dp[j]),
                     "va_pct": float(vp[j]), "gap": float(gap[j]),
                     "dense_above": bool(dp[j] > vp[j])})

    out = {"cell": CELL, "round": a.round, "n_M": int(sel.sum()),
           "n_slice": len(rows),
           "rule": "top |dense percentile - VA_nl OOF percentile| within M "
                   "(M = FIT+MINE and dense-held-out)",
           "blindness": "cards carry headline + both percentiles ONLY; no y, no "
                        "rank, no day, no per-criterion scores; asserted in code",
           "direction_counts": {
               "dense_higher_than_bank": int(sum(r["dense_above"] for r in rows)),
               "bank_higher_than_dense": int(sum(not r["dense_above"] for r in rows))},
           "rows": rows}
    (HERE / f"slice_r{a.round}.json").write_text(json.dumps(out, indent=1))
    (HERE / f"slice_r{a.round}_cards.txt").write_text("\n".join(cards))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print(f"\nwrote slice_r{a.round}.json / slice_r{a.round}_cards.txt")


if __name__ == "__main__":
    main()
