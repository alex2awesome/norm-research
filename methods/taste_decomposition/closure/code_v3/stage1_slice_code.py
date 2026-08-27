#!/usr/bin/env python3
"""Round-r VA_nl grouped-OOF inside FIT+MINE, and the disagreement slice the sealed
proposer fleet reads.  code_v3 adaptation of maps_hw_si/stage1_slice.py.

Differences from the maps version, all forced by this cell:
  * splits come from `splits.npz` (repository hash), and the mining slice M is the
    whole FIT+MINE side -- every row of this cell's population is dense-held-out;
  * the slice text is the structured SLICE CARD (cells.render_card), fetched from sk3
    on demand and cached, because the raw v3 document is up to 24,000 characters and
    a blind prefix would show a proposer nothing but diff;
  * N_SLICE = 40 (20 per direction) keeps the prompt near the 150 KB the N&C and CW
    campaigns ran at, given the much larger per-row card;
  * rows read in earlier rounds are excluded.

LABEL-BLIND: the emitted slice carries the card plus both models' percentile ranks
only, never `judgement`.

CPU only.  Usage: python stage1_slice_code.py --round 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# NOTE: maps_hw_si first, then HERE, so HERE wins position 0 -- otherwise `import cells`
# resolves to maps_hw_si/cells.py (whose load() takes a required cell arg) instead of this
# cell's shim. Caught by the readout dry run.
sys.path.insert(0, str(HERE.parent / "maps_hw_si"))
sys.path.insert(0, str(HERE))

import cells as C                                            # noqa: E402
import closure_core as L                                     # noqa: E402

N_SLICE = 40
TAG = "code_v3"


def current_blocks(d, rnd):
    """[V, A_base] plus the A-routed score columns accepted in earlier rounds."""
    blocks, tags = [d["V"], np.column_stack([d["A"], (~np.isnan(d["A"])).astype(float)])], \
                   ["V", "A_base(score+applied)"]
    # prior rounds whose A-routed criteria are already in the bank: none for the
    # decomposition round "d"; for round r it is "d" plus rounds 1..r-1. Any other label
    # (e.g. a dry-run tag) is treated as "no prior rounds" rather than crashing.
    try:
        seq = ["d"] + list(range(1, int(rnd)))
    except (TypeError, ValueError):
        seq = []
    for r in seq:
        f = HERE / f"{TAG}_r{r}_scores.npz"
        rt = HERE / f"{TAG}_r{r}_routing_final.json"
        if not (f.exists() and rt.exists()):
            continue
        z = np.load(f, allow_pickle=True)
        cids = [str(s) for s in z["a_ids"]]
        routing = json.loads(rt.read_text())
        a_ids = [x["blind_id"] for x in routing["final"] if x["final_route"] == "A"]
        idx = [cids.index(i) for i in a_ids if i in cids]
        if idx:
            X = z["X"][:, idx]
            # align to the population row order
            pos = {str(s): k for k, s in enumerate(z["row_ids"])}
            X = X[[pos[i] for i in d["ids"]]]
            blocks.append(np.column_stack([X, (~np.isnan(X)).astype(float)]))
            tags.append(f"A_round{r}({len(idx)})")
    return blocks, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="1")
    a = ap.parse_args()

    d = C.load()
    z = np.load(HERE / "splits.npz", allow_pickle=True)
    fit = z["fitmask"]
    y, groups = d["y"], d["groups"]

    blocks, tags = current_blocks(d, a.round)
    parts = []
    for M in blocks:
        keep, med = L.clean_fit(M[fit])
        if len(keep):
            parts.append(L.clean_apply(M[fit], keep, med))
    Xfit = np.column_stack(parts)
    gfit, yfit = groups[fit], y[fit]
    print(f"[{TAG} r{a.round}] FIT+MINE n={fit.sum()} blocks={tags} feats={Xfit.shape[1]}")

    lin_oof = L.linear_oof(Xfit, yfit, gfit)
    oofs = []
    for s in L.SEEDS:
        o, picks = L.gbm_oof(Xfit, yfit, gfit, seed=s)
        oofs.append(o)
        print(f"   VA_nl seed {s}: pooled OOF(fit+mine) {L.auc(yfit, o):.4f} picks={picks}",
              flush=True)
    oof_mean = np.mean(oofs, axis=0)

    wr = C.__dict__  # noqa: F841  (kept for symmetry; readouts live in cells_code)
    import cells_code as CCd
    internal = {
        "cell": TAG, "round": a.round, "blocks": tags,
        "n_fit_mine": int(fit.sum()), "n_features": int(Xfit.shape[1]),
        "VA_lin_pooled_oof_fitmine_NOT_A_RESIDUAL": L.auc(yfit, lin_oof),
        "VA_nl_pooled_oof_fitmine_per_seed_NOT_A_RESIDUAL": [L.auc(yfit, o) for o in oofs],
        "VA_nl_within_repo_fitmine": CCd.within_repo_auc(yfit, oof_mean, gfit)["nwtd"],
        "note": "pooled OOF is reported for continuity with other cells only; every "
                "decision on this cell is read within-repo",
    }
    np.savez(HERE / f"{TAG}_r{a.round}_oof_fitmine.npz",
             idx=np.where(fit)[0], va_nl_oof_mean=oof_mean,
             va_nl_oof_seeds=np.array(oofs), va_lin_oof=lin_oof)
    (HERE / f"{TAG}_r{a.round}_oof_fitmine.json").write_text(json.dumps(internal, indent=1))
    print(json.dumps(internal, indent=1))

    # ------------------------------------------------------- disagreement --
    seen = set()
    for r in (["d"] + list(range(1, int(a.round)))) if str(a.round) != "d" else []:
        p = HERE / f"{TAG}_r{r}_slice.json"
        if p.exists():
            seen |= {x["id"] for x in json.loads(p.read_text())}

    dense = d["dense"]
    oof_full = np.full(len(y), np.nan)
    oof_full[np.where(fit)[0]] = oof_mean
    mi = np.array([i for i in np.where(fit)[0] if d["ids"][i] not in seen])
    d_rank = pd.Series(dense[mi]).rank(pct=True).values
    v_rank = pd.Series(oof_full[mi]).rank(pct=True).values
    gap = d_rank - v_rank
    half = N_SLICE // 2
    picked = list(np.argsort(-gap)[:half]) + list(np.argsort(gap)[:half])

    ids = [str(d["ids"][int(mi[k])]) for k in picked]
    cards = C.fetch_texts(ids)
    trunc = d["meta"]["text_trunc"]          # TOKEN budget (cl100k), not characters
    slice_rows = []
    for k, card, rid in zip(picked, cards, ids):
        i = int(mi[k])
        # ITEM-VIEW ASSERTION (accumulated ruling): the card handed to a proposer must be
        # the card fetched for THIS row id, in this order.
        assert str(d["ids"][i]) == rid, f"item-view mismatch at {i}: {d['ids'][i]} != {rid}"
        slice_rows.append({
            "i": i, "id": rid,
            "direction": "dense_high_va_low" if gap[k] > 0 else "dense_low_va_high",
            "dense_prob": float(dense[i]), "dense_pct": float(d_rank[k]),
            "va_nl_oof": float(oof_full[i]), "va_nl_pct": float(v_rank[k]),
            "rank_gap": float(gap[k]), "text": C.tcut(card, trunc, "")})
    tk = [C.ntok(r["text"]) for r in slice_rows]
    print(f"slice card tokens: max {max(tk)} mean {sum(tk)//len(tk)} "
          f"total {sum(tk)} (budget {trunc}/card)")
    out = HERE / f"{TAG}_r{a.round}_slice.json"
    out.write_text(json.dumps(slice_rows, indent=1))
    print(f"slice: {len(slice_rows)} rows  |M|={len(mi)} (excluded {len(seen)} already read)"
          f"  median|gap|={np.median([abs(r['rank_gap']) for r in slice_rows]):.3f} "
          f"-> {out.name}")


if __name__ == "__main__":
    main()
