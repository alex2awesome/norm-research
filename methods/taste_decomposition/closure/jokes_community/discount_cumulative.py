#!/usr/bin/env python3
"""CUMULATIVE Track-B discount + matched sampling, for the full closure campaign.

`readout.py` (inherited from the map-focused batch) discounts with the CURRENT
round's B channels only, because a two-round map is a per-round object.  A closure
campaign's declared-nuisance set is CUMULATIVE -- the freeze says B-routed criteria
"join the declared-nuisance set" -- so this script re-reads every B-routed,
non-collapsed channel of rounds 1..r (plus the decomposition round's surface
components) and reports:

  1. spurious-alone AUC of the joint cumulative nuisance model (linear + HistGB,
     grouped-OOF inside FIT+MINE, refit-and-predict on MONITOR);
  2. MATCHED SAMPLING (freeze: "matched sampling once spurious-alone > .65").  Each
     positive is paired with its nearest unused negative on the joint-B percentile
     within a caliper; the readout is the fraction of matched pairs each instrument
     orders correctly.  This does not degrade as the nuisance set grows, unlike
     decile stratification, which is why the freeze switches to it above .65.
     Implementation ported verbatim from cw_community/round_readout.py.
  3. the decile-stratified readout as well, for continuity with the maps batch;
  4. the MIXED sensitivity band (ALL vs STRICT), FREEZE ADDENDUM 2;
  5. the stratification-free stacked increment over the cumulative nuisance set.

Retired parents (FREEZE ADDENDUM 3): a MIXED channel that has been decomposed is
dropped from the nuisance set once its components are scored -- read from
`<cell>_retired_channels.json` -- and the retirement is recorded, never deleted.

CPU only.  Usage: python discount_cumulative.py --cell peer_revealed --round 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L
import stage1_slice as S1
from readout import stack_oof

HERE = Path(__file__).resolve().parent
MATCH_TRIGGER = 0.65
CALIPER = 0.02


def matched_pair_auc(y, score, control, caliper=CALIPER, seed=0):
    """Pair each positive with the closest-control negative (percentile caliper),
    then report the fraction of matched pairs the score orders correctly."""
    y = np.asarray(y)
    r = np.argsort(np.argsort(control)) / max(1, len(control) - 1)
    P = np.where(y == 1)[0]
    N = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    P = P[rng.permutation(len(P))]
    used = np.zeros(len(N), bool)
    rN = r[N]
    conc, ties, n = 0.0, 0, 0
    for i in P:
        dd = np.abs(rN - r[i])
        dd[used] = np.inf
        j = int(np.argmin(dd))
        if not np.isfinite(dd[j]) or dd[j] > caliper:
            continue
        used[j] = True
        n += 1
        a, b = score[i], score[N[j]]
        if a > b:
            conc += 1
        elif a == b:
            conc += 0.5
            ties += 1
    return (float(conc / n) if n else float("nan"),
            {"n_pairs": int(n), "n_ties": int(ties), "caliper": caliper})


def round_ids(upto):
    """Decomposition round 'd' first (if present), then 1..upto."""
    return ["d"] + [str(r) for r in range(1, int(upto) + 1)]


def load_b(cell, upto):
    """Every B-routed, non-collapsed, non-retired channel of rounds d,1..upto."""
    retired = set()
    rp = HERE / f"{cell}_retired_channels.json"
    if rp.exists():
        retired = {(x["round"], x["blind_id"]) for x in json.loads(rp.read_text())["retired"]}
    cols, meta = [], []
    for r in round_ids(upto):
        f = HERE / f"{cell}_r{r}_scores.npz"
        rt = HERE / f"{cell}_r{r}_routing_final.json"
        gp = HERE / f"{cell}_r{r}_score_report.json"
        if not (f.exists() and rt.exists() and gp.exists()):
            continue
        z = np.load(f, allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        gate = json.loads(gp.read_text())["per_criterion"]
        for x in json.loads(rt.read_text())["final"]:
            if x["final_route"] != "B":
                continue
            bid = x["blind_id"]
            if bid not in cids or gate[bid]["collapsed"] or (r, bid) in retired:
                continue
            cols.append(z["X"][:, cids.index(bid)].astype(float))
            meta.append({"round": r, "blind_id": bid, "name": x["name"],
                         "mixed": bool(x.get("mixed")),
                         "upstream_parent": x.get("upstream_parent", "surface-only")})
    X = np.column_stack(cols) if cols else np.zeros((0, 0))
    return X, meta, sorted(retired)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--round", required=True)
    a = ap.parse_args()

    d = C.load(a.cell)
    sp = json.loads((HERE / f"{a.cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, groups, dense = d["y"], d["groups"], d["dense"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])

    XB, meta, retired = load_b(a.cell, a.round)
    assert XB.shape[0] == len(y), (XB.shape, len(y))
    print(f"[{a.cell} r{a.round}] cumulative nuisance set: {XB.shape[1]} channels "
          f"({sum(1 for m in meta if m['mixed'])} mixed), {len(retired)} retired")

    # current bank (V + A_base + every A-routed criterion through round r)
    blocks, tags = S1.current_blocks(d, str(int(a.round) + 1))
    rbank = L.fit_block(blocks, fitm, monm, y, groups)
    va = np.full(len(y), np.nan)
    va[fitm] = rbank["oof_nl_fitmine"]
    va[monm] = rbank["nl_mon"]

    out = {"cell": a.cell, "round": a.round, "bank_blocks": tags,
           "n_bank_features": rbank["n_features"],
           "n_B_channels": int(XB.shape[1]),
           "channels": meta, "retired": [list(x) for x in retired],
           "T_HONEST": L.auc(y[held], dense[held]),
           "VA_nl_HONEST": L.auc(y[held], va[held]),
           "T_MONITOR": L.auc(y[monm], dense[monm]),
           "VA_nl_MONITOR": L.auc(y[monm], va[monm])}
    out["Delta_HONEST"] = out["T_HONEST"] - out["VA_nl_HONEST"]
    out["Delta_MONITOR"] = out["T_MONITOR"] - out["VA_nl_MONITOR"]

    def band(idx, label):
        if len(idx) == 0:
            return None
        rb = L.fit_block([XB[:, idx]], fitm, monm, y, groups)
        jb = np.full(len(y), np.nan)
        jb[fitm] = rb["oof_nl_fitmine"]
        jb[monm] = rb["nl_mon"]
        jl = np.full(len(y), np.nan)
        jl[fitm] = rb["oof_lin_fitmine"]
        jl[monm] = rb["lin_mon"]
        blk = {"label": label, "n_channels_after_screen": rb["n_features"],
               "spurious_alone_AUC_histgb_HONEST": L.auc(y[held], jb[held]),
               "spurious_alone_AUC_linear_HONEST": L.auc(y[held], jl[held]),
               "spurious_alone_AUC_histgb_MONITOR": L.auc(y[monm], jb[monm])}
        s_alone = max(blk["spurious_alone_AUC_histgb_HONEST"],
                      blk["spurious_alone_AUC_linear_HONEST"])
        joint = jb if (blk["spurious_alone_AUC_histgb_HONEST"] >=
                       blk["spurious_alone_AUC_linear_HONEST"]) else jl
        blk["joint_used"] = "histgb" if joint is jb else "linear"
        blk["estimator"] = "matched_sampling" if s_alone > MATCH_TRIGGER else "deciles"
        for pop, mask, q in (("HONEST", held, 10), ("MONITOR", monm, 5)):
            yy, jj = y[mask], joint[mask]
            tv, ti = matched_pair_auc(yy, dense[mask], jj)
            vv, vi = matched_pair_auc(yy, va[mask], jj)
            st = L.decile_strata(jj, q=q)
            td, td_i = L.stratified_auc(yy, dense[mask], st, min_n=20)
            vd, _ = L.stratified_auc(yy, va[mask], st, min_n=20)
            blk[pop] = {
                "n": int(mask.sum()),
                "pooled_T": L.auc(yy, dense[mask]), "pooled_VA": L.auc(yy, va[mask]),
                "pooled_Delta": L.auc(yy, dense[mask]) - L.auc(yy, va[mask]),
                "matched_T_adj": tv, "matched_VA_adj": vv, "matched_Delta_adj": tv - vv,
                "matched_info": {"T": ti, "VA": vi},
                "decile_T_adj": td, "decile_VA_adj": vd, "decile_Delta_adj": td - vd,
                "decile_info": td_i,
            }
            gg = groups[mask]
            s_bd = stack_oof([jj, dense[mask]], yy, gg)
            s_bv = stack_oof([jj, va[mask]], yy, gg)
            s_bvd = stack_oof([jj, va[mask], dense[mask]], yy, gg)
            blk[pop]["stacked"] = {
                "AUC_jointB": L.auc(yy, jj),
                "dense_increment_over_B": L.auc(yy, s_bd) - L.auc(yy, jj),
                "bank_increment_over_B": L.auc(yy, s_bv) - L.auc(yy, jj),
                "dense_increment_over_B_plus_bank": L.auc(yy, s_bvd) - L.auc(yy, s_bv),
                "ci_dense_increment_over_B_plus_bank":
                    L.group_boot_ci(yy, s_bvd, s_bv, gg),
            }
        return blk

    out["ALL_B"] = band(list(range(XB.shape[1])), "all named channels")
    strict = [k for k, m in enumerate(meta) if not m["mixed"]]
    out["STRICT_no_mixed"] = band(strict, "mixed channels dropped")
    out["band_note"] = ("FREEZE ADDENDUM 2 sensitivity band; the truth for Delta_adj "
                        "lies between ALL_B and STRICT_no_mixed.")

    (HERE / f"{a.cell}_r{a.round}_cumulative_discount.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "channels"},
                     indent=1, default=float)[:4000])
    print("wrote", HERE / f"{a.cell}_r{a.round}_cumulative_discount.json")


if __name__ == "__main__":
    main()
