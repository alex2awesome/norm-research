#!/usr/bin/env python3
"""P2 -- pairwise-native Layer-1 for cw_wigleaf_curation.

The absolute instrument could not certify this cell (battery pos-vs-neg .498,
bank saturated at 83% of responses = 1.0). The pairwise instrument can (wave-1
composite .610, anchors .9963, order consistency .8406). So the Layer-1 ledger is
rebuilt at the PAIR level, where the instrument actually works.

UNIT      one pair (i, j) with opposite labels; chance = .500 by construction.
TARGET    y = 1 if the "first" piece of the oriented pair is the Top-50 piece.
A_pair    the 45 comparative verdicts as {-1, 0, +1}, oriented to piece 1.
V_pair    V(i) - V(j) over the 15 deterministic surface features. This is the
          length test: log-length alone hit .680 on the item-level eval split, so
          V_pair is how we find out whether the pairwise signal is just size.
VA_pair   concatenation.

ANTISYMMETRY IS ENFORCED. Every pair enters TWICE -- once as (i, j) and once as
(j, i) with all features negated and the label flipped. Both orientations carry
the SAME group id, so GroupKFold can never put one orientation in train and its
mirror in test. Without this a classifier can exploit a constant offset (predict
"A" always) and manufacture AUC out of position bias; with it, any constant
predictor is exactly .500 by construction.

T_pair    NOT COMPUTED HERE -- see the note printed at the end. The existing
          Wigleaf dense arm only has out-of-sample predictions for the 322
          eval+test rows, so pairs with BOTH pieces scored out-of-sample number
          ~25 of 600. A pairwise ceiling needs a cross-fitted Wigleaf dense arm
          (same template as RoyalRoad's). Reported as a gap, never faked.

  python methods/taste_decomposition/wigleaf_pairwise_layer1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_gemma_cells as L          # noqa: E402
import scaleupC_layer1 as SC            # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PW = REPO / "datasets/creative-writing/wigleaf/pairwise"
CWX = REPO / "outputs/va_gemma_banks_cw_expert"
RESULTS = Path(__file__).resolve().parent / "results"
SLUG = "cw_wigleaf_curation_pairwise"


def load_waves():
    """Merge every wave that exists; pair_ids are wave-prefixed so never collide."""
    pk, vd = {}, {}
    for pf, vf in (("packet.json", "verdicts_sol.json"),
                   ("packet_wave2.json", "verdicts_sol_wave2.json")):
        p, v = PW / pf, PW / vf
        if not (p.exists() and v.exists()):
            print(f"  (skipping {pf}: not present)")
            continue
        P = json.loads(p.read_text())
        V = json.loads(v.read_text())["verdicts"]
        pk.setdefault("criteria", P["criteria"])
        pk.setdefault("meta", {}).update(P["meta"])
        vd.update(V)
        print(f"  loaded {pf}: {len(P['meta'])} meta, {len(V)} verdicts")
    return pk, vd


def main():
    print("=== loading pairwise waves ===")
    pk, V = load_waves()
    meta, crit = pk["meta"], [m["id"] for m in pk["criteria"]]
    cname = {m["id"]: m["name"] for m in pk["criteria"]}

    # item-level V features, keyed by row id
    z = np.load(CWX / "cw_wigleaf_curation_shard0.npz", allow_pickle=True)
    v_names = [str(s) for s in z["v_names"]]
    Vmat, vids = [], []
    si = 0
    while (CWX / f"cw_wigleaf_curation_shard{si}.npz").exists():
        zz = np.load(CWX / f"cw_wigleaf_curation_shard{si}.npz", allow_pickle=True)
        Vmat.append(zz["V"]); vids += [str(s) for s in zz["ids"]]
        si += 1
    Vmat = np.vstack(Vmat)
    vpos = {d: i for i, d in enumerate(vids)}
    print(f"  V matrix {Vmat.shape} over {len(vpos)} items")

    # ---- assemble oriented pair rows (real pairs only; flips/anchors are gates) --
    rows = []
    for pid, v in V.items():
        m = meta.get(pid)
        if not m or m.get("kind") != "real":
            continue
        a_id = m["pos_id"] if m["pos_side"] == "A" else m["neg_id"]
        b_id = m["neg_id"] if m["pos_side"] == "A" else m["pos_id"]
        if a_id not in vpos or b_id not in vpos:
            continue
        av = np.array([{"A": 1.0, "B": -1.0, "TIE": 0.0}.get(v["criteria"].get(c), np.nan)
                       for c in crit])
        vv = Vmat[vpos[a_id]] - Vmat[vpos[b_id]]
        y = 1 if m["pos_side"] == "A" else 0
        rows.append({"pair": pid, "A": av, "V": vv, "y": y})
    print(f"  usable real pairs: {len(rows)}")

    # ---- ENFORCED ANTISYMMETRY: every pair twice, mirrored, same group ----------
    A, Vp, y, g = [], [], [], []
    for r in rows:
        A.append(r["A"]);      Vp.append(r["V"]);      y.append(r["y"]);      g.append(r["pair"])
        A.append(-r["A"]);     Vp.append(-r["V"]);     y.append(1 - r["y"]);  g.append(r["pair"])
    A, Vp = np.array(A, dtype=float), np.array(Vp, dtype=float)
    y, g = np.array(y), np.array(g, dtype=object)
    VA = np.column_stack([Vp, A])
    mats = {"V": Vp, "A": A, "VA": VA}
    print(f"  design matrix {VA.shape} ({len(rows)} pairs x 2 orientations), "
          f"pos rate {y.mean():.4f} (must be exactly .5000)")
    assert abs(y.mean() - 0.5) < 1e-12, "antisymmetry broken: label balance != .5"

    folds = L.outer_folds(len(y), g, n_splits=5)
    # sanity: no pair split across train/test
    for tr, te in folds:
        assert not (set(g[tr]) & set(g[te])), "a pair leaked across a fold boundary"

    res = {"cell": SLUG, "design": "P2 pairwise-native Layer-1",
           "n_pairs": len(rows), "n_rows_after_mirroring": int(len(y)),
           "chance": 0.5, "group_column": "pair (both orientations share it)",
           "antisymmetry_enforced": True,
           "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
           "sklearn_version": sklearn.__version__,
           "linear": {}, "nonlinear": {}}

    lin_oof = {}
    for k in ("V", "A", "VA"):
        auc, oof = L.linear_oof_family1(mats[k], y, g, folds)
        res["linear"][k] = auc
        lin_oof[k] = oof
        print(f"  linear  {k:2s}: {auc:.4f}")

    nl_oof = {}
    for k in ("V", "VA"):
        res["nonlinear"][k] = {}
        for s in L.GBM_SEEDS:
            r = L.gbm_oof_family1(mats[k], y, g, folds, s)
            nl_oof[(k, s)] = r.pop("oof")
            res["nonlinear"][k][str(s)] = r
            print(f"  gbm {k:2s} seed {s}: {r['auc']:.4f} (train {r['train_auc_mean']:.4f})")

    va = [res["nonlinear"]["VA"][str(s)]["auc"] for s in L.GBM_SEEDS]
    vv = [res["nonlinear"]["V"][str(s)]["auc"] for s in L.GBM_SEEDS]
    res["seed_spread"] = {"VA": va, "V": vv}
    res["seed_spread_range"] = {"VA": float(max(va) - min(va)), "V": float(max(vv) - min(vv))}
    res["ledger"] = {
        "V_lin": res["linear"]["V"], "V_nl_mean": float(np.mean(vv)),
        "A_lin": res["linear"]["A"], "VA_lin": res["linear"]["VA"],
        "VA_nl_mean": float(np.mean(va)),
        "Delta_interact": float(np.mean(va)) - res["linear"]["VA"],
        "T_pair": None, "Delta_beyond": None}
    res["group_bootstrap_delta_interact"] = SC.group_bootstrap_delta(
        y, g, lin_oof["VA"], nl_oof[("VA", 0)])
    res["group_bootstrap_ci95"] = {
        k: SC.group_bootstrap_auc(y, g, lin_oof[k]) for k in ("V", "A", "VA")}

    # length test: is the pairwise signal just size?
    li = [i for i, n in enumerate(v_names) if "log_chars" in n or "log_words" in n]
    if li:
        auc_len, _ = L.linear_oof_family1(Vp[:, li], y, g, folds)
        res["length_only_pair_auc"] = auc_len
        print(f"  length-only (V_pair log-size cols): {auc_len:.4f}")

    # ---- RULING 3: does craft survive underneath the size channel? -------------
    # Split pairs by |delta log-size|. If A/VA hold up among SIZE-MATCHED pairs
    # (below-median |delta|), craft signal exists beneath the size channel; if they
    # collapse to chance there, the cell's story is editorial scope, not craft.
    if li:
        dmag = np.abs(Vp[:, li]).mean(axis=1)          # mirrored rows share |delta|
        pair_mag = {}
        for gi, m in zip(g, dmag):
            pair_mag[gi] = m
        med = float(np.median(list(pair_mag.values())))
        strat = {"median_abs_delta_logsize": round(med, 4), "strata": {}}
        for tag, sel in (("size_matched", lambda m: m <= med),
                         ("size_divergent", lambda m: m > med)):
            mask = np.array([sel(pair_mag[gi]) for gi in g])
            gs, ys = g[mask], y[mask]
            npair = len(set(gs))
            if npair < 40 or len(np.unique(ys)) < 2:
                strat["strata"][tag] = {"n_pairs": npair, "skipped": "too few pairs"}
                continue
            fs = L.outer_folds(int(mask.sum()), gs, n_splits=5)
            e = {"n_pairs": npair}
            for k in ("V", "A", "VA"):
                e[f"{k}_lin"] = L.linear_oof_family1(mats[k][mask], ys, gs, fs)[0]
            e["VA_nl_seed0"] = L.gbm_oof_family1(mats["VA"][mask], ys, gs, fs, 0)["auc"]
            e["length_only"] = L.linear_oof_family1(Vp[mask][:, li], ys, gs, fs)[0]
            strat["strata"][tag] = e
            print(f"  [{tag}] pairs={npair} V {e['V_lin']:.4f} A {e['A_lin']:.4f} "
                  f"VA {e['VA_lin']:.4f} VA_nl {e['VA_nl_seed0']:.4f} "
                  f"len-only {e['length_only']:.4f}")
        strat["reading"] = (
            "If A_lin / VA_nl stay clearly above .500 in the SIZE-MATCHED stratum, "
            "craft signal exists beneath the size channel and is claimable. If they "
            "fall to chance there while size_divergent stays high, the cell's signal "
            "is editorial scope (length remit), not craft.")
        res["length_stratified"] = strat

    # ---- T_pair, once a cross-fitted Wigleaf dense arm exists ------------------
    CF = REPO / "datasets/creative-writing/wigleaf/dense_crossfit"
    honest = {}
    if CF.exists():
        import pandas as pd
        for k in range(5):
            f = CF / f"fold{k}" / "rm_out_seed42" / "preds_test.csv"
            if f.exists():
                d = pd.read_csv(f)
                for gid, pr in zip(d["group"].astype(str), d["prob"]):
                    honest[gid] = float(pr)
    if honest:
        tp_y, tp_d, tp_g = [], [], []
        for r0, pid in ((r, r["pair"]) for r in rows):
            m = meta[pid]
            a_id = m["pos_id"] if m["pos_side"] == "A" else m["neg_id"]
            b_id = m["neg_id"] if m["pos_side"] == "A" else m["pos_id"]
            if a_id in honest and b_id in honest:
                tp_d.append(honest[a_id] - honest[b_id])
                tp_y.append(1 if m["pos_side"] == "A" else 0)
                tp_g.append(pid)
        if len(tp_y) >= 30 and len(set(tp_y)) > 1:
            # antisymmetric by construction: score difference flips with orientation
            yy = np.array(tp_y + [1 - v for v in tp_y])
            dd = np.array(tp_d + [-v for v in tp_d])
            t_pair = float(roc_auc_score(yy, dd))
            res["ledger"]["T_pair"] = t_pair
            res["ledger"]["Delta_beyond"] = t_pair - res["ledger"]["VA_nl_mean"]
            res["T_pair_info"] = {
                "n_pairs_with_both_pieces_out_of_sample": len(tp_y),
                "of_total_pairs": len(rows),
                "source": "wigleaf/dense_crossfit fold*/rm_out_seed42/preds_test.csv "
                          "(selection-free honest set)",
                "construction": "sigmoid score difference, mirrored for antisymmetry"}
            print(f"  T_pair {t_pair:.4f} on {len(tp_y)} pairs with both pieces "
                  f"out-of-sample")
        else:
            res["T_pair_info"] = {"insufficient": len(tp_y)}

    res.setdefault("T_pair_gap", None)
    if res["ledger"].get("T_pair") is None:
        res["T_pair_gap"] = (
        "NOT COMPUTED. The existing Wigleaf dense arm has out-of-sample predictions "
        "only for the 322 eval+test rows (and eval was itself the checkpoint-selection "
        "split), so pairs with BOTH pieces scored out-of-sample number roughly 25 of "
        f"{len(rows)}. A pairwise ceiling requires a CROSS-FITTED Wigleaf dense arm, "
        "the same template used for RoyalRoad. Until that exists there is no honest "
        "T_pair and therefore no Delta_beyond at the pair level.")
    res["estimand_caveat"] = (
        "P2 licenses PAIRWISE-FRAME claims only. These AUCs are on matched, "
        "opposite-label pairs under forced choice with chance pinned at .500; they "
        "are not comparable to the item-level cell's A_lin .5407 / VA_nl .6051 / "
        "T .6054, which are unpaired grouped-OOF AUCs over all 1,568 rows.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{SLUG}_ledger.json").write_text(json.dumps(res, indent=2, default=str))
    np.savez_compressed(RESULTS / f"{SLUG}_oof.npz", pairs=g, y=y,
                        **{f"{k}_lin": lin_oof[k] for k in lin_oof},
                        VA_nl_seed0=nl_oof[("VA", 0)])
    print("\n=== P2 LEDGER ===")
    for k, v in res["ledger"].items():
        if isinstance(v, float):
            print(f"  {k:16s} {v:+.4f}")
    print(f"  seed spread VA_nl {res['seed_spread_range']['VA']:.4f}")
    if res.get("T_pair_gap"):
        print(f"  {res['T_pair_gap'][:110]}...")
    elif res.get("T_pair_info"):
        print(f"  T_pair on {res['T_pair_info'].get('n_pairs_with_both_pieces_out_of_sample')} "
              f"of {res['T_pair_info'].get('of_total_pairs')} pairs")
    print("wrote", RESULTS / f"{SLUG}_ledger.json")
    print("WIGLEAF_PAIRWISE_LAYER1_DONE")


if __name__ == "__main__":
    main()
