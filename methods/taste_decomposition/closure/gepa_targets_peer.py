#!/usr/bin/env python3
"""GEPA Stage 1 for the peer-verdict PILOT closure campaign (0-10 scale bank).

Two jobs in one pass, both required before any confirmatory number is quoted:

(1) Apply the sign-contradiction re-audit to the 56 mined A-criteria (rounds 1-4)
    using the SAME two-sided noise-scaled band the confirmatory campaigns adopted
    (Hanley-McNeil SE at AUC=.5 on FIT+MINE class counts; trigger fires only
    > 2 SE below chance).  Of 7 criteria with alone-AUC < .5, only round-2's
    "Restraint" (P13, alone-AUC .4583 FIT+MINE) clears the band; it is re-routed
    to nuisance (see notes/2026-08-09__gepa_requotes.md for the full re-audit
    reasoning). The other 6 are recorded as sign_null_band (kept in the bank).
    This yields the 55 SURVIVING mined criteria the task refers to as "the 56
    minus nuisance-routed".

(2) Bounded label-blind GEPA-phrasing targeting on those 55 survivors: fidelity
    = 0.5*(1-modal_share) + 0.3*(1-na_rate) + 0.2*min(spread,1) with spread
    normalised by scale_half=5.0 (this bank is 0-10, not CW's 0/.5/1).  Criteria
    with modal_share>.75 or na_rate>.20 are targeted for rephrasing.

CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import closure_lib as L
from stage4_round4 import load_round_blocks

HERE = Path(__file__).resolve().parent
MODAL_TRIGGER, NA_TRIGGER, SCALE_HALF = 0.75, 0.20, 5.0


def fidelity(col):
    fin = np.isfinite(col)
    if fin.sum() < 20:
        return 0.0, {"modal_share": 1.0, "na_rate": 1.0, "spread": 0.0}
    vals, cnts = np.unique(col[fin], return_counts=True)
    modal = float(cnts.max() / fin.sum())
    na = float((~fin).mean())
    spread = float(np.std(col[fin]) / SCALE_HALF)
    f = 0.5 * (1 - modal) + 0.3 * (1 - na) + 0.2 * min(spread, 1.0)
    return f, {"modal_share": modal, "na_rate": na, "spread": spread}


def hanley_mcneil_band(n1, n0, a0=0.5):
    q1, q2 = a0 / (2 - a0), 2 * a0 ** 2 / (1 + a0)
    se = float(np.sqrt((a0 * (1 - a0) + (n1 - 1) * (q1 - a0 ** 2)
                        + (n0 - 1) * (q2 - a0 ** 2)) / (n1 * n0)))
    return se, 0.5 - 2 * se, 0.5 + 2 * se


def main():
    pop, split, dsplit, summary = L.load_population(), None, None, None
    _, split, dsplit, mining = L.load_splits()
    y, nt = pop["y"], pop["ntitle"]
    fitm = split == "fit_mine"

    all_names, all_inst, all_col, all_tag = [], [], [], []
    per_round = {}
    for r in (1, 2, 3, 4):
        XA, XB, a_ids, b_ids, dropped = load_round_blocks(r)
        crits = {c["id"]: c for c in
                 json.loads((HERE / f"round{r}_track_a.json").read_text())["criteria"]}
        routing = json.loads((HERE / f"round{r}_routing_final.json").read_text())
        src_of = {e["blind_id"]: e["src_id"] for e in routing["final"]}
        z = np.load(HERE / f"round{r}_scores.npz", allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        per_round[r] = {"a_ids": a_ids, "dropped": dropped}
        for bid in a_ids:
            j = cids.index(bid)
            sid = src_of[bid]
            c = crits[sid]
            tag = f"r{r}:{bid}"
            all_names.append(c["name"])
            all_inst.append(c["instruction"])
            all_col.append(z["X"][:, j])
            all_tag.append(tag)

    print(f"56-criterion check: {len(all_tag)} A-routed, non-collapsed criteria "
          f"across rounds 1-4 (expect 56)")

    # ---------------------------------------------------- (1) sign re-audit ---
    n1 = int(y[fitm].sum())
    n0 = int((~y[fitm].astype(bool)).sum())
    se, lo, hi = hanley_mcneil_band(n1, n0)
    alone_fm = {}
    for tag, col in zip(all_tag, all_col):
        c = col[fitm].copy()
        fin = np.isfinite(c)
        if fin.sum() < 20 or len(np.unique(c[fin])) < 2:
            alone_fm[tag] = float("nan")
            continue
        c[~fin] = np.nanmedian(c)
        alone_fm[tag] = float(roc_auc_score(y[fitm], c))

    below_half = sorted([t for t in all_tag if alone_fm[t] == alone_fm[t]
                         and alone_fm[t] < 0.5], key=lambda t: alone_fm[t])
    sign_contradicting = [t for t in below_half if alone_fm[t] < lo]
    sign_null_band = [t for t in below_half if t not in sign_contradicting]

    print(f"\nTwo-sided band (Hanley-McNeil, n+={n1} n-={n0}): "
          f"SE={se:.5f}, band=[{lo:.4f}, {hi:.4f}]")
    print(f"{len(below_half)} criteria with FIT+MINE alone-AUC < .5:")
    for t in below_half:
        flag = "SIGN-CONTRADICTING -> re-route to nuisance" if t in sign_contradicting \
            else "null band -> kept in bank"
        idx = all_tag.index(t)
        print(f"  {t:10s} auc={alone_fm[t]:.4f}  {flag:40s}  {all_names[idx]}")

    nuisance_routed = set(sign_contradicting)
    surviving = [(t, n, i, c) for t, n, i, c in
                 zip(all_tag, all_names, all_inst, all_col) if t not in nuisance_routed]
    print(f"\n{len(all_tag)} total mined A criteria -> {len(nuisance_routed)} "
          f"re-routed to nuisance -> {len(surviving)} SURVIVING")

    # ---------------------------------------- (1b) removal-only Delta check ---
    XA1, _, a1, _, _ = load_round_blocks(1)
    XA2, _, a2, _, _ = load_round_blocks(2)
    XA3, _, a3, _, _ = load_round_blocks(3)
    XA4, _, a4, _, _ = load_round_blocks(4)
    restraint_j = a2.index("P13")
    XA2_dropped = np.delete(XA2, restraint_j, axis=1)

    monm = split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])
    from stage4_readout import fit_block

    def honest_delta(blocks, dense_col="same_rows_dense_heldout_1244"):
        r = fit_block(blocks, fitm, monm, y, nt)
        va_full = np.full(len(y), np.nan)
        va_full[fitm] = r["oof_nl_fitmine"]
        va_full[monm] = r["nl_mon"]
        return r, va_full

    import pandas as pd
    dense = pd.read_csv(HERE / "peer_verdict_dense_preds.csv").set_index("i").loc[
        np.arange(len(y)), "dense_prob"].values
    T_held = L.auc(y[held], dense[held])

    r_with, va_with = honest_delta([pop["V"], pop["A"], XA1, XA2, XA3, XA4])
    r_without, va_without = honest_delta([pop["V"], pop["A"], XA1, XA2_dropped, XA3, XA4])

    removal_report = {
        "T_held_1244": T_held,
        "with_restraint": {
            "n_features": r_with["n_features"],
            "VA_nl_honest_1244": L.auc(y[held], va_with[held]),
            "Delta_honest_1244": T_held - L.auc(y[held], va_with[held]),
        },
        "restraint_removed_nuisance_routed": {
            "n_features": r_without["n_features"],
            "VA_nl_honest_1244": L.auc(y[held], va_without[held]),
            "Delta_honest_1244": T_held - L.auc(y[held], va_without[held]),
        },
    }
    print("\n=== Removal-only effect (Restraint -> nuisance, before any GEPA phrasing) ===")
    print(json.dumps(removal_report, indent=2))

    # ------------------------------------------------------- (2) fidelity ----
    targets = []
    for t, n, inst, col in surviving:
        f, d = fidelity(col)
        targeted = bool(d["modal_share"] > MODAL_TRIGGER or d["na_rate"] > NA_TRIGGER)
        targets.append({"tag": t, "name": n, "instruction": inst,
                        "incumbent_fidelity": f, **d, "targeted": targeted,
                        "alone_auc_fitmine": alone_fm[t]})

    out = {
        "sign_band": {"se": se, "lo": lo, "hi": hi, "n_pos": n1, "n_neg": n0},
        "sign_contradicting_reroute": sorted(sign_contradicting),
        "sign_null_band_kept": sorted(sign_null_band),
        "n_total_mined_A": len(all_tag),
        "n_surviving": len(surviving),
        "removal_only_effect": removal_report,
        "targets": targets,
    }
    (HERE / "gepa_targets_peer.json").write_text(json.dumps(out, indent=2))
    nt_ = sum(t["targeted"] for t in targets)
    print(f"\n{len(targets)} surviving A criteria; {nt_} targeted for rephrasing "
          f"(modal_share>{MODAL_TRIGGER} or na_rate>{NA_TRIGGER})")
    for t in targets:
        if t["targeted"]:
            print(f"  {t['tag']} modal={t['modal_share']:.2f} na={t['na_rate']:.2f} "
                  f"fid={t['incumbent_fidelity']:.3f}  {t['name']}")


if __name__ == "__main__":
    main()
