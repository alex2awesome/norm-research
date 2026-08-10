#!/usr/bin/env python3
"""TERMINAL LEDGER for the jokes_community Layer-3 closure campaign.

Assembles the one table the campaign is quoted from, under the three disciplines the
freeze requires and one this campaign added:

  1. CUMULATIVE COLLAPSE GATE.  Every mined criterion flagged `collapsed` in its round's
     score report (modal_frac > .98) is EXCLUDED from the terminal bank, in every round at
     once.  `closure_core.clean_fit` does not do this on its own at n = 16,000 (it only
     drops a column with fewer than 5 off-modal rows), which is the defect this campaign
     found; the terminal bank is the gated one and the ungated figure is reported beside
     it so the correction is visible.

  2. GEPA PHRASING.  Where `gepa_select.py` accepted a rephrasing, the winner's
     corpus-wide column replaces the incumbent's.  Where nothing was accepted, the
     incumbent stands and the criterion is recorded as "phrasing pass run, no change".
     Criteria that were never targeted passed at stage 1.

  3. BOTH T CONVENTIONS, LABELLED.  T_meanAUC = mean over dense seeds of the AUC (the
     campaign convention, mirroring VA_nl's seed-mean).  T_ensemble = AUC of the seed-mean
     prediction (what the master ledger quotes).  Every Delta is emitted twice, tagged,
     and the two are never differenced against each other.

  4. STOPPING CONDITION, stated explicitly rather than inferred: which rounds were
     PROPOSING rounds, which were exempt, each round's signed MONITOR gain against
     epsilon, and whether the campaign ended by two-consecutive-sub-epsilon or by cap.

CPU only.  Usage: OMP_NUM_THREADS=6 python3 terminal_ledger.py --rounds 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent
EPS = 0.005
PROPOSING = {"1", "2", "4", "5"}      # round 3 was decomposition-only (TIER D)


def load_round_A(cell, r, collapsed_ids, winners=None):
    """A-routed, NON-COLLAPSED columns of round r, with GEPA winners swapped in."""
    f = HERE / f"{cell}_r{r}_scores.npz"
    rt = HERE / f"{cell}_r{r}_routing_final.json"
    if not (f.exists() and rt.exists()):
        return np.zeros((0, 0)), [], []
    z = np.load(f, allow_pickle=True)
    cids = [str(s) for s in z["crit_ids"]]
    keep, dropped = [], []
    for x in json.loads(rt.read_text())["final"]:
        if x["final_route"] != "A" or x["blind_id"] not in cids:
            continue
        if (r, x["blind_id"]) in collapsed_ids:
            dropped.append(x["blind_id"]); continue
        keep.append(x["blind_id"])
    if not keep:
        return np.zeros((z["X"].shape[0], 0)), [], dropped
    X = z["X"][:, [cids.index(i) for i in keep]].astype(float)
    if winners:
        wz_path = HERE / f"{cell}_gepa_winner_scores.npz"
        if wz_path.exists():
            wz = np.load(wz_path, allow_pickle=True)
            wu = [str(s) for s in wz["uid"]]
            for k, bid in enumerate(keep):
                tid = f"r{r}_{bid}"
                if tid in winners and winners[tid] in wu:
                    X[:, k] = wz["X"][:, wu.index(winners[tid])]
    return X, keep, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="jokes_community")
    ap.add_argument("--rounds", default="1,2,3,4,5")
    a = ap.parse_args()
    cell, rounds = a.cell, [r.strip() for r in a.rounds.split(",")]

    d = C.load(cell)
    sp = json.loads((HERE / f"{cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    y, g = d["y"], d["groups"]

    tgt_path = HERE / f"{cell}_gepa_targets.json"
    tgt = json.loads(tgt_path.read_text()) if tgt_path.exists() else {"criteria": []}
    collapsed = {(c["round"], c["blind_id"]) for c in tgt["criteria"] if c["COLLAPSED"]}
    sel_path = HERE / f"{cell}_gepa_selection.json"
    winners = json.loads(sel_path.read_text())["winners"] if sel_path.exists() else {}

    blocks_gated, blocks_raw, names, dropped_all = [d["V"], d["A"]], [d["V"], d["A"]], ["V", "A_base"], []
    for r in rounds:
        Xg, keep, drop = load_round_A(cell, r, collapsed, winners)
        Xr, keep_r, _ = load_round_A(cell, r, set(), None)
        dropped_all += [f"r{r}:{b}" for b in drop]
        if Xg.shape[1]:
            blocks_gated.append(Xg); names.append(f"A_round{r}")
        if Xr.shape[1]:
            blocks_raw.append(Xr)

    out = {"cell": cell, "sklearn": C.sklearn_guard(),
           "rounds_run": rounds, "epsilon": EPS,
           "proposing_rounds": sorted(PROPOSING),
           "exempt_rounds": ["3 (ADDENDUM-3 decomposition, TIER D, no proposers)"],
           "collapse_gate": {"excluded": dropped_all,
                             "rule": "modal_frac > .98 in the round score report; applied "
                                     "cumulatively, which closure_core.clean_fit does not "
                                     "do at n=16,000"},
           "gepa": {"n_targets": sum(1 for c in tgt["criteria"] if c["gepa_targeted"]),
                    "n_accepted_rephrasings": len(winners),
                    "winners": winners,
                    "note": "criteria never targeted PASSED the phrasing pass at stage 1"},
           "bank_blocks": names}

    fit = L.fit_block(blocks_gated, fitm, monm, y, g)
    raw = L.fit_block(blocks_raw, fitm, monm, y, g)
    base = L.fit_block([d["V"], d["A"]], fitm, monm, y, g)

    def full(rr):
        v = np.full(len(y), np.nan); v[fitm] = rr["oof_nl_fitmine"]; v[monm] = rr["nl_mon"]
        return v

    T_mon = C.T_by_seed(d, monm)
    T_hon = C.T_by_seed(d, held)
    for pop, mask, Tb in (("MONITOR", monm, T_mon), ("HONEST", held, T_hon)):
        vg = full(fit)[mask] if pop == "HONEST" else fit["nl_mon"]
        vr = full(raw)[mask] if pop == "HONEST" else raw["nl_mon"]
        vb = full(base)[mask] if pop == "HONEST" else base["nl_mon"]
        out[pop] = {
            "n": int(mask.sum()), "n_topics": int(len({str(x) for x in g[mask]})),
            "T_meanAUC": Tb["T"], "T_meanAUC_per_seed": Tb["per_seed"],
            "T_ensemble": Tb["T_seed_ensemble_NOT_QUOTED"],
            "VA_nl_bank0": L.auc(y[mask], vb),
            "VA_nl_terminal_collapse_gated": L.auc(y[mask], vg),
            "VA_nl_terminal_ungated": L.auc(y[mask], vr),
            "n_features_bank0": base["n_features"],
            "n_features_terminal_gated": fit["n_features"],
            "Delta_beyond_bank0_meanAUC": Tb["T"] - L.auc(y[mask], vb),
            "Delta_beyond_bank0_ensemble": Tb["T_seed_ensemble_NOT_QUOTED"] - L.auc(y[mask], vb),
            "Delta_beyond_terminal_meanAUC": Tb["T"] - L.auc(y[mask], vg),
            "Delta_beyond_terminal_ensemble": Tb["T_seed_ensemble_NOT_QUOTED"] - L.auc(y[mask], vg),
            "total_closure": L.auc(y[mask], vg) - L.auc(y[mask], vb),
        }
    out["MONITOR"]["closure_ci_vs_bank0"] = L.group_boot_ci(
        y[monm], fit["nl_mon"], base["nl_mon"], np.array([str(x) for x in g[monm]]))

    # per-round gains, restated from the round results, with the stopping arithmetic
    gains, clock, terminal = [], 0, None
    for r in rounds:
        f = HERE / f"{cell}_r{r}_results.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())["track_A"]
        gain = res["gain_MONITOR"]
        prop = r in PROPOSING
        sub = gain < EPS
        counts = prop and sub
        clock = clock + 1 if counts else (0 if prop else clock)
        gains.append({"round": r, "proposing": prop, "gain_MONITOR": gain,
                      "sub_epsilon": bool(sub), "counts_toward_stopping": bool(counts),
                      "clock_after": clock,
                      "ci": res.get("gain_ci_MONITOR")})
        if clock >= 2 and terminal is None:
            terminal = f"two consecutive sub-epsilon PROPOSING rounds (r{r})"
    if terminal is None:
        terminal = (f"CAP reached at round {rounds[-1]} WITHOUT the stopping rule firing "
                    f"(clock ended at {clock} of the 2 consecutive sub-epsilon proposing "
                    f"rounds required)")
    out["per_round_gain"] = gains
    out["stopping_condition"] = terminal
    out["stopping_clock_final"] = clock

    (HERE / f"{cell}_TERMINAL_LEDGER.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "collapse_gate"},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
