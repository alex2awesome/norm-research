#!/usr/bin/env python3
"""Fleet-based missing mass for the CW-community campaign -- BOTH tracks.

Port of methods/taste_decomposition/closure/robust_mm/m2_fleet.py with two
changes required by the FREEZE DECLARATION and its ADDENDUM:
  * species identity comes from the round's BLIND FULL-RECALL partition
    (round{r}_species.json), never from an embedding threshold;
  * the estimator runs on the Track-B pool as well as Track A (addendum), with
    the value-weighting assumption stated explicitly rather than assumed.

Reported per round and track: S_obs, f1, f2, N, Good-Turing missing mass with a
leave-one-proposer-out jackknife, cross-proposer recapture, species accumulation,
and the odds-form remaining-AUC bound  R = [M/(1-M)] * Dbar * lambda  fitted on
THIS cell's own gain series.  Chao1 is computed but never quoted (species form is
non-identified at small f2, per the pilot).

Usage: python missing_mass_cw.py --rounds 1,2,3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_PERM = 500


def species_stats(sizes):
    sizes = np.asarray(sizes)
    S, n = int(len(sizes)), int(sizes.sum())
    f1, f2 = int((sizes == 1).sum()), int((sizes == 2).sum())
    return {"S_obs": S, "f1": f1, "f2": f2, "N_proposals": n,
            "chao1_bias_corrected_NEVER_QUOTE":
                float(S + f1 * (f1 - 1) / (2 * (f2 + 1))),
            "good_turing_missing_mass": float(f1 / n) if n else float("nan")}


def analyse(pool, species, rng):
    by_pid = {p["pid"]: p for p in pool}
    lab, members = {}, {}
    for si, (key, pids) in enumerate(sorted(species.items())):
        pids = [q for q in pids if q in by_pid]
        if not pids:
            continue
        members[si] = pids
        for q in pids:
            lab[q] = si
    proposers = sorted({p["proposer"] for p in pool})
    fams = {p["proposer"]: p["family"] for p in pool}
    idx_by_prop = {q: [p["pid"] for p in pool if p["proposer"] == q] for q in proposers}

    sizes = np.array([len(v) for v in members.values()])
    st = species_stats(sizes)
    sp_props = {si: {by_pid[q]["proposer"] for q in v} for si, v in members.items()}
    sp_fams = {si: {by_pid[q]["family"] for q in v} for si, v in members.items()}
    st["species_named_by_ge2_proposers"] = int(sum(len(v) >= 2 for v in sp_props.values()))
    st["species_named_by_ge2_families"] = int(sum(len(v) >= 2 for v in sp_fams.values()))
    st["cross_proposer_recapture_rate"] = (
        st["species_named_by_ge2_proposers"] / max(1, st["S_obs"]))
    st["P"] = len(proposers)
    st["n_families"] = len(set(fams.values()))

    jk = {"S_obs": [], "missing_mass": []}
    for q in proposers:
        keep = [x for p in proposers if p != q for x in idx_by_prop[p]]
        s = np.bincount([lab[x] for x in keep if x in lab])
        s = s[s > 0]
        if not len(s):
            continue
        d = species_stats(s)
        jk["S_obs"].append(d["S_obs"])
        jk["missing_mass"].append(d["good_turing_missing_mass"])
    P = len(proposers)
    st["jackknife_leave1out"] = {
        k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
            "pseudo_sd": float(np.sqrt((P - 1) / P * np.sum(
                (np.array(v) - np.mean(v)) ** 2)))}
        for k, v in jk.items() if v}

    acc = np.zeros((N_PERM, P))
    for b in range(N_PERM):
        order = rng.permutation(P)
        seen, got = set(), []
        for j, k in enumerate(order):
            got += idx_by_prop[proposers[k]]
            seen |= {lab[x] for x in got if x in lab}
            acc[b, j] = len(seen)
    st["species_accumulation_mean"] = [float(x) for x in acc.mean(axis=0)]
    st["marginal_new_species_per_added_proposer"] = [
        float(x) for x in np.diff(np.concatenate([[0.0], acc.mean(axis=0)]))]
    st["per_family"] = {}
    for fam in sorted(set(fams.values())):
        mine = {lab[p["pid"]] for p in pool if p["family"] == fam and p["pid"] in lab}
        others = {lab[p["pid"]] for p in pool if p["family"] != fam and p["pid"] in lab}
        st["per_family"][fam] = {"n_species_touched": len(mine),
                                 "n_species_unique_to_family": len(mine - others)}
    return st


def fit_lambda(gains):
    """Geometric decay lambda from the observed per-round gain series."""
    g = [x for x in gains if x is not None]
    if len(g) < 2:
        return None
    rs = [g[i + 1] / g[i] for i in range(len(g) - 1) if g[i] > 0]
    rs = [r for r in rs if r > 0]
    return float(np.exp(np.mean(np.log(rs)))) if rs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rounds = [int(x) for x in a.rounds.split(",")]

    gains = []
    for r in rounds:
        f = HERE / f"round{r}_results.json"
        gains.append(json.loads(f.read_text())["Delta_VA_nl_monitor"]
                     if f.exists() else None)
    lam = fit_lambda(gains)
    dbar = float(np.mean([g for g in gains if g is not None])) if any(
        g is not None for g in gains) else None

    out = {"rounds": rounds, "monitor_gains": gains, "lambda_fit": lam,
           "mean_gain": dbar, "tracks": {}}
    for track in ("A", "B"):
        out["tracks"][track] = {}
        for r in rounds:
            pf = HERE / f"round{r}_fleet_{track}.json"
            sf = HERE / f"round{r}_species.json"
            if not (pf.exists() and sf.exists()):
                continue
            pool = json.loads(pf.read_text())["proposals"]
            spec = json.loads(sf.read_text())[track]
            st = analyse(pool, spec, rng)
            M = st["good_turing_missing_mass"]
            if dbar is not None and lam is not None and M < 1:
                st["remaining_auc_odds_form"] = (M / (1 - M)) * dbar * lam
            jk = st.get("jackknife_leave1out", {}).get("missing_mass")
            if jk and dbar is not None and lam is not None:
                st["remaining_auc_odds_form_jackknife_range"] = [
                    (m / (1 - m)) * dbar * lam for m in (jk["min"], jk["max"])]
            out["tracks"][track][f"round{r}"] = st
            print(f"[{track} r{r}] N={st['N_proposals']} S={st['S_obs']} "
                  f"f1={st['f1']} f2={st['f2']} M={M:.3f} "
                  f"recapture={st['cross_proposer_recapture_rate']:.2f} P={st['P']}")

    out["B_side_value_weighting_assumption"] = (
        "B-side species mass counts nameable SPURIOUS CHANNELS. Converting it to a "
        "value bound assumes unfound channels resemble found ones in influence; the "
        "species-mass figure carries no such assumption and is the primary number.")
    out["chao1_note"] = ("species-form remaining-AUC is not computed: Chao1 is "
                         "non-identified at these f2 counts (pilot Part 3.2).")
    (HERE / "missing_mass.json").write_text(json.dumps(out, indent=1))
    print("wrote missing_mass.json")


if __name__ == "__main__":
    main()
