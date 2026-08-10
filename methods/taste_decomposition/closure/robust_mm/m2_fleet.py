#!/usr/bin/env python3
"""M2(b) -- fleet-based richness: the capture-recapture quantities the pilot could not compute.

The pilot's rounds were sequential and each proposer was shown the bank and told not
to duplicate it, so observed recapture was ~0 and Good-Turing returned "missing mass
= 1.00" -- an artefact of the design, not a measurement.  The M1 sealed fleet removes
that defect: P independent proposers see the same slice with no sight of the bank or
of each other, so two proposers naming the same concept is a genuine RECAPTURE.

Computed per fleet round (tag):
  * species = single-linkage clusters of the P*k proposals at cosine tau
  * S_obs, f1, f2, bias-corrected Chao1, Good-Turing missing mass M = f1/N
  * PROPOSER-LEVEL bootstrap (resample the P proposers with replacement) -- the width
    the pilot's row-level bootstrap structurally could not produce
  * species accumulation across proposers (mean over random proposer orderings)
  * remaining-AUC bound, two forms:
      (i)  odds form   R = [M/(1-M)] * Delta_r * lambda      (design note Part 3)
      (ii) species form R = (Chao1_bc - S_obs) * value_per_species * lambda

CPU only.  Usage: python m2_fleet.py [--tau 0.79]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
sys.path.insert(0, str(HERE))
import embed_lib as E  # noqa: E402

N_BOOT = 2000
N_PERM = 500
# pilot anchors (notes/2026-08-06 Part 2): honest per-round gains and the fitted decay
PILOT_GAINS = [0.004651747187151614, 0.003796133567662507, 0.002892664193191563, 0.0004582438325847482]
PILOT_LAMBDA = 0.668378418862553
PILOT_SEQ_REMAINING = 0.002989234193022983     # the sequential extrapolation, +.003


def species_stats(sizes):
    sizes = np.asarray(sizes)
    S = int(len(sizes))
    f1 = int((sizes == 1).sum())
    f2 = int((sizes == 2).sum())
    n = int(sizes.sum())
    return {"S_obs": S, "f1": f1, "f2": f2, "N_proposals": n,
            "chao1_bias_corrected": float(S + f1 * (f1 - 1) / (2 * (f2 + 1))),
            "chao1_classic": float(S + f1 ** 2 / (2 * f2)) if f2 > 0 else None,
            "good_turing_missing_mass": float(f1 / n) if n else float("nan")}


def cluster(props, tau):
    txt = [E.crit_text(p["name"], p["instruction"]) for p in props]
    Emb = E.embed(txt)
    S = Emb @ Emb.T
    np.fill_diagonal(S, -1.0)
    return E.single_linkage(S, tau), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=E.TAU)
    ap.add_argument("--tags", default="round5,rep1,rep2,rep3")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    out = {"tau": a.tau, "n_bootstrap": N_BOOT, "tags": {}}

    for tag in a.tags.split(","):
        f = HERE / f"proposals_{tag}.json"
        if not f.exists():
            continue
        props = json.loads(f.read_text())["proposals"]
        lab, S = cluster(props, a.tau)
        proposers = sorted({p["proposer"] for p in props})
        fams = {p["proposer"]: p["family"] for p in props}
        idx_by_prop = {q: [i for i, p in enumerate(props) if p["proposer"] == q] for q in proposers}

        sizes = np.bincount(lab)
        st = species_stats(sizes)

        # cross-proposer recapture: species named by >1 proposer / >1 family
        sp_props = {s: {props[i]["proposer"] for i in np.where(lab == s)[0]} for s in range(len(sizes))}
        sp_fams = {s: {props[i]["family"] for i in np.where(lab == s)[0]} for s in range(len(sizes))}
        st["species_named_by_ge2_proposers"] = int(sum(len(v) >= 2 for v in sp_props.values()))
        st["species_named_by_ge2_families"] = int(sum(len(v) >= 2 for v in sp_fams.values()))
        st["cross_proposer_recapture_rate"] = st["species_named_by_ge2_proposers"] / st["S_obs"]

        # ---- proposer-level bootstrap ------------------------------------------
        boot = {"S_obs": [], "missing_mass": [], "chao1_bc": []}
        P = len(proposers)
        for _ in range(N_BOOT):
            pick = rng.choice(P, size=P, replace=True)
            idx = np.concatenate([idx_by_prop[proposers[k]] for k in pick])
            bsizes = np.bincount(lab[idx])
            bsizes = bsizes[bsizes > 0]
            bs = species_stats(bsizes)
            boot["S_obs"].append(bs["S_obs"])
            boot["missing_mass"].append(bs["good_turing_missing_mass"])
            boot["chao1_bc"].append(bs["chao1_bias_corrected"])
        st["proposer_bootstrap"] = {
            k: {"median": float(np.median(v)),
                "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
            for k, v in boot.items()}
        st["proposer_bootstrap_CAVEAT"] = (
            "resampling proposers WITH replacement duplicates whole proposers, which turns "
            "singletons into doubletons and biases the Good-Turing mass DOWN.  The "
            "leave-one-proposer-out jackknife below does not have this defect and is the "
            "primary proposer-level uncertainty statement.")

        # ---- leave-one-proposer-out jackknife (no artificial duplicates) --------
        jk = {"S_obs": [], "missing_mass": [], "chao1_bc": []}
        for q in proposers:
            idx = np.concatenate([idx_by_prop[r] for r in proposers if r != q])
            js = np.bincount(lab[idx])
            js = js[js > 0]
            s = species_stats(js)
            jk["S_obs"].append(s["S_obs"])
            jk["missing_mass"].append(s["good_turing_missing_mass"])
            jk["chao1_bc"].append(s["chao1_bias_corrected"])
        st["proposer_jackknife_leave1out"] = {
            k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                "pseudo_sd": float(np.sqrt((P - 1) / P * np.sum((np.array(v) - np.mean(v)) ** 2)))}
            for k, v in jk.items()}

        # ---- species accumulation across proposers ------------------------------
        acc = np.zeros((N_PERM, P))
        for b in range(N_PERM):
            order = rng.permutation(P)
            seen, idx = set(), []
            for j, k in enumerate(order):
                idx += idx_by_prop[proposers[k]]
                seen |= set(lab[idx].tolist())
                acc[b, j] = len(seen)
        st["species_accumulation_mean"] = [float(x) for x in acc.mean(axis=0)]
        st["marginal_new_species_per_added_proposer"] = [
            float(x) for x in np.diff(np.concatenate([[0.0], acc.mean(axis=0)]))]

        # ---- remaining-AUC bounds ----------------------------------------------
        M = st["good_turing_missing_mass"]
        delta_last = PILOT_GAINS[-1]
        value_per_species = sum(PILOT_GAINS) / 50.0     # 50 species bought +.0118 over 4 rounds
        st["remaining_auc_bound"] = {
            "odds_form_using_delta4": (M / (1 - M)) * delta_last * PILOT_LAMBDA if M < 1 else None,
            "odds_form_using_mean_gain": (M / (1 - M)) * float(np.mean(PILOT_GAINS)) * PILOT_LAMBDA if M < 1 else None,
            "species_form": float((st["chao1_bias_corrected"] - st["S_obs"]) * value_per_species * PILOT_LAMBDA),
            "value_per_species_used": value_per_species,
            "lambda_used": PILOT_LAMBDA,
        }
        st["per_proposer"] = {q: {"family": fams[q], "n_proposals": len(idx_by_prop[q]),
                                  "n_species_touched": len(set(lab[idx_by_prop[q]].tolist())),
                                  "n_species_unique_to_this_proposer": int(sum(
                                      1 for s in set(lab[idx_by_prop[q]].tolist())
                                      if sp_props[s] == {q}))}
                              for q in proposers}
        st["per_family"] = {}
        for fam in sorted(set(fams.values())):
            ii = [i for i, p in enumerate(props) if p["family"] == fam]
            others = {s for i, s in enumerate(lab) if props[i]["family"] != fam}
            mine = set(lab[ii].tolist())
            st["per_family"][fam] = {
                "n_proposals": len(ii), "n_species_touched": len(mine),
                "n_species_unique_to_family": len(mine - others),
                "share_of_species_unique": len(mine - others) / max(1, len(mine))}
        out["tags"][tag] = st
        print(f"{tag}: N={st['N_proposals']} S_obs={st['S_obs']} f1={st['f1']} f2={st['f2']} "
              f"M={M:.3f} chao1bc={st['chao1_bias_corrected']:.1f} "
              f"recapture={st['cross_proposer_recapture_rate']:.2f}", flush=True)

    # ------------------------------------------------ fleet vs sequential -------
    if out["tags"]:
        ms = [v["good_turing_missing_mass"] for v in out["tags"].values()]
        sp = [v["remaining_auc_bound"]["species_form"] for v in out["tags"].values()]
        od = [v["remaining_auc_bound"]["odds_form_using_mean_gain"] for v in out["tags"].values()
              if v["remaining_auc_bound"]["odds_form_using_mean_gain"] is not None]
        out["fleet_vs_sequential"] = {
            "sequential_pilot_remaining_AUC": PILOT_SEQ_REMAINING,
            "fleet_missing_mass_range": [float(min(ms)), float(max(ms))],
            "fleet_remaining_AUC_species_form_range": [float(min(sp)), float(max(sp))],
            "fleet_remaining_AUC_odds_form_range": [float(min(od)), float(max(od))] if od else None,
            "sequential_pilot_missing_mass_ARTEFACT": 0.821,
        }
        print("\n", json.dumps(out["fleet_vs_sequential"], indent=1))

    (HERE / "m2_fleet_richness.json").write_text(json.dumps(out, indent=2))
    print("wrote", HERE / "m2_fleet_richness.json")


if __name__ == "__main__":
    main()
