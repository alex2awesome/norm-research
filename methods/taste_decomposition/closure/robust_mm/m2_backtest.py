#!/usr/bin/env python3
"""M2(a) -- backtest of the missing-mass / decay estimator on the pilot's 4 rounds.

The estimator (notes/2026-08-06 Part 2/3) earns quotation rights by out-of-sample
prediction, not by fit.  So: fit on rounds 0..r only, predict round r+1's marginal
AUC gain and its new-species count, and score against the actual.

Three predictands, three very different verdicts -- all reported:
  1. marginal AUC gain g_{r+1}   (geometric + saturating-exponential, honest rows)
  2. remaining mass  sum_{r'>r}  (compared to what the pilot actually went on to get)
  3. new-species count           (Good-Turing on the mined proposals; DEGENERATE by
     design in the pilot because rounds were sequential and anti-duplication-
     instructed -- reported to show exactly what M1's sealed fleet is fixing)

Readout-noise band = group-level (`ntitle`) paired bootstrap of the per-step AUC
delta, 2,000 draws, prediction vectors held fixed.  A prediction "lands" if the
actual gain falls inside that band around the prediction.

CPU only.  Usage: python m2_backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
sys.path.insert(0, str(CLOSURE))
sys.path.insert(0, str(HERE))
import embed_lib as E  # noqa: E402

N_BOOT = 2000
LAM_LO, LAM_HI = 0.01, 0.98


def fit_geometric(g):
    """g_r = a * lam^(r-1).  Needs >= 2 points to identify lam."""
    g = np.asarray(g, dtype=float)
    if len(g) < 2:
        return None
    r = np.arange(1, len(g) + 1)

    def res(p):
        return p[0] * p[1] ** (r - 1) - g
    s = least_squares(res, x0=[max(g[0], 1e-6), 0.6],
                      bounds=([0, LAM_LO], [1, LAM_HI]))
    a, lam = float(s.x[0]), float(s.x[1])
    at_bound = lam <= LAM_LO + 1e-6 or lam >= LAM_HI - 1e-6
    return {"a": a, "lam": lam, "at_bound": bool(at_bound),
            "next": a * lam ** len(g),
            "remaining": a * lam ** len(g) / (1 - lam) if lam < 1 else float("inf")}


def fit_saturating(V):
    """V_r = Vinf - c*lam^r on the LEVEL series (r = 0..R).  Needs >= 3 levels."""
    V = np.asarray(V, dtype=float)
    if len(V) < 3:
        return None
    r = np.arange(len(V))

    def res(p):
        return p[0] - p[1] * p[2] ** r - V
    s = least_squares(res, x0=[V[-1] + 0.005, max(V[-1] - V[0], 1e-4), 0.6],
                      bounds=([0, 0, LAM_LO], [1, 1, LAM_HI]))
    Vinf, c, lam = map(float, s.x)
    R = len(V) - 1
    return {"Vinf": Vinf, "c": c, "lam": lam,
            "at_bound": bool(lam <= LAM_LO + 1e-6 or lam >= LAM_HI - 1e-6),
            "next": c * lam ** R * (1 - lam),
            "remaining": Vinf - V[-1]}


def group_boot_delta(y, pa, pb, groups, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    out = np.array(out)
    return {"lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5)),
            "sd": float(out.std()), "mean": float(out.mean())}


# ------------------------------------------------------------------ species ---
def pilot_species_series(tau=E.TAU):
    """Per-round NEW-species counts for the 56 mined criteria, measured against
    everything seen before that round (bank concepts + earlier rounds)."""
    bank = E.bank_concept_texts()
    bank_txt = [E.crit_text(k, v) for k, v in bank.items()]

    mined = {}
    for r in (1, 2, 3, 4):
        prop = {c["id"]: c for c in json.loads(
            (CLOSURE / f"round{r}_proposals_blinded.json").read_text())["criteria"]}
        routing = json.loads((CLOSURE / f"round{r}_routing_final.json").read_text())
        rep = json.loads((CLOSURE / f"round{r}_score_report.json").read_text())
        collapsed = {k for k, v in rep["per_criterion"].items() if v.get("collapsed")}
        mined[r] = [E.crit_text(prop[x["blind_id"]]["name"], prop[x["blind_id"]]["instruction"])
                    for x in routing["final"]
                    if x["final_route"] == "A" and x["blind_id"] not in collapsed]

    all_txt = bank_txt + [t for r in (1, 2, 3, 4) for t in mined[r]]
    Emb = E.embed(all_txt)
    S = Emb @ Emb.T
    nb = len(bank_txt)

    series, seen_end = [], nb
    for r in (1, 2, 3, 4):
        k = len(mined[r])
        start, end = seen_end, seen_end + k
        new, recap_prior = 0, 0
        for j in range(start, end):
            prior_max = S[j, :start].max() if start else -1
            if prior_max >= tau:
                recap_prior += 1
            else:
                new += 1
        # within-round duplicates collapse into one species
        sub = S[start:end, start:end].copy()
        np.fill_diagonal(sub, -1)
        lab = E.single_linkage(sub, tau)
        within_dupes = k - len(set(lab))
        series.append({"round": r, "n_proposals": k, "new_species_vs_prior": new - within_dupes,
                       "recaptured_from_prior": recap_prior, "within_round_dupes": within_dupes})
        seen_end = end
    return series


def main():
    z = np.load(CLOSURE / "round_preds_all.npz", allow_pickle=True)
    y, nt, held = z["y"], z["ntitle"].astype(str), z["held"].astype(bool)
    va = {r: z[f"va_nl_round{r}"] for r in range(5)}

    lev = [float(roc_auc_score(y[held], va[r][held])) for r in range(5)]
    gains = [lev[r + 1] - lev[r] for r in range(4)]
    bands = [group_boot_delta(y[held], va[r + 1][held], va[r][held], nt[held]) for r in range(4)]

    out = {
        "population": "honest dense-held-out rows (n=1,244); VA_nl = OOF in FIT+MINE, "
                      "held-out prediction on MONITOR (identical construction to the pilot)",
        "levels_r0_r4": lev,
        "marginal_gains_g1_g4": gains,
        "readout_noise_band_per_step": [
            {"step": f"r{r}->r{r+1}", "actual": gains[r], **bands[r]} for r in range(4)],
        "auc_gain_backtest": [],
        "remaining_mass_backtest": [],
    }

    # ---------------------------------------------------- (a1) AUC-gain backtest
    for r in (1, 2, 3):
        obs, actual = gains[:r], gains[r]
        geo = fit_geometric(obs)
        sat = fit_saturating(lev[:r + 1])
        band = bands[r]
        rec = {"fit_on_rounds": f"g1..g{r}", "predicting": f"g{r+1}",
               "actual": actual, "actual_band_sd": band["sd"],
               "actual_band_95": [band["lo"], band["hi"]]}
        if geo is None:
            # a single gain does not identify a decay rate: bracket over lam instead
            rec["geometric"] = {"status": "NOT IDENTIFIED (1 point)",
                                "bracket_lam_0.4_to_0.9": [obs[0] * l for l in (0.4, 0.9)]}
            rec["lands_geometric"] = None
        else:
            rec["geometric"] = geo
            rec["err_geometric"] = geo["next"] - actual
            rec["lands_geometric"] = bool(abs(geo["next"] - actual) <= 1.96 * band["sd"])
        if sat is None:
            rec["saturating"] = {"status": "NOT IDENTIFIED (<3 levels)"}
            rec["lands_saturating"] = None
        else:
            rec["saturating"] = sat
            rec["err_saturating"] = sat["next"] - actual
            rec["lands_saturating"] = bool(abs(sat["next"] - actual) <= 1.96 * band["sd"])
        # naive baselines the estimator must beat
        rec["baseline_persistence"] = {"pred": obs[-1], "err": obs[-1] - actual,
                                       "lands": bool(abs(obs[-1] - actual) <= 1.96 * band["sd"])}
        rec["baseline_zero"] = {"pred": 0.0, "err": -actual,
                                "lands": bool(abs(actual) <= 1.96 * band["sd"])}
        out["auc_gain_backtest"].append(rec)

        # -------------------------------------------- (a2) remaining-mass backtest
        realised_rest = sum(gains[r:])
        rm = {"fit_on_rounds": f"g1..g{r}", "realised_remaining_g{}+_through_g4".format(r + 1): realised_rest}
        if geo is not None:
            rm["geometric_predicted_remaining_infinite_tail"] = geo["remaining"]
            rm["geometric_predicted_remaining_truncated_to_r4"] = float(
                sum(geo["a"] * geo["lam"] ** (k - 1) for k in range(r + 1, 5)))
        if sat is not None:
            rm["saturating_predicted_remaining"] = sat["remaining"]
        out["remaining_mass_backtest"].append(rm)

    # ------------------------------------------------- (a3) new-species backtest
    sp = pilot_species_series()
    out["pilot_species_series_tau{:.2f}".format(E.TAU)] = sp
    pooled_new = [s["new_species_vs_prior"] for s in sp]
    ks = [s["n_proposals"] for s in sp]
    gt = []
    for r in (1, 2, 3):
        # Good-Turing predictor: expected new species in the next batch of size k
        # = k * (missing mass estimated from the pooled sample so far).
        n_seen = sum(ks[:r])
        f1 = sum(pooled_new[:r])          # every un-recaptured proposal is a singleton
        mhat = f1 / n_seen
        pred = ks[r] * mhat
        gt.append({"fit_on_rounds": f"r1..r{r}", "predicting": f"r{r+1}",
                   "missing_mass_hat": mhat, "predicted_new_species": pred,
                   "actual_new_species": pooled_new[r],
                   "err": pred - pooled_new[r]})
    out["new_species_backtest"] = gt
    out["new_species_backtest_verdict"] = (
        "DEGENERATE BY DESIGN: the pilot's rounds were sequential and each proposer was "
        "told not to duplicate the current bank, so observed recapture is ~0, the "
        "Good-Turing missing mass pins near 1.0, and the predictor degenerates to "
        "'the next batch is all new'.  It is trivially accurate and carries no "
        "information.  This is precisely the defect the M1 sealed fleet removes."
    )

    (HERE / "m2_backtest.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("pilot_species_series_tau0.79",)}, indent=1))
    print("\nwrote", HERE / "m2_backtest.json")


if __name__ == "__main__":
    main()
