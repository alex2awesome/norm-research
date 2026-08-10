#!/usr/bin/env python3
"""Per-round fleet missing-mass readout, BOTH tracks (freeze + FREEZE ADDENDUM).

Species come from the blind-adjudicated clustering in roundN_species_<track>.json --
never from an embedding threshold.  Given the species partition of the round's
P x k proposal pool:

  N      = proposals in the pool
  S_obs  = species observed
  f1,f2  = species seen exactly once / twice
  M_hat  = Good-Turing missing mass = f1 / N
           (probability the next independent proposal names an unseen species)

Uncertainty: LEAVE-ONE-PROPOSER-OUT jackknife (the primary statement).  A bootstrap
over proposers WITH replacement duplicates whole proposers and manufactures
doubletons, biasing M_hat down; it is reported only with that caveat.

Remaining-AUC bound, ODDS FORM only (the species form inherits Chao1's
non-identification at small f2 and is never quoted):

  R_hat = [M_hat / (1 - M_hat)] * mean_gain_per_round * lambda_hat

where mean_gain_per_round is the round's realised MONITOR_FULL VA_nl gain and
lambda_hat is the geometric decay fitted to the gain series so far (>= 2 gains).

Cross-proposer recapture = fraction of proposals whose species was also named by a
DIFFERENT proposer -- the diagnostic that showed the pilot's sequential design was
degenerate (recapture ~0 -> M_hat pinned at 1.00).

Usage: python missing_mass.py --round 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def species_stats(species, exclude_proposer=None):
    counts, prop_sets = [], []
    for s in species:
        mem = [m for m in s["members"]
               if exclude_proposer is None or not m.split(":", 1)[1].startswith(exclude_proposer)]
        if not mem:
            continue
        counts.append(len(mem))
        prop_sets.append({m.split(":", 1)[1].split("#")[0] for m in mem})
    counts = np.array(counts)
    N = int(counts.sum())
    if N == 0:
        return None
    f1 = int((counts == 1).sum())
    f2 = int((counts == 2).sum())
    recap = float(sum(c for c, ps in zip(counts, prop_sets) if len(ps) >= 2) / N)
    return {"N": N, "S_obs": int(len(counts)), "f1": f1, "f2": f2,
            "M_hat": f1 / N, "cross_proposer_recapture": recap,
            "n_species_ge2_families": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    r = a.round

    out = {"round": r, "tracks": {}}
    for track in ("a", "b"):
        p = HERE / f"round{r}_species_{track}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        sp = d["species"]
        base = species_stats(sp)
        base["n_species_ge2_families"] = int(sum(s["n_families"] >= 2 for s in sp))
        proposers = sorted({m.split(":", 1)[1].split("#")[0] for s in sp for m in s["members"]})
        jk = []
        for q in proposers:
            st = species_stats(sp, exclude_proposer=q)
            if st:
                jk.append(st["M_hat"])
        jk = np.array(jk)
        n = len(jk)
        # leave-one-out jackknife on M_hat
        theta = base["M_hat"]
        jk_mean = float(jk.mean()) if n else float("nan")
        bias = (n - 1) * (jk_mean - theta) if n else float("nan")
        se = float(np.sqrt((n - 1) / n * ((jk - jk_mean) ** 2).sum())) if n > 1 else float("nan")
        base.update({
            "P": len(proposers), "proposers": proposers,
            "n_families": len({f for s in sp for f in s["families"]}),
            "jackknife_M_hat_mean": jk_mean,
            "jackknife_bias_corrected": theta - bias if n else float("nan"),
            "jackknife_se": se,
            "jackknife_ci95": [theta - 1.96 * se, theta + 1.96 * se] if n > 1 else None,
            "loo_values": jk.tolist(),
        })

        # species accumulation curve (mean over stable-hash proposer orderings)
        import hashlib
        orders = []
        for seed in range(24):
            order = sorted(proposers, key=lambda q: hashlib.sha256(f"acc|{seed}|{q}".encode()).hexdigest())
            seen, curve = set(), []
            for q in order:
                for s in sp:
                    if any(m.split(":", 1)[1].startswith(q) for m in s["members"]):
                        seen.add(s["species_id"])
                curve.append(len(seen))
            orders.append(curve)
        acc = np.array(orders).mean(0)
        base["accumulation_mean"] = acc.round(2).tolist()
        base["marginal_new_species"] = np.diff(np.concatenate([[0], acc])).round(2).tolist()
        out["tracks"][track] = base

    # remaining-AUC odds bound (A track only -- B has no AUC target)
    res_path = HERE / f"round{r}_results.json"
    if res_path.exists() and "a" in out["tracks"]:
        res = json.loads(res_path.read_text())
        gains = [g["monitor_full"]["gain"] for g in res["gains"]]
        M = out["tracks"]["a"]["M_hat"]
        lam = None
        if len(gains) >= 2:
            g = np.array(gains)
            pos = g[g > 0]
            if len(pos) >= 2:
                lam = float(np.exp(np.polyfit(np.arange(len(pos)), np.log(pos), 1)[0]))
        last = gains[-1] if gains else 0.0
        out["remaining_auc_odds_form"] = {
            "M_hat_A": M,
            "last_gain": last,
            "lambda_hat": lam,
            "R_hat": (M / (1 - M)) * last * (lam if lam else 1.0) if M < 1 else None,
            "note": "odds form only; the species/Chao1 form is non-identified at small f2 "
                    "and is never quoted (missing-mass note s3.2).",
        }
    (HERE / f"round{r}_missing_mass.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({t: {k: v for k, v in d.items()
                          if k not in ("loo_values", "accumulation_mean", "proposers")}
                      for t, d in out["tracks"].items()}, indent=1))
    if "remaining_auc_odds_form" in out:
        print(json.dumps(out["remaining_auc_odds_form"], indent=1))


if __name__ == "__main__":
    main()
