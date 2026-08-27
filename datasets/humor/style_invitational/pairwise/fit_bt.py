#!/usr/bin/env python3
"""SI PAIRWISE — PHASE 1: Bradley-Terry fit + §6.5 commensurable readout.

MODEL.  P(i beats j, with i shown on side s) = sigmoid(theta_i - theta_j + gamma * s),
where s = +1 when i is displayed as ENTRY A and -1 when it is displayed as ENTRY B.
Penalised MLE: maximise the Bernoulli log-likelihood minus (lam/2)*||theta||^2.

  * gamma is the SIDE TERM the approval asked for. The probe measured a .53 side
    preference; estimating it here absorbs position bias into a single parameter instead
    of letting it leak into theta. The fitted gamma is reported and is itself the
    position-bias readout.
  * theta is identified only up to a constant WITHIN each week (every comparison is
    within-week), so theta is CENTRED PER WEEK after fitting. That is not a normalisation
    of convenience -- it is what the cell's construct means: y is "did the editor put this
    entry in the top tier of ITS OWN contest", so a within-contest score is the right
    object and a per-week intercept is a nuisance.
  * The L2 penalty is what keeps theta finite for items that win or lose all their
    comparisons (a week's best entry often does). lam is fixed at 1.0 and a sensitivity
    at lam in {0.3, 3.0} is reported; the readout must not hinge on it.

WHY THIS IS NOT CIRCULAR.  theta is fit from judge comparisons ONLY. No label enters the
fit at any point -- the judge is never shown tiers, and the optimiser never sees y. So
AUC(theta, y) is an honest label-blind readout. Note the asymmetry this creates in §6.5,
and it runs AGAINST theta: VA_nl is a label-FITTED grouped-OOF stack, T is a label-trained
dense model, and theta is label-free. A label-free score matching them is a stronger
result than the bare numbers suggest, and that is stated rather than left implicit.

READOUT (§6.5).  AUC of theta against y_top_tier on eval and on test SEPARATELY, beside
the rebuild note's same-rows VA_nl (eval .6165 / test .6042) and T (eval .6241 /
test .6237). Pooled and within-week both reported.

CPU only.  Usage: python fit_bt.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE / "phase1"
POP = HERE.parent / "va_v2" / "population.csv.gz"

# same-rows figures from notes/2026-08-10__si_mature_bank_rebuild.md §7d
REF = {"eval": {"VA_nl": 0.6165, "T": 0.6241, "T_seed_spread": 0.0351},
       "test": {"VA_nl": 0.6042, "T": 0.6237, "T_seed_spread": 0.0303}}


def load_answers():
    man = json.loads((ROOT / "si_prompt_manifest.json").read_text())
    jobs = {j["tag"]: j for j in man["jobs"]}
    ans, n_jobs, missing = {}, 0, 0
    for f in sorted((ROOT / "out").glob("*.json")):
        if f.stem not in jobs:
            continue
        d = json.loads(f.read_text())
        got = set()
        for a in d.get("answers", []):
            pid, ch = str(a.get("pair_id", "")).strip(), str(a.get("choice", "")).strip().upper()
            if ch in ("A", "B"):
                ans[pid] = ch
                got.add(pid)
        missing += len([p for p in jobs[f.stem]["pair_ids"] if p not in got])
        n_jobs += 1
    return ans, n_jobs, man["n_jobs"], missing


def fit(items, obs, lam=1.0):
    """obs = list of (idx_i, idx_j, side_i, y) with y=1 if i was chosen."""
    n = len(items)
    I = np.array([o[0] for o in obs])
    J = np.array([o[1] for o in obs])
    S = np.array([o[2] for o in obs], dtype=float)
    Y = np.array([o[3] for o in obs], dtype=float)

    def nll(p):
        th, g = p[:n], p[n]
        z = th[I] - th[J] + g * S
        # stable log-sigmoid
        ll = np.where(z >= 0, -np.log1p(np.exp(-z)), z - np.log1p(np.exp(z)))
        l0 = np.where(-z >= 0, -np.log1p(np.exp(z)), -z - np.log1p(np.exp(-z)))
        obj = -(Y * ll + (1 - Y) * l0).sum() + 0.5 * lam * (th ** 2).sum()
        s = 1.0 / (1.0 + np.exp(-z))
        r = (Y - s)
        gth = np.zeros(n)
        np.add.at(gth, I, -r)
        np.add.at(gth, J, r)
        gth += lam * th
        return obj, np.concatenate([gth, [-(r * S).sum()]])

    res = minimize(nll, np.zeros(n + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 3000})
    return res.x[:n], float(res.x[n]), res


def main():
    comps = json.loads((ROOT / "si_bt_comparisons.json").read_text())
    ans, n_jobs, n_jobs_total, missing = load_answers()
    pop = pd.read_csv(POP)
    pop = pop[~pop.is_fragment]
    pop = pop[pop.split.isin(["eval", "test"])]
    y = dict(zip(pop.row_id, pop.y_top_tier))
    wkm = dict(zip(pop.row_id, pop.week_id))
    spl = dict(zip(pop.row_id, pop.split))

    items = sorted({c["i"] for c in comps if c["arm"] == "GRAPH"} |
                   {c["j"] for c in comps if c["arm"] == "GRAPH"})
    idx = {r: k for k, r in enumerate(items)}

    obs, n_used, anchor_hits, anchor_n = [], 0, 0, 0
    swap_pairs = {}
    for c in comps:
        ch = ans.get(c["pair_id"])
        if ch is None:
            continue
        chose_i = (ch == c["i_side"])
        if c["arm"] == "ANCHOR_FRAGMENT":
            anchor_n += 1
            anchor_hits += int(chose_i)
            continue
        if c["arm"] == "SWAP":
            swap_pairs[c["swap_of"]] = chose_i
            # swap observations DO enter the likelihood: they are real comparisons at the
            # opposite side assignment and are exactly what identifies gamma
        s = 1.0 if c["i_side"] == "A" else -1.0
        obs.append((idx[c["i"]], idx[c["j"]], s, 1.0 if chose_i else 0.0))
        n_used += 1

    theta, gamma, res = fit(items, obs, lam=1.0)
    th = pd.Series(theta, index=items)
    wk = pd.Series({r: wkm[r] for r in items})
    theta_c = th - wk.map(th.groupby(wk).mean())      # centre within week

    out = {
        "n_jobs_read": n_jobs, "n_jobs_total": n_jobs_total,
        "n_unanswered_pair_slots": missing,
        "n_items": len(items), "n_comparisons_used": n_used,
        "fit": {"lambda": 1.0, "gamma_side_term": gamma,
                "gamma_note": "positive gamma = the entry shown as ENTRY A is favoured; "
                              "absorbed here instead of leaking into theta",
                "converged": bool(res.success), "nll": float(res.fun)},
        "anchors": {"n": anchor_n,
                    "acc": (anchor_hits / anchor_n) if anchor_n else None,
                    "kind": "ANCHOR_FRAGMENT only"},
    }
    if swap_pairs:
        cons = []
        orig = {c["pair_id"]: c for c in comps if c["arm"] == "GRAPH"}
        for pid, chose_i_sw in swap_pairs.items():
            ch = ans.get(pid)
            if ch is None:
                continue
            cons.append(int((ch == orig[pid]["i_side"]) == chose_i_sw))
        out["swap_consistency"] = {"n": len(cons),
                                   "consistency": float(np.mean(cons)) if cons else None}

    # ---------------- §6.5 commensurable readout -------------------------------
    rows = []
    for split in ("eval", "test"):
        ids = [r for r in items if spl[r] == split]
        yy = np.array([y[r] for r in ids])
        sc = theta_c.loc[ids].values
        pooled = float(roc_auc_score(yy, sc))
        # within-week
        num = tot = 0.0
        for w in {wkm[r] for r in ids}:
            sel = [k for k, r in enumerate(ids) if wkm[r] == w]
            if len(set(yy[sel])) < 2:
                continue
            num += len(sel) * roc_auc_score(yy[sel], sc[sel])
            tot += len(sel)
        rows.append({"split": split, "n": len(ids), "pos_rate": float(yy.mean()),
                     "theta_AUC_pooled": pooled,
                     "theta_AUC_within_week": float(num / tot) if tot else None,
                     "VA_nl_same_rows": REF[split]["VA_nl"],
                     "T_same_rows": REF[split]["T"],
                     "T_seed_spread": REF[split]["T_seed_spread"],
                     "theta_minus_VA_nl": pooled - REF[split]["VA_nl"],
                     "theta_minus_T": pooled - REF[split]["T"]})
    out["section_6_5"] = rows
    out["section_6_5_note"] = (
        "theta is LABEL-FREE (fit from judge comparisons only); VA_nl is a label-fitted "
        "grouped-OOF stack and T is a label-trained dense model. The asymmetry runs "
        "against theta, so a theta that matches them is a stronger result than the bare "
        "numbers show.")

    # lambda sensitivity
    sens = {}
    for lam in (0.3, 3.0):
        t2, g2, _ = fit(items, obs, lam=lam)
        s2 = pd.Series(t2, index=items)
        s2c = s2 - wk.map(s2.groupby(wk).mean())
        sens[str(lam)] = {
            "gamma": g2,
            **{sp: float(roc_auc_score([y[r] for r in items if spl[r] == sp],
                                       s2c.loc[[r for r in items if spl[r] == sp]].values))
               for sp in ("eval", "test")}}
    out["lambda_sensitivity"] = sens

    (ROOT / "si_bt_theta.json").write_text(json.dumps(
        {"theta_centered": {str(k): float(v) for k, v in theta_c.items()}}, indent=1))
    (ROOT / "si_bt_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
