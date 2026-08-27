#!/usr/bin/env python3
"""Recovery-audit Q1: dose-response of rediscovery vs held-out concept strength.

Per-concept table from m3_recall.json (primary full-recall instrument, 24 held-out
+ 24 stratum-matched retained controls), then:
  * logistic fit of match_primary on strength = |alone_AUC - .5| (pooled, since the
    measured depletion lift is ~0; also with a heldout/control covariate),
  * rank-based readout (Mann-Whitney AUC of strength for matched vs unmatched),
  * binned / top-k rates to answer: does the curve KEEP RISING at the top end or
    saturate?
CPU only, seconds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RMM = HERE.parent

rec = json.loads((RMM / "m3_recall.json").read_text())["records"]
adj = json.loads((RMM / "m3_adjudicated.json").read_text())

rows = []
for r in rec:
    rows.append({
        "rep": r["rep"], "concept": r["concept"], "kind": r["kind"],
        "stratum": r["stratum"], "alone_auc": r["alone_auc_fitmine"],
        "strength": abs(r["alone_auc_fitmine"] - 0.5),
        "matched": bool(r["match_primary"]),
        "matched_either": bool(r["match_either"]),
        "families": r["families"], "proposers": r["proposers"],
        "matched_pids": r["matched_pids"],
        "n_matching_proposers": len(set(r["proposers"])),
    })

x = np.array([r["strength"] for r in rows])
m = np.array([r["matched"] for r in rows], dtype=float)
kind = np.array([1.0 if r["kind"] == "heldout" else 0.0 for r in rows])


def logistic_fit(X, y, l2=1e-6, iters=500):
    """Newton-Raphson logistic with tiny ridge; X includes intercept col."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - l2 * w
        W = p * (1 - p)
        H = -(X.T * W) @ X - l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ w))
    # observed-information SEs
    W = p * (1 - p)
    cov = np.linalg.inv((X.T * W) @ X + l2 * np.eye(X.shape[1]))
    se = np.sqrt(np.diag(cov))
    return w, se


def wald(w, se):
    z = w / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(z), float(p)


out = {"n": len(rows), "n_heldout": int(kind.sum()), "n_control": int((1 - kind).sum())}

# --- pooled logistic: matched ~ strength ------------------------------------
X1 = np.column_stack([np.ones_like(x), x])
w1, se1 = logistic_fit(X1, m)
z1, p1 = wald(w1[1], se1[1])
out["logistic_pooled"] = {
    "coef_per_unit_strength": float(w1[1]), "se": float(se1[1]), "z": z1, "p": p1,
    "coef_per_0.01_strength": float(w1[1] * 0.01),
    "note": "matched ~ |aloneAUC-.5|, heldout+control pooled (lift ~ 0)",
}

# --- logistic with kind covariate -------------------------------------------
X2 = np.column_stack([np.ones_like(x), x, kind])
w2, se2 = logistic_fit(X2, m)
out["logistic_with_kind"] = {
    "coef_strength": float(w2[1]), "se_strength": float(se2[1]),
    "z_strength": wald(w2[1], se2[1])[0], "p_strength": wald(w2[1], se2[1])[1],
    "coef_heldout": float(w2[2]), "se_heldout": float(se2[2]),
    "p_heldout": wald(w2[2], se2[2])[1],
}

# --- heldout-only logistic ---------------------------------------------------
hx, hm = x[kind == 1], m[kind == 1]
wh, seh = logistic_fit(np.column_stack([np.ones_like(hx), hx]), hm)
out["logistic_heldout_only"] = {"coef": float(wh[1]), "se": float(seh[1]),
                                "z": wald(wh[1], seh[1])[0], "p": wald(wh[1], seh[1])[1]}
cx, cm = x[kind == 0], m[kind == 0]
wc, sec = logistic_fit(np.column_stack([np.ones_like(cx), cx]), cm)
out["logistic_control_only"] = {"coef": float(wc[1]), "se": float(sec[1]),
                                "z": wald(wc[1], sec[1])[0], "p": wald(wc[1], sec[1])[1]}

# --- rank-based: Mann-Whitney AUC of strength, matched vs unmatched ---------
def mw_auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))

out["rank_auc_strength_matched_vs_not"] = {
    "pooled": mw_auc(x[m == 1], x[m == 0]),
    "heldout": mw_auc(hx[hm == 1], hx[hm == 0]),
    "control": mw_auc(cx[cm == 1], cx[cm == 0]),
}
# permutation p for pooled rank AUC
rng = np.random.default_rng(0)
obs = out["rank_auc_strength_matched_vs_not"]["pooled"]
perm = [mw_auc(x[s == 1], x[s == 0]) for s in (rng.permutation(m) for _ in range(10000))]
out["rank_auc_pooled_perm_p"] = float(np.mean([abs(pp - .5) >= abs(obs - .5) for pp in perm]))

# --- binned rates + top-end behaviour ---------------------------------------
order = np.argsort(-x)
sorted_rows = [rows[i] for i in order]
out["sorted_by_strength_top10"] = [
    {"concept": r["concept"][:60], "kind": r["kind"], "rep": r["rep"],
     "alone_auc": round(r["alone_auc"], 4), "matched": r["matched"],
     "n_matching_proposers": r["n_matching_proposers"]}
    for r in sorted_rows[:10]]

for k in (5, 8, 10):
    top = sorted_rows[:k]
    out[f"match_rate_top{k}_pooled"] = sum(r["matched"] for r in top) / k
    toph = [r for r in sorted_rows if r["kind"] == "heldout"][:k]
    out[f"match_rate_top{k}_heldout"] = sum(r["matched"] for r in toph) / k

# quartiles of strength, pooled
qs = np.quantile(x, [0.25, 0.5, 0.75])
bins = np.digitize(x, qs)
out["quartile_bins"] = []
for b in range(4):
    sel = bins == b
    out["quartile_bins"].append({
        "strength_range": [float(x[sel].min()), float(x[sel].max())],
        "n": int(sel.sum()),
        "match_rate": float(m[sel].mean()),
        "match_rate_heldout": float(m[sel & (kind == 1)].mean()) if (sel & (kind == 1)).any() else None,
        "match_rate_control": float(m[sel & (kind == 0)].mean()) if (sel & (kind == 0)).any() else None,
    })

# isotonic check of saturation: predicted P(match) at top strength vs at .53
p_at = lambda s, w: float(1 / (1 + np.exp(-(w[0] + w[1] * s))))
out["logistic_pooled_predicted"] = {
    "at_alone_.505": p_at(.005, w1), "at_alone_.52": p_at(.02, w1),
    "at_alone_.545": p_at(.045, w1), "at_alone_.56": p_at(.06, w1),
    "at_alone_.607_bank_max": p_at(.107, w1),
}

# n matching proposers as a graded dose readout (depth, not just any-catch)
out["mean_matching_proposers_by_stratum"] = {}
for st in ("high", "mid", "low"):
    sel = [r for r in rows if r["stratum"] == st]
    out["mean_matching_proposers_by_stratum"][st] = {
        "heldout": float(np.mean([r["n_matching_proposers"] for r in sel if r["kind"] == "heldout"])),
        "control": float(np.mean([r["n_matching_proposers"] for r in sel if r["kind"] != "heldout"])),
    }

# full per-concept table
out["table"] = [
    {k: r[k] for k in ("rep", "concept", "kind", "stratum", "alone_auc", "strength",
                       "matched", "matched_either", "families", "proposers",
                       "matched_pids", "n_matching_proposers")}
    for r in sorted_rows]

(HERE / "q1_dose_response.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("table", "sorted_by_strength_top10")}, indent=1))
print("\nTOP 10 BY STRENGTH:")
for r in out["sorted_by_strength_top10"]:
    print(f"  {r['alone_auc']:.4f} {'MATCH' if r['matched'] else 'miss '} "
          f"[{r['kind']:7s}] x{r['n_matching_proposers']} {r['concept']}")
