#!/usr/bin/env python3
"""Three-y contrast, drill-down: (1) label-vs-label agreement on shared papers,
(2) per-metric (univariate AUC) comparison across preference variables.

Uses the same frozen inputs as aggregate_3y.py (union_scores.npz + cell jsonls).
Descriptive readout — no model fitting, threshold-free (AUC / rank stats only).
"""
import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

D = Path(__file__).resolve().parent
z = np.load(D / "union_scores.npz", allow_pickle=True)
X, Vf, nt = z["X"], z["V"], z["ntitle"]
a_names = [str(s) for s in z["a_names"]]
v_names = [str(s) for s in z["v_names"]]
idx = {nt[i]: i for i in range(len(nt))}

def valid_y(r):
    try:
        return int(float(r.get("judgement"))) in (0, 1)
    except (TypeError, ValueError):
        return False

ymap = {}
for name in ["verdict", "curation", "revealed"]:
    m = {}
    for line in open(D / f"{name}.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        k = r.get("ntitle")
        if k in idx and valid_y(r):
            m[k] = int(float(r["judgement"]))
    ymap[name] = m

res = {"label_agreement": {}, "per_metric": {}}

print("=" * 70)
print("PART 1 — which pieces are favored: label-vs-label agreement")
print("=" * 70)
for a, b in combinations(["verdict", "curation", "revealed"], 2):
    common = sorted(set(ymap[a]) & set(ymap[b]))
    if len(common) < 50:
        print(f"\n{a} x {b}: only {len(common)} common — skip")
        continue
    ya = np.array([ymap[a][k] for k in common])
    yb = np.array([ymap[b][k] for k in common])
    n11 = int(((ya == 1) & (yb == 1)).sum()); n10 = int(((ya == 1) & (yb == 0)).sum())
    n01 = int(((ya == 0) & (yb == 1)).sum()); n00 = int(((ya == 0) & (yb == 0)).sum())
    phi = np.corrcoef(ya, yb)[0, 1]
    p_b_given_a1 = n11 / max(n11 + n10, 1)
    p_b_given_a0 = n01 / max(n01 + n00, 1)
    # AUC of one label "predicting" the other (symmetric for binary)
    auc = roc_auc_score(yb, ya) if len(set(ya)) > 1 and len(set(yb)) > 1 else float("nan")
    res["label_agreement"][f"{a}_x_{b}"] = dict(
        n=len(common), n11=n11, n10=n10, n01=n01, n00=n00, phi=float(phi),
        p_b1_given_a1=p_b_given_a1, p_b1_given_a0=p_b_given_a0, auc=float(auc))
    print(f"\n{a} x {b}  (n={len(common)})")
    print(f"  2x2 [{a}=1&{b}=1: {n11} | {a}=1&{b}=0: {n10} | {a}=0&{b}=1: {n01} | {a}=0&{b}=0: {n00}]")
    print(f"  phi={phi:+.3f}   P({b}=1|{a}=1)={p_b_given_a1:.3f} vs P({b}=1|{a}=0)={p_b_given_a0:.3f}   label-label AUC={auc:.3f}")

print()
print("=" * 70)
print("PART 2 — which metrics are favored: univariate AUC per metric per y")
print("=" * 70)
# strict-common curation/revealed set (the apples-to-apples population) + verdict where labeled
common_cr = sorted(set(ymap["curation"]) & set(ymap["revealed"]))
rows = np.array([idx[k] for k in common_cr])
A = X[rows].astype(float)
V = Vf[rows].astype(float)
# median-impute NA per column (frozen design)
for M in (A, V):
    for j in range(M.shape[1]):
        col = M[:, j]
        nn = col[~np.isnan(col)]
        med = np.median(nn) if len(nn) else 0.0
        col[np.isnan(col)] = med

feat = np.column_stack([V, A])
fnames = [f"V:{n}" for n in v_names] + [f"A:{n}" for n in a_names]

aucs = {}
for yname in ["verdict", "curation", "revealed"]:
    mask = np.array([k in ymap[yname] for k in common_cr])
    if mask.sum() < 100:
        continue
    y = np.array([ymap[yname][k] for k, m in zip(common_cr, mask) if m])
    F = feat[mask]
    if len(set(y)) < 2:
        continue
    a_vec = []
    for j in range(F.shape[1]):
        col = F[:, j]
        a_vec.append(roc_auc_score(y, col) if col.std() > 0 else 0.5)
    aucs[yname] = np.array(a_vec)
    print(f"\n[{yname}] scored on {mask.sum()} of the {len(common_cr)} common papers")

print("\n-- rank agreement of per-metric AUC vectors (Spearman) --")
for a, b in combinations(list(aucs), 2):
    rho, p = spearmanr(aucs[a], aucs[b])
    res["per_metric"][f"spearman_{a}_x_{b}"] = dict(rho=float(rho), p=float(p))
    print(f"  {a} vs {b}: rho={rho:+.3f} (p={p:.2g})")

def top(yname, k=10):
    v = aucs[yname]
    order = np.argsort(-np.abs(v - 0.5))[:k]
    return [(fnames[j], float(v[j])) for j in order]

for yname in aucs:
    res["per_metric"][f"top_{yname}"] = top(yname)
    print(f"\n-- top-10 |AUC-.5| metrics for {yname} --")
    for n, v in top(yname):
        print(f"  {v:.3f}  {n}")

if "curation" in aucs and "revealed" in aucs:
    d = aucs["revealed"] - aucs["curation"]
    order = np.argsort(-d)
    print("\n-- biggest REVEALED-favoring metrics (AUC_rev - AUC_cur) --")
    fav_r = [(fnames[j], float(aucs['revealed'][j]), float(aucs['curation'][j])) for j in order[:10]]
    for n, r, c in fav_r:
        print(f"  rev {r:.3f} vs cur {c:.3f}  {n}")
    print("\n-- biggest CURATION-favoring metrics (AUC_cur - AUC_rev) --")
    fav_c = [(fnames[j], float(aucs['revealed'][j]), float(aucs['curation'][j])) for j in order[::-1][:10]]
    for n, r, c in fav_c:
        print(f"  rev {r:.3f} vs cur {c:.3f}  {n}")
    res["per_metric"]["revealed_favoring"] = fav_r
    res["per_metric"]["curation_favoring"] = fav_c

(D / "metric_contrast_3y.json").write_text(json.dumps(res, indent=2))
print(f"\nwrote {D / 'metric_contrast_3y.json'}")
