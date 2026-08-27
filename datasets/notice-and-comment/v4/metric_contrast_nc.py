#!/usr/bin/env python3
"""N&C multi-y drill-down: (1) label-vs-label agreement on shared comments,
(2) per-metric (univariate AUC) comparison across preference variables.

Mirrors ../../peer-review/vat_3y/metric_contrast_3y.py; uses the same frozen
inputs as aggregate_nc_multiy.py (pre-GEPA shards + labels_full). Descriptive
readout — no model fitting, threshold-free (AUC / rank stats only).

Label universes (see aggregate_nc_multiy.py):
  outcome-majority x agree-vs-disagree is the only informative pair — both are
  defined on matched-labeled comments. responded-or-not is constant (y=1) on
  the matched side, so any same-item comparison against it is structurally
  degenerate; we report that fact, not a number.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

D = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(D))
from aggregate_nc_multiy import (  # noqa: E402
    MATCHED_SHARDS, SAMPLE_JSONL, LABELS_FULL,
    load_shard_scores, load_jsonl_texts, load_label_ys,
)
from aggregate_vat_nc import v_features, V_NAMES  # noqa: E402

X_m, docket_m, agency_m, a_names = load_shard_scores(MATCHED_SHARDS)
text_m = load_jsonl_texts(SAMPLE_JSONL)
label_ys = load_label_ys(LABELS_FULL)

ymap = {"outcome-majority": {}, "agree-vs-disagree": {}}
for did in X_m:
    ls = label_ys.get(did)
    if not ls:
        continue
    if ls["outcome_majority"] in (0, 1):
        ymap["outcome-majority"][did] = ls["outcome_majority"]
    if ls["agree_vs_disagree"] in (0, 1):
        ymap["agree-vs-disagree"][did] = ls["agree_vs_disagree"]

res = {"label_agreement": {}, "per_metric": {}}

print("=" * 70)
print("PART 1 — which pieces are favored: label-vs-label agreement")
print("=" * 70)
a, b = "outcome-majority", "agree-vs-disagree"
common = sorted(set(ymap[a]) & set(ymap[b]))
ya = np.array([ymap[a][k] for k in common])
yb = np.array([ymap[b][k] for k in common])
n11 = int(((ya == 1) & (yb == 1)).sum()); n10 = int(((ya == 1) & (yb == 0)).sum())
n01 = int(((ya == 0) & (yb == 1)).sum()); n00 = int(((ya == 0) & (yb == 0)).sum())
phi = float(np.corrcoef(ya, yb)[0, 1])
p_b1_a1 = n11 / max(n11 + n10, 1)
p_b1_a0 = n01 / max(n01 + n00, 1)
auc_ll = float(roc_auc_score(yb, ya))
res["label_agreement"][f"{a}_x_{b}"] = dict(
    n=len(common), n11=n11, n10=n10, n01=n01, n00=n00, phi=phi,
    p_b1_given_a1=p_b1_a1, p_b1_given_a0=p_b1_a0, auc=auc_ll)
res["label_agreement"]["responded-or-not"] = (
    "constant y=1 on the matched-labeled side; same-item agreement vs the other "
    "two y's is structurally undefined (see aggregate_nc_multiy.py caveat)")
print(f"\n{a} x {b}  (n={len(common)})")
print(f"  2x2 [{a}=1&{b}=1: {n11} | {a}=1&{b}=0: {n10} | {a}=0&{b}=1: {n01} | {a}=0&{b}=0: {n00}]")
print(f"  phi={phi:+.3f}   P({b}=1|{a}=1)={p_b1_a1:.3f} vs P({b}=1|{a}=0)={p_b1_a0:.3f}   label-label AUC={auc_ll:.3f}")

print()
print("=" * 70)
print("PART 2 — which metrics are favored: univariate AUC per metric per y")
print("=" * 70)
A = np.array([X_m[k] for k in common], dtype=float)
V = np.array([[v_features(text_m.get(k, ""))[n] for n in V_NAMES] for k in common], dtype=float)
for M in (A, V):
    for j in range(M.shape[1]):
        col = M[:, j]
        nn = col[~np.isnan(col)]
        med = np.median(nn) if len(nn) else 0.0
        col[np.isnan(col)] = med

feat = np.column_stack([V, A])
fnames = [f"V:{n}" for n in V_NAMES] + [f"A:{n}" for n in a_names]

aucs = {}
for yname in (a, b):
    y = np.array([ymap[yname][k] for k in common])
    a_vec = []
    for j in range(feat.shape[1]):
        col = feat[:, j]
        a_vec.append(roc_auc_score(y, col) if col.std() > 0 else 0.5)
    aucs[yname] = np.array(a_vec)
    print(f"\n[{yname}] scored on all {len(common)} strict-common comments")

rho, p = spearmanr(aucs[a], aucs[b])
res["per_metric"][f"spearman_{a}_x_{b}"] = dict(rho=float(rho), p=float(p))
print(f"\n-- rank agreement of per-metric AUC vectors (Spearman) --")
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

d = aucs[b] - aucs[a]
order = np.argsort(-d)
print(f"\n-- biggest {b}-favoring metrics (AUC_agr - AUC_out) --")
fav_b = [(fnames[j], float(aucs[b][j]), float(aucs[a][j])) for j in order[:10]]
for n, vb, va in fav_b:
    print(f"  agr {vb:.3f} vs out {va:.3f}  {n}")
print(f"\n-- biggest {a}-favoring metrics (AUC_out - AUC_agr) --")
fav_a = [(fnames[j], float(aucs[b][j]), float(aucs[a][j])) for j in order[::-1][:10]]
for n, vb, va in fav_a:
    print(f"  agr {vb:.3f} vs out {va:.3f}  {n}")
res["per_metric"]["agree_favoring"] = fav_b
res["per_metric"]["outcome_favoring"] = fav_a

(D / "metric_contrast_nc.json").write_text(json.dumps(res, indent=2))
print(f"\nwrote {D / 'metric_contrast_nc.json'}")
