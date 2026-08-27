#!/usr/bin/env python3
"""Per-agency AUC of COMBINED verifiable features (V shallow + V_deep) and the
full stack (V + V_deep + A) on the agree-y — extends nc_agree_per_agency.json
(which has the arms separately) for the org-size figure.

Replication gate: recompute Vdeep per agency with this pipeline and compare to
the stored column; proceed only if median |delta| < .03, and report it.
Pipeline (matches y_audit family): docket-GroupKFold(min(5,n_dockets)),
0.5-const impute, StandardScaler + LogisticRegression(balanced, C=1),
pooled OOF AUC per agency. Same code path for every feature set.
"""
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from aggregate_nc_multiy import (MATCHED_SHARDS, SAMPLE_JSONL, LABELS_FULL,
                                 load_shard_scores, load_jsonl_texts, load_label_ys)
from aggregate_vat_nc import v_features, V_NAMES

dv = np.load(D / "nc_deepv3_scores.npz", allow_pickle=True)
dv_ids = [str(i) for i in dv["doc_id"]]
dv_by_id = {i: dv["X"][k] for k, i in enumerate(dv_ids)}

X_m, docket_m, agency_m, _ = load_shard_scores(MATCHED_SHARDS)
text_m = load_jsonl_texts(SAMPLE_JSONL)
label_ys = load_label_ys(LABELS_FULL)

ids = [i for i in dv_by_id if i in X_m and i in label_ys
       and label_ys[i]["agree_vs_disagree"] in (0, 1)]
print(f"joined ids with deepv+A+agree-y: {len(ids)}")

y_all = np.array([label_ys[i]["agree_vs_disagree"] for i in ids])
agency = np.array([agency_m[i] for i in ids])
docket = np.array([docket_m[i] for i in ids])
Vs = np.array([[v_features(text_m.get(i, ""))[n] for n in V_NAMES] for i in ids])
Vd = np.array([dv_by_id[i] for i in ids], dtype=float)
Aa = np.array([X_m[i] for i in ids], dtype=float)

FEATS = {
    "Vdeep_repro": Vd,
    "Vcombined": np.column_stack([Vs, Vd]),
    "ALL": np.column_stack([Vs, Vd, Aa]),
}


def agency_auc(M, y, groups):
    k = min(5, len(set(groups.tolist())))
    if k < 2 or len(np.unique(y)) < 2:
        return float("nan")
    pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5),
                         StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=0).split(
            np.zeros(len(y)), y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe.fit(M[tr], y[tr])
        oof[te] = pipe.predict_proba(M[te])[:, 1]
    m = ~np.isnan(oof)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], oof[m]))


stored = json.load(open(D / "nc_agree_per_agency.json"))
merged = json.load(open(D / "nc_size_vat_merged.json"))
size_by_ag = {r["agency"]: r for r in merged}

out = {}
deltas = []
for ag in sorted(set(agency)):
    sel = agency == ag
    if sel.sum() < 60:
        continue
    row = {"n": int(sel.sum())}
    for fname, M in FEATS.items():
        row[fname] = agency_auc(M[sel], y_all[sel], docket[sel])
    if ag in stored and np.isfinite(row["Vdeep_repro"]):
        row["Vdeep_stored"] = stored[ag]["Vdeep"]
        deltas.append(abs(row["Vdeep_repro"] - stored[ag]["Vdeep"]))
    out[ag] = row

deltas = np.array(deltas)
print(f"\nreplication gate: n={len(deltas)} median|delta|={np.median(deltas):.4f} "
      f"max={deltas.max():.4f} -> {'PASS' if np.median(deltas) < .03 else 'FAIL'}")

print("\n| agency | n | Vdeep stored | Vdeep repro | V-combined | ALL (V+Vdeep+A) |")
print("|---|---|---|---|---|---|")
for ag, r in sorted(out.items()):
    print(f"| {ag} | {r['n']} | {r.get('Vdeep_stored', float('nan')):.3f} | "
          f"{r['Vdeep_repro']:.3f} | {r['Vcombined']:.3f} | {r['ALL']:.3f} |")

# correlations vs org-size variables
print("\n| outcome | vs dockets rho (p, n) | vs fte rho (p, n) |")
print("|---|---|---|")
res_corr = {}
for fname in ("Vdeep_repro", "Vcombined", "ALL"):
    line = f"| {fname} |"
    for xk in ("dockets", "fte"):
        pts = [(size_by_ag[ag][xk], out[ag][fname]) for ag in out
               if ag in size_by_ag and size_by_ag[ag].get(xk) is not None
               and not (isinstance(size_by_ag[ag][xk], float) and np.isnan(size_by_ag[ag][xk]))
               and np.isfinite(out[ag][fname])]
        x = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
        rho, p = spearmanr(x, v)
        res_corr[f"{fname}__{xk}"] = dict(rho=float(rho), p=float(p), n=len(x))
        line += f" {rho:+.3f} (p={p:.3f}, n={len(x)}) |"
    print(line)

json.dump({"per_agency": out, "correlations": res_corr,
           "replication_gate": {"median_abs_delta": float(np.median(deltas)),
                                 "max_abs_delta": float(deltas.max())}},
          open(D / "nc_orgsize_combined.json", "w"), indent=2)
print(f"\nwrote {D / 'nc_orgsize_combined.json'}")
