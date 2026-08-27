#!/usr/bin/env python3
"""Aggregate peer-review V/A raw scores -> V vs V+A AUC table (peer_review_va.json).

Reads peer_review_scores.npz (from score_va_gemma.py). Reports:
  V_auc  = LR over programmatic V features only
  A_auc  = LR over A rubric scores only
  VA_auc = LR over [V features + A rubric scores]  (the nested "A bar")
plus per-metric univariate AUCs (A rubrics) and per-V-feature univariate AUCs.

Usage (base env w/ sklearn):
  python aggregate_va.py --npz peer_review_scores.npz --out ../../notebooks/data/peer_review_va.json
"""
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

def auc_of(M, y):
    if M.shape[1] == 0:
        return None
    pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5),
                         StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(pipe, M, y, cv=skf, scoring="roc_auc").mean())

def uni(M, names, y):
    out = []
    for j, nm in enumerate(names):
        col = M[:, j]; mask = ~np.isnan(col)
        if mask.sum() > 30 and len(set(y[mask].tolist())) == 2:
            out.append((str(nm), round(roc_auc_score(y[mask], col[mask]), 3), int(mask.sum())))
    out.sort(key=lambda t: -abs(t[1] - 0.5))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="peer_review_scores.npz")
    ap.add_argument("--out", default="../../notebooks/data/peer_review_va.json")
    ap.add_argument("--restrict-venue", default=None,
                    help="substring; keep only rows whose saved venue matches (confound control)")
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    X = d["X"].astype(float)          # A rubric scores  (n x n_a)
    V = d["V"].astype(float)          # V features       (n x n_v)
    y = d["y"].astype(int)
    a_names = list(d["a_names"]); v_names = list(d["v_names"])
    venues = list(d["venues"]) if "venues" in d.files else None

    if a.restrict_venue and venues is not None:
        keep = np.array([a.restrict_venue.lower() in str(v).lower() for v in venues])
        X, V, y = X[keep], V[keep], y[keep]
        venues = [venues[i] for i in range(len(keep)) if keep[i]]
        print(f"[restrict-venue={a.restrict_venue}] kept {keep.sum()}/{len(keep)} rows")

    v_auc  = auc_of(V, y)
    a_auc  = auc_of(X, y)
    va_auc = auc_of(np.hstack([V, X]), y)

    out = dict(
        domain="peer_review", label="expert-verdict (accept/reject)", input="abstract",
        split=str(d["split"]), n=int(len(y)),
        n_A=int(X.shape[1]), n_V=int(V.shape[1]),
        na_rate=float(d["na_rate"]),
        balance={int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        V_auc=v_auc, A_auc=a_auc, VA_auc=va_auc,
        A_minus_V=(round(va_auc - v_auc, 4) if (va_auc and v_auc) else None),
        top_A_metrics=uni(X, a_names, y)[:15],
        V_features=uni(V, v_names, y),
    )
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in
          ["n","n_V","n_A","na_rate","V_auc","A_auc","VA_auc","A_minus_V"]}, indent=2))
    print("top A:", out["top_A_metrics"][:6])
    print("top V:", out["V_features"][:6])
    print("AGG_DONE ->", a.out)

if __name__ == "__main__":
    main()
