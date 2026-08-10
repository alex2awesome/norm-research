#!/usr/bin/env python3
"""Multi-y study: claim metrics (verification tiers + quality + novelty) against EVERY outcome:
  peer:  y1 accept/reject (institutional)   y2 citation percentile (crowd/impact, venue x year)
  pr:    y1 k>=3 coverage (institutional)
  news:  y1 homepage placement (institutional)  y2 twitter engagement pct (crowd)
Assembles per-doc metric tables from existing outputs; computes grouped AUC per (metric-set, y).
This is E3 (outcome-conditional structure): evidence metrics should track institutional y IF the
substantiation-cluster story holds; style/quality metrics may track crowd y.
Run on sk3 after claim-quality lands: python -m methods.claim_verification.run_multi_y"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")

def load_jsonl(path, key="doc_id"):
    out = {}
    if not os.path.exists(path): return out
    for ln in open(path):
        try:
            r = json.loads(ln); out[str(r[key])] = r
        except Exception: pass
    return out

def grouped_auc(X, y, g, label):
    mk = ~np.all(np.isnan(X), axis=1)
    X, y, g = X[mk], y[mk], np.asarray(g)[mk]
    if len(set(y)) < 2 or len(y) < 100:
        print(f"  {label:42} SKIP (n={len(y)})", flush=True); return
    folds = min(5, len(set(g)))   # peer sample spans only 2 years -> 2 year-holdout folds
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    try:
        a = cross_val_score(pipe, X, y, cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                            groups=g, scoring="roc_auc")
        print(f"  {label:42} AUC={np.nanmean(a):.4f} (n={len(y)}, {folds}f)", flush=True)
    except Exception as e:
        print(f"  {label:42} ERR {str(e)[:40]}", flush=True)

def peer_study():
    print("\n===== PEER REVIEW: accept-y vs citation-y =====", flush=True)
    tiers = pd.read_csv(f"{ROOT}/outputs/tiered_peer/tiered_metrics.csv")
    tiers["id"] = tiers.id.astype(str)
    cq = load_jsonl(os.path.join(EB, "claimquality_peerintro.jsonl"))
    cit = pd.read_csv(f"{ROOT}/datasets/peer-review/openalex_citations/openalex_citations_v3.csv.gz",
                      compression="gzip")
    cit["id"] = cit.id.astype(str)
    cit = cit[pd.to_numeric(cit.percentile, errors="coerce").notna()]
    cit["pct"] = pd.to_numeric(cit.percentile)
    M = tiers.copy()
    for d in ("specificity", "ambition", "surprisingness", "falsifiability", "elegance", "mean"):
        M[f"cq_{d}"] = M.id.map(lambda i: (cq.get(i) or {}).get(f"cq_{d}"))
    M = M.merge(cit[["id", "pct", "cited_by_count"]], on="id", how="left")
    ev_cols = ["t1_support", "t1_echo", "t3_support", "t3_echo", "t4_support", "t4_echo", "novelty"]
    cq_cols = [c for c in M.columns if c.startswith("cq_")]
    y1 = M.y.values; g = M.year.astype(str).values
    print(f"[peer] merged: {len(M)} docs, {M.pct.notna().sum()} with citation pct, "
          f"{M[cq_cols[0]].notna().sum() if cq_cols else 0} with claim quality", flush=True)
    print("\n-- y = ACCEPT/REJECT (institutional) --", flush=True)
    grouped_auc(M[ev_cols].values.astype(float), y1, g, "evidence tiers")
    if cq_cols: grouped_auc(M[cq_cols].values.astype(float), y1, g, "claim quality (A)")
    if cq_cols: grouped_auc(M[ev_cols + cq_cols].values.astype(float), y1, g, "evidence + quality")
    sub = M[M.pct.notna()].copy()
    if len(sub) > 100:
        y2 = (sub.pct >= sub.pct.median()).astype(int).values
        g2 = sub.year.astype(str).values
        print("\n-- y = CITATION pct>=median (crowd/impact) --", flush=True)
        grouped_auc(sub[ev_cols].values.astype(float), y2, g2, "evidence tiers")
        if cq_cols: grouped_auc(sub[cq_cols].values.astype(float), y2, g2, "claim quality (A)")
        if cq_cols: grouped_auc(sub[ev_cols + cq_cols].values.astype(float), y2, g2, "evidence + quality")
        # accepted-only citation prediction (decouples from acceptance)
        acc = sub[sub.y == 1]
        if len(acc) > 100:
            ya = (acc.pct >= acc.pct.median()).astype(int).values
            print("\n-- y = CITATION pct (ACCEPTED papers only) --", flush=True)
            grouped_auc(acc[ev_cols].values.astype(float), ya, acc.year.astype(str).values, "evidence tiers")
            if cq_cols: grouped_auc(acc[cq_cols].values.astype(float), ya, acc.year.astype(str).values, "claim quality (A)")

def pr_study():
    print("\n===== PRESS RELEASES: coverage-y =====", flush=True)
    tiers = pd.read_csv(f"{ROOT}/outputs/tiered_pr/tiered_metrics.csv")
    tiers["id"] = tiers.id.astype(str)
    cq = load_jsonl(os.path.join(EB, "claimquality_pr.jsonl"))
    M = tiers.copy()
    for d in ("specificity", "ambition", "surprisingness", "falsifiability", "elegance", "mean"):
        M[f"cq_{d}"] = M.id.map(lambda i: (cq.get(i) or {}).get(f"cq_{d}"))
    ev_cols = ["t1_support", "t1_echo", "t3_support", "t3_echo", "t4_support", "t4_echo"]
    cq_cols = [c for c in M.columns if c.startswith("cq_")]
    y = M.y.values; g = M.company.values
    print(f"[pr] {len(M)} docs, {M[cq_cols[0]].notna().sum() if cq_cols else 0} with claim quality", flush=True)
    print("\n-- y = k>=3 COVERAGE (institutional) --", flush=True)
    grouped_auc(M[ev_cols].values.astype(float), y, g, "evidence tiers (no-leak)")
    if cq_cols: grouped_auc(M[cq_cols].values.astype(float), y, g, "claim quality (A)")
    if cq_cols: grouped_auc(M[ev_cols + cq_cols].values.astype(float), y, g, "evidence + quality")

if __name__ == "__main__":
    peer_study()
    pr_study()
    print("\nMULTI_Y_DONE", flush=True)
