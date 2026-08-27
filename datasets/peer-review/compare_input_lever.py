#!/usr/bin/env python3
"""Clean abstract-vs-fullpaper code-metric comparison on MATCHED 2400 papers.
Recomputes V_regex on the evidence abstracts; loads both code-score matrices (abstract-matched,
full-paper). All on the same 2400 ids -> no population confound. CPU-only."""
import json, pathlib, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, str(pathlib.Path("/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review")))
from score_va_gemma import v_features, V_NAMES  # reuse the 17 inline features

BASE = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review")

def auc(M, y):
    c = min(np.bincount(y).min(), 5)
    p = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5), StandardScaler(),
                      LogisticRegression(max_iter=3000, class_weight="balanced"))
    return float(cross_val_score(p, M, y, cv=StratifiedKFold(c, shuffle=True, random_state=0),
                                 scoring="roc_auc").mean())

def main():
    rows = [json.loads(l) for l in open(BASE/"peer_review_fullpaper_evidence.jsonl") if l.strip()]
    ids = [r["paper_id"] for r in rows]
    y = np.array([r["y"] for r in rows], dtype=int)
    Vreg = np.array([[v_features(r["abstract"])[n] for n in V_NAMES] for r in rows], dtype=float)

    ca = np.load(BASE/"peer_review_code_scores_abstract_matched.npz", allow_pickle=True)
    cf = np.load(BASE/"peer_review_code_scores_fullpaper.npz", allow_pickle=True)
    ca_i = {str(i): k for k, i in enumerate(ca["ids"])}
    cf_i = {str(i): k for k, i in enumerate(cf["ids"])}
    ica = np.array([ca_i[str(i)] for i in ids], dtype=int)
    icf = np.array([cf_i[str(i)] for i in ids], dtype=int)
    Vca = ca["X"][ica].astype(float)
    Vcf = cf["X"][icf].astype(float)

    print(f"matched n={len(ids)}  balance={dict(zip(*[x.tolist() for x in np.unique(y, return_counts=True)]))}")
    print(f"V_regex (17 inline, abstract)      : {auc(Vreg,y):.4f}")
    print(f"V_code ABSTRACT-matched (5)        : {auc(Vca,y):.4f}")
    print(f"V_code FULL-PAPER (5)              : {auc(Vcf,y):.4f}")
    print(f"V_regex + V_code(fullpaper)        : {auc(np.hstack([Vreg,Vcf]),y):.4f}")
    print()
    print("per-aspect (abstract vs fullpaper):")
    for j, aid in enumerate(ca["code_names"]):
        from sklearn.metrics import roc_auc_score
        a = roc_auc_score(y, Vca[:, j]); f = roc_auc_score(y, Vcf[:, j])
        print(f"  {aid}: abstract={a:.3f}  fullpaper={f:.3f}  d={f-a:+.3f}")

if __name__ == "__main__":
    main()
