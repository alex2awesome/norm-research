#!/usr/bin/env python3
"""Pooled vs venue-controlled V/A AUC from the saved raw scores + a paper_id->venue join.
Answers: how much of the pooled AUC is a venue-base-rate confound?"""
import csv, gzip, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review")
d = np.load(BASE / "peer_review_scores.npz", allow_pickle=True)
X = d["X"].astype(float); V = d["V"].astype(float); y = d["y"].astype(int)
ids = [str(i) for i in d["ids"]]

# paper_id -> venue from eval split (the sample was drawn from eval)
vmap = {}
with gzip.open(BASE / "splits/eval.csv.gz", "rt", errors="ignore") as fh:
    for r in csv.DictReader(fh):
        vmap[r.get("paper_id") or r.get("id")] = (r.get("venue") or "?", r.get("source") or "?")

def fam(venue, source):
    v = (venue or "").upper()
    if source == "f1000research" or "F1000" in v: return "F1000"
    if "ICLR" in v: return "ICLR"
    if "NEURIPS" in v: return "NeurIPS"
    if "ICML" in v: return "ICML"
    if "TMLR" in v: return "TMLR"
    if source == "peerread_legacy": return "legacy"
    return "other"

venues = [vmap.get(i, ("?", "?"))[0] for i in ids]
srcs   = [vmap.get(i, ("?", "?"))[1] for i in ids]
fams   = np.array([fam(v, s) for v, s in zip(venues, srcs)])

def auc_of(M, yy):
    if M.shape[1] == 0 or len(set(yy.tolist())) < 2: return None
    c = min(np.bincount(yy).min(), 5)
    if c < 2: return None
    pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5),
                         StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))
    return float(cross_val_score(pipe, M, yy, cv=StratifiedKFold(c, shuffle=True, random_state=0),
                                 scoring="roc_auc").mean())

print("=== sample family breakdown (n, accept_rate) ===")
for f in ["ICLR","NeurIPS","ICML","TMLR","F1000","legacy","other"]:
    m = fams == f
    if m.sum():
        print(f"  {f:8s} n={m.sum():4d}  accept={y[m].mean():.3f}")

# venue-detector sanity: does venue-family alone predict y in this sample?
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(fams.reshape(-1,1))
print(f"\n=== venue-family-ALONE AUC (confound magnitude) = {auc_of(oh, y)} ===")

def report(tag, mask):
    yy = y[mask]
    if len(set(yy.tolist())) < 2 or min(np.bincount(yy)) < 15:
        print(f"\n[{tag}] n={mask.sum()} accept={yy.mean():.3f} -- too few of a class, skipped")
        return None
    va = auc_of(np.hstack([V[mask], X[mask]]), yy)
    v  = auc_of(V[mask], yy); aa = auc_of(X[mask], yy)
    print(f"\n[{tag}] n={mask.sum()} accept={yy.mean():.3f}  "
          f"V={v}  A={aa}  V+A={va}  (A-V lift={round(va-v,4) if va and v else None})")
    return dict(tag=tag, n=int(mask.sum()), accept=float(yy.mean()), V_auc=v, A_auc=aa, VA_auc=va)

res = {}
res["pooled"] = report("POOLED (confounded)", np.ones(len(y), bool))
res["ICLR"]   = report("ICLR-only (clean expert-verdict)", fams == "ICLR")
res["F1000"]  = report("F1000-only (separate community)", fams == "F1000")
json.dump({k: v for k, v in res.items() if v}, open(BASE / "peer_review_va_venue.json", "w"), indent=2)
print("\nsaved -> peer_review_va_venue.json")
