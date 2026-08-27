#!/usr/bin/env python3
"""E1 pilot: seam-disciplined claim/sourcing battery on 600 balanced k>=3 press releases.
Arms evaluated SEPARATELY (company-grouped AUC): CODE sourcing | HYBRID retrieval | LLM support
| +stacks. Certified evidence-op marginal = real-arm minus null-twin (metric_seam pattern).
Run on sk3 (Gemma :8006): python -m methods.claim_verification.run_pr_pilot"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from claim_verification.seam_metrics import run_seam_battery

CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma",
       "doc_kind": "press release", "max_claims": 4}
N = 600

df = pd.read_parquet("datasets/press-releases/press_release_deconfounded.parquet")
# k>=3 label (established showcase recipe from run_A_layer_k3.py): count distinct covering
# outlets from the modeling dataset's news_article_domain column
import csv as _csv, gzip as _gzip, ast as _ast
_csv.field_size_limit(sys.maxsize)
id2n = {}
for r in _csv.DictReader(_gzip.open("datasets/press-releases/press_release_modeling_dataset.csv.gz", "rt")):
    dom = (r.get("news_article_domain") or "").strip()
    if dom:
        try: id2n[r["press_release_id"]] = len(set(_ast.literal_eval(dom)))
        except Exception: id2n[r["press_release_id"]] = 0
    else: id2n[r["press_release_id"]] = 0
df["n_out"] = df.id.astype(str).map(id2n).fillna(0).astype(int)
pos = df[df.n_out >= 3].copy(); pos["judgement"] = 1
neg = df[df.n_out == 0].copy(); neg["judgement"] = 0
pos = pos.sample(min(N // 2, len(pos)), random_state=0)
neg = neg.sample(min(N // 2, len(neg)), random_state=0)
samp = pd.concat([pos, neg]).sample(frac=1, random_state=0).reset_index(drop=True)
y = samp.judgement.values.astype(int)
g = samp.company.astype(str).values
print(f"[pilot] {len(samp)} docs ({y.sum()} pos), {len(set(g))} companies", flush=True)

os.makedirs("outputs/claimverify_pr", exist_ok=True)
rows = run_seam_battery(samp, "text", CFG, "outputs/claimverify_pr/cache.jsonl",
                        workers=16, with_llm=True, with_null=True)
M = pd.DataFrame(rows)
M.to_csv("outputs/claimverify_pr/metrics_600.csv", index=False)
print(f"[pilot] battery done; cols={list(M.columns)}", flush=True)

def auc(cols, label):
    X = M[cols].values.astype(float)
    if np.all(np.isnan(X)): print(f"  {label:28} SKIP(all-nan)"); return
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))
    a = cross_val_score(pipe, X, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                        groups=g, scoring="roc_auc").mean()
    print(f"  {label:28} AUC={a:.4f} ({len(cols)} feats)", flush=True)
    return a

ccols = [c for c in M.columns if c.startswith("c_")]
rcols = [c for c in M.columns if c.startswith("r_")]
scols = [c for c in M.columns if c.startswith("s_")]
nrcols = [c for c in M.columns if c.startswith("null_r_")]
nscols = [c for c in M.columns if c.startswith("null_s_")]
print("\n[pilot] === ARM AUCs (company-grouped, k>=3 coverage) ===", flush=True)
a_c = auc(ccols, "CODE sourcing (F1)")
a_r = auc(rcols, "HYBRID claim-retrieval (F2)")
a_s = auc(scols, "LLM claim-support (F3)")
auc(nrcols, "NULL-twin retrieval")
auc(nscols, "NULL-twin support")
auc(ccols + rcols, "CODE+HYBRID stack")
auc(ccols + rcols + scols, "FULL stack")
print("\n[pilot] === certified evidence-op marginal (per-metric univariate) ===", flush=True)
for real, nul in [("r_mean_top1", "null_r_mean_top1"), ("r_support_coverage", "null_r_support_coverage"),
                  ("s_support_rate", "null_s_support_rate")]:
    if real in M and nul in M:
        vr, vn = M[real].values, M[nul].values
        mk = ~(np.isnan(vr) | np.isnan(vn))
        if mk.sum() > 100:
            ar = roc_auc_score(y[mk], vr[mk]); an = roc_auc_score(y[mk], vn[mk])
            print(f"  {real:24} real={ar:.4f} null={an:.4f} marginal={ar-an:+.4f}", flush=True)
print("\n[pilot] refs: PR k>=3 V=0.628 A=0.648 dense=0.705", flush=True)
print("PR_PILOT_DONE", flush=True)
