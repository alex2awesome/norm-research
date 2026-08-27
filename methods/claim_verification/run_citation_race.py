#!/usr/bin/env python3
"""Citation-y race with STRATIFIED design (S2 coverage is outcome-correlated:
accepted 98.8% matched vs rejected 56.2% -> matched-only pooling = collider/Berkson
selection + venue/accrual leak; never pool decisions).
  PRIMARY:   accepted-only (coverage ~complete), y = above-median citation WITHIN YEAR.
  SECONDARY: rejected-matched-only (selection constant within stratum; caveated).
Features = same per-paper v2 metrics as run_race_v2 + flag counts.
Readouts threshold-free: AUC vs within-year median split + Spearman vs within-year pctile.
Run on sk3: python -m methods.claim_verification.run_citation_race"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
OUT = f"{ROOT}/outputs/expand_v2"

FEATS = ["est_rate", "asserted_only_rate", "absent_rate", "trivial_delta_rate",
         "substantive_rate", "est_performance", "est_assumption", "est_scope",
         "est_design_justification", "n_claims", "n_support_flags", "n_novelty_flags"]

def paper_metrics():
    rows = []
    for ln in open(f"{OUT}/paper_checks.jsonl"):
        try: r = json.loads(ln)
        except Exception: continue
        cl = r.get("claims", [])
        m = {"paper_id": r["doc_id"], "n_claims": len(cl)}
        ad = [c for c in cl if c.get("adequacy")]
        if ad:
            m["est_rate"] = np.mean([c["adequacy"] == "ESTABLISHED" for c in ad])
            m["asserted_only_rate"] = np.mean([c["adequacy"] == "ASSERTED_ONLY" for c in ad])
            m["absent_rate"] = np.mean([c["adequacy"] == "ABSENT" for c in ad])
            for ty in ("performance", "assumption", "scope", "design_justification"):
                sub = [c for c in ad if c["type"] == ty]
                if sub: m[f"est_{ty}"] = np.mean([c["adequacy"] == "ESTABLISHED" for c in sub])
        dl = [c for c in cl if c.get("delta")]
        if dl:
            m["trivial_delta_rate"] = np.mean([c["delta"] == "TRIVIAL_DELTA" for c in dl])
            m["substantive_rate"] = np.mean([c["delta"] == "SUBSTANTIVE_DELTA" for c in dl])
        rows.append(m)
    return pd.DataFrame(rows)

def add_flags(M):
    for tag in ("support", "novelty"):
        per = {}
        p_ = f"{OUT}/flags_{tag}.jsonl"
        if os.path.exists(p_):
            F = pd.DataFrame([json.loads(l) for l in open(p_)])
            if len(F): per = F[F.flag].groupby("paper").size().to_dict()
        M[f"n_{tag}_flags"] = M.paper_id.map(per).fillna(0)
    return M

def stratum_race(M, name):
    print(f"\n=== {name} (n={len(M)}) ===", flush=True)
    if len(M) < 300:
        print("  too small, skip", flush=True); return
    # within-year citation percentile + median split
    M = M.copy()
    M["cite_pct"] = M.groupby("year").s2_citationCount.rank(pct=True)
    M["y_cite"] = (M.cite_pct > 0.5).astype(int)
    print(f"  cites: median {M.s2_citationCount.median():.0f} "
          f"mean {M.s2_citationCount.mean():.1f} zeros {(M.s2_citationCount==0).mean():.3f}", flush=True)
    for f in FEATS:
        if f not in M: continue
        v = M[f].values.astype(float)
        mk = ~np.isnan(v)
        if mk.sum() < 300 or M.y_cite[mk].nunique() < 2: continue
        a = roc_auc_score(M.y_cite[mk], v[mk])
        rho = spearmanr(v[mk], M.cite_pct[mk]).statistic
        print(f"  {f:28} AUC={a:.4f}  rho={rho:+.3f} (n={int(mk.sum())})", flush=True)
    # multivariate, year-grouped
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    feats = [f for f in FEATS if f in M]
    X = M[feats].values.astype(float)
    mk = ~np.all(np.isnan(X), axis=1)
    folds = min(5, M.year[mk].nunique())
    if folds >= 2:
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
        a = cross_val_score(pipe, X[mk], M.y_cite[mk], groups=M.year[mk].astype(str),
                            cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                            scoring="roc_auc")
        print(f"  v2 bank multivariate (year-grouped)   AUC={np.nanmean(a):.4f}", flush=True)

def main():
    sample = pd.read_csv(f"{OUT}/sample.csv")
    sample["paper_id"] = sample.paper_id.astype(str)
    s2 = pd.DataFrame([json.loads(l) for l in
                       open(f"{ROOT}/datasets/peer-review/s2_citations_2024_25.jsonl")])
    s2 = s2.drop_duplicates("paper_id", keep="last")[
        ["paper_id", "s2_citationCount", "match_ok"]]
    M = paper_metrics().merge(sample, on="paper_id").merge(s2, on="paper_id", how="left")
    M = add_flags(M)
    print(f"[cite-race] {len(M)} papers; match_ok acc "
          f"{M[M.judgement==1].match_ok.mean():.3f} rej {M[M.judgement==0].match_ok.mean():.3f}", flush=True)
    ok = M[M.match_ok == True].dropna(subset=["s2_citationCount"])
    stratum_race(ok[ok.judgement == 1], "PRIMARY: accepted-only, cite-y (within-year)")
    stratum_race(ok[ok.judgement == 0], "SECONDARY: rejected-matched-only (selection-caveated)")
    # H_outcome contrast line: accept-AUCs from race_v2 for the same features are in
    # outputs/race_v2.log; print side-by-side reminder values
    print("\n[cite-race] accept-y reference (full set): trivial_delta .544(flip), "
          "est_rate .517, bank .530, n_novelty_flags .555(flip)", flush=True)
    print("CITATION_RACE_DONE", flush=True)

if __name__ == "__main__":
    main()
