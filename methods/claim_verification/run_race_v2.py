#!/usr/bin/env python3
"""Full-set race for the expanded v2 pipeline (n~14.5k ICLR 2024-25):
  paper metrics (adequacy rates, trivial-delta rate, by-type) vs
    (a) accept/reject (year-grouped CV; v1 baseline .504 at n=600)
    (b) reviewer support-flags at scale (v1: .532-.536 at n=600)
    (c) reviewer novelty-flags at scale (spot-check: +.145 p=.006 claim-level)
Run on sk3 after run_expand_v2: python -m methods.claim_verification.run_race_v2"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
OUT = f"{ROOT}/outputs/expand_v2"

def uni(y, v, label, flip=False):
    v = np.asarray(v, float); y = np.asarray(y, float)
    mk = ~np.isnan(v) & ~np.isnan(y)
    if mk.sum() < 200 or len(set(y[mk])) < 2:
        print(f"  {label:48} SKIP (n={int(mk.sum())})", flush=True); return None
    a = roc_auc_score(y[mk], -v[mk] if flip else v[mk])
    print(f"  {label:48} AUC={a:.4f} (n={int(mk.sum())})", flush=True)
    return a

def main():
    sample = pd.read_csv(f"{OUT}/sample.csv")
    sample["paper_id"] = sample.paper_id.astype(str)
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
            for ty in ("performance", "assumption", "scope", "design_justification", "novelty"):
                sub = [c for c in ad if c["type"] == ty]
                if sub: m[f"est_{ty}"] = np.mean([c["adequacy"] == "ESTABLISHED" for c in sub])
        dl = [c for c in cl if c.get("delta")]
        if dl:
            m["trivial_delta_rate"] = np.mean([c["delta"] == "TRIVIAL_DELTA" for c in dl])
            m["substantive_rate"] = np.mean([c["delta"] == "SUBSTANTIVE_DELTA" for c in dl])
        rows.append(m)
    M = pd.DataFrame(rows).merge(sample, on="paper_id", how="inner")
    print(f"[race] {len(M)} papers with checks; claims/paper "
          f"{M.n_claims.mean():.1f}", flush=True)
    for tag in ("support", "novelty"):
        per = {}
        p_ = f"{OUT}/flags_{tag}.jsonl"
        if os.path.exists(p_):
            F = pd.DataFrame([json.loads(l) for l in open(p_)])
            if len(F): per = F[F.flag].groupby("paper").size().to_dict()
        M[f"n_{tag}_flags"] = M.paper_id.map(per).fillna(0)
        M[f"{tag}_flagged"] = (M[f"n_{tag}_flags"] >= 1).astype(int)
    print(f"[race] flagged rates: support {M.support_flagged.mean():.3f}, "
          f"novelty {M.novelty_flagged.mean():.3f}", flush=True)

    print("\n=== accept/reject (v1 baseline .504 @600) ===", flush=True)
    y = M.judgement.values
    uni(y, M.est_rate, "est_rate -> accept")
    uni(y, M.asserted_only_rate, "low asserted_only -> accept", flip=True)
    uni(y, M.trivial_delta_rate, "low trivial_delta -> accept", flip=True)
    for ty in ("performance", "assumption", "scope", "design_justification"):
        c = f"est_{ty}"
        if c in M: uni(y, M[c], f"est_{ty} -> accept")
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    feats = [c for c in ("est_rate", "asserted_only_rate", "absent_rate", "trivial_delta_rate",
                         "substantive_rate", "est_performance", "est_assumption", "est_scope",
                         "est_design_justification", "n_claims") if c in M]
    X = M[feats].values.astype(float)
    mk = ~np.all(np.isnan(X), axis=1)
    folds = min(5, M.year[mk].nunique())
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    a = cross_val_score(pipe, X[mk], y[mk].astype(int), groups=M.year[mk].astype(str),
                        cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                        scoring="roc_auc")
    print(f"  v2 bank multivariate (year-grouped)              AUC={np.nanmean(a):.4f}", flush=True)

    print("\n=== reviewer SUPPORT flags at scale (v1 .532-.536 @600) ===", flush=True)
    uni(M.support_flagged, M.est_rate, "low est_rate -> support-flagged", flip=True)
    uni(M.support_flagged, M.asserted_only_rate, "asserted_only -> support-flagged")
    if "est_design_justification" in M:
        uni(M.support_flagged, M.est_design_justification, "low est(design_just) -> flagged", flip=True)

    print("\n=== reviewer NOVELTY flags at scale ===", flush=True)
    uni(M.novelty_flagged, M.trivial_delta_rate, "trivial_delta_rate -> novelty-flagged")
    uni(M.novelty_flagged, M.substantive_rate, "low substantive_rate -> nov-flagged", flip=True)
    hi = M[M.n_novelty_flags >= 2]; lo = M[M.n_novelty_flags == 0]
    if len(hi) > 50 and "trivial_delta_rate" in M:
        print(f"  extreme: trivial_delta nov>=2 {hi.trivial_delta_rate.mean():.3f} (n={len(hi)}) "
              f"vs 0 {lo.trivial_delta_rate.mean():.3f} (n={len(lo)})", flush=True)
    # flags themselves -> outcome (how much do complaints matter?)
    print("\n=== flags -> outcome (context) ===", flush=True)
    uni(y, M.n_support_flags, "n_support_flags -> reject", flip=True)
    uni(y, M.n_novelty_flags, "n_novelty_flags -> reject", flip=True)
    M.to_csv(f"{OUT}/race_table.csv", index=False)
    print("RACE_V2_DONE", flush=True)

if __name__ == "__main__":
    main()
