#!/usr/bin/env python3
"""Validation race for the v2 checkers against BOTH expert-revealed ground truths + outcome:
  adequacy metrics   vs reviewer SUPPORT complaints  (v1 baseline: t1_support AUC .532)
  prior-art metrics  vs reviewer NOVELTY complaints  (the patents-analog validation)
  everything         vs accept/reject                (v1 baseline: evidence tiers .504)
Also claim-level: on papers with novelty complaints, is the reviewer-disputed novel thing
among our ANTICIPATED claims? Prints money examples (claim + disclosing prior-art span).
Run on sk3 last: python -m methods.claim_verification.run_validate_v2"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from claim_verification.seam_metrics import _toks

ROOT = "/lfs/skampere3/0/alexspan/norm-research"

def jac(a, b):
    sa, sb = set(_toks(a)), set(_toks(b))
    return len(sa & sb) / max(len(sa | sb), 1)

def uni(y, v, label, flip=False):
    v = np.asarray(v, float); y = np.asarray(y, float)
    mk = ~np.isnan(v) & ~np.isnan(y)
    if mk.sum() < 80 or len(set(y[mk])) < 2:
        print(f"  {label:46} SKIP (n={int(mk.sum())})", flush=True); return
    a = roc_auc_score(y[mk], -v[mk] if flip else v[mk])
    print(f"  {label:46} AUC={a:.4f} (n={int(mk.sum())})", flush=True)

def main():
    P = pd.read_csv(f"{ROOT}/outputs/checks_v2/paper_metrics.csv")
    P["id"] = P.id.astype(str)
    t = pd.read_csv(f"{ROOT}/outputs/tiered_peer/tiered_metrics.csv")
    t["id"] = t.id.astype(str)
    M = t.merge(P, on="id", how="inner")
    # support flags (from run_reviewer_flags)
    sup = pd.read_csv(f"{ROOT}/outputs/reviewer_flags/paper_level.csv")
    sup["id"] = sup.id.astype(str)
    M = M.merge(sup[["id", "n_flags", "flagged"]], on="id", how="left")
    M["flagged"] = M.flagged.fillna(0)
    # novelty flags
    nov = [json.loads(l) for l in open(f"{ROOT}/outputs/reviewer_flags/novelty_flags.jsonl")]
    nconf = pd.DataFrame([r for r in nov if r["flag"]])
    nper = nconf.groupby("paper").size() if len(nconf) else pd.Series(dtype=int)
    M["n_nov_flags"] = M.id.map(nper).fillna(0)
    M["nov_flagged"] = (M.n_nov_flags >= 1).astype(int)
    print(f"[val2] merged {len(M)} papers; support-flagged {M.flagged.mean():.3f}, "
          f"novelty-flagged {M.nov_flagged.mean():.3f}", flush=True)

    print("\n=== 1. ADEQUACY vs reviewer SUPPORT complaints (v1 baseline .532) ===", flush=True)
    uni(M.flagged, M.est_rate, "low ESTABLISHED rate -> support-flagged", flip=True)
    uni(M.flagged, M.asserted_only_rate, "high ASSERTED_ONLY rate -> support-flagged")
    for ty in ("performance", "assumption", "scope", "design_justification"):
        c = f"est_{ty}"
        if c in M: uni(M.flagged, M[c], f"low est({ty}) -> support-flagged", flip=True)
    uni(M.flagged, M.t1_support, "v1 baseline: low t1_support -> flagged", flip=True)

    print("\n=== 2. PRIOR-ART vs reviewer NOVELTY complaints (patents analog) ===", flush=True)
    uni(M.nov_flagged, M.anticipated_rate, "high ANTICIPATED rate -> novelty-flagged")
    if "anticipated_novelty" in M:
        uni(M.nov_flagged, M.anticipated_novelty, "high anticipated(novelty-claims) -> nov-flagged")
    uni(M.nov_flagged, M.clear_rate, "high CLEAR rate -> novelty-flagged (expect <.5)")
    uni(M.nov_flagged, M.novelty, "v1 baseline: t3-novelty -> nov-flagged")
    # extreme groups
    hi = M[M.n_nov_flags >= 2]; lo = M[M.n_nov_flags == 0]
    if len(hi) > 15:
        print(f"  extreme: anticipated_rate nov-flagged>=2 {hi.anticipated_rate.mean():.3f} "
              f"(n={len(hi)}) vs unflagged {lo.anticipated_rate.mean():.3f} (n={len(lo)})", flush=True)

    print("\n=== 3. everything vs ACCEPT/REJECT (v1 baseline .504) ===", flush=True)
    for c, fl in (("est_rate", False), ("asserted_only_rate", True), ("anticipated_rate", True),
                  ("clear_rate", False), ("n_claims", False)):
        if c in M: uni(M.y, M[c], f"{'low ' if fl else ''}{c} -> accept", flip=fl)
    # multivariate, year-grouped
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    feats = [c for c in ("est_rate", "asserted_only_rate", "anticipated_rate", "clear_rate",
                         "est_performance", "est_assumption", "est_scope") if c in M]
    X = M[feats].values.astype(float)
    mk = ~np.all(np.isnan(X), axis=1)
    folds = min(5, M.year[mk].nunique())
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    try:
        a = cross_val_score(pipe, X[mk], M.y[mk], groups=M.year[mk].astype(str),
                            cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                            scoring="roc_auc")
        print(f"  v2 bank multivariate (year-grouped)          AUC={np.nanmean(a):.4f}", flush=True)
    except Exception as e:
        print(f"  multivariate ERR {str(e)[:50]}", flush=True)

    print("\n=== 4. claim-level: reviewer-disputed novel thing vs our ANTICIPATED claims ===", flush=True)
    checks = [json.loads(l) for l in open(f"{ROOT}/outputs/checks_v2/checks.jsonl")]
    bypaper = {}
    for c in checks: bypaper.setdefault(c["paper"], []).append(c)
    hit, tot = 0, 0
    examples = []
    for r in (x for x in nov if x["flag"] and len(x.get("claim", "")) > 20):
        ours = bypaper.get(r["paper"], [])
        best = max(ours, key=lambda c: jac(r["claim"], c["claim"]), default=None)
        if not best or jac(r["claim"], best["claim"]) <= 0.2: continue
        tot += 1
        if best.get("prior_art") in ("ANTICIPATED", "RELATED"):
            hit += 1
            if len(examples) < 5:
                examples.append((r["paper"], r["claim"], best["claim"],
                                 best.get("prior_art"), best.get("pa_span", "")))
    base_rate = np.mean([1 if c.get("prior_art") in ("ANTICIPATED", "RELATED") else 0
                         for c in checks if c.get("prior_art") in ("ANTICIPATED", "RELATED", "CLEAR")])
    print(f"  matched disputed->our claims: {tot}; our checker says ANTICIPATED/RELATED on "
          f"{hit}/{tot} ({hit/max(tot,1):.3f}) vs base rate {base_rate:.3f}", flush=True)
    for p, rc, oc, v, span in examples:
        print(f"    [{v}] {p}\n      reviewer disputes: {rc[:110]}\n      our claim: {oc[:110]}"
              + (f"\n      prior-art span: {span[:110]}" if span else ""), flush=True)
    M.to_csv(f"{ROOT}/outputs/checks_v2/validation_table.csv", index=False)
    print("VALIDATE_V2_DONE", flush=True)

if __name__ == "__main__":
    main()
