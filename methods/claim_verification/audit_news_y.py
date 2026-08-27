#!/usr/bin/env python3
"""Audit + extend multi_y_news:
(1) LENGTH audit of the twitter-y sourcing result (standing rule: no V claim without
    length-stratification): wordcount univariate AUC, density-only AUC, within-length-quintile AUC.
(2) PLACEMENT-y: join the same 700 fetched docs to v9 homepage rows via normalized
    anchor-text prefix (v9 has no urls), same metric sets against placement judgement.
Run on sk3: python -m methods.claim_verification.audit_news_y"""
import glob, json, os, re, sys, warnings
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
NH = f"{ROOT}/datasets/news-homepages"

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def gauc(M, cols, y, g, label):
    X = M[cols].values.astype(float)
    mk = ~np.all(np.isnan(X), axis=1) & pd.Series(y).notna().values
    if mk.sum() < 80 or len(set(np.asarray(y)[mk])) < 2:
        print(f"  {label:40} SKIP (n={int(mk.sum())})", flush=True); return
    folds = min(5, len(set(np.asarray(g)[mk])))
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    a = cross_val_score(pipe, X[mk], np.asarray(y)[mk].astype(int),
                        cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                        groups=np.asarray(g)[mk], scoring="roc_auc")
    print(f"  {label:40} AUC={np.nanmean(a):.4f} (n={int(mk.sum())})", flush=True)

def main():
    M = pd.read_csv(f"{ROOT}/outputs/multi_y_news/metrics.csv")
    # -- body word counts for the length audit --
    urls = set(M.url)
    wc = {}
    for p in glob.glob(f"{NH}/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("url") in urls and r.get("text"):
                wc[r["url"]] = len(r["text"].split())
    M["wordcount"] = M.url.map(wc)
    y = M.y_twitter.values
    g = (M.outlet.astype(str) + "|" + M.day.astype(str)).values
    print("=== (1) LENGTH AUDIT: twitter-y ===", flush=True)
    v = M.wordcount.values.astype(float); mk = ~np.isnan(v)
    print(f"  wordcount alone univariate       AUC={roc_auc_score(y[mk], v[mk]):.4f}", flush=True)
    print(f"  corr(wordcount, c_attrib_verbs)  r={M.wordcount.corr(M.c_attrib_verbs):.3f}", flush=True)
    gauc(M, ["wordcount"], y, g, "wordcount alone (grouped CV)")
    counts = [c for c in M.columns if c.startswith("c_") and not c.endswith("_density")]
    dens = [c for c in M.columns if c.endswith("_density")]
    gauc(M, counts, y, g, "sourcing RAW counts")
    gauc(M, dens, y, g, "sourcing DENSITIES only (len-free)")
    gauc(M, counts + ["wordcount"], y, g, "raw counts + wordcount")
    # residualized: within length-quintile univariate for the headline feature
    M["lq"] = pd.qcut(M.wordcount, 5, labels=False, duplicates="drop")
    aucs = []
    for q, s in M.groupby("lq"):
        if s.y_twitter.nunique() == 2 and len(s) > 40:
            aucs.append(roc_auc_score(s.y_twitter, s.c_attrib_verbs))
    print(f"  c_attrib_verbs within length-quintile AUCs: {[round(a,3) for a in aucs]} "
          f"mean={np.mean(aucs):.4f}", flush=True)
    aucs = []
    for q, s in M.groupby("lq"):
        if s.y_twitter.nunique() == 2 and len(s) > 40:
            aucs.append(roc_auc_score(s.y_twitter, s.c_attrib_density))
    print(f"  c_attrib_density within length-quintile:   {[round(a,3) for a in aucs]} "
          f"mean={np.mean(aucs):.4f}", flush=True)

    # -- (2) placement-y join --
    print("\n=== (2) PLACEMENT-y (anchor-text join to v9) ===", flush=True)
    anchors = {}
    for ln in open(f"{NH}/twitter_engagement/tweet_engagement.jsonl"):
        try:
            r = json.loads(ln)
            if r["url"] in urls and r.get("anchor_text"):
                anchors[r["url"]] = norm(r["anchor_text"])[:70]
        except Exception: pass
    print(f"  anchors for {len(anchors)}/{len(M)} docs", flush=True)
    v9 = pd.read_csv(f"{NH}/homepage_newsworthiness_clean_v9.csv.gz", compression="gzip")
    # v9 text = "HEADLINE: <h>\n\nCONTEXT: ..." — parse out the headline
    heads = v9.text.astype(str).str.extract(r"HEADLINE:\s*(.*?)(?:\n\n|$)", expand=False)
    v9["nh"] = heads.map(norm)
    pool = [(nh, j) for nh, j in zip(v9.nh, v9.judgement) if isinstance(nh, str) and len(nh) >= 25]
    print(f"  v9 headlines usable: {len(pool)}", flush=True)
    lab = {}
    # anchors have mashed kicker prefixes ("PoliticsZack Polanski...") -> containment either way
    for u, a in anchors.items():
        if len(a) < 25: continue
        js = {j for nh, j in pool if nh in a or a in nh}
        if len(js) == 1: lab[u] = js.pop()   # unambiguous match only
    M["y_place"] = M.url.map(lab)
    n = M.y_place.notna().sum()
    print(f"  placement label matched: {n}/{len(M)} (pos rate "
          f"{M.y_place.mean():.3f})", flush=True)
    if n >= 120:
        yp = M.y_place.values
        ev = ["t1_support", "t1_echo"]
        gauc(M, ev, yp, g, "claim-support (evidence)")
        gauc(M, counts + dens, yp, g, "sourcing census (CODE)")
        gauc(M, ev + counts + dens, yp, g, "evidence + sourcing")
        gauc(M, ["wordcount"], yp, g, "wordcount alone")
        for c in ["c_attrib_verbs", "c_attrib_density", "t1_support"]:
            v = M[c].values.astype(float); mk2 = ~np.isnan(v) & M.y_place.notna().values
            if mk2.sum() > 80 and len(set(yp[mk2])) == 2:
                print(f"  uni {c:22} AUC={roc_auc_score(yp[mk2].astype(int), v[mk2]):.4f}", flush=True)
    M.to_csv(f"{ROOT}/outputs/multi_y_news/metrics_audited.csv", index=False)
    print("AUDIT_NEWS_DONE", flush=True)

if __name__ == "__main__":
    main()
