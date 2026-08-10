#!/usr/bin/env python3
"""Placement-y for news (institutional y, completing the crossover):
join ALL fetched fulltext articles to v9 homepage rows via normalized headline containment,
then score the matched set (t1 claim-support via Gemma + CODE sourcing census) against
  y_place  = v9 placement judgement (institutional)
  y_twitter= within outlet x day engagement pct (crowd), where available on the SAME docs.
Run on sk3: python -m methods.claim_verification.run_placement_y [--n 700]"""
import argparse, glob, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache, verify_claim
from claim_verification.evidence_api import chunk_passages, clean_evidence_text
from claim_verification.seam_metrics import sourcing_code_metrics

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
NH = f"{ROOT}/datasets/news-homepages"
EB = f"{ROOT}/datasets/evidence_bases"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma", "doc_kind": "news article"}

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    fetched, anchors, tw = {}, {}, {}
    for p in glob.glob(f"{NH}/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("route") != "FAIL" and r.get("text_len", 0) > 600:
                fetched[r["url"]] = r["text"]
    for ln in open(f"{NH}/twitter_engagement/tweet_engagement.jsonl"):
        try:
            r = json.loads(ln)
            if r.get("anchor_text"): anchors[r["url"]] = norm(r["anchor_text"])
            tw[r["url"]] = {"n_tweets": int(r.get("n_tweets", 0)), "outlet": r.get("outlet"),
                            "day": (r.get("first_day") or "")[:10]}
        except Exception: pass
    claims = {}
    for ln in open(f"{EB}/claims_newsfull.jsonl"):
        try:
            r = json.loads(ln)
            if r.get("claims"):
                claims[str(r["doc_id"])] = [c["claim"] if isinstance(c, dict) else str(c)
                                            for c in r["claims"]][:4]
        except Exception: pass
    v9 = pd.read_csv(f"{NH}/homepage_newsworthiness_clean_v9.csv.gz", compression="gzip")
    heads = v9.text.astype(str).str.extract(r"HEADLINE:\s*(.*?)(?:\n\n|$)", expand=False)
    v9["nh"] = heads.map(norm)
    pool = [(nh, int(j)) for nh, j in zip(v9.nh, v9.judgement)
            if isinstance(nh, str) and len(nh) >= 25]
    cand = [u for u in fetched if u in claims and len(anchors.get(u, "")) >= 25]
    print(f"[place-y] fetched+claims+anchor candidates: {len(cand)}", flush=True)
    lab = {}
    for u in cand:
        a = anchors[u]
        js = {j for nh, j in pool if nh in a or a in nh}
        if len(js) == 1: lab[u] = js.pop()
    print(f"[place-y] placement-matched: {len(lab)} (pos rate "
          f"{np.mean(list(lab.values())):.3f})", flush=True)
    urls = list(lab)
    rng = np.random.default_rng(0); rng.shuffle(urls)
    urls = urls[:args.n]
    cache = Cache(f"{ROOT}/outputs/multi_y_news/cache.jsonl")   # shared verify cache
    lock = Lock(); rows = []
    def work(u):
        body = clean_evidence_text(fetched[u])
        p = chunk_passages(body, words_per=110, max_passages=14)
        t = tw.get(u, {})
        m = {"url": u, "y_place": lab[u], "outlet": t.get("outlet"), "day": t.get("day"),
             "n_tweets": t.get("n_tweets"), "wordcount": len(fetched[u].split())}
        m.update(sourcing_code_metrics(body[:12000]))
        try:
            vs = [verify_claim(c, p, CFG, cache) for c in claims[u]]
            full = sum(1 for v in vs if v["verdict"] == "FULL")
            part = sum(1 for v in vs if v["verdict"] == "PARTIAL")
            m["t1_support"] = full / len(vs); m["t1_echo"] = (full + part) / len(vs)
        except Exception:
            m["t1_support"] = np.nan; m["t1_echo"] = np.nan
        with lock:
            rows.append(m)
            if len(rows) % 50 == 0: print(f"[place-y] {len(rows)}/{len(urls)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, urls))
    M = pd.DataFrame(rows)
    M["tw_pct"] = M.groupby(["outlet", "day"]).n_tweets.rank(pct=True)
    M["y_twitter"] = (M.tw_pct >= 0.5).astype(int)
    os.makedirs(f"{ROOT}/outputs/placement_y", exist_ok=True)
    M.to_csv(f"{ROOT}/outputs/placement_y/metrics.csv", index=False)
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    ev = ["t1_support", "t1_echo"]
    counts = [c for c in M.columns if c.startswith("c_") and not c.endswith("_density")]
    dens = [c for c in M.columns if c.endswith("_density")]
    g = (M.outlet.astype(str) + "|" + M.day.astype(str)).values
    def gauc(cols, y, label):
        X = M[cols].values.astype(float)
        mk = ~np.all(np.isnan(X), axis=1) & pd.Series(y).notna().values
        if mk.sum() < 80 or len(set(np.asarray(y)[mk])) < 2:
            print(f"  {label:40} SKIP (n={int(mk.sum())})", flush=True); return
        folds = min(5, len(set(g[mk])))
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
        a = cross_val_score(pipe, X[mk], np.asarray(y)[mk].astype(int),
                            cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                            groups=g[mk], scoring="roc_auc")
        print(f"  {label:40} AUC={np.nanmean(a):.4f} (n={int(mk.sum())})", flush=True)
    print("\n[place-y] -- y = PLACEMENT (institutional) --", flush=True)
    gauc(ev, M.y_place.values, "claim-support (evidence)")
    gauc(counts + dens, M.y_place.values, "sourcing census (CODE)")
    gauc(dens, M.y_place.values, "sourcing DENSITIES only (len-free)")
    gauc(ev + counts + dens, M.y_place.values, "evidence + sourcing")
    gauc(["wordcount"], M.y_place.values, "wordcount alone")
    print("\n[place-y] -- y = TWITTER pct (crowd; same docs) --", flush=True)
    gauc(ev, M.y_twitter.values, "claim-support (evidence)")
    gauc(counts + dens, M.y_twitter.values, "sourcing census (CODE)")
    gauc(dens, M.y_twitter.values, "sourcing DENSITIES only (len-free)")
    gauc(["wordcount"], M.y_twitter.values, "wordcount alone")
    print("\n[place-y] univariate:", flush=True)
    for c in ["t1_support", "c_attrib_verbs", "c_attrib_density", "c_direct_quotes",
              "c_named_quotes", "c_numbers", "wordcount"]:
        v = M[c].values.astype(float)
        for yn, y in (("place", M.y_place.values), ("twitter", M.y_twitter.values)):
            mk = ~np.isnan(v) & pd.Series(y).notna().values
            if mk.sum() > 80 and len(set(np.asarray(y)[mk])) == 2 and np.nanstd(v[mk]) > 0:
                print(f"  {c:20} {yn:8} AUC={roc_auc_score(np.asarray(y)[mk].astype(int), v[mk]):.4f}",
                      flush=True)
    print("PLACEMENT_Y_DONE", flush=True)

if __name__ == "__main__":
    main()
