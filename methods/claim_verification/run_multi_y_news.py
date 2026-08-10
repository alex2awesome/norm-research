#!/usr/bin/env python3
"""Multi-y for NEWS (the crossover domain): the same claim metrics against TWO outcome types:
  y1 = homepage PLACEMENT (institutional; join fetched articles to v9 rows via anchor-text prefix)
  y2 = TWITTER engagement percentile within outlet x day (crowd; n_tweets/sum_likes)
Metrics per article (from claims_newsfull + tiers computed here on the fly):
  t1 own-body support (via full body), sourcing CODE census, claim-quality (cq file if present).
Only needs Gemma for t1 verdicts on the sampled docs.
Run on sk3: python -m methods.claim_verification.run_multi_y_news [--n 700]"""
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from claim_verification.core import Cache, verify_claim, _sentences
from claim_verification.evidence_api import chunk_passages, clean_evidence_text
from claim_verification.seam_metrics import sourcing_code_metrics

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
EB = os.path.join(ROOT, "datasets/evidence_bases")
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma", "doc_kind": "news article"}

def load_fetched():
    out = {}
    for p in glob.glob(f"{ROOT}/datasets/news-homepages/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("route") != "FAIL" and r.get("text_len", 0) > 600:
                out[r["url"]] = r
    return out

def load_claims():
    out = {}
    p = os.path.join(EB, "claims_newsfull.jsonl")
    for ln in open(p):
        try:
            r = json.loads(ln)
            if r.get("claims"):
                out[str(r["doc_id"])] = [c["claim"] if isinstance(c, dict) else str(c) for c in r["claims"]][:4]
        except Exception: pass
    return out

def load_twitter():
    tw = {}
    for ln in open(f"{ROOT}/datasets/news-homepages/twitter_engagement/tweet_engagement.jsonl"):
        try:
            r = json.loads(ln)
            tw[r["url"]] = {"n_tweets": int(r.get("n_tweets", 0)), "likes": int(r.get("sum_likes", 0)),
                            "outlet": r.get("outlet"), "day": (r.get("first_day") or "")[:10]}
        except Exception: pass
    return tw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    fetched = load_fetched(); claims = load_claims(); tw = load_twitter()
    # eligible: fetched + claims + twitter (for y2)
    elig = [u for u in fetched if u in claims and u in tw]
    print(f"[m-y-news] eligible (text+claims+twitter): {len(elig)}", flush=True)
    rng = np.random.default_rng(0); rng.shuffle(elig)
    docs = elig[:args.n]
    cache = Cache(f"{ROOT}/outputs/multi_y_news/cache.jsonl")
    lock = Lock(); rows = []
    def work(u):
        r = fetched[u]; cl = claims[u]
        body = clean_evidence_text(r["text"])
        pool = chunk_passages(body, words_per=110, max_passages=14)
        m = {"url": u, "outlet": tw[u]["outlet"], "day": tw[u]["day"],
             "n_tweets": tw[u]["n_tweets"], "likes": tw[u]["likes"]}
        m.update(sourcing_code_metrics(body[:12000]))          # CODE sourcing census
        try:
            vs = [verify_claim(c, pool, CFG, cache) for c in cl]
            full = sum(1 for v in vs if v["verdict"] == "FULL")
            part = sum(1 for v in vs if v["verdict"] == "PARTIAL")
            m["t1_support"] = full / len(vs); m["t1_echo"] = (full + part) / len(vs)
        except Exception:
            m["t1_support"] = np.nan; m["t1_echo"] = np.nan
        with lock:
            rows.append(m)
            if len(rows) % 50 == 0: print(f"[m-y-news] {len(rows)}/{len(docs)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, docs))
    M = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/outputs/multi_y_news", exist_ok=True)
    # y2 = within outlet x day twitter percentile (n_tweets primary)
    M["tw_pct"] = M.groupby(["outlet", "day"]).n_tweets.rank(pct=True)
    M["y_twitter"] = (M.tw_pct >= 0.5).astype(int)
    M.to_csv(f"{ROOT}/outputs/multi_y_news/metrics.csv", index=False)
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    ev = ["t1_support", "t1_echo"]
    src = [c for c in M.columns if c.startswith("c_")]
    def auc(cols, y, g, label):
        X = M[cols].values.astype(float)
        mk = ~np.all(np.isnan(X), axis=1) & pd.Series(y).notna().values
        if mk.sum() < 100: print(f"  {label:36} SKIP", flush=True); return
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
        try:
            a = cross_val_score(pipe, X[mk], np.asarray(y)[mk],
                                cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                                groups=np.asarray(g)[mk], scoring="roc_auc").mean()
            print(f"  {label:36} AUC={a:.4f} (n={int(mk.sum())})", flush=True)
        except Exception as e:
            print(f"  {label:36} ERR {str(e)[:40]}", flush=True)
    g = (M.outlet.astype(str) + "|" + M.day.astype(str)).values
    print("\n[m-y-news] -- y = TWITTER pct>=median within outlet x day (crowd) --", flush=True)
    auc(ev, M.y_twitter.values, g, "claim-support (evidence)")
    auc(src, M.y_twitter.values, g, "sourcing census (CODE)")
    auc(ev + src, M.y_twitter.values, g, "evidence + sourcing")
    print("\n[m-y-news] univariate (twitter-y):", flush=True)
    for c in ev + src[:6]:
        v = M[c].values.astype(float); mk = ~np.isnan(v)
        if mk.sum() > 100 and np.std(v[mk]) > 0:
            print(f"  {c:22} AUC={roc_auc_score(M.y_twitter.values[mk], v[mk]):.4f}", flush=True)
    print("MULTI_Y_NEWS_DONE", flush=True)

if __name__ == "__main__":
    main()
