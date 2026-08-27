#!/usr/bin/env python3
"""Cross-y homogeneity/heterogeneity for news (v3 corpus, post-July tweet drain).

Corpus v3 = fulltext(>400) x tweet-engagement x English outlets (~19K docs, ~2x v2's 9,919).
y battery, each = within outlet x day percentile >= .5, cells >= MIN_CELL:
  CROWD:  y_likes (sum_likes, the v2 honest y), y_retweets, y_replies, y_views
  EXPERT: y_persist (appearances = # homepage captures carrying the URL; editorial persistence)
Readouts (all fold-local, StratifiedGroupKFold on outlet|day; CPU-only):
  1. per-y dense TF-IDF AUC, pooled + OUTLET-HELD-OUT
  2. cross-y transfer matrix: OOF probs trained on y_a scored against y_b (same docs)
  3. label-structure matrix: Spearman between continuous percentiles
  4. census-features-per-y on the v2 subset (are craft metrics dead on ALL y's?)
Run on sk3: HOME=/lfs/... $HOME/envs/ai_usage/bin/python -m methods.claim_verification.run_news_crossy_v3
"""
import glob, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
NH = f"{ROOT}/datasets/news-homepages"
EN_OUTLETS = {"latimes", "guardian", "cnn", "nytimes", "washingtonpost", "reuters"}
MIN_CELL = 6
YCOLS = {"y_likes": "sum_likes", "y_retweets": "sum_retweets", "y_replies": "sum_replies",
         "y_views": "sum_views", "y_persist": "appearances"}

def lex_pipe():
    return Pipeline([
        ("feats", FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150000, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=150000, sublinear_tf=True)),
        ])),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

def gcv(X, y, g, label, pipe):
    folds = min(5, len(set(g)))
    a = cross_val_score(pipe, X, y, cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                        groups=g, scoring="roc_auc")
    print(f"  {label:44} AUC={a.mean():.4f}", flush=True)
    return float(a.mean())

def main():
    tw = {}
    for ln in open(f"{NH}/twitter_engagement/tweet_engagement.jsonl"):
        try:
            r = json.loads(ln)
            if r.get("outlet") in EN_OUTLETS:
                tw[r["url"]] = r
        except Exception:
            continue
    text = {}
    for p in glob.glob(f"{NH}/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("url") in tw and len(r.get("text") or "") > 400:
                text[r["url"]] = r["text"]
    rows = []
    for u, t in text.items():
        r = tw[u]
        rows.append({"url": u, "outlet": r["outlet"], "day": (r.get("first_day") or "")[:10],
                     **{yc: int(r.get(src, 0) or 0) for yc, src in YCOLS.items()}})
    M = pd.DataFrame(rows)
    cell_n = M.groupby(["outlet", "day"]).url.transform("size")
    M = M[cell_n >= MIN_CELL].reset_index(drop=True)
    print(f"[v3] corpus: {len(M)} docs | {M.outlet.value_counts().to_dict()}", flush=True)

    # within-cell percentiles (continuous) + binary y's; drop degenerate y's
    pcts, ys, keep = {}, {}, []
    for yc in YCOLS:
        zero_rate = float((M[yc] == 0).mean())
        p = M.groupby(["outlet", "day"])[yc].rank(pct=True)
        pcts[yc] = p.values
        ys[yc] = (p >= 0.5).astype(int).values
        base = ys[yc].mean()
        print(f"  {yc:12} zero-rate={zero_rate:.2f} y-rate={base:.3f}", flush=True)
        if 0.15 < base < 0.85 and zero_rate < 0.90:
            keep.append(yc)
        else:
            print(f"    -> DROPPED (degenerate)", flush=True)
    body = np.array([text[u] for u in M.url], dtype=object)
    g_pool = (M.outlet.astype(str) + "|" + M.day.astype(str)).values
    g_out = M.outlet.astype(str).values
    res = {"n": len(M), "outlets": M.outlet.value_counts().to_dict(), "kept_ys": keep}

    # sanity: outlet-id AUC per y (should be ~.50 by construction)
    from sklearn.preprocessing import LabelEncoder
    for yc in keep:
        oh = pd.get_dummies(M.outlet).values.astype(float)
        aucs = cross_val_score(LogisticRegression(max_iter=1000, class_weight="balanced"),
                               oh, ys[yc], cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                               groups=g_pool, scoring="roc_auc")
        res[f"sanity_outletid|{yc}"] = float(aucs.mean())
        print(f"  sanity outlet-id {yc}: {aucs.mean():.3f}", flush=True)

    # 1. per-y dense AUC
    for yc in keep:
        print(f"--- dense on {yc} ---", flush=True)
        res[f"dense|{yc}|pooled"] = gcv(body, ys[yc], g_pool, f"TF-IDF {yc} pooled", lex_pipe())
        res[f"dense|{yc}|outlet_held_out"] = gcv(body, ys[yc], g_out, f"TF-IDF {yc} OUTLET-HELD-OUT", lex_pipe())

    # 2. cross-y transfer: OOF probs per y, scored against every other y
    print("--- cross-y transfer (OOF) ---", flush=True)
    oof = {}
    for yc in keep:
        oof[yc] = cross_val_predict(lex_pipe(), body, ys[yc],
                                    cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                                    groups=g_pool, method="predict_proba")[:, 1]
    transfer = {}
    for ya in keep:
        for yb in keep:
            transfer[f"{ya}->{yb}"] = round(float(roc_auc_score(ys[yb], oof[ya])), 4)
    res["transfer"] = transfer
    for ya in keep:
        print("  " + ya + ": " + "  ".join(f"->{yb} {transfer[f'{ya}->{yb}']:.3f}" for yb in keep), flush=True)

    # 3. label-structure: Spearman between continuous percentiles
    lab = {f"{ya}~{yb}": round(float(spearmanr(pcts[ya], pcts[yb]).statistic), 3)
           for i, ya in enumerate(keep) for yb in keep[i + 1:]}
    res["label_spearman"] = lab
    print(f"  label spearman: {lab}", flush=True)

    # 4. census features per y on the v2 subset (existing LLM/code census, no recompute)
    try:
        V2 = pd.read_csv(f"{ROOT}/outputs/multi_y_news/doc_metrics_v2.csv")
        # CRAFT-only: pct(=within-cell likes percentile) IS the label -> leaks (AUC .9999);
        # likes/n_tweets = engagement itself; wordcount = cell-size proxy. Keep sourcing/attribution census.
        meta = {"url", "outlet", "day", "y", "n_tweets", "likes", "cell_n", "pct", "wordcount"}
        feats = [c for c in V2.columns if c not in meta and V2[c].dtype != object]
        sub = M.merge(V2[["url"] + feats], on="url", how="inner")
        print(f"--- census-per-y on v2 subset (n={len(sub)}, {len(feats)} feats) ---", flush=True)
        cens = {}
        for yc in keep:
            yv = (sub.groupby(["outlet", "day"])[yc].rank(pct=True) >= 0.5).astype(int).values
            gg = (sub.outlet.astype(str) + "|" + sub.day.astype(str)).values
            if len(set(yv)) < 2: continue
            X = sub[feats].fillna(0).values
            pipe = Pipeline([("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            from sklearn.preprocessing import StandardScaler
            pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            a = cross_val_score(pipe, X, yv, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                                groups=gg, scoring="roc_auc")
            cens[yc] = round(float(a.mean()), 4)
            print(f"  census {yc}: {a.mean():.4f}", flush=True)
        res["census_v2subset"] = cens
    except Exception as e:
        print(f"  census leg skipped: {e}", flush=True)
        res["census_v2subset"] = None

    out = f"{ROOT}/outputs/multi_y_news/news_crossy_v3.json"
    json.dump(res, open(out, "w"), indent=2, default=str)
    print(f"NEWS_CROSSY_V3_DONE -> {out}", flush=True)

if __name__ == "__main__":
    main()
