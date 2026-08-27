#!/usr/bin/env python3
"""Leak-free re-run of the news dense arms (post-Codex-audit): the original run_news_vat_v2 /
run_news_trend_decomp fit TF-IDF on ALL docs before grouped CV (transductive). Here the
vectorizers live INSIDE the CV pipeline (fold-local); bge arm reuses the saved frozen
embeddings (no fitting -> no leak, re-scored with fold-local LR as before).

Readouts vs honest y: dense TF-IDF + bge, pooled(outlet|day) and OUTLET-HELD-OUT; fold-local
OOF within-SECTION strata. CPU-only.
Run: $HOME/envs/ai_usage/bin/python -m methods.claim_verification.rerun_news_dense_foldlocal
"""
import glob, json, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
from claim_verification.run_news_trend_decomp import url_section, sec_norm

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
NH = f"{ROOT}/datasets/news-homepages"

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
    print(f"  {label:44} AUC={a.mean():.4f} ({folds} folds)", flush=True)
    return float(a.mean())

def main():
    M = pd.read_csv(f"{ROOT}/outputs/multi_y_news/doc_metrics_v2.csv")
    urls = set(M.url); text = {}
    for p in glob.glob(f"{NH}/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("url") in urls and len(r.get("text") or "") > 400:
                text[r["url"]] = r["text"]
    M = M[M.url.isin(text)].reset_index(drop=True)
    body = np.array([text[u] for u in M.url], dtype=object)
    y = M.y.astype(int).values
    g_pool = (M.outlet.astype(str) + "|" + M.day.astype(str)).values
    g_out = M.outlet.astype(str).values
    E = np.load(f"{ROOT}/outputs/multi_y_news/embeds_v2.npz", allow_pickle=True)
    eu = [str(u) for u in E["url"]]
    assert eu == [str(u) for u in M.url], "embeds_v2 url order mismatch"
    Emb = E["E"]
    print(f"[foldlocal] {len(M)} docs", flush=True)
    res = {}
    ep = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    for gname, g in [("pooled(outlet|day)", g_pool), ("OUTLET-HELD-OUT", g_out)]:
        print(f"--- {gname} ---", flush=True)
        res[f"lex|{gname}"] = gcv(body, y, g, "DENSE TF-IDF (fold-local)", lex_pipe())
        res[f"embed|{gname}"] = gcv(Emb, y, g, "DENSE bge frozen (ref)", ep)
    # fold-local OOF within-section
    oof = cross_val_predict(lex_pipe(), body, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                            groups=g_pool, method="predict_proba")[:, 1]
    sec = M.url.map(url_section).map(sec_norm)
    rows = [(s, int((sec == s).sum()), round(roc_auc_score(y[sec == s], oof[sec == s]), 3))
            for s in sorted(sec.unique())
            if (sec == s).sum() >= 150 and len(set(y[sec == s])) == 2]
    print(f"  fold-local OOF within-SECTION: {rows}", flush=True)
    res["within_section"] = rows
    json.dump(res, open(f"{ROOT}/outputs/multi_y_news/news_dense_foldlocal.json", "w"), indent=2, default=str)
    print("NEWS_DENSE_FOLDLOCAL_DONE", flush=True)

if __name__ == "__main__":
    main()
