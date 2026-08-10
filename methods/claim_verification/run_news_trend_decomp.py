#!/usr/bin/env python3
"""Decompose the news dense signal (.62/.585) into ARTICULABLE content variables.

Question (user): is the dense-only engagement signal actually articulable content structure —
trending-event membership, desk/section, platform timing — rather than tacit taste?

Features (all code-computable; trending is V-checkable):
  TRENDING  same-day story-cluster stats from bge embeddings: n_xout (# other-outlet docs same
            day with cos>=.80), n_same (any outlet), max_xout_cos; .75-threshold sensitivity.
  SECTION   URL path segment (numeric segments skipped), normalized to ~10 transportable
            categories (sports/politics/world/opinion/entertainment/business/local/lifestyle/
            espanol/other) + raw per-outlet one-hot (pooled-only upper bound).
  TIMING    tweet ts -> hour-of-day sin/cos + one-hot weekday (platform dynamics).

Readouts vs honest y (pct sum_likes within outlet x day), pooled(outlet|day) + OUTLET-HELD-OUT:
  each family alone; families combined; dense TF-IDF reference; dense+articulable;
  dense OOF-score AUC WITHIN section strata and WITHIN trending terciles (does dense survive
  content control?); univariate trending AUCs. Embeddings saved to embeds_v2.npz.

Run on sk3: CUDA_VISIBLE_DEVICES=0 $HOME/envs/ai_usage/bin/python -m methods.claim_verification.run_news_trend_decomp
"""
import glob, json, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from datetime import datetime, timezone
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
from claim_verification.evidence_api import clean_evidence_text

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
NH = f"{ROOT}/datasets/news-homepages"

SEC_MAP = [
    ("sports", {"sport", "sports", "football", "athletic", "golf", "soccer", "olympics"}),
    ("politics", {"politics", "elections", "us-politics"}),
    ("world", {"world", "world-nation", "international", "middleeast", "europe", "asia", "africa", "americas", "australia-news", "uk-news"}),
    ("opinion", {"commentisfree", "opinion", "opinions", "editorials"}),
    ("entertainment", {"entertainment-arts", "entertainment", "culture", "film", "music", "tv-and-radio", "books", "arts", "media", "stage", "games"}),
    ("business", {"business", "money", "economy", "markets", "technology", "tech"}),
    ("local_us", {"california", "us", "us-news", "nyregion", "new-york", "washington"}),
    ("lifestyle", {"travel", "food", "wellness", "style", "lifeandstyle", "wirecutter", "health", "fashion", "recipes", "realestate"}),
    ("espanol", {"espanol"}),
]

def url_section(u):
    m = re.match(r"https?://[^/]+/(.+)", str(u))
    if not m: return "none"
    for p in m.group(1).split("/"):
        if p and not re.match(r"^\d+$", p) and not re.match(r"^20\d\d", p) and p not in ("index.html",):
            return p.lower()
    return "none"

def sec_norm(s):
    for name, keys in SEC_MAP:
        if s in keys: return name
    return "other"

def gauc(X, y, g, label, sparse=False):
    y = np.asarray(y); g = np.asarray(g)
    if sparse:
        mk = np.ones(X.shape[0], bool)
    else:
        X = np.asarray(X, float)
        mk = ~np.all(np.isnan(X), axis=1)
    mk &= pd.Series(y).notna().values
    if mk.sum() < 80 or len(set(y[mk])) < 2:
        print(f"  {label:48} SKIP (n={int(mk.sum())})", flush=True); return None
    folds = min(5, len(set(g[mk])))
    clf = LogisticRegression(max_iter=3000, class_weight="balanced") if sparse else \
        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    a = cross_val_score(clf, X[mk], y[mk].astype(int),
                        cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                        groups=g[mk], scoring="roc_auc")
    print(f"  {label:48} AUC={np.nanmean(a):.4f} (n={int(mk.sum())}, {folds} folds)", flush=True)
    return float(np.nanmean(a))

def main():
    res = {}
    M = pd.read_csv(f"{ROOT}/outputs/multi_y_news/doc_metrics_v2.csv")
    urls = set(M.url)
    text, ts = {}, {}
    for p in glob.glob(f"{NH}/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("url") in urls and len(r.get("text") or "") > 400:
                text[r["url"]] = clean_evidence_text(r["text"])
    for ln in open(f"{NH}/twitter_engagement/tweet_engagement.jsonl"):
        try:
            r = json.loads(ln)
            if r.get("url") in urls and r.get("ts"): ts[r["url"]] = int(r["ts"])
        except Exception: pass
    M = M[M.url.isin(text)].reset_index(drop=True)
    M["body"] = M.url.map(text)
    print(f"[decomp] {len(M)} docs | ts coverage {M.url.isin(ts).mean():.3f}", flush=True)
    y = M.y.values
    g_pool = (M.outlet.astype(str) + "|" + M.day.astype(str)).values
    g_out = M.outlet.astype(str).values

    # ---- SECTION ----
    M["sec_raw"] = M.url.map(url_section)
    M["sec"] = M.sec_raw.map(sec_norm)
    print(f"[decomp] sections: {M.sec.value_counts().to_dict()}", flush=True)
    Sec = pd.get_dummies(M.sec).values.astype(float)
    top_raw = M.sec_raw.value_counts().head(40).index
    SecRaw = pd.get_dummies(M.sec_raw.where(M.sec_raw.isin(top_raw), "rare")).values.astype(float)

    # ---- TIMING ----
    t = M.url.map(ts)
    hours = np.array([datetime.fromtimestamp(v, tz=timezone.utc).hour if pd.notna(v) else np.nan for v in t])
    wday = np.array([datetime.fromtimestamp(v, tz=timezone.utc).weekday() if pd.notna(v) else np.nan for v in t])
    Tim = np.column_stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
                           (pd.get_dummies(pd.Series(wday)).reindex(columns=range(7), fill_value=0)
                            .values.astype(float))])

    # ---- TRENDING (bge same-day cross-outlet clusters) ----
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    mod = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).to(device).eval()
    E = []
    with torch.no_grad():
        for i in range(0, len(M), 64):
            b = tok(M.body.iloc[i:i+64].tolist(), padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(device)
            h = mod(**b).last_hidden_state[:, 0]
            E.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
    E = np.vstack(E)
    np.savez(f"{ROOT}/outputs/multi_y_news/embeds_v2.npz", E=E, url=M.url.values)
    n_xout80 = np.zeros(len(M)); n_same80 = np.zeros(len(M))
    n_xout75 = np.zeros(len(M)); max_xcos = np.zeros(len(M))
    out_arr = M.outlet.to_numpy(dtype=object)
    for d, idx in M.groupby("day").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 2: continue
        S = E[idx] @ E[idx].T
        np.fill_diagonal(S, -1)
        same_outlet = (out_arr[idx][:, None] == out_arr[idx][None, :])
        xS = np.where(same_outlet, -1, S)
        n_xout80[idx] = (xS >= .80).sum(1); n_xout75[idx] = (xS >= .75).sum(1)
        n_same80[idx] = (S >= .80).sum(1); max_xcos[idx] = xS.max(1)
    M["n_xout80"], M["n_xout75"], M["n_same80"], M["max_xcos"] = n_xout80, n_xout75, n_same80, max_xcos
    Trend = M[["n_xout80", "n_xout75", "n_same80", "max_xcos"]].values.astype(float)
    print(f"[decomp] trending: n_xout80 mean {n_xout80.mean():.2f}, "
          f"share with >=1 cross-outlet neighbor {(n_xout80 > 0).mean():.3f}, "
          f"max_xcos median {np.median(max_xcos):.3f}", flush=True)
    for c in ["n_xout80", "max_xcos"]:
        v = M[c].values
        print(f"  uni {c:12} AUC={roc_auc_score(y, v):.4f}", flush=True)
        res[f"uni|{c}"] = float(roc_auc_score(y, v))

    # ---- DENSE reference ----
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200000, sublinear_tf=True)
    Xl = hstack([wv.fit_transform(M.body), cv.fit_transform(M.body)]).tocsr()

    ART = np.column_stack([Trend, Sec, Tim])
    for gname, g in [("pooled(outlet|day)", g_pool), ("OUTLET-HELD-OUT", g_out)]:
        print(f"--- {gname} ---", flush=True)
        res[f"trend|{gname}"] = gauc(Trend, y, g, "TRENDING (cluster stats)")
        res[f"section|{gname}"] = gauc(Sec, y, g, "SECTION (normalized cats)")
        res[f"timing|{gname}"] = gauc(Tim, y, g, "TIMING (hour sin/cos + weekday)")
        res[f"articulable|{gname}"] = gauc(ART, y, g, "ARTICULABLE = trend+sec+time")
        res[f"dense|{gname}"] = gauc(Xl, y, g, "DENSE TF-IDF (reference)", sparse=True)
        res[f"dense+art|{gname}"] = gauc(hstack([Xl, csr_matrix(ART)]).tocsr(), y, g,
                                         "DENSE + articulable", sparse=True)
        res[f"embed|{gname}"] = gauc(E, y, g, "DENSE bge embed (reference)")
        if gname.startswith("pooled"):
            res["sec_raw|pooled"] = gauc(SecRaw, y, g, "SECTION raw per-outlet (upper bound)")

    # ---- does dense survive content control? OOF within strata ----
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    oof = cross_val_predict(clf, Xl, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                            groups=g_pool, method="predict_proba")[:, 1]
    print("--- dense OOF within strata ---", flush=True)
    rows = []
    for s, ss in M.assign(o=oof).groupby("sec"):
        if len(ss) >= 150 and ss.y.nunique() == 2:
            rows.append((s, len(ss), round(roc_auc_score(ss.y, ss.o), 3)))
    print(f"  dense within-SECTION: {rows}", flush=True)
    res["dense_within_section"] = rows
    M["trend_ter"] = pd.qcut(M.n_xout80.rank(method="first"), 3, labels=[0, 1, 2])
    rows = []
    for s, ss in M.assign(o=oof).groupby("trend_ter"):
        rows.append((int(s), len(ss), round(roc_auc_score(ss.y, ss.o), 3)))
    print(f"  dense within-TRENDING-tercile: {rows}", flush=True)
    res["dense_within_trend_tercile"] = rows

    M.drop(columns=["body"]).to_csv(f"{ROOT}/outputs/multi_y_news/doc_metrics_v2_decomp.csv", index=False)
    json.dump(res, open(f"{ROOT}/outputs/multi_y_news/news_trend_decomp.json", "w"), indent=2, default=str)
    print("TREND_DECOMP_DONE", flush=True)

if __name__ == "__main__":
    main()
