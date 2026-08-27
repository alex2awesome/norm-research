#!/usr/bin/env python3
"""News VAT row deconfound: publisher-held-out re-read of the sourcing census + dense baselines.

Why: the .689 census / .636 LLM-bank / .681 combined numbers were grouped-CV by outlet|day,
which blocks same-day leakage but NOT outlet identity. The corpus is 5 outlets; cnnbrasil is
44% of docs, PORTUGUESE (c_attrib_verbs mean 0.003 vs 5.7-16.6 for EN outlets — the regex
never fires), and has the lowest y-rate (.694 vs .874-.943). So census features partly encode
outlet, and outlet carries y. This run separates style-signal from outlet-fingerprint.

Readouts (all AUC, LR class_weight=balanced):
  (a) pooled reference   group=outlet|day : census / LLM / LLM+census / wordcount  [expect ~.69/.64/.68]
  (b) outlet-held-out    group=outlet     : same sets  ("what survives on an unseen publisher")
  (c) outlet-id alone    one-hot, group=day : confound magnitude ceiling
  (d) dense text         TF-IDF+LR all docs; bge-large embed+LR EN-only; pooled vs outlet-held-out
  (e) ENGLISH-only       drop cnnbrasil (n~388, 4 outlets): census / LLM / dense, both groupings
  (f) within-outlet      univariate AUC of c_attrib_density & sourced_rate per outlet (n>=100)

Run on sk3: CUDA_VISIBLE_DEVICES=0 $HOME/envs/ai_usage/bin/python -m methods.claim_verification.run_news_vat_row
"""
import glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
NH = f"{ROOT}/datasets/news-homepages"

def gauc(X, y, g, label, sparse=False):
    y = np.asarray(y); g = np.asarray(g)
    if sparse:
        mk = np.ones(X.shape[0], bool) & pd.Series(y).notna().values
    else:
        X = np.asarray(X, float)
        mk = ~np.all(np.isnan(X), axis=1) & pd.Series(y).notna().values
    if mk.sum() < 80 or len(set(y[mk])) < 2:
        print(f"  {label:46} SKIP (n={int(mk.sum())})", flush=True); return None
    folds = min(5, len(set(g[mk])))
    if sparse:
        pipe = LogisticRegression(max_iter=3000, class_weight="balanced")
    else:
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    a = cross_val_score(pipe, X[mk], y[mk].astype(int),
                        cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                        groups=g[mk], scoring="roc_auc")
    print(f"  {label:46} AUC={np.nanmean(a):.4f} (n={int(mk.sum())}, {folds} folds)", flush=True)
    return float(np.nanmean(a))

def block(M, Xl, E, tag, res):
    y = M.y_twitter.values
    g_pool = (M.outlet.astype(str) + "|" + M.day.astype(str)).values
    g_out = M.outlet.astype(str).values
    counts = [c for c in M.columns if c.startswith("c_") and not c.endswith("_density")]
    dens = [c for c in M.columns if c.endswith("_density")]
    llm = [c for c in ["sourced_rate", "asserted_rate", "nf_rate", "n_cl"] if c in M]
    for gname, g in [("pooled(outlet|day)", g_pool), ("OUTLET-HELD-OUT", g_out)]:
        print(f"--- {tag} | {gname} ---", flush=True)
        res[f"{tag}|census|{gname}"] = gauc(M[counts + dens], y, g, "sourcing census (CODE)")
        res[f"{tag}|llm|{gname}"] = gauc(M[llm], y, g, "LLM attribution bank")
        res[f"{tag}|llm+census|{gname}"] = gauc(M[llm + counts + dens], y, g, "LLM + census")
        if "wordcount" in M:
            res[f"{tag}|wordcount|{gname}"] = gauc(M[["wordcount"]], y, g, "wordcount alone")
        if Xl is not None:
            res[f"{tag}|dense_lex|{gname}"] = gauc(Xl, y, g, "DENSE TF-IDF fulltext", sparse=True)
        if E is not None:
            res[f"{tag}|dense_embed|{gname}"] = gauc(E, y, g, "DENSE bge-large fulltext")

def main():
    M = pd.read_csv(f"{ROOT}/outputs/attrib_adequacy/doc_metrics.csv")
    urls = set(M.url)
    text = {}
    for p in glob.glob(f"{NH}/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("url") in urls and len(r.get("text") or "") > 400:
                text[r["url"]] = r["text"]
    M = M[M.url.isin(text)].reset_index(drop=True)
    M["fulltext"] = M.url.map(text)
    print(f"[news-vat] {len(M)} docs | outlets {M.outlet.value_counts().to_dict()}", flush=True)
    print(f"[news-vat] y-rate by outlet: "
          f"{M.groupby('outlet').y_twitter.mean().round(3).to_dict()}", flush=True)
    res = {}

    # (c) outlet-id alone — confound magnitude (grouped by day so day-leak can't help it)
    OH = pd.get_dummies(M.outlet)
    print("--- (c) outlet-id alone ---", flush=True)
    res["outlet_id_alone"] = gauc(OH.values.astype(float), M.y_twitter.values,
                                  M.day.astype(str).values, "OUTLET-ID one-hot (group=day)")

    # dense features on fulltext
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=150000, sublinear_tf=True)
    Xl = hstack([wv.fit_transform(M.fulltext), cv.fit_transform(M.fulltext)]).tocsr()

    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    mod = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).to(device).eval()
    def embed(texts, bs=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                b = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(device)
                h = mod(**b).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
        return np.vstack(out)

    # (a)+(b)+(d) full corpus: TF-IDF dense only (bge is EN-only; PT docs would be garbage-embedded)
    block(M, Xl, None, "ALL(5 outlets)", res)

    # (e) ENGLISH-only subset
    EN = M[M.outlet != "cnnbrasil"].reset_index(drop=True)
    Xl_en = hstack([wv.fit_transform(EN.fulltext), cv.fit_transform(EN.fulltext)]).tocsr()
    E_en = embed(EN.fulltext.tolist())
    block(EN, Xl_en, E_en, "EN-only(4 outlets)", res)

    # (f) within-outlet univariate
    print("--- (f) within-outlet univariate ---", flush=True)
    for c in ["c_attrib_density", "c_attrib_verbs", "sourced_rate", "wordcount"]:
        if c not in M: continue
        rows = []
        for o, s in M.groupby("outlet"):
            v = s[c].values.astype(float); mk = ~np.isnan(v)
            if mk.sum() >= 100 and s.y_twitter[mk].nunique() == 2:
                rows.append((o, int(mk.sum()), round(roc_auc_score(s.y_twitter[mk], v[mk]), 3)))
        print(f"  {c:20} {rows}", flush=True)
        res[f"within_outlet|{c}"] = rows

    json.dump(res, open(f"{ROOT}/outputs/multi_y_news/news_vat_row.json", "w"), indent=2, default=str)
    print("NEWS_VAT_ROW_DONE", flush=True)

if __name__ == "__main__":
    main()
