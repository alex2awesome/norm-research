#!/usr/bin/env python3
"""News VAT row v2 — ENGLISH-ONLY do-over at full corpus scale, y rebuilt.

Fixes vs v1 (pilot 700 docs, 44% Portuguese):
  1. Drop cnnbrasil (attribution regexes never fire on PT) and bbc (n=10).
     EN pool with fulltext: latimes ~3.3K / guardian ~2.8K / cnn ~2.5K / nytimes ~1.3K (~9.9K).
  2. y REBUILT: pilot y ranked n_tweets within outlet x day on the SAMPLE -> 1-2 doc cells ->
     y ~ cell size (outlet y-rates .69-.94 despite within-cell construction). At full density
     cells average ~62 docs. Also n_tweets is CAPPED at 100 (median = cap) -> rank sum_likes.
     y = pct(sum_likes within outlet x day) >= .5, cells >= 6 only. By construction outlet
     carries ~no label signal -> outlet-id-alone AUC ~.50 is the sanity check.
  3. Census/dense on ALL EN docs; LLM attribution bank on a stable-hash subsample of 400/outlet
     (gemma on 8006), with the 6 blinded ANCHORS + mismatched placebo every 3rd doc as before.

Readouts (LR, AUC): outlet-id alone; census / wordcount / dense TF-IDF / dense bge-large,
each pooled(outlet|day) vs OUTLET-HELD-OUT; within-outlet univariates; LLM bank race on the
subsample. Code layers print + save BEFORE the LLM leg so a crash there loses nothing.

Run on sk3: CUDA_VISIBLE_DEVICES=0 $HOME/envs/ai_usage/bin/python -m methods.claim_verification.run_news_vat_v2
"""
import glob, hashlib, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from claim_verification.core import extract_claims, split_head_body
from claim_verification.evidence_api import clean_evidence_text
from claim_verification.seam_metrics import sourcing_code_metrics
import claim_verification.run_attrib_adequacy as aa

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
NH = f"{ROOT}/datasets/news-homepages"
EN_OUTLETS = {"latimes", "guardian", "cnn", "nytimes"}
MIN_CELL = 6
SUB_PER_OUTLET = 400

def gauc(X, y, g, label, sparse=False):
    y = np.asarray(y); g = np.asarray(g)
    if sparse:
        mk = np.ones(X.shape[0], bool)
    else:
        X = np.asarray(X, float)
        mk = ~np.all(np.isnan(X), axis=1)
    mk &= pd.Series(y).notna().values
    if mk.sum() < 80 or len(set(y[mk])) < 2:
        print(f"  {label:46} SKIP (n={int(mk.sum())})", flush=True); return None
    folds = min(5, len(set(g[mk])))
    clf = LogisticRegression(max_iter=3000, class_weight="balanced") if sparse else \
        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    a = cross_val_score(clf, X[mk], y[mk].astype(int),
                        cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                        groups=g[mk], scoring="roc_auc")
    print(f"  {label:46} AUC={np.nanmean(a):.4f} (n={int(mk.sum())}, {folds} folds)", flush=True)
    return float(np.nanmean(a))

def main():
    res = {}
    # ---- corpus ----
    tw = {}
    for ln in open(f"{NH}/twitter_engagement/tweet_engagement.jsonl"):
        try:
            r = json.loads(ln)
            if r.get("outlet") in EN_OUTLETS:
                tw[r["url"]] = {"outlet": r["outlet"], "day": (r.get("first_day") or "")[:10],
                                "n_tweets": int(r.get("n_tweets", 0)),
                                "likes": int(r.get("sum_likes", 0))}
        except Exception: pass
    print(f"[v2] EN engagement rows: {len(tw)}", flush=True)
    rows = []
    for p in glob.glob(f"{NH}/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            u = r.get("url")
            if u in tw and len(r.get("text") or "") > 400:
                body = clean_evidence_text(r["text"])
                m = {"url": u, **tw[u], "wordcount": len(body.split()), "body": body}
                m.update(sourcing_code_metrics(body[:12000]))
                rows.append(m)
    M = pd.DataFrame(rows).drop_duplicates("url").reset_index(drop=True)
    print(f"[v2] docs with fulltext: {len(M)} | {M.outlet.value_counts().to_dict()}", flush=True)

    # ---- y rebuild: sum_likes pct within outlet x day, cells >= MIN_CELL ----
    cell_n = M.groupby(["outlet", "day"]).url.transform("size")
    M = M[cell_n >= MIN_CELL].reset_index(drop=True)
    M["pct"] = M.groupby(["outlet", "day"]).likes.rank(pct=True)
    M["y"] = (M.pct >= 0.5).astype(int)
    print(f"[v2] after cell>={MIN_CELL}: n={len(M)} | y-rate by outlet: "
          f"{M.groupby('outlet').y.mean().round(3).to_dict()}", flush=True)
    print(f"[v2] likes: zero-rate {float((M.likes == 0).mean()):.3f}, "
          f"median {int(M.likes.median())}", flush=True)
    y = M.y.values
    g_pool = (M.outlet.astype(str) + "|" + M.day.astype(str)).values
    g_out = M.outlet.astype(str).values
    counts = [c for c in M.columns if c.startswith("c_") and not c.endswith("_density")]
    dens = [c for c in M.columns if c.endswith("_density")]

    # ---- sanity: outlet-id must be ~chance now (y is within-cell by construction) ----
    print("--- sanity: outlet-id alone (group=day) ---", flush=True)
    res["outlet_id_alone"] = gauc(pd.get_dummies(M.outlet).values.astype(float), y,
                                  M.day.astype(str).values, "OUTLET-ID one-hot")

    # ---- code + dense layers on full EN corpus ----
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200000, sublinear_tf=True)
    Xl = hstack([wv.fit_transform(M.body), cv.fit_transform(M.body)]).tocsr()
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
            if (i // 64) % 30 == 0: print(f"    embed {i}/{len(M)}", flush=True)
    E = np.vstack(E)

    for gname, g in [("pooled(outlet|day)", g_pool), ("OUTLET-HELD-OUT", g_out)]:
        print(f"--- EN full (n={len(M)}) | {gname} ---", flush=True)
        res[f"census|{gname}"] = gauc(M[counts + dens], y, g, "sourcing census (CODE)")
        res[f"wordcount|{gname}"] = gauc(M[["wordcount"]], y, g, "wordcount alone")
        res[f"dense_lex|{gname}"] = gauc(Xl, y, g, "DENSE TF-IDF fulltext", sparse=True)
        res[f"dense_embed|{gname}"] = gauc(E, y, g, "DENSE bge-large fulltext")
        res[f"census+wc|{gname}"] = gauc(M[counts + dens + ["wordcount"]], y, g, "census + wordcount")

    print("--- within-outlet univariates ---", flush=True)
    for c in ["c_attrib_verbs", "c_attrib_density", "c_quote_density", "c_numbers", "wordcount"]:
        rr = []
        for o, s in M.groupby("outlet"):
            v = s[c].values.astype(float); mk = ~np.isnan(v)
            if mk.sum() >= 200 and s.y[mk].nunique() == 2:
                rr.append((o, int(mk.sum()), round(roc_auc_score(s.y[mk], v[mk]), 3)))
        print(f"  {c:20} {rr}", flush=True)
        res[f"within_outlet|{c}"] = rr

    M.drop(columns=["body"]).to_csv(f"{ROOT}/outputs/multi_y_news/doc_metrics_v2.csv", index=False)
    json.dump(res, open(f"{ROOT}/outputs/multi_y_news/news_vat_v2.json", "w"), indent=2, default=str)
    print("CODE_DENSE_LAYERS_DONE (saved)", flush=True)

    # ---- LLM attribution bank on stable-hash subsample ----
    sub = (M.assign(h=M.url.map(lambda u: hashlib.sha1(u.encode()).hexdigest()))
             .sort_values("h").groupby("outlet").head(SUB_PER_OUTLET).reset_index(drop=True))
    print(f"[v2-llm] subsample {len(sub)} | {sub.outlet.value_counts().to_dict()}", flush=True)
    cache = aa.Cache(f"{ROOT}/outputs/attrib_adequacy/cache.jsonl")
    bodies = dict(zip(M.url, M.body))
    docs = list(zip(sub.url, sub.url.map(bodies)))
    lock = Lock(); out = []
    def work(i):
        u, t = docs[i]
        head, _ = split_head_body(t)
        try: claims = extract_claims(head, aa.CFG, cache)
        except Exception: return
        for c in claims:
            cl = c["claim"] if isinstance(c, dict) else str(c)
            try: r = aa.attr_check(cl, t, cache)
            except Exception: continue
            out.append({"url": u, "arm": "matched", **r})
            if i % 3 == 0:
                try: rp = aa.attr_check(cl, docs[(i + len(docs) // 2) % len(docs)][1], cache)
                except Exception: continue
                out.append({"url": u, "arm": "placebo", **rp})
        with lock:
            d = len({r_["url"] for r_ in out})
            if d % 100 == 0: print(f"[v2-llm] ~{d}/{len(docs)} docs", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(work, range(len(docs))))
    ok = sum(int(aa.attr_check(cl, art, cache)["verdict"] == want) for want, cl, art in aa.ANCHORS)
    print(f"[v2-llm] ANCHORS {ok}/{len(aa.ANCHORS)}", flush=True)
    F = pd.DataFrame(out)
    for arm, gg in F.groupby("arm"):
        print(f"  {arm:8} n={len(gg):5d} "
              f"{gg.verdict.value_counts(normalize=True).round(3).to_dict()}", flush=True)
    agg = (F[F.arm == "matched"].groupby("url").verdict
           .agg(sourced_rate=lambda v: (v == "SOURCED").mean(),
                asserted_rate=lambda v: (v == "ASSERTED_ONLY").mean(),
                nf_rate=lambda v: (v == "NOT_FOUND").mean(), n_cl="size").reset_index())
    S = sub.merge(agg, on="url", how="inner")
    print(f"[v2-llm] {len(S)} docs with aggregates; sourced_rate mean "
          f"{S.sourced_rate.mean():.3f}", flush=True)
    ys = S.y.values
    gp = (S.outlet.astype(str) + "|" + S.day.astype(str)).values
    go = S.outlet.astype(str).values
    llm = ["sourced_rate", "asserted_rate", "nf_rate", "n_cl"]
    for gname, g in [("pooled(outlet|day)", gp), ("OUTLET-HELD-OUT", go)]:
        print(f"--- LLM bank race (subsample n={len(S)}) | {gname} ---", flush=True)
        res[f"sub|llm|{gname}"] = gauc(S[llm], ys, g, "LLM attribution bank")
        res[f"sub|census|{gname}"] = gauc(S[counts + dens], ys, g, "census (same docs)")
        res[f"sub|llm+census|{gname}"] = gauc(S[llm + counts + dens], ys, g, "LLM + census")
    S.to_csv(f"{ROOT}/outputs/multi_y_news/doc_metrics_v2_llm.csv", index=False)
    json.dump(res, open(f"{ROOT}/outputs/multi_y_news/news_vat_v2.json", "w"), indent=2, default=str)
    print("NEWS_VAT_V2_DONE", flush=True)

if __name__ == "__main__":
    main()
