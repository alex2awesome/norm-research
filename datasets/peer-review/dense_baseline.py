#!/usr/bin/env python3
"""Dense ceiling for the peer-review V/A row (ICLR n=2400, same folds as aggregate_va.py).

v2 — post-Codex-audit leak fixes (2026-07-12):
  * TF-IDF vectorizers are FOLD-LOCAL (inside the CV pipeline), never fit on evaluation text.
  * Big-train arm fits vectorizers + LR on the 20,201-pool ONLY, transforms the 2,400 once.
  * No max-selection: PRIMARY (predeclared) = D_lex_big; all arms reported individually.
  * Embedding arms use a frozen pretrained bge-large (no fitting -> no leak).

Reference: V=.611 A=.676 V+A=.682 (peer_review_va.json). T_hat = D_primary - .682.
Run: CUDA_VISIBLE_DEVICES=<free> $HOME/envs/ai_usage/bin/python datasets/peer-review/dense_baseline.py
"""
import csv, gzip, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
NPZ = f"{ROOT}/datasets/peer-review/peer_review_scores_iclr.npz"
REF = {"V": 0.611, "A": 0.676, "VA": 0.682}
SKF = StratifiedKFold(5, shuffle=True, random_state=0)

def norm_prefix(t):
    return re.sub(r"[^a-z0-9]+", "", str(t).lower())[:150]

def lex_pipe():
    return Pipeline([
        ("feats", FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200000, sublinear_tf=True)),
        ])),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

def embed(texts, tok, model, device, bs=64):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512,
                    return_tensors="pt").to(device)
            h = model(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
    return np.vstack(out)

def main():
    d = np.load(NPZ, allow_pickle=True)
    ids = [str(i) for i in d["ids"]]; y = d["y"].astype(int)
    print(f"[dense-v2] npz rows {len(y)} (accept rate {y.mean():.3f})", flush=True)
    t = pd.read_csv(f"{ROOT}/datasets/peer-review/splits/train.csv.gz")
    by_id = t.drop_duplicates("id").set_index("id")
    miss = [i for i in ids if i not in by_id.index]
    assert not miss, f"{len(miss)} npz ids missing"
    texts = np.array([str(by_id.loc[i, "text"]) for i in ids], dtype=object)
    years = np.array([by_id.loc[i, "year"] for i in ids], dtype=float)
    pref = [norm_prefix(x) for x in texts]
    print(f"[dense-v2] within-2400 prefix dups: {len(pref) - len(set(pref))}", flush=True)

    print("\n=== matched-protocol (fold-local vectorizers, same folds as V/A) ===", flush=True)
    a = cross_val_score(lex_pipe(), texts, y, cv=SKF, scoring="roc_auc")
    d_lex = float(a.mean())
    print(f"  D_lex fold-local                  AUC={d_lex:.4f} (folds {[round(x,3) for x in a]})", flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).to(device).eval()
    E = embed(list(texts), tok, model, device)
    ep = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    d_emb = float(cross_val_score(ep, E, y, cv=SKF, scoring="roc_auc").mean())
    print(f"  D_embed (frozen bge, no leak)     AUC={d_emb:.4f}", flush=True)

    # per-year OOF (fold-local)
    oof = cross_val_predict(lex_pipe(), texts, y, cv=SKF, method="predict_proba")[:, 1]
    rows = [(int(yr), int((years == yr).sum()), round(roc_auc_score(y[years == yr], oof[years == yr]), 3))
            for yr in sorted(set(years[~np.isnan(years)]))
            if (years == yr).sum() > 60 and len(set(y[years == yr])) == 2]
    print(f"  D_lex OOF per-year: {rows}", flush=True)

    print("\n=== big-train (fit on POOL ONLY; transform the 2400 once) ===", flush=True)
    pool = t[t.venue.astype(str).str.lower().str.contains("iclr") & ~t.id.isin(set(ids))].copy()
    pool = pool.dropna(subset=["text", "judgement"])
    guard = pool.text.map(norm_prefix).isin(set(pref))
    pool = pool[~guard]
    print(f"[dense-v2] pool {len(pool)} (guard removed {int(guard.sum())})", flush=True)
    yp = pool.judgement.astype(int).values
    ptexts = pool.text.astype(str).tolist()
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200000, sublinear_tf=True)
    Xp = hstack([wv.fit_transform(ptexts), cv.fit_transform(ptexts)]).tocsr()   # POOL-fit
    big = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xp, yp)
    X24 = hstack([wv.transform(list(texts)), cv.transform(list(texts))]).tocsr()
    d_lex_big = float(roc_auc_score(y, big.predict_proba(X24)[:, 1]))
    print(f"  D_lex_big (PRIMARY, pool-fit n={len(pool)})  AUC={d_lex_big:.4f}", flush=True)
    Ep = embed(ptexts, tok, model, device)
    bigE = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(Ep, yp)
    d_emb_big = float(roc_auc_score(y, bigE.predict_proba(E)[:, 1]))
    print(f"  D_embed_big (pool-fit)                       AUC={d_emb_big:.4f}", flush=True)

    out = dict(domain="peer_review", input="abstract", n=int(len(y)), ref=REF, version="v2_foldlocal",
               primary_arm="D_lex_big", D_lex=d_lex, D_embed=d_emb,
               D_lex_big=d_lex_big, D_embed_big=d_emb_big,
               T_hat_primary=round(d_lex_big - REF["VA"], 4), per_year_oof=rows,
               big_train_n=int(len(pool)))
    json.dump(out, open(f"{ROOT}/notebooks/data/peer_review_dense_v2.json", "w"), indent=2)
    print(f"\n[dense-v2] V={REF['V']} A={REF['A']} V+A={REF['VA']} | PRIMARY D_lex_big={d_lex_big:.4f} "
          f"| T_hat={d_lex_big - REF['VA']:+.4f}", flush=True)
    print("DENSE_PEER_V2_DONE", flush=True)

if __name__ == "__main__":
    main()
