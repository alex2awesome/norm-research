#!/usr/bin/env python3
"""RoyalRoad re-matched EXPANSION (option b).

The n=1,274 cell of record is a topic x era-matched subsample of a 2,367-fiction
usable pool. That matching is LOAD-BEARING: unmatched, the lexical floor rises
.524 -> .606 and the era-y correlation returns at .155, crossing the S17m
"<0.6 CLEAN" line. So we cannot simply take the whole pool.

This script finds the LARGEST topic x era-matched subsample that still holds a
lexical floor < .58 (a deliberate margin under the .60 line).

Mechanism: matching granularity is the dial. Within every (topic_cluster, era)
cell the majority class is downsampled to the minority count, so:
  more clusters -> finer matching -> cleaner but smaller
  fewer clusters -> coarser matching -> larger but dirtier
We therefore sweep k (number of bge-large k-means topic clusters) and take the
SMALLEST k -- i.e. the LARGEST n -- whose lexical floor still clears the margin.

Splits are recomputed with the SAME per-fiction stable hash
md5("split::"+fiction_id)%1000, which is a pure function of the id: growing the
population cannot move any existing row across the split boundary. That is the
whole reason the rule is a hash and never a seeded shuffle
(feedback_stable_hash_splits).

  python datasets/creative-writing/build_royalroad_expanded.py --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CELL = REPO / "datasets/creative-writing/royalroad_stubs"
V1 = CELL / "built/royalroad_stubs_v1.jsonl"
OUT = CELL / "va_expanded"
LEX_MARGIN = 0.58
K_SWEEP = [6, 8, 10, 12, 16, 20, 24, 32, 40]
EMB_CHARS = 3000


def splitof(fid):
    b = int(hashlib.md5(("split::" + str(fid)).encode()).hexdigest(), 16) % 1000
    return "train" if b < 800 else ("eval" if b < 900 else "test")


def era_of(v):
    try:
        yr = int(str(int(v))[:4])
        return "old" if yr <= 2021 else str(yr)
    except Exception:
        return "NA"


def lexical_floor(d):
    tr, te = d[d.split == "train"], d[d.split != "train"]
    if te.judgement.nunique() < 2 or len(te) < 40:
        return float("nan")
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
    X = v.fit_transform(tr.text.astype(str))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, tr.judgement)
    return float(roc_auc_score(te.judgement, lr.predict_proba(v.transform(te.text.astype(str)))[:, 1]))


def register_floor(d):
    def sf(s):
        s = str(s); w = s.split(); nw = max(len(w), 1)
        sents = max(s.count(".") + s.count("!") + s.count("?"), 1)
        paras = max(len([p for p in s.split("\n\n") if p.strip()]), 1)
        return [np.log1p(len(s)), np.log1p(nw), nw / sents, nw / paras,
                sum(c.isupper() for c in s) / max(len(s), 1),
                s.count('"') / nw * 100, s.count("!") / nw * 100,
                s.count("?") / nw * 100, len(set(w)) / nw]
    tr, te = d[d.split == "train"], d[d.split != "train"]
    if te.judgement.nunique() < 2:
        return float("nan")
    Z, Zt = np.array([sf(t) for t in tr.text]), np.array([sf(t) for t in te.text])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Z, tr.judgement)
    return float(roc_auc_score(te.judgement, lr.predict_proba(Zt)[:, 1]))


def match(df, cl):
    """Downsample the majority class to the minority count inside every
    (topic_cluster, era) cell, choosing survivors by stable hash (never random)."""
    d = df.copy()
    d["cl"] = cl
    keep = []
    for (_, _), g in d.groupby(["cl", "wayback_era"]):
        p, n = g[g.judgement == 1], g[g.judgement == 0]
        m = min(len(p), len(n))
        if m == 0:
            continue
        for part in (p, n):
            order = sorted(part.fiction_id.astype(str),
                           key=lambda x: hashlib.md5(("match|" + x).encode()).hexdigest())
            keep += order[:m]
    return set(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--margin", type=float, default=LEX_MARGIN)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    v1 = pd.read_json(V1, lines=True)
    f = v1.sort_values("chapter_rank").groupby("fiction_id", as_index=False).first()
    f["judgement"] = f["y"].astype(int)
    f["wayback_era"] = f["wayback_ts"].map(era_of)
    f["fiction_id"] = f["fiction_id"].astype(str)
    f["split"] = f["fiction_id"].map(splitof)
    print(f"[pool] {len(f)} usable fictions (rank1 + wayback), "
          f"y={dict(Counter(f.judgement))}, eras={dict(Counter(f.wayback_era))}")

    from sentence_transformers import SentenceTransformer
    print("[emb] bge-large ...", flush=True)
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    emb = model.encode([t[:EMB_CHARS] for t in f.text.astype(str)],
                       batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    print(f"[emb] {emb.shape}", flush=True)

    cur_ids = set(pd.read_csv(CELL / "va/population.csv.gz").row_id.astype(str))
    sweep = []
    for k in K_SWEEP:
        cl = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096).fit(emb).labels_
        keep = match(f, cl)
        d = f[f.fiction_id.isin(keep)]
        lex, reg = lexical_floor(d), register_floor(d)
        eras = pd.get_dummies(d.wayback_era)
        ec = max(abs(np.corrcoef(eras[c].astype(float), d.judgement)[0, 1]) for c in eras.columns)
        row = {"k": k, "n": int(len(d)), "pos_rate": round(float(d.judgement.mean()), 4),
               "lexical": round(lex, 4), "register": round(reg, 4),
               "max_abs_era_y_corr": round(float(ec), 4),
               "n_new_vs_current": int(len(set(d.fiction_id) - cur_ids)),
               "clean": bool(lex < a.margin)}
        sweep.append(row)
        print(f"  k={k:3d}  n={row['n']:5d}  lex={row['lexical']:.4f}  "
              f"reg={row['register']:.4f}  era|r|={row['max_abs_era_y_corr']:.3f}  "
              f"new={row['n_new_vs_current']:5d}  {'CLEAN' if row['clean'] else 'over margin'}",
              flush=True)

    ok = [r for r in sweep if r["clean"]]
    OUT.mkdir(parents=True, exist_ok=True)
    if not ok:
        (OUT / "sweep.json").write_text(json.dumps({"sweep": sweep, "chosen": None,
                                                    "margin": a.margin}, indent=2))
        print(f"\nNO subsample clears lexical < {a.margin}. STOP: keep the n=1,274 "
              f"cell of record.")
        print("EXPANSION_VERDICT=NONE")
        return

    best = max(ok, key=lambda r: r["n"])
    print(f"\n[chosen] k={best['k']} n={best['n']} lexical={best['lexical']} "
          f"(+{best['n'] - 1274} rows vs the 1,274 cell of record)")

    if best["n"] < 1450:
        (OUT / "sweep.json").write_text(json.dumps(
            {"sweep": sweep, "chosen": best, "margin": a.margin,
             "decision": "STOP -- best clean n < 1450; keep the n=1,274 build + "
                         "cross-fit as the cell of record (coordinator's rule)"}, indent=2))
        print("EXPANSION_VERDICT=STOP_TOO_SMALL")
        return

    cl = MiniBatchKMeans(n_clusters=best["k"], random_state=0, n_init=3,
                         batch_size=4096).fit(emb).labels_
    keep = match(f, cl)
    d = f[f.fiction_id.isin(keep)].copy()
    d["topic_cluster"] = pd.Series(cl, index=f.index).loc[d.index].astype(int)
    out = pd.DataFrame({
        "row_id": d.fiction_id, "group": d.fiction_id,
        "topic_cluster": d.topic_cluster.astype(str), "wayback_era": d.wayback_era,
        "text": d.text.astype(str), "judgement": d.judgement.astype(int),
        "split": d.split,
    }).reset_index(drop=True)
    out["is_new_row"] = ~out.row_id.isin(cur_ids)
    out.to_csv(OUT / "population.csv.gz", index=False, compression="gzip")

    man = {
        "cell": "cw_royalroad_verdict_expanded",
        "n": int(len(out)), "n_pos": int(out.judgement.sum()),
        "n_new_rows": int(out.is_new_row.sum()),
        "n_carried_from_1274": int((~out.is_new_row).sum()),
        "chosen_k": best["k"], "lexical_floor": best["lexical"],
        "register_floor": best["register"],
        "max_abs_era_y_corr": best["max_abs_era_y_corr"],
        "margin": a.margin, "sweep": sweep,
        "split_rule": 'md5("split::"+fiction_id)%1000 -> <800/<900/test; a pure '
                      "function of the id, so growing the population moves no "
                      "existing row across a split boundary",
        "split_counts": out.split.value_counts().to_dict(),
        "split_pos_counts": out.groupby("split").judgement.sum().astype(int).to_dict(),
        "matching": "majority class downsampled to minority count within every "
                    "(topic_cluster, era) cell; survivors chosen by stable hash",
        "rescore_policy": "Gemma bank is rescored on NEW ROWS ONLY; the "
                          "1,274 carried rows keep their existing token-truncated "
                          "scores (never re-judged)",
    }
    (OUT / "population_manifest.json").write_text(json.dumps(man, indent=2))
    (OUT / "sweep.json").write_text(json.dumps({"sweep": sweep, "chosen": best,
                                                "margin": a.margin}, indent=2))
    print(f"[write] {OUT/'population.csv.gz'}  n={len(out)} "
          f"(new {int(out.is_new_row.sum())}, carried {int((~out.is_new_row).sum())})")
    print(f"[split] {man['split_counts']} pos {man['split_pos_counts']}")
    print(f"EXPANSION_VERDICT=GO n={len(out)} lexical={best['lexical']}")


if __name__ == "__main__":
    main()
