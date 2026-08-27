"""SE ladder step 3a: frozen bge embedding probe (2026-06-11).

Per slice: embed the raw answer body with BAAI/bge-large-en-v1.5 (frozen,
CLS pooling, L2-normalized, 512-token truncation), fit LR on train-split
embeddings, report test-split AUC + per-stratum test AUCs.

Embeddings cached to outputs/v2_analysis/se_ladder/{slice}_bge.npy
(append-only; recomputed only if missing).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python se_ladder_dense_probe.py <slice>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"
MODEL = "BAAI/bge-large-en-v1.5"


def embed(texts, batch_size=256):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t if isinstance(t, str) and t else " "
                     for t in texts[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state[:, 0]  # CLS
            h = torch.nn.functional.normalize(h, dim=-1)
            out.append(h.float().cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"  embed {i}/{len(texts)}", flush=True)
    return np.concatenate(out)


def main():
    slice_name = sys.argv[1]
    df = pd.read_parquet(OUT_DIR / f"{slice_name}_input.parquet")
    emb_path = OUT_DIR / f"{slice_name}_bge.npy"
    if emb_path.exists():
        X = np.load(emb_path)
        assert len(X) == len(df)
        print(f"[{slice_name}] loaded cached embeddings {X.shape}", flush=True)
    else:
        X = embed(df["body"].tolist())
        np.save(emb_path, X)
        print(f"[{slice_name}] wrote {emb_path} {X.shape}", flush=True)

    tr_m = (df.split == "train").values
    te_m = (df.split == "test").values
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr_m], df.label.values[tr_m])
    p = clf.predict_proba(X[te_m])[:, 1]
    yte = df.label.values[te_m]
    res = {"slice": slice_name, "model": MODEL,
           "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
           "test_auc": float(roc_auc_score(yte, p))}
    te = df[te_m].copy()
    te["p"] = p
    strata = {}
    for s, sub in te.groupby("stratum"):
        if len(sub) < 200 or sub.label.nunique() < 2:
            continue
        strata[s] = {"n_test": int(len(sub)),
                     "auc": float(roc_auc_score(sub.label, sub.p))}
    res["per_stratum_test"] = strata
    out = OUT_DIR / f"{slice_name}_bge_probe.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
