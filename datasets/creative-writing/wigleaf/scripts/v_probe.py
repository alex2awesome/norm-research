#!/usr/bin/env python3
"""Deterministic V-feature probe for Wigleaf cw-B (reuse creative_writing codegen).

Loads every creative_writing codegen program (each exposes score(text)->float),
builds a feature matrix, fits a balanced LogisticRegression on the train split,
reports AUC on eval+test. V>0 = the verifiable/codeable layer carries signal.
Mirrors codegen_v_full.py exactly.
"""
import hashlib
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = "/lfs/skampere3/0/alexspan/norm-research"
BASE = f"{ROOT}/datasets/creative-writing/wigleaf"
RNG = np.random.RandomState(0)


def load_programs(domain="creative_writing"):
    d = Path(f"{ROOT}/runs/validity_full/v2/{domain}/codegen_claude")
    progs = []
    for i, p in enumerate(sorted(d.glob("*.py"))):
        try:
            s = importlib.util.spec_from_file_location(f"m_{i}", p)
            m = importlib.util.module_from_spec(s)
            s.loader.exec_module(m)
            if hasattr(m, "score"):
                progs.append(m.score)
        except Exception:
            pass
    return progs


def mat(progs, texts):
    X = np.full((len(texts), len(progs)), 0.5, np.float32)
    for j, fn in enumerate(progs):
        for i, t in enumerate(texts):
            try:
                v = fn(t)
                X[i, j] = float(v) if v is not None else 0.5
            except Exception:
                pass
    return X


def main():
    df = pd.read_parquet(f"{BASE}/built/all.parquet")
    progs = load_programs()
    print(f"loaded {len(progs)} creative_writing codegen programs")
    tr = df[df.split == "train"]
    te = df[df.split.isin(["eval", "test"])]
    # subsample for speed (programs x texts can be large)
    if len(tr) > 2500:
        tr = tr.sample(2500, random_state=0)
    Xtr = mat(progs, tr.text.astype(str).tolist())
    Xte = mat(progs, te.text.astype(str).tolist())
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5).fit(Xtr, tr.judgement)
    auc = roc_auc_score(te.judgement, lr.predict_proba(Xte)[:, 1])
    print(f"codegen-V AUC = {auc:.3f}  ({len(progs)} programs, tr={len(tr)}, te={len(te)})")
    print(f"V>0: {'YES' if auc > 0.5 else 'NO'}  (delta over chance = {auc-0.5:+.3f})")


if __name__ == "__main__":
    main()
