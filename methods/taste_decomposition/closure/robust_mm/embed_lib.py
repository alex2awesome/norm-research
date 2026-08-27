#!/usr/bin/env python3
"""Shared embedding + species machinery for the robustified missing-mass battery.

Embedder: BAAI/bge-large-en-v1.5, CLS pooling, L2-normalised, CPU -- identical to
missing_mass.py so cosines are comparable to the pilot's published census.

Threshold calibration is inherited from the pilot's PLANTED PROBES (criteria authored
to be lexically similar to a real criterion but conceptually distinct).  Their
name+definition cosines were .739 and .615, so any threshold at or below .74 calls
genuinely distinct concepts duplicates; the defensible band is tau >= .78.  Detection
runs at tau = .79 (band midpoint) with sensitivity reported at .77 / .81.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "emb_cache.npz"
TAU = 0.79
TAU_SENS = (0.77, 0.81)
PROBE_FLOOR = 0.739          # max planted-lookalike cosine (name+definition)
BOILER = re.compile(r"^(composite:\s*|score\s+0-10\s*(on|for|how)?\s*|score\s+the\s+)", re.I)


def _key(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def embed(texts, batch=16, verbose=True):
    """Embed with an on-disk cache keyed by text hash."""
    texts = list(texts)
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = {k: v for k, v in zip(z["keys"], z["vecs"])}
    need = [t for t in texts if _key(t) not in cache]
    if need:
        if verbose:
            print(f"[embed] {len(need)} new texts (cache holds {len(cache)})", flush=True)
        import torch
        from transformers import AutoModel, AutoTokenizer
        name = "BAAI/bge-large-en-v1.5"
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModel.from_pretrained(name).eval()
        with torch.no_grad():
            for i in range(0, len(need), batch):
                enc = tok(need[i:i + batch], padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
                h = mod(**enc).last_hidden_state[:, 0]
                v = torch.nn.functional.normalize(h, dim=-1).numpy()
                for t, vv in zip(need[i:i + batch], v):
                    cache[_key(t)] = vv
        np.savez_compressed(CACHE, keys=np.array(list(cache.keys())),
                            vecs=np.array(list(cache.values())))
    return np.array([cache[_key(t)] for t in texts])


def crit_text(name, definition=""):
    d = BOILER.sub("", (definition or "")).strip()
    return f"{name}. {d}" if d else str(name)


def single_linkage(S, tau):
    n = S.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    lab = np.array([find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def chao1(sizes):
    sizes = np.asarray(sizes)
    S = int(len(sizes))
    f1 = int((sizes == 1).sum())
    f2 = int((sizes == 2).sum())
    n = int(sizes.sum())
    return {"S_obs": S, "f1": f1, "f2": f2, "n": n,
            "chao1_classic": float(S + f1 ** 2 / (2 * f2)) if f2 > 0 else None,
            "chao1_bias_corrected": float(S + f1 * (f1 - 1) / (2 * (f2 + 1))),
            "good_turing_missing_mass": float(f1 / n) if n else float("nan")}


def bank_concept_texts():
    """The 154-bank's distinct concepts, name + definition."""
    rows = [json.loads(l) for l in open(HERE.parent / "ref" / "rubrics_154.jsonl") if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["name"], r["description"])
    return by
