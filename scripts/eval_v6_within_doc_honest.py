#!/usr/bin/env python3
"""Honest v6a-vs-v6.1a within-doc eval, controlling for train/test contamination.

Problem: build_v6_training_set.py group-splits with random.seed(7)+shuffle,
but the apps list CHANGED between round 1 and round 2 -> the permutation
changed -> today's test split contains apps that were in v6a's (old) train
split.  v6a's within-doc 0.595 may be memorization.

Fix: round-2-ONLY pairs (key absent from the legacy round-1 extraction file)
did not exist when v6a was trained (model dir mtime Jun 10 17:43), and v6.1a
never saw test-split apps (proper group split).  Within-doc ranking on that
subset is clean for BOTH models.  Also report the round-1-overlap subset:
if v6a is high there but low on round-2-only, contamination is confirmed.

Run on sk3, GPU 6 (small: bge-m3 encode of a few K queries).
"""
import ast
import gzip
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE = os.path.expanduser("~/norm-research/datasets/patents/processed")
LEGACY_EXTR = f"{BASE}/oa_102_extractions.jsonl.gz"
TEST = f"{BASE}/training_pairs_v6_test.jsonl.gz"
SPECS = f"{BASE}/paragraph_keyed_specs.jsonl.gz"
MODELS = {
    "v6a": os.path.expanduser("~/norm-research/models/bge-m3-anticipation-v6a"),
    "v6.1a": os.path.expanduser(
        "~/norm-research/models/bge-m3-anticipation-v6.1a"),
}
TEXT_CAP = 1500

# ---- round-1 keys from the legacy extraction file ----
legacy_keys = set()
with gzip.open(LEGACY_EXTR, "rt") as f:
    for line in f:
        try:
            r = json.loads(line)
            ex = r.get("extraction")
            if isinstance(ex, str):
                ex = ast.literal_eval(ex)
            for e in ex or []:
                legacy_keys.add((str(r.get("app_id")),
                                 str(e.get("target_claim")),
                                 str(e.get("prior_art_pgpub_id")),
                                 (e.get("claim_element") or "")[:200]))
        except Exception:
            continue
print(f"legacy round-1 keys: {len(legacy_keys):,}", flush=True)

# ---- test split, paragraph-gold pairs only ----
pairs = []
with gzip.open(TEST, "rt") as f:
    for line in f:
        r = json.loads(line)
        if r.get("positive_kind") != "paragraph":
            continue
        k = (str(r.get("anchor_app_id")), str(r.get("anchor_target_claim")),
             str(r.get("positive_pgpub_id")),
             (r.get("anchor_text") or "")[:200])
        r["_round2_only"] = k not in legacy_keys
        pairs.append(r)
n2 = sum(p["_round2_only"] for p in pairs)
print(f"test paragraph-gold pairs: {len(pairs):,} "
      f"(round-2-only {n2:,} / round-1-overlap {len(pairs) - n2:,})",
      flush=True)

# ---- specs for the docs we need ----
need = {str(p["positive_pgpub_id"]) for p in pairs}
specs = {}  # pgpub_id -> {para_key: text}


def read_specs(fh):
    for line in fh:
        d = json.loads(line)
        if not d.get("_error") and d.get("paragraphs") \
                and str(d["pgpub_id"]) in need:
            specs[str(d["pgpub_id"])] = d["paragraphs"]


with gzip.open(SPECS, "rt") as f:
    read_specs(f)
import glob  # noqa: E402
for path in glob.glob(f"{BASE}/paragraph_keyed_specs_v2/*.jsonl"):
    with open(path) as f:
        read_specs(f)
print(f"specs loaded: {len(specs):,}/{len(need):,}", flush=True)

queries = [(p["anchor_text"][:TEXT_CAP], p["positive_key"],
            str(p["positive_pgpub_id"]), p["_round2_only"])
           for p in pairs if str(p["positive_pgpub_id"]) in specs
           and p["positive_key"] in specs[str(p["positive_pgpub_id"])]
           and len(specs[str(p["positive_pgpub_id"])]) >= 5]
print(f"eval queries: {len(queries):,}", flush=True)

for name, path in MODELS.items():
    model = SentenceTransformer(path, device="cuda")
    Q = model.encode([q[0] for q in queries], batch_size=256,
                     convert_to_tensor=True, show_progress_bar=False,
                     normalize_embeddings=True)
    ranks = {True: [], False: []}
    for qi, (_, gold_key, doc, r2only) in enumerate(queries):
        keys = list(specs[doc].keys())
        txts = [specs[doc][k][:TEXT_CAP] for k in keys]
        P = model.encode(txts, batch_size=256, convert_to_tensor=True,
                         show_progress_bar=False, normalize_embeddings=True)
        sims = (Q[qi:qi + 1] @ P.T).squeeze(0)
        order = torch.argsort(sims, descending=True).tolist()
        rank = next((j + 1 for j, idx in enumerate(order)
                     if keys[idx] == gold_key), None)
        if rank:
            ranks[r2only].append(rank)
    for r2only, label in ((True, "round-2-only (HONEST both)"),
                          (False, "round-1-overlap (v6a may have trained)")):
        rs = np.array(ranks[r2only])
        if not len(rs):
            continue
        print(f"{name:>6} {label}: n={len(rs):,}  "
              f"MRR={np.mean(1 / rs):.4f}  "
              f"top1={np.mean(rs <= 1):.3f}  top3={np.mean(rs <= 3):.3f}  "
              f"top10={np.mean(rs <= 10):.3f}", flush=True)
    del model, Q
    torch.cuda.empty_cache()
print("HONEST-EVAL-DONE", flush=True)
