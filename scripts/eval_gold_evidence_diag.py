#!/usr/bin/env python3
"""Gold-evidence diagnostic: separate 'retrieval is bad' from 'MRR is
pessimistic by construction' and measure oracle-evidence recall.

Within-doc MRR (~0.21 honest) counts ONLY the exact examiner-cited paragraph
as correct, but specs are repetitive. Here xenc-v1 (judge proxy, 0.844 AUC)
scores, for each test (element, gold-para) pair:
  - the examiner-gold paragraph
  - the v6a top-1/2/3 retrieved NON-gold paragraphs from the same doc

Readouts:
  1. disclosed-rate (score>0.5) gold vs top-retrieved-nongold — if similar,
     low MRR is mostly single-gold-among-many-disclosing (reconciles the
     wide-K null); if gold >> retrieved, retrieval genuinely misses content.
  2. Oracle-evidence recall: per (app, claim), min over its elements of the
     gold-pair score — all these claims DID fall to §102, so low min means
     decomposition/extraction/judge is the bottleneck, not retrieval.
"""
import ast
import glob
import gzip
import json
import os

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE = os.path.expanduser("~/norm-research/datasets/patents/processed")
LEGACY_EXTR = f"{BASE}/oa_102_extractions.jsonl.gz"
TEST = f"{BASE}/training_pairs_v6_test.jsonl.gz"
SPECS = f"{BASE}/paragraph_keyed_specs.jsonl.gz"
V6A = os.path.expanduser("~/norm-research/models/bge-m3-anticipation-v6a")
XENC = os.path.expanduser("~/norm-research/models/element-para-xenc-v1")
TEXT_CAP = 1500

# round-1 keys (to slice round-2-only, same as honest eval)
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
print(f"test paragraph-gold pairs: {len(pairs):,}", flush=True)

need = {str(p["positive_pgpub_id"]) for p in pairs}
specs = {}


def read_specs(fh):
    for line in fh:
        d = json.loads(line)
        if not d.get("_error") and d.get("paragraphs") \
                and str(d["pgpub_id"]) in need:
            specs[str(d["pgpub_id"])] = d["paragraphs"]


with gzip.open(SPECS, "rt") as f:
    read_specs(f)
for path in glob.glob(f"{BASE}/paragraph_keyed_specs_v2/*.jsonl"):
    with open(path) as f:
        read_specs(f)
print(f"specs loaded: {len(specs):,}/{len(need):,}", flush=True)

queries = [p for p in pairs if str(p["positive_pgpub_id"]) in specs
           and p["positive_key"] in specs[str(p["positive_pgpub_id"])]
           and len(specs[str(p["positive_pgpub_id"])]) >= 5]
print(f"eval queries: {len(queries):,}", flush=True)

# v6a top-3 non-gold retrieved paras per query
benc = SentenceTransformer(V6A, device="cuda")
Q = benc.encode([q["anchor_text"][:TEXT_CAP] for q in queries],
                batch_size=256, convert_to_tensor=True,
                show_progress_bar=False, normalize_embeddings=True)
retrieved = []  # per query: list of up to 3 non-gold para keys
for qi, p in enumerate(queries):
    doc = str(p["positive_pgpub_id"])
    keys = list(specs[doc].keys())
    txts = [specs[doc][k][:TEXT_CAP] for k in keys]
    P = benc.encode(txts, batch_size=256, convert_to_tensor=True,
                    show_progress_bar=False, normalize_embeddings=True)
    sims = (Q[qi:qi + 1] @ P.T).squeeze(0)
    order = torch.argsort(sims, descending=True).tolist()
    retrieved.append([keys[i] for i in order
                      if keys[i] != p["positive_key"]][:3])
del benc, Q
torch.cuda.empty_cache()
print("v6a retrieval done", flush=True)

# xenc scores
flat, tags = [], []  # tag = (qi, 'gold') or (qi, rank)
for qi, p in enumerate(queries):
    doc = str(p["positive_pgpub_id"])
    el = p["anchor_text"][:TEXT_CAP]
    flat.append([el, specs[doc][p["positive_key"]][:TEXT_CAP]])
    tags.append((qi, "gold"))
    for rank, k in enumerate(retrieved[qi]):
        flat.append([el, specs[doc][k][:TEXT_CAP]])
        tags.append((qi, rank))
print(f"xenc pairs: {len(flat):,}", flush=True)
xenc = CrossEncoder(XENC, max_length=512, device="cuda")
scores = xenc.predict(flat, batch_size=512, show_progress_bar=False)

gold_s = np.full(len(queries), np.nan)
ret_s = {0: np.full(len(queries), np.nan), 1: np.full(len(queries), np.nan),
         2: np.full(len(queries), np.nan)}
for (qi, tag), sc in zip(tags, scores):
    if tag == "gold":
        gold_s[qi] = sc
    else:
        ret_s[tag][qi] = sc

r2 = np.array([p["_round2_only"] for p in queries])
for label, mask in (("ALL", np.ones(len(queries), bool)),
                    ("round-2-only", r2)):
    g = gold_s[mask]
    print(f"[{label}] gold pairs: n={len(g):,} mean={np.nanmean(g):.3f} "
          f">0.5={np.nanmean(g > .5):.3f}", flush=True)
    for rank in (0, 1, 2):
        rsc = ret_s[rank][mask]
        ok = ~np.isnan(rsc)
        print(f"[{label}] v6a-top{rank + 1} nongold: n={ok.sum():,} "
              f"mean={np.nanmean(rsc):.3f} >0.5={np.nanmean(rsc[ok] > .5):.3f}",
              flush=True)
    m0 = ret_s[0][mask]
    both = ~np.isnan(g) & ~np.isnan(m0)
    print(f"[{label}] gold>top1: {np.mean(g[both] > m0[both]):.3f}  "
          f"both>0.5: {np.mean((g[both] > .5) & (m0[both] > .5)):.3f}  "
          f"gold-only>0.5: {np.mean((g[both] > .5) & (m0[both] <= .5)):.3f}  "
          f"top1-only>0.5: {np.mean((g[both] <= .5) & (m0[both] > .5)):.3f}",
          flush=True)

# oracle-evidence recall per claim (all these claims fell to §102)
claim_min = {}
for qi, p in enumerate(queries):
    k = (str(p["anchor_app_id"]), str(p["anchor_target_claim"]))
    claim_min[k] = min(claim_min.get(k, 1.0), float(gold_s[qi]))
mins = np.array(list(claim_min.values()))
print(f"oracle-evidence recall: claims={len(mins):,} "
      f"min-gold-score quartiles={np.percentile(mins, [25, 50, 75]).round(3)} "
      f"min>0.5={np.mean(mins > .5):.3f}  min>0.3={np.mean(mins > .3):.3f}",
      flush=True)
print("GOLD-DIAG-DONE", flush=True)
