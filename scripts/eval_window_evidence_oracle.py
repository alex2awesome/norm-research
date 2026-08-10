#!/usr/bin/env python3
"""Does windowed evidence (gold paragraph +/- neighbors) fix oracle recall?

Gold-evidence diag: oracle recall 26.7% (min over elements of xenc score on
the examiner's own cited paragraph). Manual audit of 20 pairs: 9 DISCLOSES /
9 PARTIAL / 2 WRONG — the PARTIAL cases' missing limitation sits in ADJACENT
paragraphs. Two windowing repairs tested on the same 2,819 gold pairs:

  A. window-max:    score(el, p-1), score(el, p), score(el, p+1) -> max
  B. window-concat: score(el, concat(p-1[:500], p[:1000], p+1[:500]))

Report pair-level disclosed-rate and per-claim oracle recall for baseline /
A / B. If recall jumps, apply windowing to V2 proper.
"""
import ast
import glob
import gzip
import json
import os

import numpy as np
from sentence_transformers import CrossEncoder

BASE = os.path.expanduser("~/norm-research/datasets/patents/processed")
TEST = f"{BASE}/training_pairs_v6_test.jsonl.gz"
SPECS = f"{BASE}/paragraph_keyed_specs.jsonl.gz"
XENC = os.path.expanduser("~/norm-research/models/element-para-xenc-v1")

pairs = []
with gzip.open(TEST, "rt") as f:
    for line in f:
        r = json.loads(line)
        if r.get("positive_kind") == "paragraph":
            pairs.append(r)
print(f"gold pairs: {len(pairs):,}", flush=True)

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

queries = [p for p in pairs if str(p["positive_pgpub_id"]) in specs
           and p["positive_key"] in specs[str(p["positive_pgpub_id"])]]
print(f"usable queries: {len(queries):,}", flush=True)

flat, tags = [], []  # tag = (qi, variant)
for qi, p in enumerate(queries):
    doc = specs[str(p["positive_pgpub_id"])]
    keys = list(doc.keys())
    i = keys.index(p["positive_key"])
    el = p["anchor_text"][:1500]
    neigh = [doc[keys[j]] for j in (i - 1, i, i + 1) if 0 <= j < len(keys)]
    gold = doc[p["positive_key"]]
    # baseline + window-max members
    for v, txt in enumerate(neigh):
        flat.append([el, txt[:1500]])
        tags.append((qi, f"n{v}"))
    flat.append([el, gold[:1500]])
    tags.append((qi, "base"))
    # window-concat
    parts = [t[:500] if t != gold else t[:1000] for t in neigh]
    flat.append([el, " ".join(parts)[:2000]])
    tags.append((qi, "concat"))
print(f"xenc pairs: {len(flat):,}", flush=True)

xenc = CrossEncoder(XENC, max_length=512, device="cuda")
scores = xenc.predict(flat, batch_size=512, show_progress_bar=True)

base = np.zeros(len(queries))
wmax = np.zeros(len(queries))
conc = np.zeros(len(queries))
for (qi, tag), sc in zip(tags, scores):
    if tag == "base":
        base[qi] = sc
    elif tag == "concat":
        conc[qi] = sc
    if tag.startswith("n"):
        wmax[qi] = max(wmax[qi], sc)
wmax = np.maximum(wmax, base)

for name, s in (("baseline (gold para)", base),
                ("window-max (p±1)", wmax),
                ("window-concat (p±1)", conc)):
    claim_min = {}
    for qi, p in enumerate(queries):
        k = (str(p["anchor_app_id"]), str(p["anchor_target_claim"]))
        claim_min[k] = min(claim_min.get(k, 1.0), float(s[qi]))
    mins = np.array(list(claim_min.values()))
    print(f"{name:>22}: pair>0.5={np.mean(s > .5):.3f}  "
          f"oracle recall(min>0.5)={np.mean(mins > .5):.3f}  "
          f"(min>0.3)={np.mean(mins > .3):.3f}  median={np.median(mins):.3f}",
          flush=True)
print("WINDOW-ORACLE-DONE", flush=True)
