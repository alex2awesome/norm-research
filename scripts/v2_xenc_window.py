#!/usr/bin/env python3
"""V2 min_of_max with windowed evidence (v6a top-3, each para +/-1 neighbor).

Window-oracle test: gold-pair recall 26.7% -> 33.7% with p±1 windows (the
manual audit's PARTIAL cases — limitation in adjacent paragraph). Question:
does the same windowing move the DISCRIMINATIVE pipeline AUC (baseline
0.5735 with xenc judge on v6a top-3)?

Element sets identical to v2_pairs.jsonl; same doc/para filters as
phase_retrieve; element score = max over (top-3 paras ∪ their ±1 neighbors)
of xenc; claim V = min over elements; pooled/within AUC.
"""
import glob
import gzip
import json
import os

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics import roc_auc_score

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
TB_DIR = f"{PROC}/truecite_testbed_v1"
V6A = f"{BASE}/models/bge-m3-anticipation-v6a"
XENC = f"{BASE}/models/element-para-xenc-v1"
MAX_TEXT = 1500
TOPK = 3

norm = lambda x: "".join(c for c in str(x) if c.isdigit()).lstrip("0")

elements, fell = {}, {}
for line in open(f"{TB_DIR}/v2_pairs.jsonl"):
    r = json.loads(line)
    k = (r["app_id"], r["ifw"], r["claim_num"])
    elements.setdefault(k, {})[r["el_idx"]] = r["element"]
    fell[k] = r["fell_102"]
print(f"claims: {len(elements):,}", flush=True)

docs_of = {}
with gzip.open(f"{TB_DIR}/testbed.jsonl.gz", "rt") as f:
    for line in f:
        r = json.loads(line)
        ds = [d["doc_id"] for d in r["art"] if d["in_gp_corpus"]]
        if ds:
            docs_of[(r["app_id"], r["ifw_number"])] = ds
need_docs = {d for k in elements for d in docs_of.get(k[:2], [])}

doc_paras = {}


def eat_gp(path, op):
    with op(path) as f:
        for line in f:
            try:
                rr = json.loads(line)
            except Exception:
                continue
            dn = norm(rr.get("pgpub_id"))
            if dn in need_docs and rr.get("paragraphs") \
                    and dn not in doc_paras:
                doc_paras[dn] = [(k, v) for k, v in rr["paragraphs"].items()
                                 if v and len(v) > 40][:200]


eat_gp(f"{PROC}/paragraph_keyed_specs.jsonl.gz", lambda p: gzip.open(p, "rt"))
for fn in sorted(glob.glob(f"{PROC}/paragraph_keyed_specs_v2/*.jsonl")):
    eat_gp(fn, open)
print(f"{len(doc_paras):,} docs", flush=True)

enc = SentenceTransformer(V6A, device="cuda")
enc.max_seq_length = 512
all_keys, all_texts = [], []
doc_pos = {}  # (doc, idx within doc) ordering for neighbor lookup
for d, ps in doc_paras.items():
    for j, (anchor, t) in enumerate(ps):
        doc_pos[(d, anchor)] = j
        all_keys.append((d, anchor))
        all_texts.append(t[:MAX_TEXT])
P = enc.encode(all_texts, batch_size=256, normalize_embeddings=True,
               show_progress_bar=False).astype(np.float32)
rows_by_doc = {}
for i, (d, _) in enumerate(all_keys):
    rows_by_doc.setdefault(d, []).append(i)

el_meta, el_texts = [], []
for k, els in elements.items():
    for ei, el in sorted(els.items()):
        el_meta.append((k, ei))
        el_texts.append(el)
E = enc.encode(el_texts, batch_size=256, normalize_embeddings=True,
               show_progress_bar=False).astype(np.float32)
del enc
torch.cuda.empty_cache()
print(f"encoded {len(el_texts):,} elements", flush=True)

# candidate set per element: top-3 global rows ∪ their within-doc ±1 neighbors
cands = []
for x, (k, ei) in enumerate(el_meta):
    rows = [i for d in docs_of.get(k[:2], []) for i in rows_by_doc.get(d, [])]
    if not rows:
        cands.append([])
        continue
    sims = P[rows] @ E[x]
    top = [rows[i] for i in np.argsort(-sims)[:TOPK]]
    cset = set()
    for gi in top:
        d, anchor = all_keys[gi]
        j = doc_pos[(d, anchor)]
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(doc_paras[d]):
                cset.add(rows_by_doc[d][jj])
    cands.append(sorted(cset))

flat, owner = [], []
for x, cs in enumerate(cands):
    for gi in cs:
        flat.append([el_texts[x][:MAX_TEXT], all_texts[gi]])
        owner.append(x)
print(f"xenc pairs: {len(flat):,}", flush=True)
xenc = CrossEncoder(XENC, max_length=512, device="cuda")
scores = xenc.predict(flat, batch_size=512, show_progress_bar=True)

el_max = {}
for x, sc in zip(owner, scores):
    el_max[x] = max(el_max.get(x, 0.0), float(sc))

claim_el = {}
for x, (k, ei) in enumerate(el_meta):
    if x in el_max:
        claim_el.setdefault(k, {})[ei] = el_max[x]
ys, ss, br = [], [], {}
for k, v in claim_el.items():
    s = float(min(v.values()))
    ys.append(int(fell[k]))
    ss.append(s)
    br.setdefault(k[:2], []).append((int(fell[k]), s))
ys, ss = np.array(ys), np.array(ss)
wr = [roc_auc_score([y for y, _ in v], [x for _, x in v])
      for v in br.values() if len({y for y, _ in v}) == 2]
print(f"v6a-top3+window±1+xenc: min_of_max pooled="
      f"{roc_auc_score(ys, ss):.4f} within={np.mean(wr):.4f} (n={len(ys)})",
      flush=True)
print("XENC-WINDOW-V2-DONE", flush=True)
