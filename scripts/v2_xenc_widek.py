#!/usr/bin/env python3
"""V2 min_of_max with xenc over a WIDER v6a evidence pool (K=3/10/30).

The judge-swap test showed xenc-v1 == Qwen-122B at pipeline level on the
same v6a top-3 pairs (0.5735 vs 0.5743). The honest within-doc eval showed
xenc reranking a wider v6a pool recovers most of its full-ranking gain
(MRR 0.207 -> 0.257 at top-20). Question: does widening the evidence pool
move the PIPELINE AUC past 0.574, or does max-pooling noise eat the gain?

Protocol: element sets identical to v2_pairs.jsonl (same decomposition);
candidate paras = v6a top-K over ALL paragraphs of the record's attached
art (same doc/para filters as v2_full_pipeline.phase_retrieve); element
score = max over K of xenc; claim V = min over elements; pooled/within AUC.
K=3 should reproduce ~0.5735 modulo retrieval tie-breaks.
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
KS = [3, 10, 30]
MAX_TEXT = 1500

norm = lambda x: "".join(c for c in str(x) if c.isdigit()).lstrip("0")

# ---- elements + labels, exactly the v2_pairs population ----
elements = {}  # (app, ifw, claim) -> {el_idx: element}
fell = {}
for line in open(f"{TB_DIR}/v2_pairs.jsonl"):
    r = json.loads(line)
    k = (r["app_id"], r["ifw"], r["claim_num"])
    elements.setdefault(k, {})[r["el_idx"]] = r["element"]
    fell[k] = r["fell_102"]
print(f"claims: {len(elements):,}  elements: "
      f"{sum(len(v) for v in elements.values()):,}", flush=True)

# ---- records -> attached docs (same filter as load_records) ----
docs_of = {}
with gzip.open(f"{TB_DIR}/testbed.jsonl.gz", "rt") as f:
    for line in f:
        r = json.loads(line)
        ds = [d["doc_id"] for d in r["art"] if d["in_gp_corpus"]]
        if ds:
            docs_of[(r["app_id"], r["ifw_number"])] = ds
need_docs = {d for k in elements for d in docs_of.get(k[:2], [])}
print(f"records: {len(docs_of):,}  need docs: {len(need_docs):,}", flush=True)

# ---- cited docs' paragraphs (same filters as phase_retrieve) ----
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
print(f"{len(doc_paras):,} docs with paragraphs", flush=True)

# ---- v6a: encode paras once, elements once; top-K per element ----
enc = SentenceTransformer(V6A, device="cuda")
enc.max_seq_length = 512
all_keys, all_texts = [], []
for d, ps in doc_paras.items():
    for anchor, t in ps:
        all_keys.append((d, anchor))
        all_texts.append(t[:MAX_TEXT])
P = enc.encode(all_texts, batch_size=256, normalize_embeddings=True,
               show_progress_bar=False).astype(np.float32)
rows_by_doc = {}
for i, (d, _) in enumerate(all_keys):
    rows_by_doc.setdefault(d, []).append(i)
print(f"encoded {len(all_texts):,} paragraphs", flush=True)

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

KMAX = max(KS)
cands = []  # per el_meta entry: list of (global_para_idx) v6a-descending, KMAX
for x, (k, ei) in enumerate(el_meta):
    rows = [i for d in docs_of.get(k[:2], []) for i in rows_by_doc.get(d, [])]
    if not rows:
        cands.append([])
        continue
    sims = P[rows] @ E[x]
    order = np.argsort(-sims)[:KMAX]
    cands.append([rows[i] for i in order])

# ---- xenc score the union of (element, candidate) pairs ----
flat, owner = [], []
for x, cs in enumerate(cands):
    for rank, gi in enumerate(cs):
        flat.append([el_texts[x][:MAX_TEXT], all_texts[gi]])
        owner.append((x, rank))
print(f"xenc pairs: {len(flat):,}", flush=True)
xenc = CrossEncoder(XENC, max_length=512, device="cuda")
scores = xenc.predict(flat, batch_size=512, show_progress_bar=True)
print("xenc scoring done", flush=True)

el_scores = {}  # x -> list of (rank, score)
for (x, rank), sc in zip(owner, scores):
    el_scores.setdefault(x, []).append((rank, float(sc)))

for K in KS:
    claim_el = {}  # claim key -> {el_idx: max score within top-K}
    for x, (k, ei) in enumerate(el_meta):
        ss = [s for rank, s in el_scores.get(x, []) if rank < K]
        if ss:
            claim_el.setdefault(k, {})[ei] = max(ss)
    ys, ss, br = [], [], {}
    for k, v in claim_el.items():
        s = float(min(v.values()))
        ys.append(int(fell[k]))
        ss.append(s)
        br.setdefault(k[:2], []).append((int(fell[k]), s))
    ys, ss = np.array(ys), np.array(ss)
    wr = [roc_auc_score([y for y, _ in v], [x for _, x in v])
          for v in br.values() if len({y for y, _ in v}) == 2]
    print(f"v6a-top{K:>2}+xenc: min_of_max pooled={roc_auc_score(ys, ss):.4f} "
          f"within={np.mean(wr):.4f} (n={len(ys)})", flush=True)
print("XENC-WIDEK-DONE", flush=True)
