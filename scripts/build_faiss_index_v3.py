#!/usr/bin/env python3
"""v3 FAISS index: extends v2's claim corpus with Google Patents supplement.

Reads:
  - patents_dataset.jsonl.gz (4.7M pre-grant publications) [same as v2]
  - google_patents_supplement.parquet (~1.7M older US + design patents) [NEW]

Builds one combined IVF index using the v2 BGE-M3 model.
"""
import argparse
import gzip
import json
import os
import sys
import time

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
GP_PARQUET = f"{BASE}/processed/google_patents_supplement.parquet"
GRANTED_PARQUET = f"{BASE}/processed/granted_patents_claim1.parquet"
PGPUB_PARQUET = f"{BASE}/processed/pgpub_claims1.parquet"
LEGACY_PARQUET = f"{BASE}/processed/claim1_lookup.parquet"
INDEX_DIR = "/lfs/skampere3/0/alexspan/norm-research/indexes/patent_claims_v3"
MODEL = "/lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2"


def iter_claims():
    """Stream from ALL text sources, dedupe by ID (first wins)."""
    seen = set()

    def emit(pid, text):
        pid = str(pid).strip()
        text = (text or "").strip()
        if not pid or not text or pid in seen:
            return None
        seen.add(pid)
        return pid, text[:2000]

    # Source 1: pre-grant publications JSONL
    print("  reading patents_dataset.jsonl.gz ...", file=sys.stderr)
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            r = emit(d.get("pgpub_id"), d.get("pg_claims"))
            if r: yield r
            # Also emit by patent_id if granted (different ID space)
            ptid = d.get("patent_id")
            if ptid:
                r2 = emit(ptid, d.get("g_claims") or d.get("pg_claims"))
                if r2: yield r2
    print(f"  after JSONL: {len(seen):,}", file=sys.stderr)

    # Source 2: granted-patent parquet (older + design + plant + reissue)
    if os.path.exists(GRANTED_PARQUET):
        print("  reading granted_patents_claim1.parquet ...", file=sys.stderr)
        t = pq.read_table(GRANTED_PARQUET)
        for row in t.to_pylist():
            r = emit(row.get("patent_id"), row.get("claim_text"))
            if r: yield r
        print(f"  after granted: {len(seen):,}", file=sys.stderr)

    # Source 3: pgpub parquet
    if os.path.exists(PGPUB_PARQUET):
        print("  reading pgpub_claims1.parquet ...", file=sys.stderr)
        t = pq.read_table(PGPUB_PARQUET)
        for row in t.to_pylist():
            r = emit(row.get("pgpub_id"), row.get("claim_text"))
            if r: yield r
        print(f"  after pgpub: {len(seen):,}", file=sys.stderr)

    # Source 4: legacy claim1_lookup
    if os.path.exists(LEGACY_PARQUET):
        print("  reading claim1_lookup.parquet ...", file=sys.stderr)
        t = pq.read_table(LEGACY_PARQUET)
        for row in t.to_pylist():
            r = emit(row.get("patent_id"), row.get("claim_1"))
            if r: yield r
        print(f"  after legacy: {len(seen):,}", file=sys.stderr)

    # Source 5: Google Patents supplement (older US + design)
    if os.path.exists(GP_PARQUET):
        print("  reading google_patents_supplement.parquet ...", file=sys.stderr)
        t = pq.read_table(GP_PARQUET)
        for row in t.to_pylist():
            r = emit(row.get("raw_id"), row.get("claim_text"))
            if r: yield r
        print(f"  after GP: {len(seen):,}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=INDEX_DIR)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=256)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading model {MODEL} ...")
    model = SentenceTransformer(MODEL)
    model.max_seq_length = args.max_seq_len
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    pgpub_ids = []
    all_embeddings = []
    buf_text, buf_pid = [], []
    n_total = 0
    t0 = time.time()

    def flush_batch():
        nonlocal buf_text, buf_pid
        if not buf_text: return
        with torch.no_grad():
            e = model.encode(buf_text, batch_size=args.batch,
                             show_progress_bar=False, convert_to_numpy=True,
                             normalize_embeddings=True)
        all_embeddings.append(e.astype("float32"))
        pgpub_ids.extend(buf_pid)
        buf_text, buf_pid = [], []

    for pid, text in iter_claims():
        buf_text.append(text)
        buf_pid.append(pid)
        n_total += 1
        if len(buf_text) >= args.batch * 8:
            flush_batch()
        if n_total % 50_000 == 0:
            print(f"  embedded {n_total:,}  ({n_total / (time.time() - t0):.0f}/s)", file=sys.stderr, flush=True)
    flush_batch()

    print(f"\nTotal embedded: {n_total:,}")
    embs = np.vstack(all_embeddings)
    print(f"  shape: {embs.shape}")

    nlist = 4096
    print(f"\nBuilding IVF index (nlist={nlist}) ...")
    quantizer = faiss.IndexFlatIP(embs.shape[1])
    index = faiss.IndexIVFFlat(quantizer, embs.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(42)
    sample = rng.choice(embs.shape[0], size=min(500_000, embs.shape[0]), replace=False)
    index.train(embs[sample])
    index.add(embs)
    index.nprobe = 32

    faiss.write_index(index, os.path.join(args.out, "index.faiss"))
    with open(os.path.join(args.out, "pgpub_ids.txt"), "w") as f:
        for pid in pgpub_ids:
            f.write(pid + "\n")
    print(f"\nSaved v3 index ({index.ntotal:,} docs) to {args.out}")


if __name__ == "__main__":
    main()
