#!/usr/bin/env python3
"""Build a FAISS IVF index over all patent claims using the v2 BGE-M3 model.

Bypasses retriv (which crashed in autofaiss float32 JSON serialization).
Direct sentence-transformers + faiss approach.

Output:
  /lfs/.../indexes/patent_claims_v2/
    index.faiss       (the IVF index)
    pgpub_ids.txt     (line-aligned pgpub_ids; index_id → pgpub_id)
"""
import argparse
import gzip
import json
import os
import sys
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
INDEX_DIR = "/lfs/skampere3/0/alexspan/norm-research/indexes/patent_claims_v2"
MODEL = "/lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2"


def iter_claims():
    """Stream JSONL → (pgpub_id, claim_text)."""
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            pid = str(d.get("pgpub_id", "")).strip()
            claims = (d.get("pg_claims") or "").strip()
            if not pid or not claims:
                continue
            yield pid, claims[:2000]  # cap text for embed speed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default=INDEX_DIR)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=256)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_len
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    dim = model.get_sentence_embedding_dimension()
    print(f"  embedding dim: {dim}")

    # Collect texts in chunks
    pgpub_ids = []
    all_embeddings = []
    t0 = time.time()
    buf_text = []
    buf_pid = []
    n_total = 0

    def flush_batch():
        nonlocal buf_text, buf_pid
        if not buf_text:
            return
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
        if len(buf_text) >= args.batch * 8:  # 2048-doc chunk
            flush_batch()
        if n_total % 50_000 == 0:
            elapsed = time.time() - t0
            rate = n_total / elapsed
            print(f"  embedded {n_total:,}  ({rate:.0f}/s, elapsed {elapsed/60:.1f} min)",
                  file=sys.stderr, flush=True)
    flush_batch()

    print(f"\nTotal embedded: {n_total:,}")
    embs = np.vstack(all_embeddings)
    print(f"  shape: {embs.shape}, dtype: {embs.dtype}")

    # Build IVF index. nlist = 4096 is reasonable for ~5M docs.
    nlist = 4096
    print(f"\nBuilding IVF index (nlist={nlist}) ...")
    quantizer = faiss.IndexFlatIP(embs.shape[1])
    index = faiss.IndexIVFFlat(quantizer, embs.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    print("  training on sample ...")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(embs.shape[0], size=min(500_000, embs.shape[0]), replace=False)
    index.train(embs[sample_idx])
    print("  adding ...")
    index.add(embs)
    index.nprobe = 16
    print(f"  index size: {index.ntotal:,}")

    # Save
    faiss.write_index(index, os.path.join(args.out, "index.faiss"))
    with open(os.path.join(args.out, "pgpub_ids.txt"), "w") as f:
        for pid in pgpub_ids:
            f.write(pid + "\n")
    print(f"\nSaved index + pgpub_ids to {args.out}")


if __name__ == "__main__":
    main()
