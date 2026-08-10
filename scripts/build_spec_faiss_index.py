#!/usr/bin/env python3
"""Build FAISS IVF index over spec chunk embeddings.

Reads:
  processed/spec_embeddings/embeddings.fp16.npy   (N x 1024 fp16)
  processed/spec_embeddings/meta.jsonl            (line-aligned with embs)

Outputs:
  indexes/spec_chunks_v1/index.faiss
  indexes/spec_chunks_v1/meta_lookup.npy   (offsets into meta.jsonl)

Uses IVF_FLAT with nlist scaled by sqrt(N). For ~200M chunks, nlist≈14k.
"""
import argparse
import os
import sys
import time

import faiss
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
EMB_DIR = f"{BASE}/processed/spec_embeddings"
INDEX_DIR = "/lfs/skampere3/0/alexspan/norm-research/indexes/spec_chunks_v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dir", default=EMB_DIR)
    p.add_argument("--out", default=INDEX_DIR)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Figure out shape from meta line count
    meta_path = f"{args.emb_dir}/meta.jsonl"
    emb_path = f"{args.emb_dir}/embeddings.fp16.npy"
    n_lines = sum(1 for _ in open(meta_path))
    print(f"Total chunks: {n_lines:,}", file=sys.stderr)

    # Determine dim from a few-byte read
    raw = np.memmap(emb_path, dtype="float16", mode="r")
    dim = raw.shape[0] // n_lines
    print(f"Embedding dim: {dim}", file=sys.stderr)

    embs = np.memmap(emb_path, dtype="float16", mode="r", shape=(n_lines, dim))

    # Choose nlist: sqrt(N) is a common heuristic
    nlist = int(np.sqrt(n_lines))
    nlist = max(1024, min(65536, nlist))
    print(f"Building IVF_FLAT (nlist={nlist})", file=sys.stderr)

    # Convert sample to fp32 for training
    rng = np.random.default_rng(42)
    sample_size = min(1_000_000, n_lines)
    sample_idx = rng.choice(n_lines, size=sample_size, replace=False)
    sample = embs[sample_idx].astype("float32")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    print(f"  training on {sample_size:,} samples ...", file=sys.stderr)
    index.train(sample)
    print(f"    done in {time.time() - t0:.0f}s", file=sys.stderr)

    # Add in batches (memory-friendly)
    print("  adding all embeddings ...", file=sys.stderr)
    batch = 1_000_000
    for i in range(0, n_lines, batch):
        chunk = embs[i:i + batch].astype("float32")
        index.add(chunk)
        if (i // batch) % 5 == 0:
            print(f"    {min(i + batch, n_lines):,}/{n_lines:,}", file=sys.stderr, flush=True)
    index.nprobe = 32

    out_index = f"{args.out}/index.faiss"
    faiss.write_index(index, out_index)
    sz = os.path.getsize(out_index) / 1e9
    print(f"\nDone. Saved {out_index} ({sz:.1f} GB)")


if __name__ == "__main__":
    main()
