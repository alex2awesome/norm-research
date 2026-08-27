#!/usr/bin/env python3
"""Embed spec chunks with v2 BGE-M3 model and save embeddings as numpy + jsonl meta.

Reads:
  processed/spec_chunks/*.parquet  (output of paragraph_chunk_specs.py)

Outputs:
  processed/spec_embeddings/embeddings.fp16.npy     (N x 1024 float16)
  processed/spec_embeddings/meta.jsonl              (line-aligned with embeddings)
    each line: {"source": "g"|"pg", "doc_id": str, "chunk_idx": int}

Streaming: handles arbitrary corpus size, embeds + flushes incrementally.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
CHUNK_DIR = f"{BASE}/processed/spec_chunks"
OUT_DIR = f"{BASE}/processed/spec_embeddings"
MODEL = "/lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--out", default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading model {args.model} ...", file=sys.stderr)
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_len
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    dim = model.get_sentence_embedding_dimension()

    chunk_files = sorted(glob.glob(f"{CHUNK_DIR}/*.parquet"))
    print(f"Found {len(chunk_files)} chunk shards", file=sys.stderr)

    meta_path = f"{args.out}/meta.jsonl"
    embs_path = f"{args.out}/embeddings.fp16.npy"
    # Stream embedding and append to memmap; first count total rows for shape.
    print("Counting chunks ...", file=sys.stderr)
    n_total = 0
    for f in chunk_files:
        n_total += pq.read_metadata(f).num_rows
    print(f"  total chunks: {n_total:,}", file=sys.stderr)

    # Allocate fp16 memmap
    embs = np.memmap(embs_path, dtype="float16", mode="w+", shape=(n_total, dim))
    meta_f = open(meta_path, "w")

    t0 = time.time()
    cursor = 0
    for f in chunk_files:
        tbl = pq.read_table(f)
        rows = tbl.to_pylist()
        # Process in batches
        for i in range(0, len(rows), args.batch * 4):
            chunk = rows[i:i + args.batch * 4]
            texts = [r["text"] for r in chunk]
            with torch.no_grad():
                e = model.encode(texts, batch_size=args.batch,
                                 show_progress_bar=False, convert_to_numpy=True,
                                 normalize_embeddings=True)
            e = e.astype("float16")
            embs[cursor:cursor + len(chunk)] = e
            for r in chunk:
                meta_f.write(json.dumps({
                    "source": r["source"], "doc_id": r["doc_id"],
                    "chunk_idx": int(r["chunk_idx"]),
                }) + "\n")
            cursor += len(chunk)
            if cursor % 50_000 < args.batch * 4:
                rate = cursor / max(1, time.time() - t0)
                eta_h = (n_total - cursor) / rate / 3600
                print(f"  {cursor:,}/{n_total:,}  ({rate:.0f}/s, ETA {eta_h:.1f}h)",
                      file=sys.stderr, flush=True)
    meta_f.close()
    embs.flush()
    print(f"\nDone. Wrote {cursor:,} embeddings to {embs_path}")


if __name__ == "__main__":
    main()
