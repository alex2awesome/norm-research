#!/usr/bin/env python3
"""Bake top-K retrievals into the patents file as new columns.

For each row in patents_final_outcome_cpc_balanced_with_rejections.csv.gz:
  1. Encode its claim text with v2 BGE-M3.
  2. FAISS top-K against the claim DB (excluding its own pgpub_id).
  3. Look up top-K (pgpub_id, claim_text, score) tuples.
  4. Append as new columns: top1_pgpub_id, top1_score, top1_claim_text, ..., topK.

Output: patents_final_outcome_cpc_balanced_with_rejections_with_retrievals.csv.gz
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
INPUT = f"{BASE}/patents_final_outcome_cpc_balanced_with_rejections.csv.gz"
OUTPUT = f"{BASE}/patents_final_outcome_cpc_balanced_with_rejections_with_retrievals.csv.gz"
INDEX_DIR = "/lfs/skampere3/0/alexspan/norm-research/indexes/patent_claims_v2"
MODEL = "/lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"

TOP_K = 5
CLAIM_TRUNC = 1500
NPROBE = 32  # higher → better recall, slower


def load_claim_text_lookup(pgpub_ids_needed):
    """Build pgpub_id → first 1500 chars of pg_claims, from JSONL."""
    out = {}
    needed = set(pgpub_ids_needed)
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            pid = str(d.get("pgpub_id", "")).strip()
            if pid in needed and pid not in out:
                t = (d.get("pg_claims") or "").strip()
                if t:
                    out[pid] = t[:CLAIM_TRUNC]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=INPUT)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--index_dir", default=INDEX_DIR)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    print(f"Loading FAISS index from {args.index_dir} ...")
    index = faiss.read_index(os.path.join(args.index_dir, "index.faiss"))
    index.nprobe = NPROBE
    pgpub_ids = open(os.path.join(args.index_dir, "pgpub_ids.txt")).read().splitlines()
    pid_to_idx = {p: i for i, p in enumerate(pgpub_ids)}
    print(f"  index size: {index.ntotal:,}")

    print(f"\nLoading v2 model {args.model} ...")
    model = SentenceTransformer(args.model)
    model.max_seq_length = 256
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    # Stream input, batch encode + search.
    print(f"\nReading input {args.input} ...")
    rows = []
    with gzip.open(args.input, "rt") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"  {len(rows):,} rows")

    # Embed queries in batches and do FAISS search.
    print("Embedding queries + searching ...")
    top_k_results = []  # list of (D, I) per row
    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        texts = []
        for r in chunk:
            t = r["text"]
            # Strip ABSTRACT prefix to get to claims; truncate
            claim_start = t.find("CLAIMS:\n")
            t = t[claim_start + 8:] if claim_start >= 0 else t
            texts.append(t[:2000])
        with torch.no_grad():
            q_emb = model.encode(texts, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        q_emb = q_emb.astype("float32")
        # Search top-K+1 (to exclude self)
        D, I = index.search(q_emb, TOP_K + 1)
        # Filter out self-hits
        for j, r in enumerate(chunk):
            self_pid = r["pgpub_id"]
            self_idx = pid_to_idx.get(self_pid, -1)
            keep_d, keep_i = [], []
            for k in range(TOP_K + 1):
                if I[j, k] == self_idx:
                    continue
                keep_d.append(D[j, k])
                keep_i.append(I[j, k])
                if len(keep_d) == TOP_K:
                    break
            top_k_results.append((keep_d, keep_i))
        if (i // args.batch + 1) % 50 == 0:
            done = i + len(chunk)
            rate = done / (time.time() - t0)
            eta_min = (len(rows) - done) / rate / 60
            print(f"  {done:,}/{len(rows):,}  ({rate:.0f}/s, ETA {eta_min:.1f} min)",
                  file=sys.stderr, flush=True)

    # Build the set of pgpub_ids we need to look up text for
    needed_pgpubs = set()
    for D, I in top_k_results:
        for idx in I:
            needed_pgpubs.add(pgpub_ids[idx])
    print(f"\nNeed text for {len(needed_pgpubs):,} retrieved pgpub_ids")

    print("Looking up retrieved claim text from JSONL ...")
    claim_text = load_claim_text_lookup(needed_pgpubs)
    print(f"  resolved {len(claim_text):,} pgpub_ids")

    # Write output
    print(f"\nWriting {args.output} ...")
    base_fields = list(rows[0].keys())
    new_fields = []
    for k in range(1, TOP_K + 1):
        new_fields.extend([f"top{k}_pgpub_id", f"top{k}_score", f"top{k}_claim_text"])
    out_fields = base_fields + new_fields

    with gzip.open(args.output, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r, (D, I) in zip(rows, top_k_results):
            row_out = dict(r)
            for k in range(TOP_K):
                if k < len(D):
                    pid = pgpub_ids[I[k]]
                    row_out[f"top{k+1}_pgpub_id"] = pid
                    row_out[f"top{k+1}_score"] = float(D[k])
                    row_out[f"top{k+1}_claim_text"] = claim_text.get(pid, "")
                else:
                    row_out[f"top{k+1}_pgpub_id"] = ""
                    row_out[f"top{k+1}_score"] = 0.0
                    row_out[f"top{k+1}_claim_text"] = ""
            w.writerow(row_out)
    print(f"\nDone. Wrote {len(rows):,} rows.")


if __name__ == "__main__":
    main()
