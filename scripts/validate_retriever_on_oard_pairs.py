#!/usr/bin/env python3
"""Validate the v2/v3 retriever by measuring whether it surfaces the actual
prior art that examiners cited in §102 rejections.

Held-out (rejected_app, cited_ref) pairs are drawn from OARD where
action_type='102' implicitly — we already have these as our training pairs.
We sample 5K pairs, then for each query the FAISS index for top-50 hits.

Metrics:
  - MRR (mean reciprocal rank of the cited ref)
  - Recall@1, @5, @10, @50
  - Per-CPC-section recall (does it work cross-section?)
  - Top-50 same-CPC-section fraction (topic shortcut detection)

Also reports raw similarity score (cosine via inner-product of normalized
embeddings) between the query's claim and the cited ref's claim-1.
A low score for many cited refs would suggest the anticipation isn't in
the claim — it's buried in the spec — and the corpus needs expansion.
"""
import argparse
import csv
import gzip
import json
import os
import random
import sys
import time
from collections import defaultdict

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
TRAINING_PAIRS = f"{BASE}/processed/anticipation_training_pairs_v2.jsonl.gz"
INDEX_DIR_DEFAULT = "/lfs/skampere3/0/alexspan/norm-research/indexes/patent_claims_v2"
MODEL = "/lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2"
CPC_LOOKUP = f"{BASE}/processed/granted_patents_claim1.parquet"  # has CPC info? probably not
# We'll get CPC from the balanced file's pgpub_id → cpc mapping instead
BALANCED = f"{BASE}/patents_final_outcome_cpc_balanced_with_rejections.csv.gz"


def load_pgpub_to_cpc():
    """Build pgpub_id → cpc_section map from the balanced file."""
    print("Loading pgpub_id → cpc_section from balanced file ...", file=sys.stderr)
    out = {}
    with gzip.open(BALANCED, "rt") as f:
        for r in csv.DictReader(f):
            pid = r.get("pgpub_id", "").strip()
            sec = r.get("cpc_section", "").strip()
            if pid and sec:
                out[pid] = sec
    print(f"  {len(out):,} pgpub→cpc", file=sys.stderr)
    return out


def sample_test_pairs(n_target, rng_seed=42):
    """Sample held-out (anchor, positive) pairs from the training set.

    Since the model trained on the same pairs, this is OPTIMISTIC eval —
    the model has seen these pairs during fine-tune. Numbers will be inflated.
    For honest eval we'd need to hold out by app_id before training, which we
    haven't done. Use these numbers as an upper bound.
    """
    print(f"Sampling {n_target:,} pairs from training set (optimistic eval) ...", file=sys.stderr)
    rng = random.Random(rng_seed)
    pairs = []
    with gzip.open(TRAINING_PAIRS, "rt") as f:
        for line in f:
            r = json.loads(line)
            # Sample with reservoir
            if len(pairs) < n_target:
                pairs.append(r)
            else:
                idx = rng.randrange(len(pairs))
                if idx < n_target:
                    pairs[idx] = r
    print(f"  {len(pairs):,} pairs sampled", file=sys.stderr)
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", default=INDEX_DIR_DEFAULT)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n_pairs", type=int, default=5000)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    print(f"Loading FAISS index from {args.index_dir} ...")
    index = faiss.read_index(os.path.join(args.index_dir, "index.faiss"))
    index.nprobe = 32
    pgpub_ids = open(os.path.join(args.index_dir, "pgpub_ids.txt")).read().splitlines()
    pid_to_idx = {p: i for i, p in enumerate(pgpub_ids)}
    print(f"  index size: {index.ntotal:,}")

    print(f"\nLoading model {args.model} ...")
    model = SentenceTransformer(args.model)
    model.max_seq_length = 256
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    pgpub_to_cpc = load_pgpub_to_cpc()

    pairs = sample_test_pairs(args.n_pairs)

    # Drop pairs where the positive isn't in the index
    pairs = [
        p for p in pairs
        if p["positive_pgpub_id"] in pid_to_idx
        and p["anchor_pgpub_id"] in pid_to_idx
    ]
    print(f"  {len(pairs):,} pairs survive index-membership filter")

    print("\nEmbedding anchors + searching ...")
    ranks = []  # rank of the positive cite (1-indexed); -1 if not in top-K
    same_section_top_k = []  # fraction of top-K from same CPC section
    cross_section_hits = 0
    same_section_hits = 0
    unfindable_count = 0
    positive_scores = []  # raw similarity scores between anchor and positive

    t0 = time.time()
    for i in range(0, len(pairs), args.batch):
        chunk = pairs[i:i + args.batch]
        texts = [p["anchor_text"][:2000] for p in chunk]
        with torch.no_grad():
            q_emb = model.encode(texts, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        q_emb = q_emb.astype("float32")
        D, I = index.search(q_emb, args.top_k + 1)

        for j, pair in enumerate(chunk):
            anchor_pid = pair["anchor_pgpub_id"]
            positive_pid = pair["positive_pgpub_id"]
            positive_idx = pid_to_idx.get(positive_pid, -1)
            anchor_idx = pid_to_idx.get(anchor_pid, -1)

            # Rank of positive in top-K (excluding self-hit)
            rank = -1
            seen_self = False
            for rk, doc_idx in enumerate(I[j]):
                if doc_idx == anchor_idx:
                    seen_self = True
                    continue
                if doc_idx == positive_idx:
                    rank = rk + 1 - (1 if seen_self else 0)
                    break
            ranks.append(rank)

            # Raw similarity score: explicit lookup
            if positive_idx >= 0:
                # Find positive's position in the search results
                for rk, doc_idx in enumerate(I[j]):
                    if doc_idx == positive_idx:
                        positive_scores.append(float(D[j, rk]))
                        break
                else:
                    # Not in top-K — score might be very low. Skip.
                    pass

            # Topic confound: same-CPC-section fraction
            anchor_sec = pgpub_to_cpc.get(anchor_pid, "?")
            top_secs = [pgpub_to_cpc.get(pgpub_ids[I[j, k]], "?") for k in range(args.top_k)]
            same_sec = sum(1 for s in top_secs if s == anchor_sec)
            same_section_top_k.append(same_sec / args.top_k)

            # Was the positive in same section as anchor?
            pos_sec = pgpub_to_cpc.get(positive_pid, "?")
            if pos_sec != "?" and anchor_sec != "?":
                if pos_sec == anchor_sec:
                    same_section_hits += 1
                else:
                    cross_section_hits += 1

        if (i // args.batch + 1) % 20 == 0:
            print(f"  {i + len(chunk):,}/{len(pairs):,}  rate={i / max(1, (time.time() - t0)):.0f}/s",
                  file=sys.stderr, flush=True)

    ranks_arr = np.array(ranks)
    found = ranks_arr > 0

    print(f"\n=== Results ===")
    print(f"  n_pairs evaluated:        {len(ranks_arr):,}")
    print(f"  Recall@1:                 {(ranks_arr == 1).mean() * 100:.2f}%")
    print(f"  Recall@5:                 {((ranks_arr <= 5) & found).mean() * 100:.2f}%")
    print(f"  Recall@10:                {((ranks_arr <= 10) & found).mean() * 100:.2f}%")
    print(f"  Recall@50:                {((ranks_arr <= 50) & found).mean() * 100:.2f}%")
    mrr_input = np.where(found, 1.0 / ranks_arr, 0.0)
    print(f"  MRR:                      {mrr_input.mean():.4f}")
    print()
    print(f"  Avg same-CPC-section fraction in top-K: {np.mean(same_section_top_k) * 100:.1f}%")
    print(f"  (50%+ would suggest strong topic-section shortcut)")
    print()
    print(f"  Same-section anchor/positive pairs:  {same_section_hits:,}")
    print(f"  Cross-section anchor/positive pairs: {cross_section_hits:,}")
    if same_section_hits + cross_section_hits > 0:
        print(f"  Per-class recall@10:")
        for label, mask in [
            ("same-section", [True if pgpub_to_cpc.get(p["positive_pgpub_id"], "?")
                              == pgpub_to_cpc.get(p["anchor_pgpub_id"], "?")
                              and pgpub_to_cpc.get(p["positive_pgpub_id"], "?") != "?"
                              else False for p in pairs[:len(ranks_arr)]]),
            ("cross-section", [True if pgpub_to_cpc.get(p["positive_pgpub_id"], "?")
                               != pgpub_to_cpc.get(p["anchor_pgpub_id"], "?")
                               and pgpub_to_cpc.get(p["positive_pgpub_id"], "?") != "?"
                               and pgpub_to_cpc.get(p["anchor_pgpub_id"], "?") != "?"
                               else False for p in pairs[:len(ranks_arr)]]),
        ]:
            m = np.array(mask)
            if m.sum() > 0:
                r10 = (((ranks_arr <= 10) & found) & m).sum() / m.sum() * 100
                print(f"    {label} ({m.sum()} pairs): {r10:.2f}%")
    print()
    if positive_scores:
        ps = np.array(positive_scores)
        print(f"  Raw similarity scores anchor↔positive (only for in-top-K hits):")
        print(f"    n={len(ps)}, mean={ps.mean():.3f}, median={np.median(ps):.3f},")
        print(f"    p10={np.percentile(ps, 10):.3f}, p90={np.percentile(ps, 90):.3f}")
        low = (ps < 0.5).sum()
        print(f"    Below 0.5 similarity (suspicious): {low} ({low / len(ps) * 100:.1f}%)")
        print(f"    Low-similarity cited refs may have anticipation in spec, not claim.")


if __name__ == "__main__":
    main()
