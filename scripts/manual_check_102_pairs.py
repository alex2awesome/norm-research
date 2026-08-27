#!/usr/bin/env python3
"""Print 10 §102 pairs side-by-side for manual inspection.

For each: shows anchor (rejected app) claim text and cited ref claim text,
plus cosine similarity. Lets us judge: would a human examiner say claim-vs-claim
shows the anticipation? If not, the anticipation must be in the spec.
"""
import gzip
import json
import random
import sys

import torch
from sentence_transformers import SentenceTransformer

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
PAIRS = f"{BASE}/processed/anticipation_training_pairs_v2.jsonl.gz"
MODEL = "/lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation-v2"


def main():
    rng = random.Random(7)
    print("Sampling pairs ...", file=sys.stderr)
    pairs = []
    with gzip.open(PAIRS, "rt") as f:
        for line in f:
            if len(pairs) < 30_000:
                pairs.append(json.loads(line))
            else:
                idx = rng.randrange(len(pairs))
                if idx < 30_000:
                    pairs[idx] = json.loads(line)
    sample = rng.sample(pairs, 10)

    print("Loading model ...", file=sys.stderr)
    model = SentenceTransformer(MODEL)
    model.max_seq_length = 256
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    anchor_texts = [p["anchor_text"][:2000] for p in sample]
    cite_texts = [p["positive_text"][:2000] for p in sample]
    with torch.no_grad():
        a_emb = model.encode(anchor_texts, normalize_embeddings=True, convert_to_numpy=True)
        c_emb = model.encode(cite_texts, normalize_embeddings=True, convert_to_numpy=True)
    sims = (a_emb * c_emb).sum(axis=1)

    for i, (p, sim) in enumerate(zip(sample, sims)):
        print(f"\n{'=' * 80}")
        print(f"PAIR #{i + 1}  similarity={sim:.3f}  "
              f"app={p['rejected_app_id']} → cited={p['positive_pgpub_id']}")
        print(f"{'-' * 80}")
        print(f"ANCHOR (rejected app's claims, first 800 chars):")
        print(f"  {p['anchor_text'][:800]}")
        print(f"{'-' * 80}")
        print(f"POSITIVE (cited ref's claim-1, first 800 chars):")
        print(f"  {p['positive_text'][:800]}")


if __name__ == "__main__":
    main()
