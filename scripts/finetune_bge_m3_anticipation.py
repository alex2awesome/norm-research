#!/usr/bin/env python3
"""Fine-tune BAAI/bge-m3 on anticipation pairs using sentence-transformers.

Loss: MultipleNegativesRanking (InfoNCE w/ in-batch negatives).
Each batch element provides (anchor, positive); other batch positives become
hard negatives for each anchor.

Inputs:
  - processed/anticipation_training_pairs.jsonl.gz from extract_anticipation_training_pairs.py

Outputs:
  - models/bge-m3-anticipation/ (HuggingFace SentenceTransformer dir)
"""
import argparse
import gzip
import json
import os
import random

import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
DEFAULT_PAIRS = f"{BASE}/processed/anticipation_training_pairs.jsonl.gz"
OUT = "/lfs/skampere3/0/alexspan/norm-research/models/bge-m3-anticipation"
SEED = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="BAAI/bge-m3")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_pct", type=float, default=0.05)
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap on training pairs (default: use all)")
    p.add_argument("--out", default=OUT)
    p.add_argument("--pairs", default=DEFAULT_PAIRS,
                   help="Path to anticipation_training_pairs(.v2).jsonl.gz")
    args = p.parse_args()

    print(f"Loading pairs from {args.pairs} ...")
    examples = []
    with gzip.open(args.pairs, "rt") as f:
        for line in f:
            r = json.loads(line)
            a = r["anchor_text"][:2000]
            p_ = r["positive_text"][:2000]
            if a and p_:
                examples.append(InputExample(texts=[a, p_]))
    random.Random(SEED).shuffle(examples)
    if args.max_pairs:
        examples = examples[: args.max_pairs]
    print(f"  loaded {len(examples):,} pairs")

    print(f"\nLoading base model {args.base_model} ...")
    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_len

    train_dataloader = DataLoader(
        examples, shuffle=True, batch_size=args.batch_size, num_workers=2
    )
    loss = losses.MultipleNegativesRankingLoss(model)

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)
    print(f"\nTraining: {steps_per_epoch} steps/epoch × {args.epochs} = {total_steps} total, warmup {warmup_steps}")

    os.makedirs(args.out, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=args.out,
        checkpoint_path=os.path.join(args.out, "checkpoints"),
        checkpoint_save_steps=10_000,
        save_best_model=True,
    )
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
