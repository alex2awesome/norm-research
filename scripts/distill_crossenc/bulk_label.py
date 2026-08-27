#!/usr/bin/env python3
"""Bulk-label competition pairs with a trained cross-encoder checkpoint.

Streams (editorial_text, candidate_text) pairs through the model and writes
pair_id, prob, label (prob >= --threshold). Used for the LC scale-out:
machine-label the 102K-pair phase1b LC pool with the Claude-distilled v0.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PoolDataset(Dataset):
    def __init__(self, df, tok, max_len):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(str(r.editorial_text), str(r.candidate_text),
                       truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--threshold", type=float, default=0.254)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=20000,
                    help="write partial output every N pairs (resumable-ish)")
    args = ap.parse_args()

    df = pd.read_parquet(args.pool)
    if args.platform:
        df = df[df.platform == args.platform].reset_index(drop=True)
    print(f"labeling {len(df)} pairs from {args.pool}")

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=1, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    loader = DataLoader(PoolDataset(df, tok, args.max_len),
                        batch_size=args.batch, num_workers=4)
    probs = []
    for bi, enc in enumerate(loader):
        enc = {k: v.to("cuda") for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1)
        probs.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
        n_done = len(probs)
        if bi % 50 == 0:
            print(f"{n_done}/{len(df)}", flush=True)
        if n_done % args.checkpoint_every < args.batch:
            part = df.iloc[:n_done][["pair_id", "platform", "canonical_pid"]].copy()
            part["prob"] = probs
            part.to_parquet(args.out + ".partial")

    out = df[["pair_id", "platform", "canonical_pid", "cosine"]].copy() \
        if "cosine" in df.columns else df[["pair_id", "platform", "canonical_pid"]].copy()
    out["prob"] = probs
    out["label"] = (out.prob >= args.threshold).astype(int)
    out.to_parquet(args.out)
    Path(args.out + ".partial").unlink(missing_ok=True)
    print(f"wrote {len(out)} -> {args.out}; pos_rate={out.label.mean():.3f}")


if __name__ == "__main__":
    main()
