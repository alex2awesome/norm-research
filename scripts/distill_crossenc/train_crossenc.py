#!/usr/bin/env python3
"""Distill Claude R4 editorial-similarity labels into a cross-encoder.

Replaces Qwen as the bulk labeler (Qwen failed the kappa gate: 0.388 vs the 0.6
threshold). Trained purely on Claude labels; evaluated with problem-grouped
splits so no canonical_pid crosses train/val/test.

Usage (sk3, 1 GPU):
  python3 train_crossenc.py --table distill_training_table.parquet \
      --model answerdotai/ModernBERT-base --out runs/distill_v0

Reports per-platform AUC and kappa@0.5 on the held-out test split — the
distillation gate is kappa >= 0.6 vs held-out Claude labels (same bar Qwen failed).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(r.editorial_code, r.candidate_code, truncation=True,
                       max_length=self.max_len, padding="max_length",
                       return_tensors="pt")
        return ({k: v.squeeze(0) for k, v in enc.items()},
                torch.tensor(float(r.label)))


def grouped_split(df, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(gss.split(df, groups=df.canonical_pid))
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest, groups=rest.canonical_pid))
    return df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for enc, _ in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1)
        probs.extend(torch.sigmoid(logits).float().cpu().numpy())
    return np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    df = pd.read_parquet(args.table)
    df["canonical_pid"] = df.canonical_pid.fillna("pid_" + df.pair_id.astype(str))
    train, val, test = grouped_split(df, args.seed)
    print(f"train {len(train)} / val {len(val)} / test {len(test)} "
          f"(problems: {train.canonical_pid.nunique()}/{val.canonical_pid.nunique()}"
          f"/{test.canonical_pid.nunique()})")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, torch_dtype=torch.bfloat16).to(device)

    loaders = {
        "train": DataLoader(PairDataset(train, tok, args.max_len),
                            batch_size=args.batch, shuffle=True, num_workers=2),
        "val": DataLoader(PairDataset(val, tok, args.max_len),
                          batch_size=args.batch * 2, num_workers=2),
        "test": DataLoader(PairDataset(test, tok, args.max_len),
                           batch_size=args.batch * 2, num_workers=2),
    }

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(loaders["train"]) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (enc, y) in enumerate(loaders["train"]):
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)
            logits = model(**enc).logits.squeeze(-1).float()
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss {running / (step + 1):.4f}", flush=True)
        val_probs = predict(model, loaders["val"], device)
        val_auc = roc_auc_score(val.label, val_probs)
        print(f"epoch {epoch} VAL AUC {val_auc:.4f}", flush=True)
        if val_auc > best_val:
            best_val = val_auc
            model.save_pretrained(out_dir / "best")
            tok.save_pretrained(out_dir / "best")

    # final eval on test with the best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        out_dir / "best", num_labels=1, torch_dtype=torch.bfloat16).to(device)
    test_probs = predict(model, loaders["test"], device)
    test = test.copy()
    test["prob"] = test_probs
    test["pred"] = (test_probs >= 0.5).astype(int)

    metrics = {"val_auc_best": best_val, "n_train": len(train),
               "n_val": len(val), "n_test": len(test), "per_platform": {}}
    metrics["test_auc"] = roc_auc_score(test.label, test.prob)
    metrics["test_kappa"] = cohen_kappa_score(test.label, test.pred)
    for plat, g in test.groupby("platform"):
        if g.label.nunique() < 2:
            continue
        metrics["per_platform"][plat] = {
            "n": len(g),
            "auc": roc_auc_score(g.label, g.prob),
            "kappa": cohen_kappa_score(g.label, g.pred),
            "pos_rate": float(g.label.mean()),
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
