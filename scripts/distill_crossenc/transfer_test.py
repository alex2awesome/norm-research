#!/usr/bin/env python3
"""LC-only training -> zero-shot transfer eval on CC / Luogu / AC Claude labels.

Answers: does an editorial-similarity cross-encoder trained on (Python) LC
transfer to the C++-heavy platforms, or is per-platform data required?

Train: LC rows only (problem-grouped 85/15 train/val within LC).
Test:  ALL non-LC rows (CC 147 + Luogu 150 + AC 229 from the training table —
       none seen in training). Reports AUC + kappa@0.5 + val-tuned kappa.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_crossenc import PairDataset, predict  # noqa: E402
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,  # noqa: E402
                          get_linear_schedule_with_warmup)


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
    lc = df[df.platform == "lc"]
    other = df[df.platform != "lc"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(lc, groups=lc.canonical_pid))
    train, val = lc.iloc[tr_idx], lc.iloc[va_idx]
    print(f"LC train {len(train)} / LC val {len(val)} / transfer test {len(other)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, torch_dtype=torch.bfloat16).to(device)

    tl = DataLoader(PairDataset(train, tok, args.max_len), batch_size=args.batch,
                    shuffle=True, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(tl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        for step, (enc, y) in enumerate(tl):
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1).float()
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(tl)} loss {loss.item():.4f}",
                      flush=True)

    out = {"n_train_lc": len(train), "n_val_lc": len(val)}
    val = val.copy()
    val["prob"] = predict(model, DataLoader(PairDataset(val, tok, args.max_len),
                                            batch_size=args.batch * 2), device)
    out["lc_val_auc"] = roc_auc_score(val.label, val.prob)
    ths = np.quantile(val.prob, np.linspace(0.05, 0.95, 37))
    best_th = max(ths, key=lambda th: cohen_kappa_score(
        val.label, (val.prob >= th).astype(int)))
    out["lc_val_kappa_tuned"] = cohen_kappa_score(
        val.label, (val.prob >= best_th).astype(int))
    out["tuned_th"] = float(best_th)

    other = other.copy()
    other["prob"] = predict(model, DataLoader(PairDataset(other, tok, args.max_len),
                                              batch_size=args.batch * 2), device)
    out["transfer"] = {}
    for plat, g in other.groupby("platform"):
        out["transfer"][plat] = {
            "n": len(g), "pos_rate": float(g.label.mean()),
            "auc": roc_auc_score(g.label, g.prob),
            "kappa_0.5": cohen_kappa_score(g.label, (g.prob >= 0.5).astype(int)),
            "kappa_lc_tuned_th": cohen_kappa_score(
                g.label, (g.prob >= best_th).astype(int)),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
