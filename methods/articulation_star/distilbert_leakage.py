"""DistilBERT leakage classifier — trains a small text classifier to predict
the artifact's label from rationale text alone, on the kept rationales from
the training-time iters. Used two ways:

  1. **As a measurement.** The classifier's held-out accuracy on rationales
     IS the canonical leakage rate. If a tiny frozen classifier can predict
     the label from rationale-only text at, say, 65%, then rationales leak
     at 65%.

  2. **As an exclusionary filter.** At combine time we can drop rationales
     where p(y_true | rationale) > tau (say 0.9), since those are
     "too-easily decodable" and probably leak via cheap cues.

The probe is frozen between training-time iters and articulation-time iters
of the STaR loop: we train it ONCE on iter-0 kept rationales, then evaluate
it on iter-1, iter-2, etc. If accuracy on iter-2 rationales is higher than
on iter-0, the rationales are getting MORE label-decodable over time → loop
is amplifying leakage. If accuracy stays flat, loop isn't amplifying.

Modes:
  --mode train      train classifier on iter-0 kept rationales
  --mode score      score classifier on (any iter's) kept rationales OR
                    on test-eval rationales
  --mode summarize  print per-iter accuracy table

CLI examples:
  python -m methods.articulation_star.distilbert_leakage \\
    --task creative_writing --run_name v1_overnight_logprob --mode train

  python -m methods.articulation_star.distilbert_leakage \\
    --task creative_writing --run_name v1_overnight_logprob --mode score \\
    --target test --stage base
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from .config import LoopConfig, TASKS


MODEL_ID = "distilbert-base-uncased"
MAX_LEN = 384


class RationaleDataset(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(
            r["completion"],
            truncation=True, padding="max_length", max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(r["y"], dtype=torch.long),
        }


def _classifier_path(cfg: LoopConfig) -> Path:
    return Path(cfg.output_root) / cfg.task / cfg.run_name / "distilbert_leakage"


def train(cfg: LoopConfig, train_iter: int = 0, val_frac: float = 0.1,
          n_epochs: int = 3, lr: float = 2e-5, batch_size: int = 32) -> Path:
    """Train DistilBERT to predict y from rationale text on iter-N's kept rationales."""
    kept_path = cfg.iter_dir(train_iter) / "rationales_kept.jsonl"
    rows = [json.loads(l) for l in kept_path.open()]
    random.seed(13); random.shuffle(rows)
    n_val = max(int(len(rows) * val_frac), 16)
    val = rows[:n_val]; tr = rows[n_val:]
    print(f"[distilbert.train] iter={train_iter} train={len(tr)} val={len(val)}")
    print(f"  train label dist: {{1: {sum(1 for r in tr if r['y']==1)}, 0: {sum(1 for r in tr if r['y']==0)}}}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2,
    ).to(device)

    train_ds = RationaleDataset(tr, tok)
    val_ds = RationaleDataset(val, tok)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_steps = len(train_dl) * n_epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=n_steps // 10, num_training_steps=n_steps)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0; n_batches = 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            total_loss += out.loss.item(); n_batches += 1
        # val
        model.eval(); correct = 0; total = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                preds = out.logits.argmax(-1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])
        model.train()
        print(f"  epoch {epoch+1}: train_loss={total_loss/n_batches:.4f} val_acc={correct/total:.1%}")

    out_dir = _classifier_path(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[distilbert.train] saved to {out_dir}")
    return out_dir


def score(cfg: LoopConfig, target: str, stage: str | None = None,
          iter_idx: int | None = None, batch_size: int = 32) -> Path:
    """Score rationales with the trained classifier.

    target='test'  -> score outputs/test_eval/rationales_<stage>.jsonl
    target='iter'  -> score outputs/iter_NN/rationales_kept.jsonl
    """
    path = _classifier_path(cfg)
    if not (path / "config.json").exists():
        raise SystemExit(f"No trained classifier at {path}. Run --mode train first.")

    if target == "test":
        if stage is None:
            raise SystemExit("--stage required for target=test")
        test_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
        in_path = test_dir / f"rationales_{stage}.jsonl"
        out_path = test_dir / f"distilbert_scores_{stage}.jsonl"
    else:
        if iter_idx is None:
            raise SystemExit("--iter required for target=iter")
        in_path = cfg.iter_dir(iter_idx) / "rationales_kept.jsonl"
        out_path = cfg.iter_dir(iter_idx) / "distilbert_scores_kept.jsonl"

    rows = [json.loads(l) for l in in_path.open()]
    print(f"[distilbert.score] target={target} stage={stage} iter={iter_idx} n={len(rows)}")

    tok = AutoTokenizer.from_pretrained(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

    ds = RationaleDataset(rows, tok)
    dl = DataLoader(ds, batch_size=batch_size)

    out = []
    correct = 0
    with torch.no_grad():
        i = 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(-1)
            for j in range(len(preds)):
                r = rows[i]
                rec = {
                    "row_id": r.get("row_id", i),
                    "y": r["y"],
                    "stage": stage,
                    "p_pos": float(probs[j, 1].item()),
                    "p_neg": float(probs[j, 0].item()),
                    "pred": int(preds[j].item()),
                    "p_y_true": float(probs[j, r["y"]].item()),
                    "correct": int(preds[j].item() == r["y"]),
                }
                out.append(rec)
                correct += rec["correct"]
                i += 1

    acc = correct / len(rows)
    # AUC
    pairs = sorted([(r["p_pos"], r["y"]) for r in out])
    npos = sum(1 for _, y in pairs if y == 1); nneg = len(pairs) - npos
    rank_sum_pos = 0; cnt = 0
    for sc, y in pairs:
        cnt += 1
        if y == 1: rank_sum_pos += cnt
    auc = (rank_sum_pos - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else float("nan")
    # fraction with p_y_true > 0.9 (would-be filtered out as too-easy)
    too_easy = sum(1 for r in out if r["p_y_true"] > 0.9) / len(out)
    print(f"  acc={acc:.1%} auc={auc:.3f} too_easy_(p>.9)={too_easy:.1%}")

    with out_path.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"  saved -> {out_path}")
    return out_path


def summarize(cfg: LoopConfig, stages: list[str]) -> None:
    test_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
    print()
    print("=" * 78)
    print(f"DISTILBERT LEAKAGE PROBE  ({cfg.task} / {cfg.run_name})")
    print("=" * 78)
    print("stage     n     acc    AUC   too_easy(p>.9)  too_easy(p>.95)")
    print("-" * 78)
    for s in stages:
        p = test_dir / f"distilbert_scores_{s}.jsonl"
        if not p.exists():
            print(f"{s:<8}  (missing)")
            continue
        rows = [json.loads(l) for l in p.open()]
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n
        pairs = sorted([(r["p_pos"], r["y"]) for r in rows])
        npos = sum(1 for _, y in pairs if y == 1); nneg = n - npos
        rs = 0; c = 0
        for sc, y in pairs:
            c += 1
            if y == 1: rs += c
        auc = (rs - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else float("nan")
        te9 = sum(1 for r in rows if r["p_y_true"] > 0.9) / n
        te95 = sum(1 for r in rows if r["p_y_true"] > 0.95) / n
        print(f"{s:<8} {n:>4}  {acc:>5.1%}  {auc:>5.3f}     {te9:>10.1%}     {te95:>13.1%}")
    print("=" * 78)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative_writing")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--mode", choices=["train", "score", "summarize"], required=True)
    ap.add_argument("--train_iter", type=int, default=0)
    ap.add_argument("--target", choices=["test", "iter"], default="test")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--iter", type=int, default=None, dest="iter_idx")
    ap.add_argument("--stages", default="base,iter00,iter01,iter02")
    ap.add_argument("--n_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    cfg = LoopConfig(task=a.task, run_name=a.run_name)
    if a.mode == "train":
        train(cfg, train_iter=a.train_iter, n_epochs=a.n_epochs, lr=a.lr)
    elif a.mode == "score":
        score(cfg, target=a.target, stage=a.stage, iter_idx=a.iter_idx)
    elif a.mode == "summarize":
        summarize(cfg, a.stages.split(","))
