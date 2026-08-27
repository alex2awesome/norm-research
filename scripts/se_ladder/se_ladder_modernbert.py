"""SE ladder step 3b: ModernBERT-base fine-tune (2026-06-11).

Per slice: fine-tune answerdotai/ModernBERT-base as a binary classifier on
the raw answer body (1024-token truncation), train split subsampled to
<=30K rows, 2 epochs, bf16, report test-split AUC + per-stratum.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python se_ladder_modernbert.py <slice>
Writes outputs/v2_analysis/se_ladder/{slice}_modernbert.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"
MODEL = "answerdotai/ModernBERT-base"
MAX_LEN = 1024
BS = 32
EPOCHS = 2
LR = 3e-5
MAX_TRAIN = 30_000


class DS(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        t = self.texts[i]
        return (t if isinstance(t, str) and t else " "), self.labels[i]


def main():
    slice_name = sys.argv[1]
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    df = pd.read_parquet(OUT_DIR / f"{slice_name}_input.parquet")
    tr = df[df.split == "train"]
    if len(tr) > MAX_TRAIN:
        tr = tr.sample(n=MAX_TRAIN, random_state=0)
    te = df[df.split == "test"]
    print(f"[{slice_name}] train={len(tr)} test={len(te)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, dtype=torch.bfloat16).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_steps = (len(tr) // BS + 1) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=0.06)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc

    train_dl = DataLoader(DS(tr.body.tolist(), tr.label.tolist()),
                          batch_size=BS, shuffle=True, collate_fn=collate,
                          num_workers=4)
    t0 = time.time()
    model.train()
    step = 0
    for ep in range(EPOCHS):
        for batch in train_dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"[{slice_name}] ep{ep} step {step}/{n_steps} "
                      f"loss={out.loss.item():.4f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    # eval
    model.eval()
    test_dl = DataLoader(DS(te.body.tolist(), te.label.tolist()),
                         batch_size=64, shuffle=False, collate_fn=collate,
                         num_workers=4)
    probs = []
    with torch.no_grad():
        for batch in test_dl:
            labels = batch.pop("labels")
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(**batch).logits.float()
            probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    p = np.concatenate(probs)
    res = {"slice": slice_name, "model": MODEL, "max_len": MAX_LEN,
           "epochs": EPOCHS, "n_train": int(len(tr)), "n_test": int(len(te)),
           "test_auc": float(roc_auc_score(te.label.values, p)),
           "train_minutes": round((time.time() - t0) / 60, 1)}
    te2 = te.copy()
    te2["p"] = p
    strata = {}
    for s, sub in te2.groupby("stratum"):
        if len(sub) < 200 or sub.label.nunique() < 2:
            continue
        strata[s] = {"n_test": int(len(sub)),
                     "auc": float(roc_auc_score(sub.label, sub.p))}
    res["per_stratum_test"] = strata
    out = OUT_DIR / f"{slice_name}_modernbert.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
