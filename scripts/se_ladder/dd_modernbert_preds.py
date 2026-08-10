"""Deep dive task 3 prep: re-run ModernBERT FT on so_python, SAVING test
predictions (the ladder run discarded them). Same config as
se_ladder_modernbert.py (2 ep, lr 3e-5, 1024 tok, bf16, <=30K train,
seed 0). Writes {slice}_modernbert_preds.parquet (row_id, p) +
{slice}_modernbert_rerun.json.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python dd_modernbert_preds.py so_python
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

    model.eval()
    test_dl = DataLoader(DS(te.body.tolist(), te.label.tolist()),
                         batch_size=64, shuffle=False, collate_fn=collate,
                         num_workers=4)
    probs = []
    with torch.no_grad():
        for batch in test_dl:
            batch.pop("labels")
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(**batch).logits.float()
            probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    p = np.concatenate(probs)
    auc = float(roc_auc_score(te.label.values, p))
    pd.DataFrame({"row_id": te.row_id.values, "p": p}).to_parquet(
        OUT_DIR / f"{slice_name}_modernbert_preds.parquet", index=False)
    res = {"slice": slice_name, "test_auc": auc,
           "train_minutes": round((time.time() - t0) / 60, 1),
           "note": "re-run with saved test preds; same config as ladder"}
    (OUT_DIR / f"{slice_name}_modernbert_rerun.json").write_text(
        json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
