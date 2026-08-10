#!/usr/bin/env python3
"""Distill Qwen-122B element-disclosure judge into a cross-encoder.

Data: element_para_xenc_v1/{train,val,test}.jsonl.gz — (element, paragraph)
pairs from the REAL v6a retrieval distribution, soft labels from judge
confidence (disclosed -> conf/100, else (100-conf)/100).

Same recipe as train_cross_encoder_v4_clean.py (bge-reranker-base, BCE),
but with soft labels and a held-out AUC eval against the judge's hard labels.

Output: models/element-para-xenc-v1/ + test AUC printed at end.
"""
import gzip
import json
import os
import random

import numpy as np
import torch
from sentence_transformers import CrossEncoder, InputExample
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

random.seed(7)
torch.manual_seed(7)

BASE = ("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/"
        "element_para_xenc_v1")
OUT = "/lfs/skampere3/0/alexspan/norm-research/models/element-para-xenc-v1"
BASE_MODEL = "BAAI/bge-reranker-base"

BATCH = 64
EPOCHS = 2
MAX_LEN = 512
LR = 2e-5
WARMUP_PCT = 0.05
MAX_TEXT = 1500


def load(split):
    rows = []
    with gzip.open(f"{BASE}/{split}.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            if len(d["paragraph"]) < 40 or len(d["element"]) < 5:
                continue  # degenerate
            rows.append(d)
    return rows


train = load("train")
val = load("val")
test = load("test")
print(f"train {len(train):,} / val {len(val):,} / test {len(test):,}",
      flush=True)

examples = [InputExample(texts=[d["element"][:MAX_TEXT],
                                d["paragraph"][:MAX_TEXT]],
                         label=float(d["soft_label"])) for d in train]
random.shuffle(examples)

model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_LEN,
                     device="cuda")
dl = DataLoader(examples, batch_size=BATCH, shuffle=True, num_workers=0)
total = len(dl) * EPOCHS
print(f"training: {len(dl)} steps/epoch x {EPOCHS} = {total}; "
      f"warmup {int(total * WARMUP_PCT)}", flush=True)
model.fit(train_dataloader=dl, epochs=EPOCHS,
          warmup_steps=int(total * WARMUP_PCT),
          optimizer_params={"lr": LR}, show_progress_bar=True)
model.save(OUT)
print(f"saved {OUT}", flush=True)


def eval_split(name, rows):
    pairs = [[d["element"][:MAX_TEXT], d["paragraph"][:MAX_TEXT]]
             for d in rows]
    scores = model.predict(pairs, batch_size=256, show_progress_bar=False)
    y = np.array([int(d["disclosed"]) for d in rows])
    auc = roc_auc_score(y, scores)
    # agreement at 0.5 threshold vs judge hard label
    agree = ((scores > 0.5).astype(int) == y).mean()
    print(f"{name}: AUC vs judge = {auc:.4f}  agree@0.5 = {agree:.4f} "
          f"(judge disclosed-rate {y.mean():.3f}, "
          f"CE rate {(scores > 0.5).mean():.3f})", flush=True)


eval_split("val", val)
eval_split("test", test)
print("XENC-V1-DONE", flush=True)
