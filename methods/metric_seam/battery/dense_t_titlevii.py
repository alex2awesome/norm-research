#!/usr/bin/env python3
"""Dense/T ceiling for the seam-slice legal y-ladder: Llama-3.1-8B + LoRA (SEQ_CLS) on
title_vii_balanced_v2.jsonl with a COURT-GROUPED 80/20 split (fullpool_llama8b recipe,
split discipline upgraded to match the seam program's checks). One GPU."""
import json, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from peft import LoraConfig, get_peft_model

MODEL = "/lfs/skampere3/0/alexspan/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
DATA = "/lfs/skampere3/0/alexspan/norm-research/datasets/legal-outcome-prediction/title_vii_balanced_v2.jsonl"
OUT = "/lfs/skampere3/0/alexspan/norm-research/methods/metric_seam/battery/y_gepa/legal_dense_t.json"
DEVICE = "cuda"; MAXLEN = 1024; EPOCHS = 2; BS = 16; LR = 1e-4

rows = [json.loads(l) for l in open(DATA)]
def lab(r):
    b = r.get("binary_label")
    if b in (0, 1): return b
    return {"PLAINTIFF_WIN": 1, "DEFENDANT_WIN": 0}.get(r.get("outcome"))
rows = [r for r in rows if lab(r) in (0, 1) and len(r.get("facts") or "") >= 200]
X = [r["facts"] for r in rows]; y = np.array([lab(r) for r in rows])
g = np.array([r.get("court_id", "?") for r in rows])
tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, groups=g))
print(f"n={len(rows)} train={len(tr_i)} test={len(te_i)} courts(train/test disjoint)={len(set(g[tr_i]) & set(g[te_i]))==0}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
coll = DataCollatorWithPadding(tok)
def make_ds(texts, labels):
    enc = tok(texts, truncation=True, max_length=MAXLEN)
    class D(Dataset):
        def __len__(s): return len(labels)
        def __getitem__(s, i): return {"input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i], "labels": int(labels[i])}
    return D()

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, torch_dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id
model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"])).to(DEVICE)
tr = DataLoader(make_ds([X[i] for i in tr_i], y[tr_i]), batch_size=BS, shuffle=True, collate_fn=coll)
te = DataLoader(make_ds([X[i] for i in te_i], y[te_i]), batch_size=32, shuffle=False, collate_fn=coll)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
steps = len(tr) * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
best = 0.0
for ep in range(EPOCHS):
    model.train()
    for b in tr:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        out = model(**b); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad()
    model.eval(); preds = []
    with torch.no_grad():
        for b in te:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            preds += torch.softmax(model(**b).logits.float(), -1)[:, 1].cpu().tolist()
    auc = roc_auc_score(y[te_i], preds)
    print(f"epoch {ep}: test AUC {auc:.4f}", flush=True)
    best = max(best, auc)
json.dump(dict(n=len(rows), n_test=len(te_i), auc_best=round(best, 4),
               split="court-grouped 80/20", recipe="Llama-3.1-8B LoRA SEQ_CLS, fullpool recipe"),
          open(OUT, "w"), indent=1)
print(f"DENSE_T_DONE best={best:.4f}", flush=True)
