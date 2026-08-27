#!/usr/bin/env python3
"""Candidate-only dense-ceiling experiments per cell.

Two ladders:
  (a) Frozen bge-code-v1 embedding of candidate_code -> LR with
      StratifiedGroupKFold by canonical_pid (5-fold x 5 seeds).
  (b) Fine-tuned ModernBERT-base single-input classifier:
      grouped 5-fold, fine-tune per fold, 3 seeds; bf16.

Outputs per-cell results to OUT_DIR. Writes JSON with mean/std per metric per cell.

Designed for sk3, 1 GPU. Stacking onto a shared GPU is fine — bf16 inference and
ModernBERT-base both fit in ~10-15GB.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)


# ----------------------------- frozen probe ----------------------------------

def last_token_pool(last_hidden, attn_mask):
    left_pad = (attn_mask[:, -1].sum() == attn_mask.shape[0])
    if left_pad:
        return last_hidden[:, -1]
    seq_lens = attn_mask.sum(dim=1) - 1
    bs = last_hidden.size(0)
    return last_hidden[torch.arange(bs, device=last_hidden.device), seq_lens]


def embed_texts(texts, tok, model, batch_size=32, max_len=1024, device="cuda"):
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = [t if isinstance(t, str) and t.strip() else "EMPTY"
                     for t in texts[i:i + batch_size]]
            enc = tok(batch, max_length=max_len, padding=True, truncation=True,
                      return_tensors="pt", pad_to_multiple_of=8).to(device)
            o = model(**enc)
            emb = F.normalize(
                last_token_pool(o.last_hidden_state, enc["attention_mask"]).float(),
                p=2, dim=1)
            out[i:i + len(batch)] = emb.cpu().numpy()
    return out


def frozen_probe(df, embeds, n_splits=5, seeds=(0, 1, 2, 3, 4)):
    """LR on candidate-only embeddings, StratifiedGroupKFold."""
    aucs = []
    fold_aucs = []
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y = df.label.values
        groups = df.canonical_pid.values
        per_seed = []
        for fold, (tr, te) in enumerate(sgkf.split(embeds, y, groups)):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            scaler = StandardScaler().fit(embeds[tr])
            Xtr = scaler.transform(embeds[tr])
            Xte = scaler.transform(embeds[te])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(y[te], p)
            per_seed.append(auc)
            fold_aucs.append({"seed": seed, "fold": fold, "auc": auc, "n_te": len(te)})
        if per_seed:
            aucs.append(float(np.mean(per_seed)))
    return {
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "std_auc": float(np.std(aucs)) if aucs else None,
        "per_seed_means": aucs,
        "fold_details": fold_aucs,
    }


# -------------------------- ModernBERT fine-tune -----------------------------

class CodeDataset(Dataset):
    def __init__(self, df, tok, max_len):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(str(r.candidate_code), truncation=True,
                       max_length=self.max_len, padding="max_length",
                       return_tensors="pt")
        return ({k: v.squeeze(0) for k, v in enc.items()},
                torch.tensor(float(r.label)))


@torch.no_grad()
def mb_predict(model, loader, device):
    model.eval()
    probs = []
    for enc, _ in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1)
        probs.extend(torch.sigmoid(logits).float().cpu().numpy())
    return np.array(probs)


def finetune_modernbert_fold(train_df, test_df, model_name, device,
                              max_len=1024, batch=8, epochs=3, lr=2e-5,
                              seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, torch_dtype=torch.bfloat16).to(device)

    tr_loader = DataLoader(CodeDataset(train_df, tok, max_len),
                           batch_size=batch, shuffle=True, num_workers=2)
    te_loader = DataLoader(CodeDataset(test_df, tok, max_len),
                           batch_size=batch * 2, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total = len(tr_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        running = 0.0
        for step, (enc, y) in enumerate(tr_loader):
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

    probs = mb_predict(model, te_loader, device)
    auc = roc_auc_score(test_df.label.values, probs) if len(set(test_df.label)) > 1 else None
    del model
    torch.cuda.empty_cache()
    return auc


def finetune_modernbert(df, model_name="answerdotai/ModernBERT-base",
                         device="cuda", n_splits=5, seeds=(0, 1, 2),
                         max_len=1024, batch=8, epochs=3, lr=2e-5):
    aucs = []
    fold_aucs = []
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y = df.label.values
        groups = df.canonical_pid.values
        per_seed = []
        for fold, (tr, te) in enumerate(sgkf.split(df, y, groups)):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            tr_df = df.iloc[tr].copy()
            te_df = df.iloc[te].copy()
            t0 = time.time()
            auc = finetune_modernbert_fold(tr_df, te_df, model_name, device,
                                            max_len=max_len, batch=batch,
                                            epochs=epochs, lr=lr, seed=seed)
            dur = time.time() - t0
            if auc is None:
                continue
            per_seed.append(auc)
            fold_aucs.append({"seed": seed, "fold": fold, "auc": auc,
                              "n_tr": len(tr), "n_te": len(te), "seconds": dur})
            print(f"  seed={seed} fold={fold} n_tr={len(tr)} n_te={len(te)} "
                  f"AUC={auc:.4f}  {dur:.1f}s", flush=True)
        if per_seed:
            aucs.append(float(np.mean(per_seed)))
    return {
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "std_auc": float(np.std(aucs)) if aucs else None,
        "per_seed_means": aucs,
        "fold_details": fold_aucs,
    }


# ---------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells", nargs="+", required=True,
                    help="cell names without 'cell_' prefix and '.parquet' suffix")
    ap.add_argument("--bge-model", default="BAAI/bge-code-v1")
    ap.add_argument("--mb-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--bge-max-len", type=int, default=1024)
    ap.add_argument("--mb-max-len", type=int, default=1024)
    ap.add_argument("--mb-batch", type=int, default=8)
    ap.add_argument("--mb-epochs", type=int, default=3)
    ap.add_argument("--mb-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--skip-finetune", action="store_true",
                    help="Run only frozen probe (debug / fast path)")
    args = ap.parse_args()

    cells_dir = Path(args.cells_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # Load all cell tables first so we can embed once per text pool.
    cells = {}
    for name in args.cells:
        p = cells_dir / f"cell_{name}.parquet"
        df = pd.read_parquet(p)
        # canonical_pid fallback
        df["canonical_pid"] = df["canonical_pid"].fillna("pid_" + df["pair_id"].astype(str))
        cells[name] = df
        print(f"loaded {name}: n={len(df)} pos={df.label.mean():.3f} "
              f"groups={df.canonical_pid.nunique()}", flush=True)

    # ------------------- frozen probe (one model load) -------------------
    print(f"\n=== Frozen probe: {args.bge_model} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(args.bge_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.bge_model, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to(device).eval()
    probe_results = {}
    embed_cache = {}
    for name, df in cells.items():
        print(f"  embedding {name} ({len(df)} texts)...", flush=True)
        t0 = time.time()
        emb = embed_texts(df.candidate_code.astype(str).tolist(),
                          tok, model, batch_size=32, max_len=args.bge_max_len,
                          device=device)
        print(f"    embed done {time.time()-t0:.1f}s", flush=True)
        embed_cache[name] = emb
        res = frozen_probe(df, emb, n_splits=args.n_splits, seeds=args.probe_seeds)
        print(f"  PROBE {name}: AUC {res['mean_auc']:.4f} ± {res['std_auc']:.4f}",
              flush=True)
        probe_results[name] = res
        # checkpoint
        with open(out_dir / "frozen_probe.json", "w") as f:
            json.dump(probe_results, f, indent=2)

    del model, tok
    torch.cuda.empty_cache()

    # ------------------- ModernBERT fine-tune ----------------------------
    ft_results = {}
    if not args.skip_finetune:
        print(f"\n=== Fine-tune: {args.mb_model} ===", flush=True)
        for name, df in cells.items():
            print(f"\n--- {name} ---", flush=True)
            res = finetune_modernbert(df, model_name=args.mb_model,
                                       device=device, n_splits=args.n_splits,
                                       seeds=args.mb_seeds,
                                       max_len=args.mb_max_len,
                                       batch=args.mb_batch,
                                       epochs=args.mb_epochs)
            print(f"  FT {name}: AUC {res['mean_auc']:.4f} ± {res['std_auc']:.4f}",
                  flush=True)
            ft_results[name] = res
            # checkpoint
            with open(out_dir / "finetune.json", "w") as f:
                json.dump(ft_results, f, indent=2)

    # ------------------- combined output ---------------------------------
    combined = {
        "args": vars(args),
        "cells": {name: {"n": int(len(df)),
                          "pos_rate": float(df.label.mean()),
                          "groups": int(df.canonical_pid.nunique())}
                   for name, df in cells.items()},
        "frozen_probe": probe_results,
        "finetune": ft_results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
