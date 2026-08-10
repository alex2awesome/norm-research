"""Method B: ModernBERT-base cross-encoder fine-tune on Phase-1 stratified subsample.

Cross-encoder format (tokenizer pair-encode):
    [CLS] editorial_text [SEP] candidate_text [SEP]
Truncated to MAX_LEN tokens total (longest_first, candidate side).

Binary classification head, BCE/CE loss (CrossEntropyLoss on 2-logit head with class
weights for balance), AdamW lr=2e-5 wd=0.01, bf16, 3 epochs, batch 16 grad-accum 2.

LEAKAGE NOTE: cross-encoder is fine; both texts are inputs the model sees end-to-end.
We do NOT add cosine or any pairwise feature beyond the raw text.

CV: 5-fold StratifiedKFold(seed=42), SAME splits per (pooled / lc / luogu) cell as
the Bank LR reference for direct comparison.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
OUT_DIR = f"{ROOT}/outputs/v2_analysis"
SUB = f"{OUT_DIR}/comp_qwen_phase1_stratified_subsample.parquet"
OUT_JSON = f"{OUT_DIR}/comp_qwen_phase1_modernbert_auc.json"
WORK_DIR = f"{OUT_DIR}/comp_qwen_phase1_modernbert_runs"

MODEL_ID = "answerdotai/ModernBERT-base"
MAX_LEN = 4096  # spec says 4096 (ModernBERT-base native context 8K)
N_SPLITS = 5
SEED = 42

EPOCHS = 3
LR = 2e-5
WD = 0.01
PER_DEVICE_BATCH = 16
GRAD_ACCUM = 2  # effective batch 32

# Allow caller to restrict which cells run; default all three.
CELL_ARG = os.environ.get("MB_CELL", "pooled,lc,luogu").split(",")
# Allow fold subset to parallelize across multiple GPU/processes.
FOLDS_ARG = os.environ.get("MB_FOLDS")  # e.g. "0,1" -> only run folds 0 and 1
N_SPLITS_ENV = int(os.environ.get("MB_NSPLITS", N_SPLITS))


def select_indices(df: pd.DataFrame, cell: str) -> np.ndarray:
    if cell == "pooled":
        return np.arange(len(df))
    if cell == "lc":
        return np.where(df["platform"].to_numpy() == "lc")[0]
    if cell == "luogu":
        return np.where(df["platform"].to_numpy() == "luogu")[0]
    raise ValueError(cell)


def tokenize_pair(tokenizer, ed_texts, cand_texts, max_len=MAX_LEN):
    return tokenizer(
        list(ed_texts),
        list(cand_texts),
        truncation="longest_first",
        max_length=max_len,
        padding=False,
    )


def run_fold(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
             tokenizer, cell: str, fold_id: int, work_dir: str) -> dict:
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    def to_ds(d):
        ds = Dataset.from_pandas(
            d[["editorial_text", "candidate_text", "qwen_label"]].rename(
                columns={"qwen_label": "labels"}),
            preserve_index=False,
        )

        def _tok(batch):
            return tokenize_pair(tokenizer, batch["editorial_text"], batch["candidate_text"])
        ds = ds.map(_tok, batched=True, remove_columns=["editorial_text", "candidate_text"])
        return ds

    train_ds = to_ds(train_df)
    val_ds = to_ds(val_df)

    # Class weights for any residual imbalance (subsample is roughly balanced)
    n_pos = int(train_df["qwen_label"].sum())
    n_neg = len(train_df) - n_pos
    pos_w = float(n_neg) / max(1, n_pos)
    class_weights = torch.tensor([1.0, pos_w], dtype=torch.float)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=2, dtype=torch.bfloat16
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=2, torch_dtype=torch.bfloat16
        )

    outdir = os.path.join(work_dir, f"{cell}_fold{fold_id}")
    args = TrainingArguments(
        output_dir=outdir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WD,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        bf16=True,
        tf32=True,
        report_to="none",
        dataloader_num_workers=2,
        seed=SEED,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def compute_metrics(p):
        logits, lbl = p
        probs = torch.softmax(torch.tensor(logits).float(), dim=-1).numpy()[:, 1]
        try:
            auc = roc_auc_score(lbl, probs)
        except Exception:
            auc = float("nan")
        return {"auc": auc}

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            cw = class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=cw)
            loss = loss_fn(logits.float(), labels)
            return (loss, outputs) if return_outputs else loss

    # transformers 5.x uses processing_class; 4.x uses tokenizer.
    try:
        trainer = WTrainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
            processing_class=tokenizer, data_collator=collator,
            compute_metrics=compute_metrics,
        )
    except TypeError:
        trainer = WTrainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
            tokenizer=tokenizer, data_collator=collator,
            compute_metrics=compute_metrics,
        )
    trainer.train()
    pred = trainer.predict(val_ds)
    probs = torch.softmax(torch.tensor(pred.predictions).float(), dim=-1).numpy()[:, 1]
    final_auc = float(roc_auc_score(val_df["qwen_label"].to_numpy(), probs))
    best_auc = final_auc
    per_epoch = []
    for entry in trainer.state.log_history:
        if "eval_auc" in entry:
            best_auc = max(best_auc, float(entry["eval_auc"]))
            per_epoch.append({"epoch": entry.get("epoch"), "eval_auc": float(entry["eval_auc"])})

    del trainer, model
    torch.cuda.empty_cache()

    return {
        "cell": cell, "fold": fold_id,
        "n_train": int(len(train_df)), "n_val": int(len(val_df)),
        "final_auc": final_auc, "best_epoch_auc": best_auc,
        "per_epoch_auc": per_epoch,
    }


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    print(f"loading {SUB}...", flush=True)
    df = pd.read_parquet(SUB)
    print(f"rows={len(df)} cols={list(df.columns)}", flush=True)

    # ---- PRE-FLIGHT LEAKAGE CHECK ----
    print("--- pre-flight leakage check ---", flush=True)
    print("label col: qwen_label", flush=True)
    print("inputs: editorial_text + candidate_text (cross-encoder pair)", flush=True)
    bad = [c for c in df.columns if any(t in c.lower() for t in ["cos", "sim", "embed"])]
    assert not bad, f"LEAKAGE GUARD: blocked input cols {bad}"
    print("OK: no cosine/sim/embed column present", flush=True)

    print(f"cuda available: {torch.cuda.is_available()} ndev={torch.cuda.device_count()}",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    out_path = OUT_JSON
    # Resume if partial file exists
    out = {"method": "modernbert_crossenc", "model": MODEL_ID, "max_len": MAX_LEN,
           "n_splits": N_SPLITS_ENV, "seed": SEED,
           "epochs": EPOCHS, "lr": LR, "weight_decay": WD,
           "per_device_batch": PER_DEVICE_BATCH, "grad_accum": GRAD_ACCUM,
           "leakage_note": "cross-encoder; no extra pairwise feature",
           "cells": {}}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                prior = json.load(f)
            if prior.get("model") == MODEL_ID:
                out = prior
                out.setdefault("cells", {})
                print(f"resuming with prior cells: {list(out['cells'].keys())}",
                      flush=True)
        except Exception:
            pass

    folds_filter = None
    if FOLDS_ARG:
        folds_filter = set(int(x) for x in FOLDS_ARG.split(","))
        print(f"FOLD filter: {folds_filter}", flush=True)

    for cell in CELL_ARG:
        idx = select_indices(df, cell)
        sub = df.iloc[idx].reset_index(drop=True)
        y = sub["qwen_label"].to_numpy(dtype=int)
        print(f"\n=== cell={cell} n={len(sub)} pos_rate={y.mean():.3f} ===", flush=True)

        skf = StratifiedKFold(n_splits=N_SPLITS_ENV, shuffle=True, random_state=SEED)
        if cell not in out["cells"]:
            out["cells"][cell] = {"n": int(len(sub)), "pos_rate": float(y.mean()),
                                  "folds": []}
        done_folds = {r["fold"] for r in out["cells"][cell].get("folds", [])}

        for fold_id, (tr, va) in enumerate(skf.split(np.zeros(len(sub)), y)):
            if folds_filter is not None and fold_id not in folds_filter:
                continue
            if fold_id in done_folds:
                print(f"skip cell={cell} fold={fold_id} (already done)", flush=True)
                continue
            t0 = time.time()
            print(f"\n--- cell={cell} fold={fold_id} n_tr={len(tr)} n_va={len(va)} ---",
                  flush=True)
            res = run_fold(sub, tr, va, tokenizer, cell, fold_id, WORK_DIR)
            res["wall_seconds"] = time.time() - t0
            out["cells"][cell]["folds"].append(res)
            # recompute summary
            aucs = [r["final_auc"] for r in out["cells"][cell]["folds"]]
            best = [r["best_epoch_auc"] for r in out["cells"][cell]["folds"]]
            out["cells"][cell]["auc_mean"] = float(np.mean(aucs))
            out["cells"][cell]["auc_std"] = float(np.std(aucs))
            out["cells"][cell]["best_auc_mean"] = float(np.mean(best))
            out["cells"][cell]["best_auc_std"] = float(np.std(best))
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"--> fold {fold_id} final={res['final_auc']:.4f} "
                  f"best={res['best_epoch_auc']:.4f} wall={res['wall_seconds']:.0f}s",
                  flush=True)
            print(f"running mean={out['cells'][cell]['auc_mean']:.4f} "
                  f"(n={len(aucs)})", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
