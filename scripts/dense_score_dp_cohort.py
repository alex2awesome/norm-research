#!/usr/bin/env python3
"""Score the 804-dp patents validity cohort with the dense Llama-8B reward model.

STAGED 2026-06-12 — fire when a GPU is free:
  CUDA_VISIBLE_DEVICES=5 python scripts/dense_score_dp_cohort.py

Cohort definition (mirrors scripts/per_class_a_analysis.py):
  outputs/v2_db/cells_v1/task=patents parquet cells, judge_model ==
  "qwen_thinking_fp8_20x1_r2post", applicable & score notna -> 804 unique
  datapoint_ids (one of which, "d0", is a cells-DB artifact that does not
  exist in datapoints.json and is dropped with a warning -> 803 scored).

Model loading mirrors methods/dense/train_reward_model.py::score_texts EXACTLY:
  - tokenizer from the adapter dir (saved at best-model time), padding_side
    "left", pad = eos if missing
  - base model from adapter_config.json["base_model_name_or_path"]
    (meta-llama/Llama-3.1-8B), AutoModelForSequenceClassification num_labels=1,
    bf16, device_map="auto"
  - PeftModel.from_pretrained(base, adapter); score = sigmoid(logit)
  - max_length 1024, truncation, dynamic padding (same as trainer inference)

Leakage check (train.csv has NO id column, so matching is by md5 of the
whitespace-normalized lowercased first 2000 chars of text):
  - in_dense_train_pool: text appears in the dense sweep's FULL train split
    (datasets/patents/patents_first_draft_cpc_balanced/train.csv)
  - in_dense_train: text appears in the ACTUAL training subsample used by the
    scored adapter, reproduced exactly as the trainer does:
    train_df.sample(n=max(1, int(round(len*frac))), random_state=42)
    with frac = --train-subset-percentage (0.7 for subset_0_7)
  - in_dense_eval / in_dense_test: same check vs eval.csv / test.csv
  CPU-validated 2026-06-12: 803 cohort dp -> 77 in full train pool, 52 in the
  0.7 subsample, 7 in eval, 10 in test, 709 in none (different source pool).

Output: runs/validity_full/v2/patents/dense_scores_dp.json — list of
  {dp_id, app_id, judgement, dense_score, in_dense_train,
   in_dense_train_pool, in_dense_eval, in_dense_test}
plus printed AUCs: all dp, and the non-overlap subsets.

Use --dry-run to run everything except model loading/scoring (CPU-safe).
"""
import argparse
import glob
import hashlib
import json
import logging
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE = "runs/validity_full/v2/patents"
CELLS_GLOB = "outputs/v2_db/cells_v1/task=patents/**/*.parquet"
JUDGE = "qwen_thinking_fp8_20x1_r2post"
SPLIT_DIR = "datasets/patents/patents_first_draft_cpc_balanced"
TRAINER_SEED = 42  # train_reward_model.py default --seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="runs/patents_first_draft_cpc_sweep_llama8b/subset_0_7/best_model",
                   help="LoRA adapter dir (best_model of a sweep subset)")
    p.add_argument("--out", default=f"{BASE}/dense_scores_dp.json")
    p.add_argument("--max-length", type=int, default=1024,
                   help="Must match the sweep's --max_length (1024)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-subset-percentage", type=float, default=0.7,
                   help="The --train_subset_percentage the adapter was trained with "
                        "(0.7 for subset_0_7, 0.6 for subset_0_6, ...)")
    p.add_argument("--dry-run", action="store_true",
                   help="Build cohort + leakage flags only; skip model loading/scoring (CPU-safe)")
    return p.parse_args()


def norm_key(t: str, n: int = 2000) -> str:
    """Hash of normalized text prefix; robust to the 12000-char cap on dp texts."""
    return hashlib.md5(re.sub(r"\s+", " ", str(t)[:n]).strip().lower().encode()).hexdigest()


def build_cohort() -> List[dict]:
    """803 dp records {dp_id, app_id, judgement, text} in sorted dp_id order."""
    files = glob.glob(CELLS_GLOB, recursive=True)
    if not files:
        raise FileNotFoundError(f"No cells parquet under {CELLS_GLOB}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    cells = df[(df.judge_model == JUDGE) & df.applicable & df.score.notna()]
    cohort_ids = sorted(cells.datapoint_id.unique())
    logger.info("Judge matrix cohort: %d dp x %d aspects (judge=%s)",
                cells.datapoint_id.nunique(), cells.aspect_id.nunique(), JUDGE)

    dps = json.load(open(f"{BASE}/datapoints.json"))
    by_id = {d["datapoint_id"]: d for d in dps}
    pgpub: Dict[str, int] = json.load(open(f"{BASE}/dp_to_pgpub.json"))

    missing = [d for d in cohort_ids if d not in by_id]
    if missing:
        logger.warning("Dropping %d cohort dp not in datapoints.json: %s", len(missing), missing)
    records = []
    for d in cohort_ids:
        if d not in by_id:
            continue
        records.append({
            "dp_id": d,
            "app_id": pgpub.get(d),
            "judgement": int(by_id[d]["judgement"]),
            "text": by_id[d]["text"],
        })
    logger.info("Cohort joined: %d dp (%d with app_id) | labels: %s",
                len(records), sum(r["app_id"] is not None for r in records),
                pd.Series([r["judgement"] for r in records]).value_counts().to_dict())
    return records


def add_leakage_flags(records: List[dict], frac: float) -> None:
    """Flag overlap with the dense sweep's train pool / actual subsample / eval / test."""
    train_df = pd.read_csv(f"{SPLIT_DIR}/train.csv", usecols=["text"])
    # Reproduce the trainer's subsample EXACTLY (train_reward_model.py train()):
    #   subset_size = max(1, int(round(len(train_df) * frac)))
    #   train_df.sample(n=subset_size, random_state=args.seed)
    if frac is not None and frac < 1.0:
        n = max(1, int(round(len(train_df) * frac)))
        sub_df = train_df.sample(n=n, random_state=TRAINER_SEED)
    else:
        sub_df = train_df
    logger.info("Dense train pool: %d rows | actual training subsample (frac=%.2f): %d rows",
                len(train_df), frac, len(sub_df))
    pool_keys = set(train_df["text"].map(norm_key))
    sub_keys = set(sub_df["text"].map(norm_key))
    eval_keys = set(pd.read_csv(f"{SPLIT_DIR}/eval.csv", usecols=["text"])["text"].map(norm_key))
    test_keys = set(pd.read_csv(f"{SPLIT_DIR}/test.csv", usecols=["text"])["text"].map(norm_key))

    for r in records:
        k = norm_key(r["text"])
        r["in_dense_train_pool"] = k in pool_keys
        r["in_dense_train"] = k in sub_keys
        r["in_dense_eval"] = k in eval_keys
        r["in_dense_test"] = k in test_keys
    n_pool = sum(r["in_dense_train_pool"] for r in records)
    n_sub = sum(r["in_dense_train"] for r in records)
    n_ev = sum(r["in_dense_eval"] for r in records)
    n_te = sum(r["in_dense_test"] for r in records)
    logger.info("LEAKAGE: %d/%d in full train pool | %d in actual train subsample | "
                "%d in eval | %d in test | %d in no split",
                n_pool, len(records), n_sub, n_ev, n_te,
                sum(not (r["in_dense_train_pool"] or r["in_dense_eval"] or r["in_dense_test"])
                    for r in records))


def score_texts(adapter_dir: str, texts: List[str], max_length: int, batch_size: int) -> List[float]:
    """Mirror of methods/dense/train_reward_model.py::score_texts."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading tokenizer from %s", adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(adapter_config_path) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]
    logger.info("Loading base model %s (bf16, num_labels=1) + LoRA adapter", base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    scores: List[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)
            scores.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
        if (start // batch_size) % 10 == 0:
            logger.info("Scored %d/%d", min(start + batch_size, len(texts)), len(texts))
    return scores


def report_auc(records: List[dict]) -> None:
    from sklearn.metrics import roc_auc_score
    y = np.array([r["judgement"] for r in records])
    s = np.array([r["dense_score"] for r in records])
    logger.info("AUC all %d dp: %.4f", len(records), roc_auc_score(y, s))
    for flag, name in [("in_dense_train", "actual train subsample"),
                       ("in_dense_train_pool", "full train pool")]:
        keep = np.array([not r[flag] for r in records])
        if keep.sum() and len(set(y[keep])) == 2:
            logger.info("AUC excluding %s (%d dp kept): %.4f",
                        name, int(keep.sum()), roc_auc_score(y[keep], s[keep]))


def main():
    args = parse_args()
    records = build_cohort()
    add_leakage_flags(records, args.train_subset_percentage)

    if args.dry_run:
        logger.info("--dry-run: skipping model load + scoring. Staging validated.")
        return

    texts = [r["text"] for r in records]
    scores = score_texts(args.adapter, texts, args.max_length, args.batch_size)
    assert len(scores) == len(records)
    for r, s in zip(records, scores):
        r["dense_score"] = float(s)

    report_auc(records)

    out_records = [{k: r[k] for k in
                    ("dp_id", "app_id", "judgement", "dense_score", "in_dense_train",
                     "in_dense_train_pool", "in_dense_eval", "in_dense_test")}
                   for r in records]
    meta = {
        "adapter": args.adapter,
        "judge_cohort": JUDGE,
        "max_length": args.max_length,
        "train_subset_percentage": args.train_subset_percentage,
        "n_dp": len(out_records),
    }
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "scores": out_records}, f, indent=2)
    logger.info("Wrote %d scores to %s", len(out_records), args.out)


if __name__ == "__main__":
    main()
