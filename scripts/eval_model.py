#!/usr/bin/env python3
"""
Global evaluation runner for dense reward models.

Computes standard metrics on a dataset split, then discovers and runs
dataset-specific evaluations from datasets/<dataset>/evals.py.

Usage:
    python scripts/eval_model.py \
        --model_dir runs/peer_review_sweep_llama-8b/subset_1p0/best_model \
        --data_path datasets/peer-review/peer_review_modeling_dataset.csv.gz \
        --split test

    # Or point directly at a split CSV:
    python scripts/eval_model.py \
        --model_dir runs/peer_review_sweep_llama-8b/subset_1p0/best_model \
        --split_csv datasets/peer-review/peer_review_modeling_dataset/test.csv

    # Evaluate on all splits:
    python scripts/eval_model.py \
        --model_dir runs/peer_review_sweep_llama-8b/subset_1p0/best_model \
        --data_path datasets/peer-review/peer_review_modeling_dataset.csv.gz \
        --split all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "methods" / "dense"))

from eval_utils import compute_metrics, run_dataset_evals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(data_path: str, split: str, split_dir: str = None) -> pd.DataFrame:
    """Load a specific split, creating it if needed via the training code's logic."""
    dp = Path(data_path).resolve()
    if split_dir:
        sd = Path(split_dir).resolve()
    else:
        stem = dp.stem
        if stem.endswith(".csv"):
            stem = stem[: -len(".csv")]
        sd = dp.parent / stem

    split_path = sd / f"{split}.csv"
    gz_split_path = sd / f"{split}.csv.gz"

    if split_path.exists():
        logger.info("Loading %s split from %s", split, split_path)
        return pd.read_csv(split_path)
    elif gz_split_path.exists():
        logger.info("Loading %s split from %s", split, gz_split_path)
        return pd.read_csv(gz_split_path)

    # Fall back: create the split using the training code's logic
    logger.info("Split files not found at %s; creating fixed split from %s", sd, data_path)
    from train_reward_model import get_or_create_fixed_split

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "judgement"])
    df["judgement"] = df["judgement"].astype(int)

    class _Args:
        def __init__(self):
            self.data_path = str(dp)
            self.split_dir = str(sd)
            self.seed = 42
    train_df, eval_df, test_df = get_or_create_fixed_split(df, _Args())
    return {"train": train_df, "eval": eval_df, "test": test_df}[split]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(model_dir: str, df: pd.DataFrame, data_path: str, output_path: str = None,
             max_length: int = 1024, batch_size: int = 64):
    """Run full evaluation: standard metrics + dataset-specific sliced evals."""
    from train_reward_model import score_texts

    # --- Score all texts ---
    texts = df["text"].tolist()
    labels = df["judgement"].astype(int).to_numpy()
    logger.info("Scoring %d texts with model from %s", len(texts), model_dir)
    probs = np.array(score_texts(model_dir, texts, max_length=max_length, batch_size=batch_size))

    # --- Overall metrics ---
    overall = compute_metrics(labels, probs)
    logger.info(
        "Overall | n=%d | Loss: %.4f | Acc: %.4f | P: %.4f | R: %.4f | F1: %.4f | AUC: %.4f",
        overall["n"], overall["loss"], overall["accuracy"],
        overall["precision"], overall["recall"], overall["f1"], overall["auc"],
    )

    # --- Dataset-specific sliced evals ---
    ds_results = run_dataset_evals(
        df=df,
        labels=labels,
        probs=probs,
        data_path=data_path,
    )

    # Merge overall into results
    results = {"overall": overall}
    if "slices" in ds_results:
        results["slices"] = ds_results["slices"]
    if "summary" in ds_results:
        results["summary"] = ds_results["summary"]

    # --- Save results ---
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results written to %s", out)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a dense reward model with dataset-specific evals")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model (best_model/ directory)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the original dataset CSV (for split creation and evals.py discovery)")
    parser.add_argument("--split_csv", type=str, default=None, help="Direct path to a split CSV (overrides --data_path + --split)")
    parser.add_argument("--split", type=str, default="test", help="Which split to evaluate: train, eval, test, or all")
    parser.add_argument("--split_dir", type=str, default=None, help="Directory containing split CSVs")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write results JSON (defaults to model_dir)")
    parser.add_argument("--max_length", type=int, default=1024, help="Max token length")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    args = parser.parse_args()

    if not args.data_path and not args.split_csv:
        parser.error("Provide either --data_path or --split_csv")

    output_base = Path(args.output_dir or args.model_dir)
    data_path = args.data_path or args.split_csv

    splits_to_run = [args.split] if args.split != "all" else ["train", "eval", "test"]

    for split in splits_to_run:
        if args.split_csv and split == args.split:
            logger.info("Loading data from %s", args.split_csv)
            df = pd.read_csv(args.split_csv)
        else:
            df = load_split(data_path, split, args.split_dir)

        df = df.dropna(subset=["text", "judgement"])
        df["judgement"] = df["judgement"].astype(int)

        output_path = output_base / f"eval_results_{split}.json"
        run_eval(
            model_dir=args.model_dir,
            df=df,
            data_path=data_path,
            output_path=str(output_path),
            max_length=args.max_length,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
