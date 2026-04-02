import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from eval_utils import compute_metrics, run_dataset_evals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FIXED_TRAIN_FRACTION = 0.8
FIXED_EVAL_FRACTION = 0.1
FIXED_TEST_FRACTION = 0.1
EVALS_PER_EPOCH = 5


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train a dense reward model with LoRA")

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CSV with 'text' and 'judgement' columns",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max token length for input text",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Deprecated. Ignored because this script uses a fixed 80/10/10 train/eval/test split on disk.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Directory containing persistent train/eval/test split CSVs. Defaults to a split folder next to data_path.",
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization (QLoRA). Recommended for 70B.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce GPU memory at the cost of ~30%% slower training.",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        help="Modules to apply LoRA to",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=100,
        help="Batch size for validation/evaluation dataloader.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--train_subset_size",
        type=int,
        default=None,
        help="Limit number of training examples (after split) to this value for scaling tests.",
    )
    parser.add_argument(
        "--train_subset_percentage",
        type=float,
        default=None,
        help="Limit number of training examples (after split) to this fraction in (0, 1].",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Log running training loss every N optimizer steps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class_weight_auto",
        "--class-weight-auto",
        action="store_true",
        help="Automatically compute pos_weight for BCEWithLogitsLoss to handle class imbalance. "
             "Weight = num_neg / num_pos.",
    )
    parser.add_argument(
        "--bradley-terry",
        "--bradley_terry",
        action="store_true",
        help="Train with pairwise Bradley-Terry objective using sampled (label=1, label=0) pairs each epoch.",
    )
    parser.add_argument(
        "--use_optuna",
        "--use-optuna",
        action="store_true",
        help="Enable Optuna hyperparameter search instead of a single training run.",
    )
    parser.add_argument(
        "--optuna_trials",
        type=int,
        default=10,
        help="Number of Optuna trials when --use_optuna is enabled.",
    )
    parser.add_argument(
        "--optuna_timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds.",
    )
    parser.add_argument(
        "--optuna_study_name",
        type=str,
        default="reward_model_optuna",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--optuna_storage",
        type=str,
        default=None,
        help="Optuna storage URL. Defaults to sqlite in output_dir.",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./reward_model_output", help="Output directory"
    )

    args = parser.parse_args()
    if args.train_subset_size is not None and args.train_subset_percentage is not None:
        raise ValueError("Specify only one of --train_subset_size or --train_subset_percentage.")
    if args.train_subset_percentage is not None and not (0 < args.train_subset_percentage <= 1.0):
        raise ValueError("--train_subset_percentage must be in the interval (0, 1].")
    return args


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────


class RewardDataset(Dataset):
    """
    Simple dataset: tokenize text, return input_ids, attention_mask, and label.
    The label is the binary judgement (0 or 1).
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class BradleyTerryPairDataset(Dataset):
    """Dataset of paired examples for Bradley-Terry training."""

    def __init__(self, positive_texts, negative_texts, tokenizer, max_length):
        self.positive_texts = positive_texts
        self.negative_texts = negative_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.positive_texts)

    def __getitem__(self, idx):
        pos = self.tokenizer(
            self.positive_texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        neg = self.tokenizer(
            self.negative_texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "pos_input_ids": pos["input_ids"].squeeze(0),
            "pos_attention_mask": pos["attention_mask"].squeeze(0),
            "neg_input_ids": neg["input_ids"].squeeze(0),
            "neg_attention_mask": neg["attention_mask"].squeeze(0),
        }


def setup_run_logger(output_dir: str) -> str:
    """Attach a file handler so logs are persisted to disk."""
    log_path = os.path.join(output_dir, "training_run.log")
    os.makedirs(output_dir, exist_ok=True)
    for handler in list(logger.handlers):
        if getattr(handler, "_is_run_log", False):
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    # mark handler to avoid duplicates if train() called again
    file_handler._is_run_log = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)
    logger.info("File logging enabled at %s", log_path)
    return log_path


def _default_split_dir(data_path: str) -> Path:
    """Choose a deterministic on-disk location for the fixed split."""
    data_file = Path(data_path).resolve()
    stem = data_file.stem
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    return data_file.parent / stem


def get_or_create_fixed_split(df: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load existing on-disk train/eval/test split or create it once and persist it."""
    split_dir = Path(args.split_dir).resolve() if args.split_dir else _default_split_dir(args.data_path)
    train_path = split_dir / "train.csv"
    eval_path = split_dir / "eval.csv"
    test_path = split_dir / "test.csv"
    metadata_path = split_dir / "split_metadata.json"

    split_exists = train_path.exists() and eval_path.exists() and test_path.exists()
    if split_exists:
        logger.info("Using existing on-disk split from %s", split_dir)
        train_df = pd.read_csv(train_path)
        eval_df = pd.read_csv(eval_path)
        test_df = pd.read_csv(test_path)
        total = max(1, len(train_df) + len(eval_df) + len(test_df))
        observed_train_fraction = len(train_df) / total
        observed_eval_fraction = len(eval_df) / total
        observed_test_fraction = len(test_df) / total
        if (
            not np.isclose(observed_train_fraction, FIXED_TRAIN_FRACTION, atol=2e-2)
            or not np.isclose(observed_eval_fraction, FIXED_EVAL_FRACTION, atol=2e-2)
            or not np.isclose(observed_test_fraction, FIXED_TEST_FRACTION, atol=2e-2)
        ):
            raise RuntimeError(
                "Existing split ratios do not match expected train/eval/test 80/10/10 "
                f"(observed train={observed_train_fraction:.4f}, eval={observed_eval_fraction:.4f}, "
                f"test={observed_test_fraction:.4f}) in {split_dir}."
            )
        return train_df, eval_df, test_df

    existing_count = sum(int(p.exists()) for p in (train_path, eval_path, test_path))
    if 0 < existing_count < 3:
        raise RuntimeError(
            f"Incomplete split in {split_dir}. Expected {train_path.name}, {eval_path.name}, and {test_path.name}."
        )

    split_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Creating fixed train/eval/test split (train=%.0f%%, eval=%.0f%%, test=%.0f%%) in %s",
        FIXED_TRAIN_FRACTION * 100.0,
        FIXED_EVAL_FRACTION * 100.0,
        FIXED_TEST_FRACTION * 100.0,
        split_dir,
    )
    stratify_full = df["judgement"] if df["judgement"].nunique() > 1 else None
    train_eval_df, test_df = train_test_split(
        df,
        test_size=FIXED_TEST_FRACTION,
        random_state=args.seed,
        stratify=stratify_full,
    )
    eval_fraction_within_remaining = FIXED_EVAL_FRACTION / (FIXED_TRAIN_FRACTION + FIXED_EVAL_FRACTION)
    stratify_train_eval = train_eval_df["judgement"] if train_eval_df["judgement"].nunique() > 1 else None
    train_df, eval_df = train_test_split(
        train_eval_df,
        test_size=eval_fraction_within_remaining,
        random_state=args.seed,
        stratify=stratify_train_eval,
    )
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    test_df.to_csv(test_path, index=False)
    metadata = {
        "data_path": str(Path(args.data_path).resolve()),
        "seed": args.seed,
        "train_fraction": FIXED_TRAIN_FRACTION,
        "eval_fraction": FIXED_EVAL_FRACTION,
        "test_fraction": FIXED_TEST_FRACTION,
        "train_size": int(len(train_df)),
        "eval_size": int(len(eval_df)),
        "test_size": int(len(test_df)),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote split files: %s, %s, and %s", train_path, eval_path, test_path)
    return train_df, eval_df, test_df


def sample_bt_pairs(train_df: pd.DataFrame, num_pairs: int, seed: int) -> Tuple[List[str], List[str], Dict[str, object]]:
    """Sample label-1 vs label-0 pairs, with replacement as needed."""
    positives = train_df[train_df["judgement"] == 1]["text"].tolist()
    negatives = train_df[train_df["judgement"] == 0]["text"].tolist()
    if not positives or not negatives:
        raise ValueError(
            "Bradley-Terry mode requires both label=1 and label=0 in the training split. "
            "Try a larger subset or disable --bradley-terry."
        )

    rng = np.random.default_rng(seed)
    pos_replace = len(positives) < num_pairs
    neg_replace = len(negatives) < num_pairs
    pos_indices = rng.choice(len(positives), size=num_pairs, replace=pos_replace)
    neg_indices = rng.choice(len(negatives), size=num_pairs, replace=neg_replace)
    pos_texts = [positives[i] for i in pos_indices]
    neg_texts = [negatives[i] for i in neg_indices]
    stats: Dict[str, object] = {
        "available_positives": len(positives),
        "available_negatives": len(negatives),
        "pairs_per_epoch": num_pairs,
        "positive_replacement": pos_replace,
        "negative_replacement": neg_replace,
    }
    return pos_texts, neg_texts, stats


def _eval_trigger_steps(optimizer_steps_in_epoch: int) -> set[int]:
    """Return optimizer-step indices to run eval at ~20%,40%,60%,80%,100% of epoch."""
    if optimizer_steps_in_epoch <= 0:
        return set()
    return {int(np.ceil((i * optimizer_steps_in_epoch) / EVALS_PER_EPOCH)) for i in range(1, EVALS_PER_EPOCH + 1)}


# ─────────────────────────────────────────────
# 3. MODEL SETUP
# ─────────────────────────────────────────────


def build_model(args):
    """
    Load a pretrained LM, add a scalar classification head (num_labels=1),
    apply LoRA, and return the model + tokenizer.

    Architecture:
        [pretrained LM backbone] -> [last hidden state of final token] -> [linear head] -> scalar

    Using AutoModelForSequenceClassification with num_labels=1 gives us a
    regression head (single scalar output) instead of classification logits.
    We train this with BCE loss against the binary label.
    """

    logger.info("Loading tokenizer for %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Tokenizer ready (pad_token_id=%s)", tokenizer.pad_token_id)

    quantization_config = None
    if args.quantize:
        logger.info("Enabling 4-bit quantization (QLoRA)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        logger.info("Running in bf16 without quantization")

    # When running under FSDP (via accelerate), let FSDP handle device placement.
    # device_map="auto" conflicts with FSDP sharding.
    use_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
    device_map = None if use_fsdp else "auto"

    logger.info("Loading base model %s (device_map=%s)", args.model_name, device_map)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.quantize:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )

    logger.info(
        "Applying LoRA (r=%s, alpha=%s, dropout=%.3f) to modules: %s",
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        ", ".join(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    return model, tokenizer


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info("Training configuration: %s", json.dumps({k: str(v) for k, v in vars(args).items()}))

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = setup_run_logger(args.output_dir)
    validation_metrics_path = os.path.join(args.output_dir, "validation_metrics.csv")
    with open(validation_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "global_step",
                "train_batch",
                "val_loss",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_auc",
            ]
        )

    logger.info("[Stage] Loading dataset from %s", args.data_path)
    df = pd.read_csv(args.data_path)
    assert {"text", "judgement"}.issubset(df.columns), "CSV must have text and judgement columns"

    df = df.dropna(subset=["text", "judgement"])
    df["judgement"] = df["judgement"].astype(int)
    logger.info("Dataset size: %d", len(df))
    logger.info("Label distribution: %s", df["judgement"].value_counts().to_dict())

    logger.info("[Stage] Resolving fixed train/eval/test split on disk")
    train_df, eval_df, test_df = get_or_create_fixed_split(df, args)
    original_train_size = len(train_df)
    if args.train_subset_percentage is not None and args.train_subset_percentage < 1.0:
        subset_size = max(1, int(round(len(train_df) * args.train_subset_percentage)))
        logger.info(
            "Applying train subset percentage: requested %.3f of %d -> %d examples",
            args.train_subset_percentage,
            len(train_df),
            subset_size,
        )
        train_df = train_df.sample(n=subset_size, random_state=args.seed)
    if args.train_subset_size is not None and args.train_subset_size < len(train_df):
        logger.info(
            "Applying train subset limit: requested %d examples from %d",
            args.train_subset_size,
            len(train_df),
        )
        train_df = train_df.sample(n=args.train_subset_size, random_state=args.seed)
    train_size = len(train_df)
    eval_size = len(eval_df)
    test_size = len(test_df)
    logger.info("Train: %d (original %d) | Eval: %d | Test: %d", train_size, original_train_size, eval_size, test_size)
    logger.info("Training objective mode=%s", "bradley-terry" if args.bradley_terry else "pointwise")
    if args.bradley_terry and train_df["judgement"].nunique() < 2:
        raise ValueError(
            "Bradley-Terry mode requires both label classes in train split. "
            "Current train split has one class after subsetting."
        )
    selection_split_name = "eval" if getattr(args, "is_optuna_trial", False) else "test"
    selection_df = eval_df if selection_split_name == "eval" else test_df
    logger.info("Model-selection split: %s (eval reserved for Optuna trials)", selection_split_name)

    logger.info("[Stage] Building model + tokenizer")
    model, tokenizer = build_model(args)

    logger.info("[Stage] Building torch datasets")
    selection_dataset = RewardDataset(
        selection_df["text"].tolist(),
        selection_df["judgement"].tolist(),
        tokenizer,
        args.max_length,
    )

    logger.info("[Stage] Creating dataloaders")
    selection_loader = DataLoader(selection_dataset, batch_size=args.eval_batch_size, shuffle=False)
    if args.bradley_terry:
        pairs_per_epoch = max(1, len(train_df))
        train_batches_per_epoch = max(1, int(np.ceil(pairs_per_epoch / args.batch_size)))
        logger.info(
            "BT mode dataloaders | Pairs/epoch: %d | Est. train batches/epoch: %d | %s loader batches: %d | Train batch size (pairs): %d | Eval batch size: %d",
            pairs_per_epoch,
            train_batches_per_epoch,
            selection_split_name.capitalize(),
            len(selection_loader),
            args.batch_size,
            args.eval_batch_size,
        )
        pointwise_train_loader = None
    else:
        train_dataset = RewardDataset(
            train_df["text"].tolist(),
            train_df["judgement"].tolist(),
            tokenizer,
            args.max_length,
        )
        pointwise_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        train_batches_per_epoch = len(pointwise_train_loader)
        logger.info(
            "Train loader batches: %d | %s loader batches: %d | Train batch size: %d | Eval batch size: %d",
            len(pointwise_train_loader),
            selection_split_name.capitalize(),
            len(selection_loader),
            args.batch_size,
            args.eval_batch_size,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("[Stage] Initializing optimizer/scheduler (%d trainable tensors)", len(trainable_params))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = max(1, (train_batches_per_epoch // args.gradient_accumulation_steps) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(
        "Total optimizer steps: %d | Warmup steps: %d | Grad accum: %d",
        total_steps,
        warmup_steps,
        args.gradient_accumulation_steps,
    )

    if args.class_weight_auto:
        num_pos = train_df["judgement"].sum()
        num_neg = len(train_df) - num_pos
        pos_weight_val = num_neg / max(num_pos, 1)
        logger.info("Class weight auto: %d pos, %d neg -> pos_weight=%.3f", num_pos, num_neg, pos_weight_val)
        pos_weight = torch.tensor([pos_weight_val], device=model.device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    best_test_auc = 0.0
    best_epoch = 0
    history: List[Dict[str, float]] = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        running_loss = 0.0
        steps_since_log = 0
        epoch_optimizer_step = 0
        logger.info("Starting epoch %d/%d", epoch + 1, args.epochs)
        if args.bradley_terry:
            pos_texts, neg_texts, bt_stats = sample_bt_pairs(
                train_df=train_df,
                num_pairs=pairs_per_epoch,
                seed=args.seed + epoch,
            )
            logger.info(
                "BT epoch %d/%d | positives=%d negatives=%d pairs=%d | replacement pos=%s neg=%s",
                epoch + 1,
                args.epochs,
                bt_stats["available_positives"],
                bt_stats["available_negatives"],
                bt_stats["pairs_per_epoch"],
                bt_stats["positive_replacement"],
                bt_stats["negative_replacement"],
            )
            bt_train_dataset = BradleyTerryPairDataset(
                pos_texts,
                neg_texts,
                tokenizer,
                args.max_length,
            )
            train_loader = DataLoader(bt_train_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = pointwise_train_loader
            if train_loader is None:
                raise RuntimeError("Pointwise train loader is unexpectedly None.")
        optimizer_steps_this_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
        eval_trigger_steps = _eval_trigger_steps(optimizer_steps_this_epoch)
        logger.info(
            "Validation cadence this epoch: %d checkpoints at optimizer steps %s (%.0f%% intervals)",
            len(eval_trigger_steps),
            sorted(eval_trigger_steps),
            100.0 / EVALS_PER_EPOCH,
        )
        last_eval_metrics = None
        last_eval_epoch_step = None

        for step, batch in enumerate(train_loader):
            if args.bradley_terry:
                pos_input_ids = batch["pos_input_ids"].to(model.device)
                pos_attention_mask = batch["pos_attention_mask"].to(model.device)
                neg_input_ids = batch["neg_input_ids"].to(model.device)
                neg_attention_mask = batch["neg_attention_mask"].to(model.device)

                pos_outputs = model(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
                neg_outputs = model(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
                pair_logits = pos_outputs.logits.squeeze(-1) - neg_outputs.logits.squeeze(-1)
                pair_targets = torch.ones_like(pair_logits)
                loss = loss_fn(pair_logits, pair_targets) / args.gradient_accumulation_steps
            else:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits, labels) / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            running_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_optimizer_step += 1
                steps_since_log += 1
                should_log = steps_since_log >= args.log_every or global_step == 1
                if should_log:
                    avg_recent_loss = running_loss / max(1, steps_since_log)
                    batch_progress = (step + 1) / max(1, len(train_loader))
                    logger.info(
                        "Epoch %d Step %d | Batch %d/%d (%.1f%%) | Recent avg loss (%d steps): %.4f",
                        epoch + 1,
                        global_step,
                        step + 1,
                        len(train_loader),
                        batch_progress * 100.0,
                        steps_since_log,
                        avg_recent_loss,
                    )
                    running_loss = 0.0
                    steps_since_log = 0

                if epoch_optimizer_step in eval_trigger_steps:
                    logger.info(
                        "Running validation at epoch optimizer step %d/%d (global step %d)",
                        epoch_optimizer_step,
                        optimizer_steps_this_epoch,
                        global_step,
                    )
                    val_metrics = evaluate(model, selection_loader, loss_fn)
                    logger.info(
                        "Val @ step %d | Loss: %.4f | Acc: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
                        global_step,
                        val_metrics["loss"],
                        val_metrics["accuracy"],
                        val_metrics["precision"],
                        val_metrics["recall"],
                        val_metrics["f1"],
                        val_metrics["auc"],
                    )
                    with open(validation_metrics_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                epoch + 1,
                                global_step,
                                step + 1,
                                f"{val_metrics['loss']:.6f}",
                                f"{val_metrics['accuracy']:.6f}",
                                f"{val_metrics['precision']:.6f}",
                                f"{val_metrics['recall']:.6f}",
                                f"{val_metrics['f1']:.6f}",
                                f"{val_metrics['auc']:.6f}",
                            ]
                        )
                    last_eval_metrics = val_metrics
                    last_eval_epoch_step = epoch_optimizer_step
                    model.train()

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        if running_loss > 0 and steps_since_log > 0:
            logger.info(
                "Epoch %d residual window (%d steps) | Avg loss: %.4f",
                epoch + 1,
                steps_since_log,
                running_loss / steps_since_log,
            )

        if last_eval_metrics is not None and last_eval_epoch_step == epoch_optimizer_step:
            logger.info(
                "Using already-computed validation metrics at epoch end from step %d/%d",
                last_eval_epoch_step,
                optimizer_steps_this_epoch,
            )
            test_metrics = last_eval_metrics
        else:
            logger.info("Evaluating on %s split at epoch end...", selection_split_name)
            test_metrics = evaluate(model, selection_loader, loss_fn)
        logger.info(
            "Epoch %s/%s | Train Loss: %.4f | %s Loss: %.4f | %s Acc: %.4f | %s Precision: %.4f | %s Recall: %.4f | %s F1: %.4f | %s AUC: %.4f",
            epoch + 1,
            args.epochs,
            avg_train_loss,
            selection_split_name.capitalize(),
            test_metrics["loss"],
            selection_split_name.capitalize(),
            test_metrics["accuracy"],
            selection_split_name.capitalize(),
            test_metrics["precision"],
            selection_split_name.capitalize(),
            test_metrics["recall"],
            selection_split_name.capitalize(),
            test_metrics["f1"],
            selection_split_name.capitalize(),
            test_metrics["auc"],
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                **{f"{selection_split_name}_{k}": v for k, v in test_metrics.items()},
            }
        )

        if test_metrics["auc"] > best_test_auc:
            best_test_auc = test_metrics["auc"]
            best_epoch = epoch + 1
            logger.info("New best %s AUC: %.4f at epoch %d. Saving model.", selection_split_name, best_test_auc, best_epoch)
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Wrote training history (%d epochs) to %s", len(history), args.output_dir)
    logger.info("Wrote step-level validation metrics to %s", validation_metrics_path)

    final_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(
        "Training complete | Train samples: %d | Eval samples: %d | Test samples: %d | Best %s AUC: %.4f (epoch %d) | Final artifacts: %s | Log file: %s",
        train_size,
        eval_size,
        test_size,
        selection_split_name,
        best_test_auc,
        best_epoch,
        final_dir,
        log_path,
    )

    # Run dataset-specific sliced evaluation on best model
    best_model_dir = os.path.join(args.output_dir, "best_model")
    if os.path.isdir(best_model_dir):
        logger.info("[Stage] Running dataset-specific evaluations on %s split", selection_split_name)
        sel_probs = np.array(
            score_texts(best_model_dir, selection_df["text"].tolist(),
                        max_length=args.max_length, batch_size=args.eval_batch_size)
        )
        sel_labels = selection_df["judgement"].astype(int).to_numpy()
        dataset_eval_results = run_dataset_evals(
            df=selection_df,
            labels=sel_labels,
            probs=sel_probs,
            data_path=args.data_path,
            output_path=os.path.join(args.output_dir, f"dataset_eval_{selection_split_name}.json"),
        )
    else:
        dataset_eval_results = {}

    # Free GPU memory between Optuna trials
    del model, optimizer, scheduler, trainable_params
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("GPU memory released after trial cleanup")

    return {
        "best_selection_auc": float(best_test_auc),
        "best_epoch": int(best_epoch),
        "selection_split": selection_split_name,
        "best_model_dir": os.path.join(args.output_dir, "best_model"),
        "final_model_dir": final_dir,
        "train_size": int(train_size),
        "eval_size": int(eval_size),
        "test_size": int(test_size),
        "output_dir": args.output_dir,
    }


def _trial_args(args, trial, output_dir: str) -> argparse.Namespace:
    trial_cfg = vars(args).copy()
    trial_cfg["use_optuna"] = False
    trial_cfg["is_optuna_trial"] = True
    trial_cfg["output_dir"] = output_dir
    # For quantized (large) models, lock batch_size=1 to avoid OOM and tune
    # gradient_accumulation_steps for effective batch size instead.
    if args.quantize:
        trial_cfg["batch_size"] = 1
        trial_cfg["gradient_accumulation_steps"] = trial.suggest_categorical(
            "gradient_accumulation_steps", [4, 8, 16, 32]
        )
        trial_cfg["max_length"] = trial.suggest_categorical("max_length", [256, 512, 1024])
    else:
        # Keep batch_size small to avoid OOM; use gradient_accumulation_steps
        # to control effective batch size instead (equivalent, but constant memory).
        trial_cfg["batch_size"] = trial.suggest_categorical("batch_size", [1, 2])
        trial_cfg["gradient_accumulation_steps"] = trial.suggest_categorical(
            "gradient_accumulation_steps", [2, 4, 8, 16, 32]
        )
        trial_cfg["max_length"] = trial.suggest_categorical("max_length", [256, 512, 1024])
    trial_cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    trial_cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    trial_cfg["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    # lora_r beyond 16 rarely helps; keep alpha ~ 2*r as a common heuristic
    trial_cfg["lora_r"] = trial.suggest_categorical("lora_r", [8, 16, 32])
    trial_cfg["lora_alpha"] = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    trial_cfg["lora_dropout"] = trial.suggest_float("lora_dropout", 0.0, 0.2)
    module_subsets = _build_lora_target_module_subsets(args.lora_target_modules)
    module_subset_name = trial.suggest_categorical("lora_target_modules_subset", list(module_subsets.keys()))
    trial_cfg["lora_target_modules"] = module_subsets[module_subset_name]
    return argparse.Namespace(**trial_cfg)


def _build_lora_target_module_subsets(base_modules: List[str]) -> Dict[str, List[str]]:
    """Build named LoRA target module subsets for Optuna search."""
    unique_modules = list(dict.fromkeys(base_modules))
    attention_like = [m for m in unique_modules if m in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    mlp_like = [m for m in unique_modules if m in {"gate_proj", "up_proj", "down_proj"}]
    qv_only = [m for m in unique_modules if m in {"q_proj", "v_proj"}]

    subsets: Dict[str, List[str]] = {"all_modules": unique_modules}
    if attention_like:
        subsets["attention_only"] = attention_like
    if mlp_like:
        subsets["mlp_only"] = mlp_like
    if qv_only:
        subsets["qv_only"] = qv_only
    return subsets


def _metrics_from_probs(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Compute binary metrics from probabilities.

    Delegates to eval_utils.compute_metrics (single source of truth).
    """
    m = compute_metrics(labels, probs)
    # Return only the original keys for backward compatibility
    return {k: m[k] for k in ("loss", "accuracy", "precision", "recall", "f1", "auc")}


def evaluate_saved_model_on_split(model_dir: str, split_df: pd.DataFrame, args, split_name: str) -> Dict[str, float]:
    """Evaluate a saved checkpoint on a dataframe split and return metrics."""
    labels = split_df["judgement"].astype(int).to_numpy()
    probs = np.array(
        score_texts(
            model_dir,
            split_df["text"].tolist(),
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
        )
    )
    metrics = _metrics_from_probs(labels, probs)
    logger.info(
        "Best-trial %s metrics | Loss: %.4f | Acc: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
        split_name,
        metrics["loss"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc"],
    )
    return metrics


def run_optuna(args):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is not installed. Install it (e.g., pip install optuna) or run without --use_optuna."
        ) from exc

    os.makedirs(args.output_dir, exist_ok=True)
    storage = args.optuna_storage or f"sqlite:///{os.path.join(args.output_dir, 'optuna_study.db')}"
    study = optuna.create_study(
        study_name=args.optuna_study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    logger.info(
        "Starting Optuna study '%s' | trials=%d | timeout=%s | storage=%s",
        args.optuna_study_name,
        args.optuna_trials,
        str(args.optuna_timeout),
        storage,
    )

    def objective(trial):
        trial_output_dir = os.path.join(args.output_dir, f"trial_{trial.number:04d}")
        trial_run_args = _trial_args(args, trial, trial_output_dir)
        result = train(trial_run_args)
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("trial_output_dir", trial_output_dir)
        trial.set_user_attr("best_model_dir", result["best_model_dir"])
        auc = result["best_selection_auc"]
        if np.isnan(auc):
            return 0.0
        return float(auc)

    study.optimize(objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout)

    best_trial_output_dir = study.best_trial.user_attrs.get(
        "trial_output_dir", os.path.join(args.output_dir, f"trial_{study.best_trial.number:04d}")
    )
    best_model_dir = study.best_trial.user_attrs.get(
        "best_model_dir", os.path.join(best_trial_output_dir, "best_model")
    )
    if not os.path.isdir(best_model_dir):
        fallback_model_dir = os.path.join(best_trial_output_dir, "final_model")
        if not os.path.isdir(fallback_model_dir):
            raise FileNotFoundError(
                f"Could not find best checkpoint for best trial at {best_model_dir} or {fallback_model_dir}."
            )
        best_model_dir = fallback_model_dir

    logger.info("Evaluating best Optuna trial model on held-out test split from %s", best_model_dir)
    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=["text", "judgement"])
    df["judgement"] = df["judgement"].astype(int)
    _, _, test_df = get_or_create_fixed_split(df, args)
    best_trial_test_metrics = evaluate_saved_model_on_split(best_model_dir, test_df, args, split_name="test")

    # Run dataset-specific sliced evaluation on test split
    logger.info("[Stage] Running dataset-specific evaluations on test split")
    test_probs = np.array(
        score_texts(best_model_dir, test_df["text"].tolist(),
                    max_length=args.max_length, batch_size=args.eval_batch_size)
    )
    test_labels = test_df["judgement"].astype(int).to_numpy()
    dataset_eval_results = run_dataset_evals(
        df=test_df,
        labels=test_labels,
        probs=test_probs,
        data_path=args.data_path,
        output_path=os.path.join(args.output_dir, "dataset_eval_test.json"),
    )

    summary = {
        "study_name": args.optuna_study_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_model_dir": best_model_dir,
        "best_trial_test_metrics": best_trial_test_metrics,
    }
    summary_path = os.path.join(args.output_dir, "optuna_best_params.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    test_metrics_path = os.path.join(args.output_dir, "optuna_best_trial_test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(
            {
                "best_trial_number": int(study.best_trial.number),
                "best_trial_model_dir": best_model_dir,
                "metrics": best_trial_test_metrics,
            },
            f,
            indent=2,
        )

    trials_path = os.path.join(args.output_dir, "optuna_trials.csv")
    study.trials_dataframe().to_csv(trials_path, index=False)
    logger.info(
        "Optuna complete | best eval AUC=%.4f | test AUC (best trial)=%.4f | summary=%s | test_metrics=%s | trials=%s",
        study.best_value,
        best_trial_test_metrics["auc"],
        summary_path,
        test_metrics_path,
        trials_path,
    )
    return summary


def evaluate(model, dataloader, loss_fn):
    """Run evaluation and return loss, accuracy, precision, recall, F1, and AUC."""

    model.eval()
    logger.info("Running evaluation over %d batches", len(dataloader))
    all_logits = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1).to(torch.float32)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            all_logits.extend(logits.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / max(1, len(dataloader)),
        "accuracy": accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "auc": auc,
    }


# ─────────────────────────────────────────────
# 5. INFERENCE
# ─────────────────────────────────────────────


def score_texts(model_dir, texts, max_length=1024, batch_size=16):
    """
    Load a saved reward model and score a list of texts.
    Returns a list of scalar scores (probabilities after sigmoid).
    """
    from peft import PeftModel

    logger.info("[Stage] Loading reward model from %s for scoring", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        logger.info("Detected LoRA adapter; loading base model and merging adapters")
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        logger.info("No adapter config found; loading plain model")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    all_scores: List[float] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            all_scores.extend(probs.float().cpu().numpy().tolist())

    return all_scores


# ─────────────────────────────────────────────
# 6. PAIRWISE EVALUATION
# ─────────────────────────────────────────────


def pairwise_accuracy(model_dir, pairs_csv, max_length=1024):
    """
    Evaluate reward model on preference pairs.

    Input CSV must have columns: text_a, text_b, preferred
    where preferred is 'a' or 'b'.
    """
    logger.info("[Stage] Pairwise evaluation using %s on %s", model_dir, pairs_csv)
    df = pd.read_csv(pairs_csv)
    all_texts = df["text_a"].tolist() + df["text_b"].tolist()
    all_scores = score_texts(model_dir, all_texts, max_length)

    n = len(df)
    scores_a = all_scores[:n]
    scores_b = all_scores[n:]

    correct = 0
    for i in range(n):
        preferred = df.iloc[i]["preferred"]
        if preferred == "a" and scores_a[i] > scores_b[i]:
            correct += 1
        elif preferred == "b" and scores_b[i] > scores_a[i]:
            correct += 1

    return correct / n if n else 0.0


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.use_optuna:
        run_optuna(cli_args)
    else:
        train(cli_args)
