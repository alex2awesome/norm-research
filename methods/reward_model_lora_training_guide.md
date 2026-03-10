# Fine-Tuning a Dense Reward Model with LoRA

## Overview

You are training a **scalar reward model** from a pretrained language model. The model takes a piece of text and outputs a single scalar score predicting human judgement (binary: 0 or 1). This is used as a dense reward signal in a reward function scaling laws experiment.

**Input**: A CSV file with columns `text` (the item being evaluated) and `judgement` (binary 0/1 label — e.g., did the paper get accepted, did the PR get merged, did the patent get granted).

**Output**: A fine-tuned model that, given new text, outputs a scalar reward score.

---

## Environment Setup

```bash
pip install torch transformers datasets peft accelerate bitsandbytes scikit-learn pandas --break-system-packages
```

For 70B models, you will also need:
```bash
pip install flash-attn --no-build-isolation --break-system-packages
```

### FSDP-ready Accelerate configs

Two reference configs live in `methods/fsdp_configs/` so you can shard 70B checkpoints with `accelerate` + PyTorch FSDP:

- `h200_fsdp.yaml`: 2×H200 (141 GB) with aggressive activation checkpointing/limit-all-gathers to stay under ~120 GB/GPU.
- `b200_fsdp.yaml`: 2×B200 (192 GB) with looser wrapping thresholds and forward prefetch to leverage the extra memory bandwidth.

Launch training with:

```bash
accelerate launch \
  --config_file methods/fsdp_configs/h200_fsdp.yaml \
  methods/train_reward_model.py \
  --data_path path/to/dataset.csv \
  --model_name meta-llama/Llama-3.1-70B \
  --quantize \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir runs/l70b_h200
```

Swap in the B200 config when running on that node class. Adjust batch/accumulation/sequence length as your memory headroom allows; FSDP + QLoRA keeps the base weights sharded/int4 while LoRA adapters train in bf16.

---

## Full Training Script

Create `train_reward_model.py`:

```python
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train a dense reward model with LoRA")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV with 'text' and 'judgement' columns")
    parser.add_argument("--max_length", type=int, default=1024, help="Max token length for input text")
    parser.add_argument("--val_fraction", type=float, default=0.15, help="Fraction of data for validation")

    # Model
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace model identifier")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization (QLoRA). Recommended for 70B.")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Modules to apply LoRA to")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="./reward_model_output")

    return parser.parse_args()


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

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Reward models use the LAST token's hidden state as the sequence representation.
    # For decoder-only models, we need left-padding so the last non-pad token is at the end.
    tokenizer.padding_side = "left"

    # Many decoder-only models don't have a pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Quantization config (for QLoRA on large models) ---
    quantization_config = None
    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # --- Load model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,               # Single scalar output (regression head)
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",          # Automatically distribute across GPUs
        trust_remote_code=True,
    )

    # Set pad_token_id on the model config so it knows which token to ignore.
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Prepare for quantized training if applicable ---
    if args.quantize:
        model = prepare_model_for_kbit_training(model)

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,            # Sequence classification (includes regression)
        r=args.lora_r,                         # Rank: controls capacity. 8-64 typical. Higher = more expressive but more params.
        lora_alpha=args.lora_alpha,            # Scaling: effective LR multiplier is alpha/r. Common: alpha = 2*r.
        lora_dropout=args.lora_dropout,        # Regularization. 0.05-0.1 typical.
        target_modules=args.lora_target_modules,
        bias="none",                           # Don't train bias terms
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Should show ~0.1-2% of params are trainable

    return model, tokenizer


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)

    assert "text" in df.columns and "judgement" in df.columns, \
        "CSV must have 'text' and 'judgement' columns"

    # Drop rows with missing values
    df = df.dropna(subset=["text", "judgement"])
    df["judgement"] = df["judgement"].astype(int)

    logger.info(f"Dataset size: {len(df)}")
    logger.info(f"Label distribution: {df['judgement'].value_counts().to_dict()}")

    # --- Split ---
    train_df, val_df = train_test_split(
        df, test_size=args.val_fraction, random_state=args.seed, stratify=df["judgement"]
    )
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # --- Build model ---
    model, tokenizer = build_model(args)

    # --- Create datasets ---
    train_dataset = RewardDataset(
        train_df["text"].tolist(), train_df["judgement"].tolist(), tokenizer, args.max_length
    )
    val_dataset = RewardDataset(
        val_df["text"].tolist(), val_df["judgement"].tolist(), tokenizer, args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Optimizer ---
    # Only optimize LoRA parameters (already handled by PEFT, but explicit filter is safer)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # --- Scheduler ---
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Loss ---
    # BCE with logits for binary classification from a single scalar output.
    loss_fn = nn.BCEWithLogitsLoss()

    # --- Training ---
    best_val_auc = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)  # Shape: (batch_size,)

            loss = loss_fn(logits, labels)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validation ---
        val_metrics = evaluate(model, val_loader, loss_fn)
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # --- Save best model ---
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            logger.info(f"New best AUC: {best_val_auc:.4f}. Saving model.")
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))

    # --- Save training history ---
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # --- Save final model ---
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

    logger.info(f"Training complete. Best Val AUC: {best_val_auc:.4f}")


def evaluate(model, dataloader, loss_fn):
    """Run evaluation and return loss, accuracy, and AUC."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, preds),
        "auc": roc_auc_score(all_labels, probs),
    }


# ─────────────────────────────────────────────
# 5. INFERENCE
# ─────────────────────────────────────────────

def score_texts(model_dir, texts, max_length=1024, batch_size=16):
    """
    Load a saved reward model and score a list of texts.
    Returns a list of scalar scores (probabilities after sigmoid).

    Usage:
        scores = score_texts("./reward_model_output/best_model", ["some text", "other text"])
    """
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Detect if the saved model is LoRA (adapter_config.json present) or full
    if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        with open(os.path.join(model_dir, "adapter_config.json")) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto",
        )

    model.eval()
    all_scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoding = tokenizer(
            batch_texts, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            all_scores.extend(probs.cpu().numpy().tolist())

    return all_scores


# ─────────────────────────────────────────────
# 6. PAIRWISE EVALUATION
# ─────────────────────────────────────────────

def pairwise_accuracy(model_dir, pairs_csv, max_length=1024):
    """
    Evaluate reward model on preference pairs.

    Input CSV must have columns: text_a, text_b, preferred
    where preferred is 'a' or 'b' (which text the human preferred).

    Returns: fraction of pairs where the model assigns a higher score
    to the preferred text.
    """
    df = pd.read_csv(pairs_csv)
    all_texts = df["text_a"].tolist() + df["text_b"].tolist()
    all_scores = score_texts(model_dir, all_texts, max_length)

    n = len(df)
    scores_a = all_scores[:n]
    scores_b = all_scores[n:]

    correct = 0
    for i in range(n):
        if df.iloc[i]["preferred"] == "a" and scores_a[i] > scores_b[i]:
            correct += 1
        elif df.iloc[i]["preferred"] == "b" and scores_b[i] > scores_a[i]:
            correct += 1

    return correct / n


if __name__ == "__main__":
    args = parse_args()
    train(args)
```

---

## Usage Examples

### Basic training (8B model, single GPU or multi-GPU)

```bash
python train_reward_model.py \
    --data_path data/academic_peer_review.csv \
    --model_name meta-llama/Llama-3.1-8B \
    --max_length 1024 \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir ./outputs/peer_review_8b
```

### QLoRA for 70B model

```bash
python train_reward_model.py \
    --data_path data/academic_peer_review.csv \
    --model_name meta-llama/Llama-3.1-70B \
    --quantize \
    --max_length 1024 \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --output_dir ./outputs/peer_review_70b
```

### Scaling experiment (vary training set size)

```bash
for N in 50 100 250 500 1000 2500 5000; do
    # Sample N rows from the full dataset (stratified)
    python -c "
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/full_dataset.csv')
sample, _ = train_test_split(df, train_size=$N, stratify=df['judgement'], random_state=42)
sample.to_csv('data/sample_${N}.csv', index=False)
"
    # Train
    python train_reward_model.py \
        --data_path data/sample_${N}.csv \
        --model_name meta-llama/Llama-3.1-8B \
        --epochs 5 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 \
        --seed 42 \
        --output_dir ./outputs/scaling_n${N}_seed42

    # Repeat with different seeds for variance estimation
    for SEED in 1 2 3 4; do
        python train_reward_model.py \
            --data_path data/sample_${N}.csv \
            --seed $SEED \
            --output_dir ./outputs/scaling_n${N}_seed${SEED} \
            --model_name meta-llama/Llama-3.1-8B \
            --epochs 5 \
            --batch_size 4 \
            --gradient_accumulation_steps 4 \
            --learning_rate 2e-4
    done
done
```

### Inference: score new texts

```python
from train_reward_model import score_texts

scores = score_texts(
    model_dir="./outputs/peer_review_8b/best_model",
    texts=["This paper presents a novel approach...", "We study the thing..."],
)
print(scores)  # e.g., [0.87, 0.23]
```

### Pairwise evaluation on held-out preference pairs

```python
from train_reward_model import pairwise_accuracy

acc = pairwise_accuracy(
    model_dir="./outputs/peer_review_8b/best_model",
    pairs_csv="data/held_out_pairs.csv",
)
print(f"Pairwise accuracy: {acc:.4f}")
```

---

## Design Decisions and Rationale

### Why `AutoModelForSequenceClassification` with `num_labels=1`?

This is the standard approach for scalar reward models. It adds a linear projection from the model's hidden dimension to a single scalar on top of the pretrained LM. With `num_labels=1`, HuggingFace treats this as regression (no softmax), so `outputs.logits` is a raw scalar. We apply sigmoid ourselves for binary classification and use `BCEWithLogitsLoss` for numerical stability.

### Why LoRA on attention AND MLP layers?

The default in many tutorials is to only target `q_proj` and `v_proj`. For reward modeling, we've included all attention projections plus the MLP gate/up/down projections. This gives the model more capacity to learn nuanced preference patterns. For your scaling experiment, you might want to ablate this — try attention-only LoRA vs. full LoRA to see if the extra capacity matters at different data scales.

### Why left-padding?

Decoder-only models (GPT-style, LLaMA) generate tokens left-to-right. The classification head uses the last token's hidden state as the sequence representation. If you right-pad, the "last token" is a pad token, which carries no information. Left-padding ensures the last token is always the final real token of the input.

### Key hyperparameters to tune for your scaling experiment

| Parameter | Low-data regime (n < 500) | High-data regime (n > 2000) |
|-----------|--------------------------|----------------------------|
| `lora_r` | 8 (less overfitting) | 16-64 (more capacity) |
| `epochs` | 5-10 (more passes needed) | 3 (sufficient) |
| `learning_rate` | 1e-4 (more conservative) | 2e-4 to 5e-4 |
| `weight_decay` | 0.05-0.1 (stronger reg.) | 0.01 |
| `lora_dropout` | 0.1 (stronger reg.) | 0.05 |

### Matching the dense model to the rubric judge

For your controlled comparison (same base model for both approaches), set `--model_name` to the same model you use as the LLM-as-a-judge. For example, if your rubric approach uses `meta-llama/Llama-3.1-8B-Instruct` as the judge, use `meta-llama/Llama-3.1-8B` (base, not instruct) as the reward model backbone. Using the base model is preferable for reward modeling since the instruct tuning can interfere with the reward head training.

### On early stopping

The script saves the best model by validation AUC. For your scaling experiment, you should report the best validation AUC (not final epoch) at each data budget. This ensures you're comparing the best each approach can do rather than measuring overfitting differences.

---

## Hardware Requirements (Approximate)

| Model | Method | GPU Memory | Recommended |
|-------|--------|-----------|-------------|
| 8B | LoRA (bf16) | ~20 GB | 1x A100 40GB |
| 8B | QLoRA (4-bit) | ~8 GB | 1x A10 24GB or RTX 4090 |
| 70B | QLoRA (4-bit) | ~40 GB | 1x A100 80GB |
| 70B | QLoRA (4-bit) | ~24 GB per GPU | 2x A100 40GB (device_map="auto") |
