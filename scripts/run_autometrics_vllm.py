#!/usr/bin/env python3
"""Run iterative AutoMetrics with VLLM backend on a single GPU."""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ──
REPO_ROOT = Path(__file__).resolve().parent.parent
AUTOMETRICS_PKG = REPO_ROOT / "methods" / "autometrics"

sys.path.insert(0, str(AUTOMETRICS_PKG))

# Clear cached autometrics modules
for k in list(sys.modules):
    if k == "autometrics" or k.startswith("autometrics."):
        del sys.modules[k]

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

import dspy
import pandas as pd

# Import run_iterative directly to avoid heavy Autometrics imports (pyserini/Java)
from autometrics.iterative_refinement.runner import run_iterative
from autometrics.dataset.Dataset import Dataset
from autometrics.backends import create_backend
from autometrics.task_descriptions import get_task_description

# ── Dataset configs ──
DATASET_CONFIGS = {
    "press-release": {
        "dataset_name": "PressReleaseModeling",
        "split_dir": REPO_ROOT / "datasets" / "press-releases" / "press_release_modeling_dataset.csv",
        "output_subdir": "press_release_vllm_70b",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    },
    "peer-review": {
        "dataset_name": "PeerReviewAcceptance",
        "split_dir": REPO_ROOT / "datasets" / "peer-review" / "peer_review_modeling_dataset",
        "output_subdir": "peer_review_vllm_70b",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    },
    "code-review": {
        "dataset_name": "CodeReviewAcceptance",
        "split_dir": REPO_ROOT / "datasets" / "code-review" / "code_review_dense_4096tok",
        "output_subdir": "code_review_vllm_70b",
        "id_column": "paper_id",
        "text_column": "text",
        "label_column": "judgement",
    },
    "notice-and-comment": {
        "dataset_name": "NoticeAndComment",
        "split_dir": REPO_ROOT / "datasets" / "notice-and-comment" / "notice_and_comment_len_balanced",
        "output_subdir": "notice_and_comment_vllm_70b",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
        "add_synthetic_id": True,
    },
}

# ── CLI args ──
parser = argparse.ArgumentParser(description="Run iterative AutoMetrics with VLLM backend")
parser.add_argument("--dataset", type=str, default="press-release",
                    choices=list(DATASET_CONFIGS.keys()),
                    help="Dataset to run on")
parser.add_argument("--disable-early-stopping", action="store_true",
                    help="Run all iterations without early stopping")
parser.add_argument("--num-iterations", type=int, default=25)
parser.add_argument("--early-stop-patience", type=int, default=2)
parser.add_argument("--num-metrics", type=int, default=5,
                    help="Number of single-dimension metrics to propose per iteration")
parser.add_argument("--num-rubrics", type=int, default=5,
                    help="Number of holistic rubrics to propose per iteration")
parser.add_argument("--no-interactions", action="store_true",
                    help="Disable pairwise interaction terms in the regression")
parser.add_argument("--max-interaction-metrics", type=int, default=8,
                    help="Max metrics that participate in interactions (default 8, caps at C(K,2) pairs)")
parser.add_argument("--model-type", type=str, default="logistic",
                    choices=["logistic", "gated_mlp"],
                    help="Aggregation model: 'logistic' (L1 LR) or 'gated_mlp' (gated interaction MLP)")
parser.add_argument("--gated-mlp-hidden-dim", type=int, default=64)
parser.add_argument("--gated-mlp-lambda-feature", type=float, default=0.1,
                    help="L1 penalty on feature gates (higher = more sparsity)")
parser.add_argument("--gated-mlp-lambda-interaction", type=float, default=0.05,
                    help="L1 penalty on interaction gates")
parser.add_argument("--gated-mlp-epochs", type=int, default=200)
parser.add_argument("--gated-mlp-gate-threshold", type=float, default=0.1,
                    help="Gate value below which a feature/interaction is considered inactive")
parser.add_argument("--eval-fraction", type=float, default=0.4,
                    help="Fraction of combined train+eval to use as eval")
parser.add_argument("--max-text-tokens", type=int, default=512,
                    help="Truncate text to first N whitespace tokens")
parser.add_argument("--eval-selection-max", type=int, default=10_000,
                    help="Max eval_selection samples per iteration (default 10000, 0=unlimited)")
parser.add_argument("--continue-run-id", type=str, default=None,
                    help="Resume from an existing run ID (reuses its label_cache)")
parser.add_argument("--balance-classes", action="store_true",
                    help="Downsample majority class to match minority class size")
parser.add_argument("--max-model-len", type=int, default=16384,
                    help="VLLM max model context length (default 16384)")
parser.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism (default 1)")
parser.add_argument("--proposer-model", type=str, default=None,
                    help="External API model for metric proposer (e.g., 'openai/gpt-5.4-mini'). "
                         "If not set, uses the VLLM backend for both scoring and proposing.")
args = parser.parse_args()

# ── Resolve dataset config ──
dcfg = DATASET_CONFIGS[args.dataset]
MODEL = os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
SPLIT_DIR = dcfg["split_dir"]
DATA_PATH = SPLIT_DIR / "train.csv.gz"
BASE_OUTPUT_DIR = REPO_ROOT / "outputs" / "iterative_autometrics" / dcfg["output_subdir"]

ID_COLUMN = dcfg["id_column"]
TEXT_COLUMN = dcfg["text_column"]
LABEL_COLUMN = dcfg["label_column"]
DATASET_NAME = dcfg["dataset_name"]

TASK_DESCRIPTION = get_task_description(DATASET_NAME)

# ── Resolve output dir with run ID ──
if args.continue_run_id:
    OUTPUT_DIR = BASE_OUTPUT_DIR / args.continue_run_id
    if not OUTPUT_DIR.exists():
        print(f"ERROR: Run directory does not exist: {OUTPUT_DIR}")
        sys.exit(1)
    # Archive old JSONL/JSON files, keep label_cache for reuse
    archive_dir = OUTPUT_DIR / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for f in list(OUTPUT_DIR.glob("*.jsonl")) + list(OUTPUT_DIR.glob("*.json")):
        shutil.move(str(f), str(archive_dir / f.name))
    print(f"Continuing run {args.continue_run_id} (archived previous outputs to {archive_dir})")
    print(f"  Label cache preserved at {OUTPUT_DIR / 'label_cache'}")
    run_id = args.continue_run_id
else:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = BASE_OUTPUT_DIR / run_id
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"New run: {run_id}")

print(f"Output dir: {OUTPUT_DIR}")

# ── Load data ──
print(f"Loading data from splits in {SPLIT_DIR} ...")
split_dfs = []
for split in ("train", "eval", "test"):
    for ext in (".csv.gz", ".csv"):
        p = SPLIT_DIR / f"{split}{ext}"
        if p.exists():
            split_dfs.append(pd.read_csv(p, low_memory=False))
            break
    else:
        raise FileNotFoundError(f"Missing {split} split in {SPLIT_DIR}")

df = pd.concat(split_dfs, ignore_index=True)

# Add synthetic ID column if needed (for datasets without a natural ID)
if dcfg.get("add_synthetic_id"):
    df[ID_COLUMN] = [f"row_{i}" for i in range(len(df))]

df = df.dropna(subset=[ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN])
print(f"Loaded {len(df)} rows")

# ── Balance classes by downsampling majority class ──
if args.balance_classes:
    pos = df[df[LABEL_COLUMN] == 1]
    neg = df[df[LABEL_COLUMN] == 0]
    minority_size = min(len(pos), len(neg))
    majority_label = 1 if len(pos) > len(neg) else 0
    print(f"Balancing classes: {len(pos)} positive, {len(neg)} negative → {minority_size} each")
    if len(pos) > len(neg):
        pos = pos.sample(n=minority_size, random_state=42)
    else:
        neg = neg.sample(n=minority_size, random_state=42)
    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced dataset: {len(df)} rows ({(df[LABEL_COLUMN]==1).sum()} pos, {(df[LABEL_COLUMN]==0).sum()} neg)")

dataset = Dataset(
    dataframe=df,
    target_columns=[LABEL_COLUMN],
    ignore_columns=[ID_COLUMN],
    metric_columns=[],
    name=DATASET_NAME,
    data_id_column=ID_COLUMN,
    input_column=TEXT_COLUMN,
    output_column=TEXT_COLUMN,
    reference_columns=[],
    metrics=[],
    task_description=TASK_DESCRIPTION,
)

# ── Create VLLM backend ──
print(f"Initializing VLLM backend with model: {MODEL}")
backend = create_backend(
    "vllm",
    model_name_or_path=MODEL,
    tensor_parallel_size=args.tensor_parallel_size,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=0.95,
    attention_config={"backend": "FLASH_ATTN"},
)
print("VLLM backend ready.")

# ── Generator/Judge LLM ──
if args.proposer_model:
    # Use external API model for metric proposer (e.g., GPT-5.4 mini)
    print(f"Using external proposer model: {args.proposer_model}")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        key_file = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY found in env or key file")
    os.environ["OPENAI_API_KEY"] = api_key
    generator_llm = dspy.LM(model=args.proposer_model)
else:
    generator_llm = dspy.LM(model=f"openai/{MODEL}", api_key="unused", api_base="http://localhost:1")
judge_llm = generator_llm

# ── Run ──
print("Starting iterative AutoMetrics ...")
results = run_iterative(
    dataset=dataset,
    target_measure=LABEL_COLUMN,
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    num_iterations=args.num_iterations,
    disable_early_stopping=args.disable_early_stopping,
    early_stop_patience=args.early_stop_patience,
    data_path=str(DATA_PATH),
    split_dir=str(SPLIT_DIR),
    id_column=ID_COLUMN,
    text_column=TEXT_COLUMN,
    label_column=LABEL_COLUMN,
    output_dir=str(OUTPUT_DIR),
    k_pairs=5,
    num_metrics=args.num_metrics,
    num_rubrics=args.num_rubrics,
    label_batch_size=200,
    verbose=True,
    eval_fraction=args.eval_fraction,
    eval_gate_fraction=0.2,
    max_text_tokens=args.max_text_tokens,
    eval_selection_max=args.eval_selection_max or None,
    tqdm_scoring=True,
    scoring_backend=backend,
    use_interactions=not args.no_interactions,
    max_interaction_metrics=args.max_interaction_metrics,
    model_type=args.model_type,
    gated_mlp_hidden_dim=args.gated_mlp_hidden_dim,
    gated_mlp_lambda_feature=args.gated_mlp_lambda_feature,
    gated_mlp_lambda_interaction=args.gated_mlp_lambda_interaction,
    gated_mlp_epochs=args.gated_mlp_epochs,
    gated_mlp_gate_threshold=args.gated_mlp_gate_threshold,
    seed=42,
)

print(f"\n=== Done (run_id={run_id}) ===")
print(f"Output written to: {OUTPUT_DIR}")
for k, v in results.items():
    if isinstance(v, (int, float, str, bool)):
        print(f"  {k}: {v}")
