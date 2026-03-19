#!/usr/bin/env python3
"""Run Metric Tree with VLLM backend — entry point following run_autometrics_vllm.py pattern."""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Paths ──
REPO_ROOT = Path(__file__).resolve().parent.parent
AUTOMETRICS_PKG = REPO_ROOT / "methods" / "autometrics"
METRIC_TREE_PKG = REPO_ROOT / "methods"

sys.path.insert(0, str(AUTOMETRICS_PKG))
sys.path.insert(0, str(METRIC_TREE_PKG))

# Clear cached modules
for k in list(sys.modules):
    if k == "autometrics" or k.startswith("autometrics."):
        del sys.modules[k]
    if k == "metric_tree" or k.startswith("metric_tree."):
        del sys.modules[k]

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

import dspy
import numpy as np
import pandas as pd

from autometrics.backends import create_backend
from autometrics.generator.ContrastiveRubricProposer import ContrastiveRubricProposer
from autometrics.iterative_refinement.label_cache import LabelCache
from autometrics.iterative_refinement.runner import _coerce_binary_labels
from autometrics.task_descriptions import get_task_description
from autometrics.util.splits import load_fixed_split

from metric_tree.config import TreeConfig
from metric_tree.tree_builder import build_metric_tree
from metric_tree.inference import predict_batch, predict_root_only
from metric_tree.ensemble import build_metric_tree_ensemble, ensemble_predict
from metric_tree.analysis import (
    analyze_tree_complexity,
    compute_articulability_gap,
    export_tree_summary,
    measure_depth_distribution,
)

# ── CLI args ──
parser = argparse.ArgumentParser(description="Run Metric Tree with VLLM backend")

# Dataset args
parser.add_argument("--data-path", type=str, default=None,
                    help="Path to full dataset CSV (used for split creation)")
parser.add_argument("--split-dir", type=str, default=None,
                    help="Directory containing train/eval/test CSVs")
parser.add_argument("--id-column", type=str, default="id")
parser.add_argument("--text-column", type=str, default="output")
parser.add_argument("--label-column", type=str, default="newsworthiness_score")
parser.add_argument("--dataset-name", type=str, default="PressReleaseModeling")

# Tree config
parser.add_argument("--max-depth", type=int, default=3)
parser.add_argument("--min-subset-size", type=int, default=20)
parser.add_argument("--n-metrics", type=int, default=5,
                    help="Number of metrics to propose per node")
parser.add_argument("--n-rubrics", type=int, default=5,
                    help="Number of rubrics to propose per node")
parser.add_argument("--no-interactions", action="store_true",
                    help="Disable pairwise interaction terms")
parser.add_argument("--use-learned-router", action="store_true",
                    help="Use learned routing instead of confidence-based")
parser.add_argument("--confidence-threshold", type=float, default=0.7)
parser.add_argument("--eval-fraction", type=float, default=0.4)
parser.add_argument("--max-text-tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)

# Ensemble
parser.add_argument("--n-trees", type=int, default=1,
                    help="Number of trees in ensemble (1 = single tree)")
parser.add_argument("--no-ensemble", action="store_true",
                    help="Explicitly disable ensemble even if --n-trees > 1")

# VLLM backend
parser.add_argument("--model", type=str, default=None,
                    help="Model name (default: $VLLM_MODEL or Llama-3.3-70B-Instruct)")
parser.add_argument("--tensor-parallel-size", type=int, default=1)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
parser.add_argument("--max-model-len", type=int, default=4096)

# Output
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--verbose", action="store_true", default=True)
parser.add_argument("--quiet", action="store_true")

args = parser.parse_args()

# ── Resolve defaults ──
MODEL = args.model or os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
VERBOSE = args.verbose and not args.quiet

# Dataset defaults: press releases
if args.split_dir is None:
    args.split_dir = str(REPO_ROOT / "datasets" / "press-releases" / "press_release_modeling_dataset.csv")
if args.data_path is None:
    args.data_path = str(Path(args.split_dir) / "train.csv.gz")
if args.output_dir is None:
    args.output_dir = str(REPO_ROOT / "outputs" / "metric_tree" / args.dataset_name)

TASK_DESCRIPTION = get_task_description(args.dataset_name)

# ── Load data ──
print(f"Loading data from splits in {args.split_dir} ...")
split_dir = Path(args.split_dir)
split_dfs = []
for split in ("train", "eval", "test"):
    for ext in (".csv.gz", ".csv"):
        p = split_dir / f"{split}{ext}"
        if p.exists():
            split_dfs.append(pd.read_csv(p, low_memory=False))
            break
    else:
        raise FileNotFoundError(f"Missing {split} split in {split_dir}")

train_df, eval_df, test_df = split_dfs

# Rename columns for press releases dataset
if args.dataset_name == "PressReleaseModeling":
    rename_map = {}
    if "press_release_id" in train_df.columns:
        rename_map["press_release_id"] = args.id_column
    if "text" in train_df.columns:
        rename_map["text"] = args.text_column
    if "judgement" in train_df.columns:
        rename_map["judgement"] = args.label_column
    if rename_map:
        train_df = train_df.rename(columns=rename_map)
        eval_df = eval_df.rename(columns=rename_map)
        test_df = test_df.rename(columns=rename_map)

# Drop rows with missing required columns
for df_name, df in [("train", train_df), ("eval", eval_df), ("test", test_df)]:
    required = [args.id_column, args.text_column, args.label_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} split missing columns: {missing}. Available: {list(df.columns)}")

train_df = train_df.dropna(subset=[args.id_column, args.text_column, args.label_column])
eval_df = eval_df.dropna(subset=[args.id_column, args.text_column, args.label_column])
test_df = test_df.dropna(subset=[args.id_column, args.text_column, args.label_column])

print(f"Loaded train={len(train_df)}, eval={len(eval_df)}, test={len(test_df)}")

# ── Create VLLM backend ──
print(f"Initializing VLLM backend with model: {MODEL}")
backend = create_backend(
    "vllm",
    model_name_or_path=MODEL,
    tensor_parallel_size=args.tensor_parallel_size,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
)
print("VLLM backend ready.")

# ── Load tokenizer for token-precise truncation ──
from metric_tree.token_utils import get_tokenizer

tokenizer = get_tokenizer(MODEL)
# Token budgets are computed dynamically at each call site (score_subset, proposer)
# based on the actual rubric/context sizes. We just pass max_model_len through.
token_budgets = {"max_model_len": args.max_model_len}
print(f"Token truncation: max_model_len={args.max_model_len}, budgets computed dynamically per call")

# ── Generator/Judge LLM (fallback for DSPy signatures) ──
generator_llm = dspy.LM(model=f"openai/{MODEL}", api_key="unused", api_base="http://localhost:1")
judge_llm = generator_llm

# ── Build config ──
config = TreeConfig(
    max_depth=args.max_depth,
    min_subset_size=args.min_subset_size,
    n_metrics_to_propose=args.n_metrics,
    n_rubrics_to_propose=args.n_rubrics,
    use_interactions=not args.no_interactions,
    use_learned_router=args.use_learned_router,
    confidence_threshold=args.confidence_threshold,
    eval_fraction=args.eval_fraction,
    max_text_tokens=args.max_text_tokens,
    random_seed=args.seed,
    output_dir=args.output_dir,
    verbose=VERBOSE,
)

# ── Create proposer ──
proposer = ContrastiveRubricProposer(
    generator_llm=generator_llm,
    seed=config.random_seed,
    scoring_backend=backend,
)

# ── Build tree(s) ──
cache_dir = str(Path(args.output_dir) / "label_cache")
n_trees = args.n_trees if not args.no_ensemble else 1

print(f"\nBuilding Metric Tree (max_depth={config.max_depth}, n_trees={n_trees})...")

if n_trees > 1:
    trees = build_metric_tree_ensemble(
        train_df=train_df,
        eval_df=eval_df,
        config=config,
        proposer=proposer,
        task_description=TASK_DESCRIPTION,
        n_trees=n_trees,
        id_column=args.id_column,
        text_column=args.text_column,
        label_column=args.label_column,
        judge_llm=judge_llm,
        cache_dir=cache_dir,
        scoring_backend=backend,
        tokenizer=tokenizer,
        token_budgets=token_budgets,
    )
    tree = trees[0]  # primary tree for analysis
else:
    tree = build_metric_tree(
        train_df=train_df,
        eval_df=eval_df,
        config=config,
        proposer=proposer,
        task_description=TASK_DESCRIPTION,
        id_column=args.id_column,
        text_column=args.text_column,
        label_column=args.label_column,
        judge_llm=judge_llm,
        cache_dir=cache_dir,
        scoring_backend=backend,
        tokenizer=tokenizer,
        token_budgets=token_budgets,
    )
    trees = [tree]

# ── Analyze tree ──
print("\n=== Tree Complexity ===")
complexity_df = analyze_tree_complexity(tree)
print(complexity_df.to_string(index=False))

# ── Predict on test set ──
print(f"\nPredicting on {len(test_df)} test examples...")
label_cache = LabelCache(cache_dir)

if n_trees > 1:
    test_predictions = ensemble_predict(
        trees=trees,
        df=test_df,
        label_cache=label_cache,
        id_column=args.id_column,
        text_column=args.text_column,
        label_column=args.label_column,
        judge_llm=judge_llm,
        task_description=TASK_DESCRIPTION,
        batch_size=config.label_batch_size,
        scoring_backend=backend,
        verbose=VERBOSE,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
    )
else:
    test_predictions = predict_batch(
        tree=tree,
        df=test_df,
        label_cache=label_cache,
        id_column=args.id_column,
        text_column=args.text_column,
        label_column=args.label_column,
        judge_llm=judge_llm,
        task_description=TASK_DESCRIPTION,
        batch_size=config.label_batch_size,
        scoring_backend=backend,
        verbose=VERBOSE,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
    )

# ── Evaluate ──
test_labels = _coerce_binary_labels(test_df, args.label_column)[args.label_column].values
test_preds = test_predictions["prediction"].values
test_accuracy = float((test_preds == test_labels).mean())
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Depth distribution
if "resolving_node" in test_predictions.columns:
    depth_dist = measure_depth_distribution(tree, test_predictions)
    print("\n=== Resolution Depth Distribution ===")
    print(depth_dist.to_string(index=False))

# Articulability gap — score ALL test points through root only
print("\nComputing root-only predictions for articulability gap...")
root_only_preds = predict_root_only(
    tree=tree,
    df=test_df,
    label_cache=label_cache,
    id_column=args.id_column,
    text_column=args.text_column,
    label_column=args.label_column,
    judge_llm=judge_llm,
    task_description=TASK_DESCRIPTION,
    batch_size=config.label_batch_size,
    scoring_backend=backend,
    verbose=VERBOSE,
    max_model_len=args.max_model_len,
    tokenizer=tokenizer,
)
gap_results = compute_articulability_gap(tree, test_predictions, test_labels, root_only_preds)
print(f"\n=== Articulability Gap ===")
print(f"  Root accuracy:  {gap_results['root_accuracy']:.4f}")
print(f"  Tree accuracy:  {gap_results['tree_accuracy']:.4f}")
print(f"  Gap:            {gap_results['articulability_gap']:.4f}")
print(f"  Per-depth accuracy: {gap_results['per_depth_accuracy']}")

# ── Export ──
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

export_tree_summary(tree, str(output_dir / "tree_summary.txt"))
complexity_df.to_csv(str(output_dir / "tree_complexity.csv"), index=False)
test_predictions.to_csv(str(output_dir / "test_predictions.csv"), index=False)

with open(output_dir / "results.json", "w") as f:
    json.dump({
        "test_accuracy": test_accuracy,
        "articulability_gap": gap_results,
        "n_trees": n_trees,
        "n_nodes": len(tree.all_nodes),
        "n_metrics": len(tree.all_metrics),
        "config": {
            "max_depth": config.max_depth,
            "min_subset_size": config.min_subset_size,
            "n_metrics_to_propose": config.n_metrics_to_propose,
            "use_interactions": config.use_interactions,
            "use_learned_router": config.use_learned_router,
            "random_seed": config.random_seed,
        },
    }, f, indent=2, default=str)

print(f"\n=== Done ===")
print(f"Output written to: {output_dir}")
