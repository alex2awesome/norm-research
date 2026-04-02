#!/usr/bin/env python3
"""Run Partitioned Metric Tree with VLLM backend — binary features + base-rate leaves."""

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

import numpy as np
import pandas as pd

from autometrics.backends import create_backend
from autometrics.generator.ContrastiveRubricProposer import ContrastiveRubricProposer
from autometrics.iterative_refinement.label_cache import LabelCache
from autometrics.iterative_refinement.runner import _coerce_binary_labels
from autometrics.task_descriptions import get_task_description

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

# ── Dataset configs ──
DATASET_CONFIGS = {
    "press-release": {
        "dataset_name": "PressReleaseModeling",
        "split_dir": REPO_ROOT / "datasets" / "press-releases" / "press_release_modeling_dataset.csv",
        "output_subdir": "press_release_partition_tree",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    },
    "peer-review": {
        "dataset_name": "PeerReviewAcceptance",
        "split_dir": REPO_ROOT / "datasets" / "peer-review" / "peer_review_modeling_dataset",
        "output_subdir": "peer_review_partition_tree",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    },
}

# ── CLI args ──
parser = argparse.ArgumentParser(description="Run Partitioned Metric Tree with VLLM backend")

# Dataset
parser.add_argument("--dataset", type=str, default="peer-review",
                    choices=list(DATASET_CONFIGS.keys()),
                    help="Dataset to run on")

# Tree config
parser.add_argument("--max-depth", type=int, default=3)
parser.add_argument("--n-binary-metrics", type=int, default=3,
                    help="K: binary metrics per level (→ 2^K partitions)")
parser.add_argument("--n-propose", type=int, default=5,
                    help="Number of metrics to propose (select top K by MI)")
parser.add_argument("--min-partition-size", type=int, default=20)
parser.add_argument("--min-contrastive-pairs", type=int, default=3)
parser.add_argument("--min-minority-fraction", type=float, default=0.0,
                    help="Min minority class fraction to extend a partition (0=extend all, e.g. 0.15=prune pure partitions)")
parser.add_argument("--clustering-depth", type=int, default=2,
                    help="Depths < this use clustering (descriptive) features; >= this use discriminative")
parser.add_argument("--eval-fraction", type=float, default=0.4)

# Router: per-node text classifier for selective deepening
parser.add_argument("--use-router", action="store_true",
                    help="Train per-node text classifiers for selective deepening")
parser.add_argument("--router-threshold", type=float, default=0.5,
                    help="p(minority) > threshold => continue deeper (0.3=aggressive, 0.7=conservative)")
parser.add_argument("--router-epochs", type=int, default=20)
parser.add_argument("--router-hidden-dim", type=int, default=128)
parser.add_argument("--router-min-examples", type=int, default=40)
parser.add_argument("--max-text-tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)

# Restructuring
parser.add_argument("--restructure-iterations", type=int, default=0,
                    help="Number of restructuring iterations (0=disabled)")
parser.add_argument("--restructure-na-threshold", type=float, default=0.05,
                    help="Max NA rate to consider a feature applicable at a node")
parser.add_argument("--restructure-k-min", type=int, default=3,
                    help="Min features per node after restructuring")
parser.add_argument("--restructure-k-max", type=int, default=6,
                    help="Max features per node after restructuring")

# Ensemble
parser.add_argument("--n-trees", type=int, default=1)

# VLLM backend (scoring)
parser.add_argument("--model", type=str, default=None,
                    help="VLLM model for scoring (default: Llama-3.3-70B-Instruct)")
parser.add_argument("--tensor-parallel-size", type=int, default=1)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
parser.add_argument("--max-model-len", type=int, default=16384)
parser.add_argument("--label-batch-size", type=int, default=8192,
                    help="Batch size for scoring (lower = less GPU memory pressure)")
parser.add_argument("--enforce-eager", action="store_true",
                    help="Disable CUDA graphs in VLLM (slower but more stable)")
parser.add_argument("--disable-prefix-caching", action="store_true",
                    help="Disable prefix caching in VLLM")

# Proposer model (metric generation)
parser.add_argument("--proposer-model", type=str, default=None,
                    help="External API model for metric proposer (e.g., 'openai/gpt-5.4-mini'). "
                         "If not set, uses the VLLM backend for both scoring and proposing.")

# Output
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--verbose", action="store_true", default=True)
parser.add_argument("--quiet", action="store_true")

args = parser.parse_args()

# ── Resolve dataset config ──
dcfg = DATASET_CONFIGS[args.dataset]
MODEL = args.model or os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
VERBOSE = args.verbose and not args.quiet

SPLIT_DIR = dcfg["split_dir"]
ID_COLUMN = dcfg["id_column"]
TEXT_COLUMN = dcfg["text_column"]
LABEL_COLUMN = dcfg["label_column"]
DATASET_NAME = dcfg["dataset_name"]
TASK_DESCRIPTION = get_task_description(DATASET_NAME)

if args.output_dir is None:
    args.output_dir = str(REPO_ROOT / "outputs" / "metric_tree" / dcfg["output_subdir"])


def main():
    """Main entry point — wrapped for multiprocessing (tensor parallelism) compatibility."""
    # Make args/config accessible
    global train_df, eval_df, test_df

    # ── Load data ──
    print(f"Loading data from splits in {SPLIT_DIR} ...")
    split_dir = Path(SPLIT_DIR)
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
    
    for df_name, df in [("train", train_df), ("eval", eval_df), ("test", test_df)]:
        required = [ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} split missing columns: {missing}. Available: {list(df.columns)}")
    
    train_df = train_df.dropna(subset=[ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN])
    eval_df = eval_df.dropna(subset=[ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN])
    test_df = test_df.dropna(subset=[ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN])
    
    print(f"Loaded train={len(train_df)}, eval={len(eval_df)}, test={len(test_df)}")
    
    # ── Create VLLM backend ──
    print(f"Initializing VLLM backend with model: {MODEL}")
    vllm_kwargs = dict(
        model_name_or_path=MODEL,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_config={"backend": "FLASH_ATTN"},
    )
    if args.enforce_eager:
        vllm_kwargs["enforce_eager"] = True
        print("  enforce_eager=True (CUDA graphs disabled)")
    if args.disable_prefix_caching:
        vllm_kwargs["enable_prefix_caching"] = False
        print("  prefix_caching disabled")
    backend = create_backend("vllm", **vllm_kwargs)
    print("VLLM backend ready.")
    
    # ── Load tokenizer ──
    from metric_tree.token_utils import get_tokenizer
    tokenizer = get_tokenizer(MODEL)
    print(f"Token truncation: max_model_len={args.max_model_len}")
    
    # ── Build config ──
    config = TreeConfig(
        max_depth=args.max_depth,
        n_binary_metrics_per_level=args.n_binary_metrics,
        n_rubrics_to_propose=args.n_propose,
        min_partition_size=args.min_partition_size,
        min_contrastive_pairs=args.min_contrastive_pairs,
        min_minority_fraction=args.min_minority_fraction,
        clustering_depth=args.clustering_depth,
        use_router=args.use_router,
        router_threshold=args.router_threshold,
        router_n_epochs=args.router_epochs,
        router_hidden_dim=args.router_hidden_dim,
        router_min_examples=args.router_min_examples,
        eval_fraction=args.eval_fraction,
        max_text_tokens=args.max_text_tokens,
        random_seed=args.seed,
        restructure_iterations=args.restructure_iterations,
        restructure_na_threshold=args.restructure_na_threshold,
        restructure_k_min=args.restructure_k_min,
        restructure_k_max=args.restructure_k_max,
        label_batch_size=args.label_batch_size,
        output_dir=args.output_dir,
        verbose=VERBOSE,
    )
    
    # ── Create proposer ──
    import dspy

    if args.proposer_model:
        # Use external API model for metric generation (e.g., GPT-5.4 mini)
        print(f"Using external proposer model: {args.proposer_model}")
        # Load API key from file or environment
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            key_file = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")
            if key_file.exists():
                api_key = key_file.read_text().strip()
        if not api_key:
            raise RuntimeError("No OPENAI_API_KEY found in env or key file")
        os.environ["OPENAI_API_KEY"] = api_key
        proposer_llm = dspy.LM(model=args.proposer_model)
        proposer = ContrastiveRubricProposer(
            generator_llm=proposer_llm,
            seed=config.random_seed,
            # No scoring_backend → uses DSPy path with the external API
        )
    else:
        # Use VLLM backend for both scoring and proposing
        generator_llm = dspy.LM(model=f"openai/{MODEL}", api_key="unused", api_base="http://localhost:1")
        proposer = ContrastiveRubricProposer(
            generator_llm=generator_llm,
            seed=config.random_seed,
            scoring_backend=backend,
        )
    
    # ── Build tree(s) ──
    cache_dir = str(Path(args.output_dir) / "label_cache")
    n_trees = args.n_trees
    
    print(f"\nBuilding Partition Metric Tree (max_depth={config.max_depth}, K={config.n_binary_metrics_per_level}, n_trees={n_trees})...")
    
    if n_trees > 1:
        trees = build_metric_tree_ensemble(
            train_df=train_df,
            eval_df=eval_df,
            config=config,
            proposer=proposer,
            task_description=TASK_DESCRIPTION,
            n_trees=n_trees,
            id_column=ID_COLUMN,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            cache_dir=cache_dir,
            scoring_backend=backend,
            tokenizer=tokenizer,
            max_model_len=args.max_model_len,
        )
        tree = trees[0]
    else:
        tree = build_metric_tree(
            train_df=train_df,
            eval_df=eval_df,
            config=config,
            proposer=proposer,
            task_description=TASK_DESCRIPTION,
            id_column=ID_COLUMN,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            cache_dir=cache_dir,
            scoring_backend=backend,
            tokenizer=tokenizer,
            max_model_len=args.max_model_len,
        )
        trees = [tree]
    
    # ── Restructure tree (if enabled) ──
    if config.restructure_iterations > 0:
        from metric_tree.restructure import restructure_tree
        print(f"\n=== Restructuring Tree ({config.restructure_iterations} iterations) ===")
        label_cache_restructure = LabelCache(cache_dir)
        tree, iteration_results = restructure_tree(
            initial_tree=tree,
            train_df=train_df,
            eval_df=eval_df,
            test_df=test_df,
            label_cache=label_cache_restructure,
            config=config,
            id_column=ID_COLUMN,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            task_description=TASK_DESCRIPTION,
            scoring_backend=backend,
            proposer=proposer,
            tokenizer=tokenizer,
            max_model_len=args.max_model_len,
        )
        trees = [tree]
    
        # Print iteration results
        print("\n=== Restructuring Results ===")
        for r in iteration_results:
            print(f"  Iteration {r['iteration']} ({r['stage']}): "
                  f"eval_acc={r['eval_accuracy']:.4f}, eval_auc={r['eval_auc']:.4f}, "
                  f"test_acc={r['test_accuracy']:.4f}, test_auc={r['test_auc']:.4f}, "
                  f"n_metrics={r['n_metrics']}, n_nodes={r['n_nodes']}")
    
    # ── Analyze tree ──
    print("\n=== Tree Complexity ===")
    complexity_df = analyze_tree_complexity(tree)
    print(complexity_df.to_string(index=False))
    
    # ── Save tree before prediction (so crash during predict doesn't lose the tree) ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from metric_tree.serialization import save_tree
    save_tree(tree, str(output_dir / "saved_tree"))
    print(f"Full tree saved to {output_dir / 'saved_tree'}")
    
    export_tree_summary(tree, str(output_dir / "tree_summary.txt"))
    complexity_df.to_csv(str(output_dir / "tree_complexity.csv"), index=False)
    
    # ── Predict on test set ──
    print(f"\nPredicting on {len(test_df)} test examples...")
    label_cache = LabelCache(cache_dir)
    
    if n_trees > 1:
        test_predictions = ensemble_predict(
            trees=trees,
            df=test_df,
            label_cache=label_cache,
            id_column=ID_COLUMN,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            task_description=TASK_DESCRIPTION,
            scoring_backend=backend,
            batch_size=config.label_batch_size,
            verbose=VERBOSE,
            max_model_len=args.max_model_len,
            tokenizer=tokenizer,
        )
    else:
        test_predictions = predict_batch(
            tree=tree,
            df=test_df,
            label_cache=label_cache,
            id_column=ID_COLUMN,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            task_description=TASK_DESCRIPTION,
            scoring_backend=backend,
            batch_size=config.label_batch_size,
            verbose=VERBOSE,
            max_model_len=args.max_model_len,
            tokenizer=tokenizer,
        )
    
    # ── Evaluate ──
    from sklearn.metrics import roc_auc_score
    
    test_labels = _coerce_binary_labels(test_df, LABEL_COLUMN)[LABEL_COLUMN].values
    test_preds = test_predictions["prediction"].values
    test_probs = test_predictions["probability"].values
    test_accuracy = float((test_preds == test_labels).mean())
    
    try:
        test_auc = float(roc_auc_score(test_labels, test_probs))
    except ValueError:
        test_auc = 0.5
    
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Depth distribution
    if "resolving_node" in test_predictions.columns:
        depth_dist = measure_depth_distribution(tree, test_predictions)
        print("\n=== Resolution Depth Distribution ===")
        print(depth_dist.to_string(index=False))
    
    # Articulability gap
    print("\nComputing root-only predictions for articulability gap...")
    root_only_preds = predict_root_only(
        tree=tree,
        df=test_df,
        label_cache=label_cache,
        id_column=ID_COLUMN,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        task_description=TASK_DESCRIPTION,
        scoring_backend=backend,
        batch_size=config.label_batch_size,
        verbose=VERBOSE,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
    )
    gap_results = compute_articulability_gap(tree, test_predictions, test_labels, root_only_preds)
    print(f"\n=== Articulability Gap ===")
    print(f"  Root accuracy:  {gap_results['root_accuracy']:.4f}")
    print(f"  Tree accuracy:  {gap_results['tree_accuracy']:.4f}")
    print(f"  Tree AUC:       {gap_results.get('tree_auc', 0):.4f}")
    print(f"  Gap:            {gap_results['articulability_gap']:.4f}")
    print(f"  Per-depth accuracy: {gap_results['per_depth_accuracy']}")
    
    # ── Export predictions ──
    test_predictions.to_csv(str(output_dir / "test_predictions.csv"), index=False)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "test_accuracy": test_accuracy,
            "test_auc": test_auc,
            "articulability_gap": gap_results,
            "n_trees": n_trees,
            "n_nodes": len(tree.all_nodes),
            "n_metrics": len(tree.all_metrics),
            "config": {
                "max_depth": config.max_depth,
                "n_binary_metrics_per_level": config.n_binary_metrics_per_level,
                "n_rubrics_to_propose": config.n_rubrics_to_propose,
                "min_partition_size": config.min_partition_size,
                "min_contrastive_pairs": config.min_contrastive_pairs,
                "min_minority_fraction": config.min_minority_fraction,
                "use_router": config.use_router,
                "router_threshold": config.router_threshold,
                "restructure_iterations": config.restructure_iterations,
                "restructure_na_threshold": config.restructure_na_threshold,
                "restructure_k_min": config.restructure_k_min,
                "restructure_k_max": config.restructure_k_max,
                "random_seed": config.random_seed,
            },
        }, f, indent=2, default=str)
    
    # Tree text visualization
    from metric_tree.visualization import format_tree_text
    tree_text = format_tree_text(tree)
    print(f"\n=== Tree Structure ===\n{tree_text}")
    
    print(f"\n=== Done ===")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
