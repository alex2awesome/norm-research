#!/usr/bin/env python3
"""Resume metric tree by loading a saved tree and running inference only.

Use this when a metric tree run crashed during the inference phase but the
tree was already saved (which `run_metric_tree.py` does before predicting).
"""

import argparse
import sys
from pathlib import Path

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

import pandas as pd

from autometrics.backends import create_backend
from autometrics.iterative_refinement.label_cache import LabelCache
from autometrics.iterative_refinement.runner import _coerce_binary_labels

from metric_tree.serialization import load_tree
from metric_tree.inference import predict_batch, predict_root_only
from metric_tree.analysis import (
    analyze_tree_complexity,
    compute_articulability_gap,
    measure_depth_distribution,
)


# Same dataset configs as run_metric_tree.py
DATASET_CONFIGS = {
    "press-release": {
        "split_dir": REPO_ROOT / "datasets" / "press-releases" / "press_release_modeling_dataset.csv",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    },
    "peer-review": {
        "split_dir": REPO_ROOT / "datasets" / "peer-review" / "peer_review_modeling_dataset",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    },
    "code-review": {
        "split_dir": REPO_ROOT / "datasets" / "code-review" / "code_review_dense_4096tok",
        "id_column": "paper_id",
        "text_column": "text",
        "label_column": "judgement",
    },
    "notice-and-comment": {
        "split_dir": REPO_ROOT / "datasets" / "notice-and-comment" / "notice_and_comment_len_balanced",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    },
}

NOTICE_AND_COMMENT_AGENCIES = [
    "cms", "epa", "fws", "fda", "faa", "aphis", "noaa", "ed", "ams", "nhtsa", "uscis", "dot",
    "cdc", "blm", "fs", "osha", "fsis", "irs",
]
for _ag in NOTICE_AND_COMMENT_AGENCIES:
    DATASET_CONFIGS[f"notice-and-comment-{_ag}"] = {
        "split_dir": REPO_ROOT / "datasets" / "notice-and-comment" / "agencies" / _ag,
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    }


def load_test_split(split_dir: Path) -> pd.DataFrame:
    """Load test split from disk (handles both directory and file layouts)."""
    if split_dir.is_dir():
        for ext in (".csv.gz", ".csv"):
            p = split_dir / f"test{ext}"
            if p.exists():
                return pd.read_csv(p, low_memory=False)
        raise FileNotFoundError(f"No test split in {split_dir}")
    else:
        # Single CSV file — use the standard split loading
        from autometrics.iterative_refinement.runner import load_fixed_split
        _, _, test_df = load_fixed_split(
            data_path=str(split_dir),
            split_dir=None,
            create_if_missing=False,
            seed=42,
            label_column="judgement",
        )
        return test_df


def main():
    parser = argparse.ArgumentParser(description="Resume metric tree inference from saved tree")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--tree-dir", type=str, required=True,
                        help="Directory containing saved_tree/ subdirectory")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--label-batch-size", type=int, default=8192)
    args = parser.parse_args()

    dcfg = DATASET_CONFIGS[args.dataset]
    tree_dir = Path(args.tree_dir)
    saved_tree_dir = tree_dir / "saved_tree"
    label_cache_dir = tree_dir / "label_cache"

    if not saved_tree_dir.exists():
        print(f"ERROR: {saved_tree_dir} does not exist")
        sys.exit(1)
    if not label_cache_dir.exists():
        print(f"ERROR: {label_cache_dir} does not exist")
        sys.exit(1)

    # Load tree
    print(f"Loading tree from {saved_tree_dir}...")
    tree = load_tree(str(saved_tree_dir))
    print(f"Tree loaded: {len(tree.all_nodes)} nodes, {len(tree.all_metrics)} metrics")

    # Load test data
    print(f"Loading test split from {dcfg['split_dir']}...")
    test_df = load_test_split(dcfg["split_dir"])
    print(f"Test set: {len(test_df)} examples")

    # Tree complexity
    print("\n=== Tree Complexity ===")
    complexity_df = analyze_tree_complexity(tree)
    print(complexity_df.to_string(index=False))

    # Create VLLM backend
    print(f"\nCreating VLLM backend with {args.model}...")
    backend = create_backend(
        "vllm",
        model_name_or_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = backend._tokenizer if hasattr(backend, "_tokenizer") else None

    # Get task description from tree (saved in metadata)
    task_description = tree.task_description if hasattr(tree, "task_description") else None
    if not task_description:
        from autometrics.task_descriptions import get_task_description
        task_description_map = {
            "press-release": "PressReleaseModeling",
            "peer-review": "PeerReviewAcceptance",
            "code-review": "CodeReviewAcceptance",
            "notice-and-comment": "NoticeAndComment",
        }
        task_description = get_task_description(task_description_map[args.dataset])

    # Run inference
    label_cache = LabelCache(str(label_cache_dir))

    print(f"\nPredicting on {len(test_df)} test examples...")
    test_predictions = predict_batch(
        tree=tree,
        df=test_df,
        label_cache=label_cache,
        id_column=dcfg["id_column"],
        text_column=dcfg["text_column"],
        label_column=dcfg["label_column"],
        task_description=task_description,
        scoring_backend=backend,
        batch_size=args.label_batch_size,
        verbose=True,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
    )

    # Evaluate
    from sklearn.metrics import roc_auc_score

    test_labels = _coerce_binary_labels(test_df, dcfg["label_column"])[dcfg["label_column"]].values
    test_preds = test_predictions["prediction"].values
    test_probs = test_predictions["probability"].values
    test_accuracy = float((test_preds == test_labels).mean())

    try:
        test_auc = float(roc_auc_score(test_labels, test_probs))
    except ValueError:
        test_auc = 0.5

    print(f"\n{'=' * 60}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test AUC:      {test_auc:.4f}")
    print(f"{'=' * 60}")

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
        id_column=dcfg["id_column"],
        text_column=dcfg["text_column"],
        label_column=dcfg["label_column"],
        task_description=task_description,
        scoring_backend=backend,
        batch_size=args.label_batch_size,
        verbose=True,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
    )
    root_acc = float((root_only_preds["prediction"].values == test_labels).mean())
    try:
        root_auc = float(roc_auc_score(test_labels, root_only_preds["probability"].values))
    except ValueError:
        root_auc = 0.5
    print(f"Root-only test accuracy: {root_acc:.4f}")
    print(f"Root-only test AUC:      {root_auc:.4f}")
    print(f"Articulability gap (full - root): {test_auc - root_auc:.4f}")

    # Save results
    import json
    results = {
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "root_only_accuracy": root_acc,
        "root_only_auc": root_auc,
        "articulability_gap": test_auc - root_auc,
        "n_test": len(test_df),
        "n_nodes": len(tree.all_nodes),
        "n_metrics": len(tree.all_metrics),
    }
    results_path = tree_dir / "resumed_inference_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
