#!/usr/bin/env python
"""Run local explanation approaches (rationalization or STaR-local)."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "methods" / "autometrics"))

from methods.autometrics.autometrics.task_descriptions import get_task_description
from methods.local_explanations.config import (
    TASK_METADATA_REGISTRY,
    LocalExplanationConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset configs (same as run_metric_tree.py)
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "peer-review": {
        "split_dir": REPO_ROOT / "datasets" / "peer-review" / "splits",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
        "dataset_name": "PeerReviewAcceptance",
    },
    "press-release": {
        "split_dir": REPO_ROOT / "datasets" / "press-releases" / "press_release_modeling_dataset",
        "id_column": "id",
        "text_column": "text",
        "label_column": "label",
        "dataset_name": "PressReleaseModeling",
    },
    "code-review": {
        "split_dir": REPO_ROOT / "datasets" / "code-review" / "splits",
        "id_column": "id",
        "text_column": "text",
        "label_column": "label",
        "dataset_name": "CodeReviewAcceptance",
    },
    "math-stackexchange": {
        "split_dir": REPO_ROOT / "datasets" / "math-stackexchange" / "splits_150k",
        "id_column": "id",
        "text_column": "text",
        "label_column": "label",
        "dataset_name": "MathAnswerQuality",
    },
    "notice-and-comment": {
        "split_dir": REPO_ROOT / "datasets" / "notice-and-comment" / "notice_and_comment_len_balanced",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
        "dataset_name": "NoticeAndComment",
    },
}


def load_splits(split_dir: Path):
    """Load train/eval/test CSV splits."""
    import pandas as pd

    dfs = {}
    for split in ("train", "eval", "test"):
        for ext in (".csv.gz", ".csv"):
            p = split_dir / f"{split}{ext}"
            if p.exists():
                dfs[split] = pd.read_csv(p, low_memory=False)
                logger.info("Loaded %s: %d rows from %s", split, len(dfs[split]), p)
                break
        if split not in dfs:
            raise FileNotFoundError(f"Missing {split} split in {split_dir}")
    return dfs["train"], dfs["eval"], dfs["test"]


def create_generate_fn(proposer_model: str):
    """Create a generate_fn callable from a proposer model string.

    Supports "openai/gpt-5-mini", "openai/gpt-5", etc. GPT-5 reasoning
    models require temperature=1.0 (or None) and max_tokens >= 16000.
    """
    import dspy

    is_reasoning = any(
        tag in proposer_model.lower()
        for tag in ("gpt-5", "gpt5", "o1", "o3", "o4")
    )
    if is_reasoning:
        lm = dspy.LM(proposer_model, max_tokens=16000, temperature=1.0)
    else:
        lm = dspy.LM(proposer_model, max_tokens=2048, temperature=0.7)

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = lm(messages=messages)
        if isinstance(response, list):
            return response[0] if response else ""
        return str(response)

    return generate


def main():
    parser = argparse.ArgumentParser(description="Run local explanation approaches")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument(
        "--approach", type=str, default="star_local",
        choices=["rationalization", "star_local"],
    )
    parser.add_argument("--proposer-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--n-canonical-features", type=int, default=30)
    parser.add_argument("--max-text-tokens", type=int, default=512)
    parser.add_argument("--eval-fraction", type=float, default=0.4)
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--agency", type=str, default=None,
        help="For notice-and-comment: pick one agency dir under datasets/notice-and-comment/agencies/<agency>/",
    )
    args = parser.parse_args()

    # Resolve dataset config
    dcfg = dict(DATASET_CONFIGS[args.dataset])  # shallow copy so we can override
    task_description = get_task_description(dcfg["dataset_name"])
    task_metadata = TASK_METADATA_REGISTRY[dcfg["dataset_name"]]

    # notice-and-comment: override split_dir to per-agency path
    if args.agency:
        if args.dataset != "notice-and-comment":
            raise SystemExit("--agency is only supported for --dataset notice-and-comment")
        dcfg["split_dir"] = (
            REPO_ROOT / "datasets" / "notice-and-comment" / "agencies" / args.agency
        )
        logger.info("Agency override: split_dir=%s", dcfg["split_dir"])

    # Load data
    train_df, eval_df, test_df = load_splits(dcfg["split_dir"])

    # Auto-generate id column if missing (e.g. NC agency splits are just text,judgement)
    import pandas as pd
    for split_name, _df in [("train", train_df), ("eval", eval_df), ("test", test_df)]:
        if dcfg["id_column"] not in _df.columns:
            _df[dcfg["id_column"]] = [f"{split_name}_{i}" for i in range(len(_df))]
            logger.info("Auto-generated %s id column for %s split", dcfg["id_column"], split_name)

    # Optional class balancing
    if args.balance_classes:
        label_col = dcfg["label_column"]
        counts = train_df[label_col].value_counts()
        min_count = counts.min()
        train_df = train_df.groupby(label_col).apply(
            lambda x: x.sample(n=min_count, random_state=42)
        ).reset_index(drop=True)
        logger.info("Balanced training set to %d per class (%d total)", min_count, len(train_df))

    # Create VLLM backend
    from methods.autometrics.autometrics.backends import create_backend
    backend = create_backend(
        "vllm",
        model_name_or_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Create generate_fn for proposer/clustering LLM calls
    generate_fn = create_generate_fn(args.proposer_model)

    # Build config
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.dataset}_{args.agency}" if args.agency else args.dataset
    output_dir = args.output_dir or str(
        REPO_ROOT / "outputs" / "local_explanations" / f"{tag}_{args.approach}_{run_id}"
    )

    config = LocalExplanationConfig(
        approach=args.approach,
        n_canonical_features=args.n_canonical_features,
        max_text_tokens=args.max_text_tokens,
        eval_fraction=args.eval_fraction,
        output_dir=output_dir,
    )

    # Run
    from methods.local_explanations.runner import run_rationalization, run_star_local

    common_kwargs = dict(
        train_df=train_df,
        eval_df=eval_df,
        test_df=test_df,
        config=config,
        scoring_backend=backend,
        generate_fn=generate_fn,
        task_description=task_description,
        task_metadata=task_metadata,
        id_column=dcfg["id_column"],
        text_column=dcfg["text_column"],
        label_column=dcfg["label_column"],
        max_model_len=args.max_model_len,
    )

    if args.approach == "rationalization":
        results = run_rationalization(**common_kwargs)
    else:
        results = run_star_local(**common_kwargs)

    # Print summary
    logger.info("=" * 60)
    logger.info("FINAL RESULTS (%s on %s)", args.approach, args.dataset)
    logger.info("=" * 60)
    test = results.get("test_metrics", {})
    logger.info("Test AUC:      %.4f", test.get("roc_auc", 0))
    logger.info("Test Accuracy: %.4f", test.get("accuracy", 0))
    logger.info("Test F1:       %.4f", test.get("f1", 0))
    logger.info("Active features: %d / %d", len(results.get("active_features", [])), results.get("n_canonical_features", 0))
    logger.info("Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
