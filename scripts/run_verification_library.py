#!/usr/bin/env python3
"""Run verification library discovery algorithm."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from methods.verification_library.config import (
    DATASET_REGISTRY,
    VerificationLibraryConfig,
)
from methods.verification_library.runner import run_verification_library, run_verification_library_v2

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    "peer-review": {
        "split_dir": REPO_ROOT / "datasets" / "peer-review" / "splits",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    },
    "code-review": {
        "split_dir": REPO_ROOT / "datasets" / "code-review" / "splits",
        "id_column": "id",
        "text_column": "text",
        "label_column": "label",
    },
    "notice-and-comment": {
        "split_dir": REPO_ROOT / "datasets" / "notice-and-comment" / "notice_and_comment_len_balanced",
        "id_column": "id",
        "text_column": "text",
        "label_column": "judgement",
    },
    "press-release": {
        "split_dir": REPO_ROOT / "datasets" / "press-releases" / "press_release_modeling_dataset",
        "id_column": "id",
        "text_column": "text",
        "label_column": "label",
    },
}


def load_splits(split_dir: Path):
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


def main():
    parser = argparse.ArgumentParser(description="Run verification library discovery")
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to run on",
    )
    parser.add_argument("--generation-model", default="qwen/qwen3-coder-next",
                        help="Model for bulk program generation (cheap)")
    parser.add_argument("--generation-base-url", default="https://openrouter.ai/api/v1",
                        help="API base URL for generation model. Use http://localhost:PORT/v1 for local vLLM")
    parser.add_argument("--refactoring-model", default="claude-sonnet-4-20250514",
                        help="Model for refactoring (quality matters, few calls)")
    parser.add_argument("--refactoring-base-url", default="",
                        help="API base URL for refactoring model. Empty = Anthropic native API")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Generate programs in batches of this size (parallel)")
    parser.add_argument("--refactor-size", type=int, default=3,
                        help="Refactor library after every N new successful programs")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--sandbox-timeout", type=float, default=5.0)
    parser.add_argument("--test-set-size", type=int, default=500)
    parser.add_argument(
        "--combination-method", default="logistic",
        choices=["vote", "mean", "logistic"],
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--approach", default="approach2",
        choices=["v1", "v2", "approach1", "approach2"],
        help="v1=basic raw-text programs, v2=extraction-aware, "
             "approach1=full-example agnostic programs (with optional z guidance), "
             "approach2=per-z program synthesis from STaR rationales",
    )
    args = parser.parse_args()

    ds_cfg = DATASET_CONFIGS[args.dataset]
    dataset_config = DATASET_REGISTRY[args.dataset]

    # Override columns from dataset config
    dataset_config.text_column = ds_cfg["text_column"]
    dataset_config.label_column = ds_cfg["label_column"]
    dataset_config.id_column = ds_cfg["id_column"]

    config = VerificationLibraryConfig(
        approach=args.approach,
        batch_size=args.batch_size,
        refactor_size=args.refactor_size,
        max_examples=args.max_examples,
        generation_model=args.generation_model,
        generation_base_url=args.generation_base_url,
        generation_concurrency=args.concurrency,
        refactoring_model=args.refactoring_model,
        refactoring_base_url=args.refactoring_base_url,
        sandbox_timeout=args.sandbox_timeout,
        test_set_size=args.test_set_size,
        combination_method=args.combination_method,
        output_dir=args.output_dir or f"outputs/verification_library/{args.dataset}",
        random_seed=args.seed,
    )

    train_df, eval_df, test_df = load_splits(ds_cfg["split_dir"])

    logger.info(
        "Running verification library on %s: %d train, %d test, "
        "generation=%s, refactoring=%s",
        args.dataset, len(train_df), len(test_df),
        config.generation_model, config.refactoring_model,
    )

    approach_runners = {
        "v1": run_verification_library,
        "v2": run_verification_library_v2,
        "approach1": run_verification_library,   # approach1 uses v1 runner with z-guidance
        "approach2": run_verification_library,   # approach2 needs dedicated runner (TODO)
    }
    runner_fn = approach_runners.get(args.approach, run_verification_library)
    summary = asyncio.run(
        runner_fn(train_df, test_df, config, dataset_config)
    )

    logger.info("=== DONE ===")
    logger.info("Total programs: %d (%d successful)", summary["total_programs"], summary["successful_programs"])
    logger.info("Final ensemble accuracy: %.4f", summary["final_ensemble_accuracy"])
    logger.info("Final ensemble AUC: %.4f", summary["final_ensemble_auc"])
    logger.info("Library functions: %d", summary["final_library_functions"])
    logger.info("Converged: %s", summary["converged"])


if __name__ == "__main__":
    main()
