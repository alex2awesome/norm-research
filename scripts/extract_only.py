#!/usr/bin/env python
"""Run only Step 1 (STaR-Local extraction) and exit. Caches to
explanation_cache/star_local_v1.jsonl. Subsequent pipeline steps can reuse.
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "methods" / "autometrics"))

from methods.autometrics.autometrics.task_descriptions import get_task_description
from methods.local_explanations.config import (
    TASK_METADATA_REGISTRY, LocalExplanationConfig,
)
from methods.local_explanations.explainer import LocalExplainer
from scripts.run_local_explanations import DATASET_CONFIGS, load_splits

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _truncate(df, text_col, max_tokens):
    if max_tokens <= 0:
        return df
    df = df.copy()
    df[text_col] = df[text_col].apply(lambda t: " ".join(str(t).split()[:max_tokens]))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--agency", default=None, help="For notice-and-comment per-agency")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model",
                   default="/lfs/skampere3/0/shared_hf_cache/models--nvidia--Llama-3.3-70B-Instruct-FP8/snapshots/fde04ee76a27704c88f569542ef023b57d4d0362")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-text-tokens", type=int, default=512)
    args = p.parse_args()

    dcfg = dict(DATASET_CONFIGS[args.dataset])
    if args.agency:
        dcfg["split_dir"] = (REPO_ROOT / "datasets" / "notice-and-comment" / "agencies" / args.agency)
    task_description = get_task_description(dcfg["dataset_name"])
    task_metadata = TASK_METADATA_REGISTRY[dcfg["dataset_name"]]

    train_df, _e, _t = load_splits(dcfg["split_dir"])
    id_col = dcfg["id_column"]
    text_col = dcfg["text_column"]
    if id_col not in train_df.columns:
        train_df[id_col] = [f"train_{i}" for i in range(len(train_df))]
    train_df = _truncate(train_df, text_col, args.max_text_tokens)
    logger.info("Extracting on %d train docs", len(train_df))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from methods.autometrics.autometrics.backends import create_backend
    backend = create_backend(
        "vllm", model_name_or_path=args.model,
        tensor_parallel_size=1, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    cfg = LocalExplanationConfig(
        approach="star_local",
        max_text_tokens=args.max_text_tokens,
        output_dir=str(out),
    )
    explainer = LocalExplainer(
        cfg, scoring_backend=backend,
        task_description=task_description, task_metadata=task_metadata,
        cache_dir=str(out / "explanation_cache"),
    )
    doc_ids = train_df[id_col].astype(str).tolist()
    texts = train_df[text_col].tolist()
    results = explainer.explain_star_local_batch(texts=texts, doc_ids=doc_ids)
    logger.info("Extraction complete: %d docs cached → %s",
                len(results), out / "explanation_cache" / "star_local_v1.jsonl")


if __name__ == "__main__":
    main()
