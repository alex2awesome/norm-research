#!/usr/bin/env python3
"""Run iterative AutoMetrics with VLLM backend on a single GPU."""

import os
import sys
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

# ── Config ──
MODEL = os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
SPLIT_DIR = REPO_ROOT / "datasets" / "press-releases" / "press_release_modeling_dataset.csv"
DATA_PATH = SPLIT_DIR / "train.csv.gz"
OUTPUT_DIR = REPO_ROOT / "outputs" / "iterative_autometrics" / "press_release_vllm_70b"

ID_COLUMN = "id"
TEXT_COLUMN = "output"
LABEL_COLUMN = "newsworthiness_score"

TASK_DESCRIPTION = (
    "Evaluate press-release text for the human judgment label. The model should score "
    "each press release according to the rubric-driven criteria the LLM proposes."
)

# ── Load data ──
print(f"Loading data from splits in {SPLIT_DIR} ...")
split_dfs = []
for split in ("train", "eval", "test"):
    split_path = SPLIT_DIR / f"{split}.csv.gz"
    split_dfs.append(pd.read_csv(split_path, low_memory=False))
df = pd.concat(split_dfs, ignore_index=True)
df = (
    df
    .rename(columns={"press_release_id": ID_COLUMN, "text": TEXT_COLUMN, "judgement": LABEL_COLUMN})
    .dropna(subset=[ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN])
)
print(f"Loaded {len(df)} rows")

dataset = Dataset(
    dataframe=df,
    target_columns=[LABEL_COLUMN],
    ignore_columns=[ID_COLUMN],
    metric_columns=[],
    name="PressReleaseModeling",
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
    tensor_parallel_size=1,
    max_model_len=32768,
    gpu_memory_utilization=0.95,
)
print("VLLM backend ready.")

# ── Generator/Judge LLM (needed as fallback signatures; backend handles actual calls) ──
generator_llm = dspy.LM(model=f"openai/{MODEL}", api_key="unused", api_base="http://localhost:1")
judge_llm = generator_llm

# ── Run ──
print("Starting iterative AutoMetrics ...")
results = run_iterative(
    dataset=dataset,
    target_measure=LABEL_COLUMN,
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    num_iterations=3,
    data_path=str(DATA_PATH),
    split_dir=str(SPLIT_DIR),
    id_column=ID_COLUMN,
    text_column=TEXT_COLUMN,
    label_column=LABEL_COLUMN,
    output_dir=str(OUTPUT_DIR),
    k_pairs=5,
    num_metrics=5,
    num_rubrics=5,
    label_batch_size=200,
    verbose=True,
    max_train_set_size=0.1,
    eval_sample_fraction=0.15,
    tqdm_scoring=True,
    scoring_backend=backend,
    seed=42,
)

print("\n=== Done ===")
print(f"Output written to: {OUTPUT_DIR}")
for k, v in results.items():
    if isinstance(v, (int, float, str, bool)):
        print(f"  {k}: {v}")
