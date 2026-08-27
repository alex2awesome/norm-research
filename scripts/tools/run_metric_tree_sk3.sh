#!/usr/bin/env bash
# Run Metric Tree on sk3 with Llama-3.3-70B via VLLM (single GPU).
#
# Usage:
#   ./run_metric_tree_sk3.sh
#   ./launch_when_gpus_free.sh 1 ./run_metric_tree_sk3.sh
set -euo pipefail

export VLLM_MODEL="meta-llama/Llama-3.3-70B-Instruct"
export FLASHINFER_DISABLE_VERSION_CHECK=1

python scripts/run_metric_tree.py \
  --model "$VLLM_MODEL" \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95 \
  --split-dir datasets/press-releases/press_release_modeling_dataset.csv \
  --dataset-name PressReleaseModeling \
  --id-column id \
  --text-column output \
  --label-column newsworthiness_score \
  --max-depth 2 \
  --n-metrics 5 \
  --n-rubrics 5 \
  --min-subset-size 20 \
  --n-trees 1 \
  --seed 42 \
  --output-dir outputs/metric_tree/press_release_70b
