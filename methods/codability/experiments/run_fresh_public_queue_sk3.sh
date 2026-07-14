#!/usr/bin/env bash
set -euo pipefail

ROOT=/lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
DATA=notebooks/data/two_faces_20260702
PACKET=$DATA/fresh_item_partitions_v1
TARGETS=$DATA/fresh_target_scores_v1
N_TARGET_SHARDS=$DATA/fresh_name_target_score_shards_v1
G_TARGET_SHARDS=$DATA/fresh_gestalt_target_score_shards_v1
ARM_SCORES=$DATA/fresh_name_arm_scores_v1
ARM_SHARDS=$DATA/fresh_name_arm_score_shards_v1
LOGS=logs/fresh_public_queue_v1

cd "$ROOT"
mkdir -p "$LOGS" "$TARGETS" "$N_TARGET_SHARDS" "$G_TARGET_SHARDS" "$ARM_SCORES" "$ARM_SHARDS"
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export CUDA_VISIBLE_DEVICES=7
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "waiting for externally launched llama70 N repetition 1"
while pgrep -f '[s]core_fresh_target_views --model-job llama70_n_target.*--repetition 1' >/dev/null; do
  sleep 10
done

echo "sharding all name targets in a target-view-specific, job-scoped root"
$PY -m methods.codability.experiments.shard_fresh_score_artifact \
  $TARGETS/llama70_n_target/*.npz $TARGETS/qwen7_target/*.npz $TARGETS/gemma31_target/*.npz \
  --out-dir "$N_TARGET_SHARDS" > "$LOGS/shard_name_targets.json"

for REP in 0 1; do
  echo "running clean llama70 G repetition $REP"
  $PY -u -m methods.codability.experiments.score_fresh_target_views \
    --model-job llama70_g_target \
    --packet-root "$PACKET" \
    --packet-manifest "$PACKET/packet_manifest.json" \
    --out-dir "$TARGETS" \
    --manifest methods/codability/experiments/fresh_gestalt_target_manifest_v1.json \
    --repetition "$REP" > "$LOGS/llama70_g_rep${REP}.log" 2>&1
done

echo "sharding clean llama70 G targets"
$PY -m methods.codability.experiments.shard_fresh_score_artifact \
  $TARGETS/llama70_g_target/*.npz --out-dir "$G_TARGET_SHARDS" \
  > "$LOGS/shard_llama70_g.json"

echo "running Llama-8B sparse development baseline"
$PY -u -m methods.codability.experiments.score_fresh_name_arms \
  --model-job llama8_big_sparse --phase development \
  --arm-bank "$DATA/fresh_name_arm_bank_v1.json" \
  --packet-root "$PACKET" --packet-manifest "$PACKET/packet_manifest.json" \
  --target-manifest methods/codability/experiments/fresh_target_view_manifest_v1.json \
  --out-dir "$ARM_SCORES" > "$LOGS/llama8_big_development.log" 2>&1

echo "running Llama-3B source arms and matched controls on public development"
$PY -u -m methods.codability.experiments.score_fresh_name_arms \
  --model-job llama3_small --phase development \
  --arm-bank "$DATA/fresh_name_arm_bank_v1.json" \
  --packet-root "$PACKET" --packet-manifest "$PACKET/packet_manifest.json" \
  --target-manifest methods/codability/experiments/fresh_target_view_manifest_v1.json \
  --out-dir "$ARM_SCORES" > "$LOGS/llama3_small_development.log" 2>&1

echo "sharding executor development matrices"
$PY -m methods.codability.experiments.shard_fresh_score_artifact \
  $ARM_SCORES/llama8_big_sparse/*.npz $ARM_SCORES/llama3_small/*.npz \
  --out-dir "$ARM_SHARDS" > "$LOGS/shard_executors.json"

echo "freezing public-only arm selection; no lockbox executor is called"
$PY -m methods.codability.experiments.fresh_name_arm_selection \
  --target-shard-root "$N_TARGET_SHARDS" \
  --executor-shard-root "$ARM_SHARDS" \
  --arm-bank "$DATA/fresh_name_arm_bank_v1.json" \
  --packet-manifest "$PACKET/packet_manifest.json" \
  --partition residual_prompt_selection --n-boot 5000 --seed 1207 \
  --out "$DATA/fresh_name_arm_selection_v1.json" \
  > "$LOGS/selection.log" 2>&1

echo "updating aggregate-only target health report"
$PY -m methods.codability.experiments.fresh_target_score_report \
  --scores-dir "$TARGETS" --out "$TARGETS/target_health_report.json" \
  > "$LOGS/target_health.log" 2>&1

echo "PUBLIC_QUEUE_COMPLETE"
