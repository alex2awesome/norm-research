#!/usr/bin/env bash
set -euo pipefail

# Batched same-version Llama-3.1 8B -> 70B confirmation for three prior-selected constructs.
# Calibration and lockbox remain separate authenticated phases.  This launcher never touches the
# breadth campaign's GPU 0 or the user-prohibited physical GPUs 1--4.

ROOT=${ROOT:-/lfs/skampere3/0/alexspan/norm-research}
PY=${PY:-/lfs/skampere3/0/alexspan/miniconda3/bin/python}
DATA=notebooks/data/two_faces_20260702
CONF=$DATA/concluding_policy_confirmation_v2
MANIFEST=methods/codability/experiments/concluding_policy_execution_manifest_v2.json
SELECTION=methods/codability/experiments/concluding_policy_selection_v1.json
BANK=$DATA/concluding_policy_arm_bank_v1.json
PACKET_ROOT=$DATA/tacit_breadth_item_partitions_v2
PACKET=$PACKET_ROOT/packet_manifest.json
READOUT=methods/codability/experiments/concluding_policy_target_manifest_v1.json
RELEASE=$CONF/calibration_release.json
MAX_ACCOUNT_GPUS=4
MAX_IDLE_MEMORY_MIB=${MAX_IDLE_MEMORY_MIB:-5000}
ALLOWED_DEVICES=(5 6 7)

export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export METRIC_IMPLEMENTER_LFS_HOME=/lfs/skampere3/0/alexspan
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_GPU_MEM_UTIL VLLM_BLOCK_SIZE VLLM_ENFORCE_EAGER FLASHINFER_CUDA_ARCHS

device_is_allowed() {
  local requested=$1 allowed
  for allowed in "${ALLOWED_DEVICES[@]}"; do
    [[ "$requested" == "$allowed" ]] && return 0
  done
  return 1
}

account_gpu_count() {
  local pid uuid owner
  declare -A occupied=()
  while IFS=, read -r pid uuid; do
    pid=${pid//[[:space:]]/}
    uuid=${uuid//[[:space:]]/}
    [[ -n "$pid" && -n "$uuid" ]] || continue
    owner=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')
    [[ "$owner" == "$(id -un)" ]] && occupied["$uuid"]=1
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits)
  echo "${#occupied[@]}"
}

check_one_gpu_request() {
  local device=$1 used account_count uuid busy_pids
  if ! device_is_allowed "$device"; then
    echo "concluding confirmation permits physical GPUs 5, 6, or 7 only; requested=$device" >&2
    exit 3
  fi
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$device" \
    | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print}')
  busy_pids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits \
    | awk -F, -v uuid="$uuid" '$2 ~ uuid {gsub(/[[:space:]]/, "", $1); print $1}')
  if [[ -n "$busy_pids" ]]; then
    echo "requested GPU $device has active compute PIDs: $busy_pids" >&2
    exit 3
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$device" \
    | awk 'NR==1 {print $1}')
  if [[ -z "$used" ]] || (( used >= MAX_IDLE_MEMORY_MIB )); then
    echo "requested GPU $device is not free (${used:-unknown} MiB used)" >&2
    exit 3
  fi
  account_count=$(account_gpu_count)
  if (( account_count + 1 > MAX_ACCOUNT_GPUS )); then
    echo "account GPU cap would be exceeded: $account_count + 1 > $MAX_ACCOUNT_GPUS" >&2
    exit 4
  fi
  echo "account GPU union after request: $((account_count + 1))/$MAX_ACCOUNT_GPUS"
}

score_job() {
  local job=$1 phase=$2 repetition=$3 device=$4 raw=$5
  local release_args=()
  check_one_gpu_request "$device"
  [[ "$phase" == "lockbox" ]] && release_args=(--lockbox-release-artifact "$RELEASE")
  CUDA_VISIBLE_DEVICES="$device" "$PY" -u \
    -m methods.codability.experiments.score_fresh_name_arms \
    --model-job "$job" --phase "$phase" --arm-bank "$BANK" \
    --packet-root "$PACKET_ROOT" --packet-manifest "$PACKET" \
    --target-manifest "$READOUT" --execution-manifest "$MANIFEST" \
    --selection-artifact "$SELECTION" --out-dir "$raw" \
    --repetition "$repetition" "${release_args[@]}"
}

analyze_phase() {
  local phase=$1 shards=$2 report=$3
  local release_args=() write_args=()
  if [[ "$phase" == "lockbox" ]]; then
    release_args=(--lockbox-release-artifact "$RELEASE")
  else
    write_args=(--write-lockbox-release)
  fi
  "$PY" -m methods.codability.experiments.run_policy_isomorphism \
    --executor-shard-root "$shards" --target-shard-root "$shards" \
    --scale-comparator-use-target --arm-bank "$BANK" \
    --partition "tacit_breadth_$([[ "$phase" == "calibration" ]] && echo search || echo validation)" \
    --packet-root "$PACKET_ROOT" --packet-manifest "$PACKET" \
    --execution-manifest "$MANIFEST" --selection-artifact "$SELECTION" \
    --small-job llama31_8b_executor --big-job llama31_70b_name_target \
    --target-arm-id name --n-boot 10000 --seed 1207 \
    --mae-margin 0.02 --rho-margin 0.05 --flip-margin 0.02 --bias-margin 0.02 \
    --functional-rho-floor 0.70 --confidence 0.95 \
    --fiber-mutual-rho-floor 0.90 --fiber-mutual-rho-sensitivity-floor 0.85 \
    --fiber-min-rank-valid-fraction 0.99 --fiber-distinctness-floor 0.35 \
    --include-controls --out "$report" "${release_args[@]}" "${write_args[@]}"
}

run_phase() {
  local phase=$1 device=$2
  local raw=$CONF/${phase}_raw_scores
  local shards=$CONF/${phase}_shards
  local report=$CONF/${phase}_report.json
  mkdir -p "$raw" "$shards" "$CONF/logs"
  for repetition in 0 1; do
    score_job llama31_70b_name_target "$phase" "$repetition" "$device" "$raw"
  done
  for repetition in 0 1; do
    score_job llama31_8b_executor "$phase" "$repetition" "$device" "$raw"
  done
  "$PY" -m methods.codability.experiments.shard_fresh_score_artifact \
    "$raw"/llama31_70b_name_target/*.npz "$raw"/llama31_8b_executor/*.npz \
    --out-dir "$shards" --execution-manifest "$MANIFEST"
  analyze_phase "$phase" "$shards" "$report"
}

cd "$ROOT"
mkdir -p "$CONF/logs"
MODE=${1:-}
DEVICE=${CONCLUDING_POLICY_DEVICE:-5}
echo "timestamp=$(date --iso-8601=seconds) host=$(hostname) mode=$MODE device=$DEVICE"
case "$MODE" in
  calibration)
    run_phase calibration "$DEVICE"
    ;;
  lockbox)
    [[ -f "$RELEASE" ]] || { echo "missing calibration release: $RELEASE" >&2; exit 6; }
    run_phase lockbox "$DEVICE"
    ;;
  full)
    run_phase calibration "$DEVICE"
    run_phase lockbox "$DEVICE"
    ;;
  *)
    echo "usage: $0 {calibration|lockbox|full}" >&2
    exit 64
    ;;
esac
