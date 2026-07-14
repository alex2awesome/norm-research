#!/usr/bin/env bash
set -euo pipefail

# Bounded launcher for the frozen same-version Llama-3.1 8B -> 70B H49 sentinel.  The two
# scientific phases remain separate: calibration creates the authenticated release artifact;
# lockbox scoring is impossible until that artifact exists and validates.

ROOT=/lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
DATA=notebooks/data/two_faces_20260702
CONF=$DATA/same_version_upper_confirmation_v1
MANIFEST=methods/codability/experiments/same_version_upper_execution_manifest_v1.json
SELECTION=methods/codability/experiments/same_version_upper_selection_v1.json
BANK=$DATA/fresh_name_arm_bank_v1.json
PACKET_ROOT=$DATA/same_version_upper_item_partitions_v1
PACKET=$PACKET_ROOT/packet_manifest.json
READOUT=methods/codability/experiments/fresh_llama70_name_target_manifest_v1.json
RELEASE=$CONF/calibration_release.json
MAX_ACCOUNT_GPUS=4

export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export METRIC_IMPLEMENTER_LFS_HOME=/lfs/skampere3/0/alexspan
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_GPU_MEM_UTIL VLLM_BLOCK_SIZE VLLM_ENFORCE_EAGER FLASHINFER_CUDA_ARCHS

account_gpu_count() {
  local count=0 pid uuid owner
  declare -A occupied=()
  while IFS=, read -r pid uuid; do
    pid=${pid//[[:space:]]/}
    uuid=${uuid//[[:space:]]/}
    [[ -n "$pid" && -n "$uuid" ]] || continue
    owner=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')
    if [[ "$owner" == "$(id -un)" ]]; then
      occupied["$uuid"]=1
    fi
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits)
  count=${#occupied[@]}
  echo "$count"
}

first_free_gpu() {
  local index used
  while IFS=, read -r index used; do
    index=${index//[[:space:]]/}
    used=${used//[[:space:]]/}
    if (( used < 1000 )); then
      echo "$index"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  return 1
}

check_one_gpu_request() {
  local device=$1 used account_count
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$device" \
    | awk 'NR==1 {print $1}')
  if [[ -z "$used" ]] || (( used >= 1000 )); then
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

smoke_target() {
  local device=$1
  check_one_gpu_request "$device"
  CUDA_VISIBLE_DEVICES="$device" "$PY" -c '
from vllm import LLM, SamplingParams
model = "/lfs/skampere3/0/shared_hf_cache/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
engine = LLM(model=model, tensor_parallel_size=1, dtype="bfloat16",
             gpu_memory_utilization=0.9, max_model_len=8192)
result = engine.generate(["Answer with exactly YES or NO: Is water wet?"],
                         SamplingParams(temperature=0.0, max_tokens=1))
assert result and result[0].outputs
print("H49_TP1_SMOKE_OK", result[0].outputs[0].text)
'
}

score_job() {
  local job=$1 phase=$2 repetition=$3 device=$4 raw=$5
  local release_args=()
  check_one_gpu_request "$device"
  if [[ "$phase" == "lockbox" ]]; then
    release_args=(--lockbox-release-artifact "$RELEASE")
  fi
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
  local release_args=()
  local write_args=()
  if [[ "$phase" == "lockbox" ]]; then
    release_args=(--lockbox-release-artifact "$RELEASE")
  else
    write_args=(--write-lockbox-release)
  fi
  "$PY" -m methods.codability.experiments.run_policy_isomorphism \
    --executor-shard-root "$shards" --target-shard-root "$shards" \
    --scale-comparator-use-target --arm-bank "$BANK" \
    --partition "same_version_upper_$phase" \
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

queue_calibration() {
  local deadline_seconds=${H49_QUEUE_DEADLINE_SECONDS:-14400}
  local started now device
  started=$(date +%s)
  while true; do
    now=$(date +%s)
    if (( now - started >= deadline_seconds )); then
      echo "H49_QUEUE_DEADLINE_EXPIRED after ${deadline_seconds}s" >&2
      exit 5
    fi
    if (( $(account_gpu_count) < MAX_ACCOUNT_GPUS )); then
      device=$(first_free_gpu || true)
      if [[ -n "$device" ]]; then
        echo "H49_QUEUE_ACQUIRED_GPU=$device timestamp=$(date --iso-8601=seconds)"
        smoke_target "$device"
        run_phase calibration "$device"
        echo "H49_CALIBRATION_COMPLETE timestamp=$(date --iso-8601=seconds)"
        return 0
      fi
    fi
    sleep 60
  done
}

cd "$ROOT"
mkdir -p "$CONF/logs"
MODE=${1:-}
DEVICE=${H49_DEVICE:-0}
echo "timestamp=$(date --iso-8601=seconds) host=$(hostname) mode=$MODE device=$DEVICE"
case "$MODE" in
  smoke)
    smoke_target "$DEVICE"
    ;;
  calibration)
    run_phase calibration "$DEVICE"
    ;;
  lockbox)
    [[ -f "$RELEASE" ]] || { echo "missing calibration release: $RELEASE" >&2; exit 6; }
    run_phase lockbox "$DEVICE"
    ;;
  queue-calibration)
    queue_calibration
    ;;
  *)
    echo "usage: $0 {smoke|calibration|lockbox|queue-calibration}" >&2
    exit 64
    ;;
esac
