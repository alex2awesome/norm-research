#!/usr/bin/env bash
set -euo pipefail

# Auditable launcher for the open search phase of the 11-task x R1/R2/R3 tacit-policy
# reconstruction.  It reuses the integrated scorer/sharder/runner and intentionally contains no
# validation-partition command.  Selection, validation-manifest freeze, and release happen only
# after the public search report has been inspected.

ROOT=${ROOT:-/lfs/skampere3/0/alexspan/norm-research}
PY=${PY:-/lfs/skampere3/0/alexspan/miniconda3/bin/python}
DATA=notebooks/data/two_faces_20260702
CONF=$DATA/tacit_breadth_confirmation_v3
MANIFEST=$CONF/search_execution_manifest_v2.json
PANEL=$DATA/tacit_breadth_metric_panel_v3.json
BANK=$DATA/tacit_breadth_arm_bank_v3.json
PACKET_ROOT=$DATA/tacit_breadth_item_partitions_v2
PACKET=$PACKET_ROOT/packet_manifest.json
READOUT=methods/codability/experiments/tacit_breadth_readout_manifest_v1.json
SCORES=$CONF/search_scores
SHARDS=$CONF/search_shards
REPORT=$CONF/calibration_report.json

PANEL_SHA=ea34fddad96558ad5261455b394b6aba7378b737b3f1d326a2f47d1abcdce479
BANK_SHA=e61999c68eb04d582893ec2bc2a19ee02a8ced79bb80615300707aee89dd1d32
PACKET_SHA=2bdadf79072155587f5c1a03eb30cee14ea76b78d8b4c6260f51037d284225ea
# This campaign is deliberately serialized on physical GPU 0.  The user has prohibited this
# thread from using GPUs 1--4, and no implicit fallback to another device is allowed.
ALLOWED_DEVICE=0
# This campaign uses one GPU.  The account-wide guard still honors the user's separate four-GPU
# ceiling, so unrelated jobs on allowed-by-their-own-thread devices do not block a free GPU 0.
MAX_ACCOUNT_GPUS=4

sha256() {
  sha256sum "$1" | awk '{print $1}'
}

require_hash() {
  local path=$1
  local expected=$2
  local observed
  observed=$(sha256 "$path")
  if [[ "$observed" != "$expected" ]]; then
    echo "hash mismatch: $path expected=$expected observed=$observed" >&2
    exit 2
  fi
}

check_gpu_request() {
  local requested=$1
  if [[ "$requested" != "$ALLOWED_DEVICE" ]]; then
    echo "tacit breadth may use physical GPU $ALLOWED_DEVICE only; requested=$requested" >&2
    exit 3
  fi
  declare -A occupied=()
  declare -A busy=()
  local pid uuid owner index used
  while IFS=, read -r pid uuid; do
    pid=${pid//[[:space:]]/}
    uuid=${uuid//[[:space:]]/}
    busy["$uuid"]=1
    owner=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')
    if [[ "$owner" == "$(id -un)" ]]; then
      occupied["$uuid"]=1
    fi
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits)

  while IFS=, read -r index uuid used; do
    index=${index//[[:space:]]/}
    uuid=${uuid//[[:space:]]/}
    used=${used//[[:space:]]/}
    if [[ ",${requested}," == *",${index},"* ]]; then
      if [[ -n "${busy[$uuid]:-}" ]] || (( used > 1000 )); then
        echo "requested GPU $index is not free: compute process or ${used} MiB used" >&2
        exit 3
      fi
      occupied["$uuid"]=1
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)

  if (( ${#occupied[@]} > MAX_ACCOUNT_GPUS )); then
    echo "account GPU cap would be exceeded: ${#occupied[@]} > $MAX_ACCOUNT_GPUS" >&2
    exit 4
  fi
  echo "account GPU union after request: ${#occupied[@]}/$MAX_ACCOUNT_GPUS"
}

score_job() {
  local job=$1
  local devices=$2
  local domains=${3:-}
  check_gpu_request "$devices"
  local args=(
    --model-job "$job"
    --phase calibration
    --arm-bank "$BANK"
    --packet-root "$PACKET_ROOT"
    --packet-manifest "$PACKET"
    --target-manifest "$READOUT"
    --execution-manifest "$MANIFEST"
    --out-dir "$SCORES"
    --repetition 0
  )
  if [[ -n "$domains" ]]; then
    args+=(--domains "$domains")
  fi
  echo "model_job=$job devices=$devices domains=${domains:-ALL}"
  CUDA_VISIBLE_DEVICES="$devices" "$PY" -u \
    -m methods.codability.experiments.score_fresh_name_arms "${args[@]}"
}

cd "$ROOT"
mkdir -p "$SCORES/llama31_70b_name_target" "$SCORES/llama31_8b_executor" \
  "$SHARDS" "$CONF/logs"
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export METRIC_IMPLEMENTER_LFS_HOME=/lfs/skampere3/0/alexspan
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset VLLM_GPU_MEM_UTIL VLLM_BLOCK_SIZE VLLM_ENFORCE_EAGER FLASHINFER_CUDA_ARCHS

require_hash "$PANEL" "$PANEL_SHA"
require_hash "$BANK" "$BANK_SHA"
require_hash "$PACKET" "$PACKET_SHA"

MODE=${1:-}
echo "timestamp=$(date --iso-8601=seconds) host=$(hostname) mode=$MODE"
case "$MODE" in
  target)
    score_job llama31_70b_name_target "${TARGET_DEVICE:-0}"
    ;;
  executor-a)
    score_job llama31_8b_executor "${EXECUTOR_A_DEVICE:-0}" \
      "grant-funding,humor,peer-review,press-releases"
    ;;
  executor-b)
    score_job llama31_8b_executor "${EXECUTOR_B_DEVICE:-0}" \
      "math-stackexchange,news-homepages,patents"
    ;;
  executor-c)
    score_job llama31_8b_executor "${EXECUTOR_C_DEVICE:-0}" \
      "code-review,creative-writing,legal-outcome-prediction,notice-and-comment"
    ;;
  shard)
    "$PY" -m methods.codability.experiments.shard_fresh_score_artifact \
      "$SCORES"/llama31_70b_name_target/*.npz \
      "$SCORES"/llama31_8b_executor/*.npz \
      --out-dir "$SHARDS" --execution-manifest "$MANIFEST"
    ;;
  analyze)
    "$PY" -m methods.codability.experiments.run_policy_isomorphism \
      --executor-shard-root "$SHARDS" --target-shard-root "$SHARDS" \
      --scale-comparator-use-target \
      --arm-bank "$BANK" --partition tacit_breadth_search \
      --packet-root "$PACKET_ROOT" --packet-manifest "$PACKET" \
      --execution-manifest "$MANIFEST" \
      --small-job llama31_8b_executor --big-job llama31_70b_name_target \
      --target-arm-id name --n-boot 2000 --seed 1207 \
      --mae-margin 0.02 --rho-margin 0.05 --flip-margin 0.02 --bias-margin 0.02 \
      --functional-rho-floor 0.70 --confidence 0.95 \
      --fiber-mutual-rho-floor 0.90 --fiber-mutual-rho-sensitivity-floor 0.85 \
      --fiber-min-rank-valid-fraction 0.99 --fiber-distinctness-floor 0.35 \
      --include-controls --out "$REPORT"
    ;;
  supervise)
    echo "waiting for the authenticated 70B target process"
    while pgrep -f \
      '[s]core_fresh_name_arms --model-job llama31_70b_name_target.*--phase calibration' \
      >/dev/null; do
      sleep 60
    done
    target_count=$(find "$SCORES/llama31_70b_name_target" -maxdepth 1 \
      -name '*.npz' -type f 2>/dev/null | wc -l | awk '{print $1}')
    if [[ "$target_count" != "11" ]]; then
      echo "target phase incomplete: expected 11 domain artifacts, found $target_count" >&2
      exit 5
    fi
    if grep -Eq 'Traceback|EngineCore failed|RuntimeError|ValueError' \
      "$CONF/logs/target_70b_rep0.log"; then
      echo "target log contains a failure marker" >&2
      exit 6
    fi

    echo "target complete; starting three domain-disjoint executor shards sequentially on GPU 0"
    bash "$0" executor-a > "$CONF/logs/executor_8b_a_rep0.log" 2>&1
    bash "$0" executor-b > "$CONF/logs/executor_8b_b_rep0.log" 2>&1
    bash "$0" executor-c > "$CONF/logs/executor_8b_c_rep0.log" 2>&1

    executor_count=$(find "$SCORES/llama31_8b_executor" -maxdepth 1 \
      -name '*.npz' -type f 2>/dev/null | wc -l | awk '{print $1}')
    if [[ "$executor_count" != "11" ]]; then
      echo "executor phase incomplete: expected 11 domain artifacts, found $executor_count" >&2
      exit 8
    fi
    echo "executor complete; authenticating shards and producing the open search report"
    bash "$0" shard > "$CONF/logs/shard_search.log" 2>&1
    bash "$0" analyze > "$CONF/logs/analyze_search.log" 2>&1
    echo "OPEN_SEARCH_COMPLETE; validation remains sealed"
    ;;
  *)
    echo "usage: $0 {target|executor-a|executor-b|executor-c|shard|analyze|supervise}" >&2
    exit 64
    ;;
esac
