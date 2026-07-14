#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=$(cd -- "$script_dir/.." && pwd)

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU_CSV <run_v14_value_campaign arguments...>" >&2
  exit 2
fi

physical_csv="$1"
shift
IFS=',' read -r -a physical_ids <<< "$physical_csv"
if [[ ${#physical_ids[@]} -eq 0 ]]; then
  echo "no physical GPUs declared" >&2
  exit 2
fi

for gpu in "${physical_ids[@]}"; do
  case "$gpu" in
    0|5|6|7) ;;
    1|2|3|4)
      echo "HARD STOP: sk3 physical GPU $gpu is permanently forbidden" >&2
      exit 64
      ;;
    *)
      echo "invalid sk3 physical GPU $gpu; v14 permits only 0,5,6,7" >&2
      exit 64
      ;;
  esac
done

# Hold one nonblocking process lock per physical device before CUDA is exposed.
lock_index=0
for gpu in "${physical_ids[@]}"; do
  lock_fd=$((200 + lock_index))
  eval "exec ${lock_fd}>/tmp/cr3-v14-sk3-gpu-${gpu}.lock"
  if ! flock -n "$lock_fd"; then
    echo "physical GPU $gpu already has a v14 lane lock" >&2
    exit 75
  fi
  lock_index=$((lock_index + 1))
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export METRIC_IMPLEMENTER_LFS_HOME=/lfs/skampere3/0/alexspan
export CUDA_VISIBLE_DEVICES="$physical_csv"
export V14_PHYSICAL_GPUS="$physical_csv"
export V14_MODEL_PATH_OVERRIDES_JSON=${V14_MODEL_PATH_OVERRIDES_JSON:-'{"meta-llama/Llama-3.3-70B-Instruct":"/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b","mistralai/Mistral-Small-24B-Instruct-2501":"/lfs/skampere3/0/shared_hf_cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724"}'}
V14_PYTHON=${V14_PYTHON:-/lfs/skampere3/0/alexspan/miniconda3/bin/python}
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$code_root"

exec "$V14_PYTHON" -m methods.metric_implementer.experiments.run_v14_value_campaign \
  --physical-gpus "$physical_csv" "$@"
