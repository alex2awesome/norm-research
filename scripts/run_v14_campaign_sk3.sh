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
host=$(hostname -s)
case "$host" in
  sk2*|skampere2*) lfs_root=/lfs/skampere2/0; host_family=sk2 ;;
  sk3*|skampere3*) lfs_root=/lfs/skampere3/0; host_family=sk3 ;;
  *) echo "unsupported campaign host $host; expected sk2 or sk3" >&2; exit 64 ;;
esac
IFS=',' read -r -a physical_ids <<< "$physical_csv"
if [[ ${#physical_ids[@]} -eq 0 ]]; then
  echo "no physical GPUs declared" >&2
  exit 2
fi

for gpu in "${physical_ids[@]}"; do
  if [[ "$host_family" == sk3 ]]; then
    case "$gpu" in
      0|5|6|7) ;;
      1|2|3|4) echo "HARD STOP: sk3 physical GPU $gpu is permanently forbidden" >&2; exit 64 ;;
      *) echo "invalid sk3 physical GPU $gpu; permitted: 0,5,6,7" >&2; exit 64 ;;
    esac
  else
    case "$gpu" in
      0|1|2|3|4|5|6|7) ;;
      *) echo "invalid sk2 physical GPU $gpu; permitted: 0-7" >&2; exit 64 ;;
    esac
  fi
done

# Hold one nonblocking process lock per physical device before CUDA is exposed.
lock_index=0
for gpu in "${physical_ids[@]}"; do
  lock_fd=$((200 + lock_index))
  eval "exec ${lock_fd}>/tmp/cr3-v14-${host}-gpu-${gpu}.lock"
  if ! flock -n "$lock_fd"; then
    echo "physical GPU $gpu already has a v14 lane lock" >&2
    exit 75
  fi
  lock_index=$((lock_index + 1))
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HOME="$lfs_root/alexspan"
export HF_HOME="$HOME/.cache/huggingface"
export METRIC_IMPLEMENTER_LFS_HOME="$HOME"
export CUDA_VISIBLE_DEVICES="$physical_csv"
export V14_PHYSICAL_GPUS="$physical_csv"
export VLLM_TP_SIZE=${VLLM_TP_SIZE:-${#physical_ids[@]}}
export V14_MODEL_PATH_OVERRIDES_JSON=${V14_MODEL_PATH_OVERRIDES_JSON:-'{"meta-llama/Llama-3.3-70B-Instruct":"/lfs/skampere3/0/shared_hf_cache/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b","mistralai/Mistral-Small-24B-Instruct-2501":"/lfs/skampere3/0/shared_hf_cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724","Qwen/Qwen2.5-32B-Instruct":"/lfs/skampere3/0/shared_hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd","mistralai/Mistral-7B-Instruct-v0.3":"/lfs/skampere3/0/shared_hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"}'}
V14_PYTHON=${V14_PYTHON:-$HOME/miniconda3/bin/python}
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$code_root"

exec "$V14_PYTHON" -m methods.metric_implementer.experiments.run_v14_value_campaign \
  --physical-gpus "$physical_csv" "$@"
