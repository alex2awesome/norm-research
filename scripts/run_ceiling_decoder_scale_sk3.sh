#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BASE_LADDER_ROOT SCALE_ROOT" >&2
  exit 2
fi
base_root=$1
scale_root=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$scale_root"

models=(
  meta-llama/Llama-3.1-8B-Instruct
  mistralai/Mistral-Small-24B-Instruct-2501
  Qwen/Qwen2.5-32B-Instruct
  meta-llama/Llama-3.3-70B-Instruct
)
for model in "${models[@]}"; do
  safe=${model//\//__}
  root="$scale_root/$safe"
  mkdir -p "$root"
  [[ -e "$root/designs" ]] || ln -s "$base_root/designs" "$root/designs"
  [[ -e "$root/design_index.parquet" ]] || ln -s "$base_root/design_index.parquet" "$root/design_index.parquet"
  [[ -e "$root/reference" ]] || ln -s "$base_root/reference" "$root/reference"
  [[ -e "$root/planted" ]] || ln -s "$base_root/planted" "$root/planted"
  "$script_dir/run_ceiling_ladder_sk3.sh" constructor "$root" 5 \
    --constructor-model "$model"
  "$script_dir/run_ceiling_ladder_sk3.sh" executor "$root" 7
  "$script_dir/run_ceiling_ladder_sk3.sh" aggregate "$root"
done
