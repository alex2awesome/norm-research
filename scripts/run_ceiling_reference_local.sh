#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUT_ROOT" >&2
  exit 2
fi
out_root=$(cd -- "$1" && pwd)
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${CEILING_LOCAL_PYTHON:-python}
mkdir -p "$out_root/reference_logs"

mapfile_cmd='import pandas as pd,sys; print("\n".join(pd.read_parquet(sys.argv[1]).sort_values(["task","metric_key"]).metric_key))'
metric_keys=()
while IFS= read -r key; do metric_keys+=("$key"); done < <(
  "$python_bin" -c "$mapfile_cmd" "$out_root/design_index.parquet"
)
if [[ ${#metric_keys[@]} -ne 35 ]]; then
  echo "reference launch requires exactly 35 frozen metrics" >&2
  exit 1
fi

worker() {
  local part=$1
  local log="$out_root/reference_logs/final_part_${part}.log"
  for ((index=part; index<${#metric_keys[@]}; index+=2)); do
    "$python_bin" -m methods.metric_implementer.experiments.ceiling_ladder \
      --phase reference --out-root "$out_root" \
      --metric-keys "${metric_keys[$index]}" --void-on-reference-failure
  done >>"$log" 2>&1
}

cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
worker 0 & pid0=$!
worker 1 & pid1=$!
printf '%s\n' "$pid0" >"$out_root/reference_logs/final_part_0.pid"
printf '%s\n' "$pid1" >"$out_root/reference_logs/final_part_1.pid"
wait "$pid0"
wait "$pid1"
"$python_bin" -m methods.metric_implementer.experiments.ceiling_ladder \
  --phase reference-assemble --out-root "$out_root"
