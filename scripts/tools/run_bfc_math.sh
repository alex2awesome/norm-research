#!/usr/bin/env bash
# Executor-bottleneck TVD recovery R across the math Ω's (criteria + aigner steps). One free GPU.
set -uo pipefail
cd /lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan VLLM_GPU_MEM_UTIL=0.3
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
[ -z "${FREE:-}" ] && { echo "NO FREE GPU at $(date)"; exit 1; }
export CUDA_VISIBLE_DEVICES="$FREE"; echo "GPU $FREE $(date)"
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
CRIT=outputs/metric_implementer_scale/real_omega
STEP=outputs/metric_implementer_scale/real_omega_steps
MATH="datasets/math/stackexchange/math_se_modeling.csv.gz"
OUT=outputs/metric_implementer_scale/gamma_runs

run() {  # $1 omega  $2 corpus  $3 task  $4 tag
  name=$(basename "$1" .md.txt)
  echo ""; echo "##### bfc $name [$4] #####"
  "$PY" -m methods.metric_implementer.experiments.brute_force_class --rubric-file "$1" --pool "$2" \
    --text-col text --task "$3" --model meta-llama/Llama-3.1-8B-Instruct --n-items 200 --max-chars 3000 \
    --budget 4 --out "$OUT/bfc_${4}_${name}.npz" 2>&1 \
    | grep -E "EXACT within|greedy\(k\)|OPT_.|CERTIFICATE|global best|PRUNE|γ on|per-criterion|X_[0-9]:|Error|Traceback"
}

run "$CRIT/arnold_on_teaching_mathematics.md.txt"             "$MATH" math criteria
run "$CRIT/gowers_chow_solution_to_exposition_problem.md.txt" "$MATH" math criteria
run "$STEP/aigner_ziegler_proofs_from_the_book.md.txt"        "$MATH" math steps
echo ""; echo "ALL DONE $(date)"
