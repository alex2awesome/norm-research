#!/usr/bin/env bash
# Executor-bottleneck TVD recovery R = I_TVD(M; M̂_S) on a clean Ω: brute-force every subset, compile to
# ONE verdict, exact OPT + greedy + PRUNE. Gives the real R (cap ½) and the Ǐ−R executor gap.
# Args: $1 omega_file  $2 corpus  $3 task. One free GPU; HOME=/lfs.
set -uo pipefail
cd /lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan VLLM_GPU_MEM_UTIL=0.3
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
[ -z "${FREE:-}" ] && { echo "NO FREE GPU at $(date)"; exit 1; }
export CUDA_VISIBLE_DEVICES="$FREE"
echo "GPU $FREE  $(date)  Ω=$1"
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
"$PY" -m methods.metric_implementer.experiments.brute_force_class \
  --rubric-file "$1" --pool "$2" --text-col text --task "$3" \
  --model meta-llama/Llama-3.1-8B-Instruct --n-items 200 --max-chars 3000 --budget 4 \
  --out "outputs/metric_implementer_scale/gamma_runs/bfc_$(basename "$1" .md.txt).npz"
echo "DONE $(date)"
