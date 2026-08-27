#!/usr/bin/env bash
# Executor-bottleneck TVD recovery R across the creative-corpus Ω's (criteria). One free GPU.
set -uo pipefail
cd /lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan VLLM_GPU_MEM_UTIL=0.3
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
[ -z "${FREE:-}" ] && { echo "NO FREE GPU at $(date)"; exit 1; }
export CUDA_VISIBLE_DEVICES="$FREE"; echo "GPU $FREE $(date)"
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
CRIT=outputs/metric_implementer_scale/real_omega
CW="datasets/creative-writing/writingprompts_modeling_clean.csv.gz"
OUT=outputs/metric_implementer_scale/gamma_runs

run() {  # $1 omega
  name=$(basename "$1" .md.txt)
  echo ""; echo "##### bfc $name [criteria] (creative) #####"
  "$PY" -m methods.metric_implementer.experiments.brute_force_class --rubric-file "$1" --pool "$CW" \
    --text-col text --task creative-writing --model meta-llama/Llama-3.1-8B-Instruct --n-items 200 \
    --max-chars 3000 --budget 4 --out "$OUT/bfc_criteria_$name.npz" 2>&1 \
    | grep -E "EXACT within|greedy\(k\)|OPT_|global best|PRUNE|γ on|per-criterion|X_[0-9]:|Error|Traceback"
}

run "$CRIT/ap_english_literature_rubric.md.txt"
run "$CRIT/aristotle_poetics.md.txt"
run "$CRIT/abbott_cambridge_introduction_narrative.md.txt"
run "$CRIT/andrew_stanton_ted_clues_great_story.md.txt"
echo ""; echo "ALL DONE $(date)"
