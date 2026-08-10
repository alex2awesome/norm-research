#!/usr/bin/env bash
# Large-scale from-scratch γ/recovery across ALL real GEPA-mined Ω's, BOTH granularities (criteria + steps),
# each on its matching corpus, N=250. One free GPU. Reports spectral γ (trustworthy), R(OPT), N_eff,
# saturation. Skips empty Ω files. HOME pinned to /lfs.
set -uo pipefail
cd /lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan VLLM_GPU_MEM_UTIL=0.3
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
if [ -z "${FREE:-}" ]; then echo "NO FREE GPU (all >2000 MiB) - aborting at $(date)"; exit 1; fi
export CUDA_VISIBLE_DEVICES="$FREE"
echo "using GPU $FREE at $(date)"
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
OUT=outputs/metric_implementer_scale/gamma_runs; mkdir -p "$OUT"
CRIT=outputs/metric_implementer_scale/real_omega
STEP=outputs/metric_implementer_scale/real_omega_steps
MATH="datasets/math/stackexchange/math_se_modeling.csv.gz"
CW="datasets/creative-writing/writingprompts_modeling_clean.csv.gz"

run() {  # $1 omega_file  $2 corpus  $3 task  $4 tag
  if [ ! -s "$1" ] || [ "$(grep -c '^- ' "$1")" -lt 2 ]; then echo "SKIP (empty/<2 Ω): $1"; return; fi
  name=$(basename "$1" .md.txt)
  echo ""; echo "########## $name [$4] ($3) ##########"
  "$PY" -m methods.metric_implementer.experiments.real_gamma --pool "$2" --text-col text --task "$3" \
    --crit-file "$1" --holistic-m --n-items 250 --max-k 14 --min-std 0.04 \
    --out "$OUT/gamma_${4}_${name}.npz" 2>&1 \
    | grep -E "loaded|LIVE|behavioral dedup|spectral γ|greedy picks|OPT_Ω picks|OPTIMALITY|complexity|accumulation|synergy pairs|degenerate|Error|Traceback|assert" || echo "(no matching output - check full run)"
}

for m in aigner_ziegler_proofs_from_the_book arnold_on_teaching_mathematics gowers_chow_solution_to_exposition_problem; do
  run "$CRIT/$m.md.txt" "$MATH" math criteria
  run "$STEP/$m.md.txt" "$MATH" math steps
done
for m in ap_english_literature_rubric aristotle_poetics abbott_cambridge_introduction_narrative andrew_stanton_ted_clues_great_story; do
  run "$CRIT/$m.md.txt" "$CW" creative-writing criteria
  run "$STEP/$m.md.txt" "$CW" creative-writing steps
done
echo ""; echo "ALL DONE at $(date)"
