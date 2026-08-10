#!/usr/bin/env bash
# cw_multilevel_alpha_supervisor.sh — CW multi-level saturation HUNT, ONE GPU at a time, crash-resilient.
#
# Runs the metric α-probe (→ M_i-bearing checkpoints) sequentially across R3 → R1(sample) → R2(all),
# then a final M_i value census over every checkpoint. The "SATURATED hunt": value census flags metrics
# where a few criteria recover M_i (α_V≪1, MV0 small) — the certifiable case (0 found in the 11 largest).
#
# Crash-resilience: each level loops until its summary marker exists. Every attempt re-detects a free GPU
# (migrating if the held one was lost) and passes --skip-existing, so a crashed/hung engine loses at most
# the in-progress metric and resumes cleanly. `timeout --kill-after` reclaims a hung (even D-state-prone)
# engine. Excludes GPU 0 (reserved).
#
# Launch:  nohup bash scripts/tools/cw_multilevel_alpha_supervisor.sh > /dev/null 2>&1 &
# Tail:    tail -f /lfs/skampere3/0/alexspan/logs/cw_multilevel_alpha_supervisor.log
set -uo pipefail
REPO=/lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export HUGGINGFACE_HUB_CACHE=/lfs/skampere3/0/shared_hf_cache/hub
export HF_HUB_OFFLINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_MOE_FP8=0
export TOKENIZERS_PARALLELISM=false
export VLLM_GPU_MEM_UTIL=0.93
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
MODEL=meta-llama/Llama-3.1-8B-Instruct
TASK=creative-writing
BUCKET=general
OUT=/lfs/skampere3/0/alexspan/outputs/alpha_probe_metric
VCOUT=/lfs/skampere3/0/alexspan/outputs/value_census_cw_all
LOG=/lfs/skampere3/0/alexspan/logs/cw_multilevel_alpha_supervisor.log
mkdir -p "$OUT" "$VCOUT" "$(dirname "$LOG")"
cd "$REPO" || exit 2

log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

grab_gpu(){  # first free GPU (<2000 MiB), excluding index 0 (reserved)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1!=0 && $2+0 < 2000 {print $1; exit}'
}

COMMON=(--task "$TASK" --r2-bucket "$BUCKET" --target-model "$MODEL" --no-glm
        --n-probes 300 --gepa-reserve 60 --out-dir "$OUT" --skip-existing)
# LEVEL|extra-args|summary-marker   (order: R3 smallest→fastest first to validate on real data)
SPECS=(
  "R3|--level R3 --n-metrics 0|${OUT}/${TASK}_R3_metric_alpha_summary.json"
  "R1|--level R1 --n-metrics 150|${OUT}/${TASK}_R1_metric_alpha_summary.json"
  "R2|--level R2 --n-metrics 0 --largest-first|${OUT}/${TASK}_R2_metric_alpha_summary.json"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r LVL EXTRA MARKER <<< "$spec"
  log "===== LEVEL $LVL start (marker $MARKER) ====="
  attempt=0
  while [ ! -f "$MARKER" ]; do
    attempt=$((attempt+1))
    if [ "$attempt" -gt 80 ]; then log "LEVEL $LVL: gave up after 80 attempts"; break; fi
    GPU="$(grab_gpu)"
    if [ -z "$GPU" ]; then log "$LVL: no free GPU — wait 5m (attempt $attempt)"; sleep 300; continue; fi
    export CUDA_VISIBLE_DEVICES="$GPU"
    log "$LVL attempt $attempt on GPU $GPU :: run_alpha_probe ${COMMON[*]} $EXTRA"
    timeout --kill-after=90 7200 "$PY" -m methods.metric_implementer.experiments.run_alpha_probe \
        "${COMMON[@]}" $EXTRA >> "$LOG" 2>&1
    rc=$?
    log "$LVL attempt $attempt ended rc=$rc (0=clean-finish, 124=timeout/hang-reclaim, else=crash)"
    sleep 15   # brief backoff before re-grabbing a GPU and resuming
  done
  [ -f "$MARKER" ] && log "===== LEVEL $LVL DONE =====" || log "===== LEVEL $LVL INCOMPLETE (see above) ====="
done

log "===== VALUE CENSUS over all CW checkpoints (CPU) ====="
"$PY" -m methods.metric_implementer.experiments.run_value_census --ckpt-dir "$OUT" --out-dir "$VCOUT" >> "$LOG" 2>&1 \
  && log "value census → $VCOUT/value_census_summary.json" \
  || log "value census FAILED (run manually: run_value_census --ckpt-dir $OUT)"
log "===== ALL DONE ====="
