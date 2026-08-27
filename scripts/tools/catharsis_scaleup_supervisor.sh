#!/usr/bin/env bash
# catharsis_scaleup_supervisor.sh — Catharsis (closest-to-saturation) Ω-SCALE-UP. Approved #1.
#
# The saturation hunt found 0/591 genuine SATURATED CW metrics; Catharsis (α_V=0.71, the lone LONG-TAIL)
# is the closest. This tests the real bound question: does α_V PLATEAU as Ω grows (→ a recovery ceiling
# exists) or keep CLIMBING (→ genuinely inexhaustible, conclusion firm)? Runs Catharsis at M_freegen ∈
# {60,300,600} (per_call=10, matching the sweep baseline), value-censuses each, prints α_V vs M_freegen.
# One GPU, crash-resilient (per-attempt GPU re-detect + retry until each M's marker exists). Excludes GPU 0.
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
SCALE_OUT=/lfs/skampere3/0/alexspan/outputs/catharsis_scaleup
LOG=/lfs/skampere3/0/alexspan/logs/catharsis_scaleup.log
mkdir -p "$SCALE_OUT" "$(dirname "$LOG")"
cd "$REPO" || exit 2
log(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
grab_gpu(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -F', ' '$1!=0 && $2+0 < 2000 {print $1; exit}'; }

COMMON=(--task creative-writing --level R2 --r2-bucket general --target-name Catharsis
        --n-metrics 1 --target-model "$MODEL" --no-glm --n-probes 300 --gepa-reserve 60 --per-call 10)
MS=(60 300 600)
for M in "${MS[@]}"; do
  OUTM="$SCALE_OUT/M$M"; mkdir -p "$OUTM"
  MARKER="$OUTM/creative-writing_R2_metric_alpha_summary.json"
  log "===== Catharsis M_freegen=$M (marker $MARKER) ====="
  att=0
  while [ ! -f "$MARKER" ]; do
    att=$((att+1)); [ "$att" -gt 25 ] && { log "M=$M gave up after 25 attempts"; break; }
    GPU="$(grab_gpu)"
    if [ -z "$GPU" ]; then log "M=$M no free GPU — wait 5m (attempt $att)"; sleep 300; continue; fi
    export CUDA_VISIBLE_DEVICES="$GPU"
    log "M=$M attempt $att on GPU $GPU"
    timeout --kill-after=90 5400 "$PY" -m methods.metric_implementer.experiments.run_alpha_probe \
        "${COMMON[@]}" --M-freegen "$M" --out-dir "$OUTM" >> "$LOG" 2>&1
    log "M=$M attempt $att rc=$? (124=timeout/hang-reclaim)"
    sleep 15
  done
  [ -f "$MARKER" ] && log "===== M=$M DONE =====" || log "===== M=$M INCOMPLETE ====="
done

log "===== value census per M ====="
for M in "${MS[@]}"; do
  "$PY" -m methods.metric_implementer.experiments.run_value_census \
    --ckpt-dir "$SCALE_OUT/M$M" --out-dir "$SCALE_OUT/M$M/vc" >> "$LOG" 2>&1 \
    && log "M=$M VC done" || log "M=$M VC failed"
done

log "===== α_V vs M_freegen (plateau test) ====="
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import json, os, glob
base="/lfs/skampere3/0/alexspan/outputs/catharsis_scaleup"
print("M_freegen | α_i    α_V    gap    MV0    rec%  verdict")
for M in (60,300,600):
    fs=glob.glob(f"{base}/M{M}/vc/value_census_summary.json")
    if not fs: print(f"{M:9d} | (no vc)"); continue
    d=json.load(open(fs[0]))
    r=d["results"][0] if d["results"] else {}
    print("%9d | %.3f  %.3f  %+.3f  %.3f  %3.0f%%  %s" % (
        M, r.get("alpha_i",float("nan")), r.get("alpha_V_terminal",float("nan")),
        r.get("breadth_gap",float("nan")), r.get("MV0",float("nan")),
        r.get("frac_M_i_recovered",0)*100, r.get("decision",{}).get("verdict","?")))
PYEOF
log "===== CATHARSIS SCALEUP ALL DONE ====="
