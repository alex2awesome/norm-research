#!/bin/bash
# Close out the PROSE round on a free GPU (2026-06-23). GPU 2 got grabbed mid-init last attempt, so
# this pins a target GPU, waits for enough FREE memory (robust to transient contention), and uses a
# modest VLLM_GPU_MEM_UTIL (Llama-8B is ~16GB; 0.50 reservation is plenty and coexists if grabbed).
# Phase 1: compiler sweep on saved real_test3 CW npz (confirms T_prose binding + I/T_prose ratio).
# Phase 2: discriminative GEPA re-run, warm-started from the liked p̂ (w_disc=0.55, disc_healthy=0.15).
set -u
export HOME=/lfs/skampere3/0/alexspan          # AFS-token-safe; ~/.z-ai-api-key.txt lives here
TARGET_GPU="${TARGET_GPU:-3}"
export CUDA_VISIBLE_DEVICES=$TARGET_GPU
export VLLM_GPU_MEM_UTIL=0.50                   # small model; robust to a partial grab
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
cd /lfs/skampere3/0/alexspan/norm-research
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
OUT=/lfs/skampere3/0/alexspan/tmp_vinfo/prose_round
mkdir -p "$OUT"
NPZ=/lfs/skampere3/0/alexspan/tmp_vinfo/real_test3/creative-writing.npz
POOL=/lfs/skampere3/0/alexspan/norm-research/methods/metric_implementer/trial/pool_creative_writing.jsonl.gz

# wait for the target GPU to have enough free memory (up to ~40 min), else abort
need=150000   # MiB; with 0.50 util we ask for ~89GB, so 150GB free is ample slack
for i in $(seq 1 80); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $TARGET_GPU | tr -d ' ')
  if [ "${free:-0}" -gt "$need" ]; then echo "GPU $TARGET_GPU free=${free}MiB -> go"; break; fi
  echo "wait #$i: GPU $TARGET_GPU free=${free}MiB < $need; sleeping 30s"; sleep 30
  if [ $i -eq 80 ]; then echo "ABORT: GPU $TARGET_GPU never freed"; exit 2; fi
done

echo "=== PHASE 1: compiler sweep  $(date) ==="
"$PY" -m methods.metric_implementer.experiments.compiler_sweep \
  --npz "$NPZ" --pool "$POOL" --text-col text --task creative-writing \
  --model meta-llama/Llama-3.1-8B-Instruct --max-chars 3000 \
  --out "$OUT/sweep.json" > "$OUT/sweep.log" 2>&1
echo "sweep exit=$?  $(date)"; tail -n 22 "$OUT/sweep.log"

echo "=== PHASE 2: discriminative GEPA re-run (warm-start p̂ + disc boost)  $(date) ==="
"$PY" -m methods.metric_implementer.experiments.run_real_test \
  --tasks creative-writing --target-model meta-llama/Llama-3.1-8B-Instruct \
  --reconstructor-backend zai_anthropic --reconstructor-model glm-5 \
  --warm-start-npz "$NPZ" --compiler conjunction \
  --n-items 50 --rounds 3 --budget 4 --out-dir "$OUT" > "$OUT/gepa.log" 2>&1
echo "gepa exit=$?  $(date)"; tail -n 30 "$OUT/gepa.log"
echo "=== ALL DONE  $(date) ==="
