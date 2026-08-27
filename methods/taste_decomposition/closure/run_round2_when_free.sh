#!/bin/bash
# Race-free launcher for the Layer-3 round-2 Gemma scoring on sk3.
#
# The check-then-launch pattern loses the GPU between the nvidia-smi and the
# launch.  vLLM's startup check ("Free memory ... is less than desired GPU memory
# utilization") fails FAST and BEFORE any allocation, so a failed attempt is
# harmless to whoever holds the GPU.  We therefore just retry until one attempt
# gets through.
#
# GPU1 is EXCLUDED unconditionally: it belongs to the v3 code training.
# Nothing is ever killed.

export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CDIR=/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure
PY=$HOME/envs/gemma4/bin/python
MASTER=$CDIR/round2_launcher.log
NEED_MIB=100000          # need ~100GB free for Gemma-4-31B bf16 + KV cache
EXCLUDE_GPU=1            # v3 training -- never touch

cd "$CDIR" || exit 1
echo "[launcher] start $(date)" >> "$MASTER"

for attempt in $(seq 1 240); do
  if [ -f "$CDIR/round2_scores.npz" ]; then
    echo "[launcher] scores already present, stopping" >> "$MASTER"
    exit 0
  fi

  # pick the GPU with the most free memory, excluding the training GPU
  best=$(nvidia-smi --query-gpu=index,memory.total,memory.used \
           --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' -v ex="$EXCLUDE_GPU" '$1 != ex {free=$2-$3; print $1, free}' \
         | sort -k2 -nr | head -1)
  gid=$(echo "$best" | awk '{print $1}')
  gfree=$(echo "$best" | awk '{print $2}')

  if [ -n "$gfree" ] && [ "$gfree" -ge "$NEED_MIB" ]; then
    # vLLM's gpu_memory_utilization is a fraction of TOTAL memory, so on a shared
    # GPU it must be sized from what is ACTUALLY free (take 90% of free), never
    # from a fixed 0.85 -- that asks for 151GB and fails on any occupied card.
    gtotal=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gid")
    util=$(awk -v f="$gfree" -v t="$gtotal" 'BEGIN{printf "%.2f", (f*0.90)/t}')
    echo "[launcher] attempt $attempt: trying GPU $gid (${gfree}MiB free of ${gtotal}, util=$util) $(date)" >> "$MASTER"
    CUDA_VISIBLE_DEVICES=$gid GEMMA_UTIL=$util $PY score_round2_gemma.py > "$CDIR/score_round2.log" 2>&1
    if grep -q ROUND2_SCORE_DONE "$CDIR/score_round2.log"; then
      echo "[launcher] SUCCESS on GPU $gid $(date)" >> "$MASTER"
      exit 0
    fi
    echo "[launcher] attempt $attempt failed on GPU $gid: $(grep -Eo 'ValueError: Free memory[^\"]*' "$CDIR/score_round2.log" | head -1)" >> "$MASTER"
  else
    echo "[launcher] attempt $attempt: best non-training GPU $gid has only ${gfree}MiB free, waiting" >> "$MASTER"
  fi
  sleep 180
done

echo "[launcher] GAVE UP after 240 attempts $(date)" >> "$MASTER"
exit 1
