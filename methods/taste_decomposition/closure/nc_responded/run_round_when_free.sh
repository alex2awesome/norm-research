#!/bin/bash
# Race-free launcher for the N&C RESPONDED closure Gemma scoring on sk3.
#
# The check-then-launch pattern loses the GPU between the nvidia-smi and the launch.
# vLLM's startup check ("Free memory ... is less than desired GPU memory utilization")
# fails FAST and BEFORE any allocation, so a failed attempt is harmless to whoever
# holds the GPU.  We therefore just retry until one attempt gets through.
#
# GPU ledger protocol: claim BEFORE launching, release when done.
# Co-tenant GPUs are NEVER touched and nothing is ever killed.
#
# Usage: ROUND=1 bash run_round_when_free.sh

set -u
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/alexspan/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn

R=${ROUND:?set ROUND}
CDIR=/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/nc_responded
PY=$HOME/envs/gemma4/bin/python
LEDGER=/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt
MASTER=$CDIR/round${R}_launcher.log
NEED_MIB=100000          # ~100GB free for Gemma-4-31B bf16 + KV cache

# GPUs CURRENTLY claimed by OTHER agents.  A ledger CLAIM is only live until the
# matching RELEASE, so a plain "every GPU ever mentioned" grep over-excludes and
# eventually starves this job (observed at round 4: 5 of 8 GPUs excluded while 4 of
# them sat at 0 MiB).  Walk the ledger in order and keep, per GPU, only the LAST
# event; exclude a GPU iff its last event is another agent's CLAIM.
recompute_exclude() {
  awk '
    /^#/ {next}
    {
      gid=""
      if (match($0, /GPU=[0-9]+/)) gid=substr($0, RSTART+4, RLENGTH-4)
      if (gid=="") next
      if ($0 ~ /RELEASE/ || $0 ~ /ALL JOBS COMPLETE/ || $0 ~ /CAMPAIGN COMPLETE/) { last[gid]="free"; next }
      if ($0 ~ /agent=claude-closure-nc/) { last[gid]="mine"; next }
      last[gid]="other"
    }
    END { for (g in last) if (last[g]=="other") printf "%s ", g }
  ' "$LEDGER" 2>/dev/null
}
EXCLUDE=$(recompute_exclude)

cd "$CDIR" || exit 1
echo "[launcher r$R] start $(date); ledger-excluded GPUs: [$EXCLUDE]" >> "$MASTER"

for attempt in $(seq 1 480); do
  EXCLUDE=$(recompute_exclude)
  if [ -f "$CDIR/round${R}_scores.npz" ]; then
    echo "[launcher r$R] scores already present, stopping" >> "$MASTER"
    exit 0
  fi

  best=$(nvidia-smi --query-gpu=index,memory.total,memory.used \
           --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' -v ex="$EXCLUDE" '
             BEGIN{n=split(ex,a," "); for(i=1;i<=n;i++) bad[a[i]]=1}
             !($1 in bad){free=$2-$3; print $1, free}' \
         | sort -k2 -nr | head -1)
  gid=$(echo "$best" | awk '{print $1}')
  gfree=$(echo "$best" | awk '{print $2}')

  if [ -n "${gfree:-}" ] && [ "$gfree" -ge "$NEED_MIB" ]; then
    gtotal=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gid")
    # gpu_memory_utilization is a fraction of TOTAL memory, so on a shared GPU it
    # must be sized from what is ACTUALLY free (90% of free), never a fixed .85.
    util=$(awk -v f="$gfree" -v t="$gtotal" 'BEGIN{printf "%.2f", (f*0.90)/t}')
    echo "$(date -u +%FT%TZ) | cell=nc_responded (layer3 closure confirmatory) | GPU=$gid | agent=claude-closure-nc | purpose=gemma4-31b round-$R criterion scoring" >> "$LEDGER"
    echo "[launcher r$R] attempt $attempt: GPU $gid (${gfree}MiB free of ${gtotal}, util=$util) $(date)" >> "$MASTER"
    CUDA_VISIBLE_DEVICES=$gid GEMMA_UTIL=$util ROUND=$R $PY score_round_gemma.py \
        > "$CDIR/score_round${R}.log" 2>&1
    if grep -q "ROUND${R}_SCORE_DONE" "$CDIR/score_round${R}.log"; then
      echo "[launcher r$R] SUCCESS on GPU $gid $(date)" >> "$MASTER"
      echo "$(date -u +%FT%TZ) | RELEASE GPU=$gid | agent=claude-closure-nc | round $R done" >> "$LEDGER"
      exit 0
    fi
    echo "$(date -u +%FT%TZ) | RELEASE GPU=$gid | agent=claude-closure-nc | attempt failed" >> "$LEDGER"
    echo "[launcher r$R] attempt $attempt failed on GPU $gid: $(grep -Eo 'ValueError: Free memory[^\"]*' "$CDIR/score_round${R}.log" | head -1)" >> "$MASTER"
  else
    echo "[launcher r$R] attempt $attempt: best free GPU $gid has only ${gfree:-0}MiB, waiting" >> "$MASTER"
  fi
  sleep 180
done

echo "[launcher r$R] GAVE UP after 480 attempts $(date)" >> "$MASTER"
exit 1
