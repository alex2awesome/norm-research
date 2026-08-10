#!/usr/bin/env bash
# Self-contained watcher: wait for a TRULY idle GPU (<2000 MiB used), then run the cross-corpus
# recovery (recon_channel) on exactly that one GPU. HOME pinned to /lfs (AFS-token safe).
set -uo pipefail
cd /lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan
export VLLM_GPU_MEM_UTIL=0.3
PY=/lfs/skampere3/0/alexspan/miniconda3/bin/python
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$2 < 2000 {print $1; exit}')
  if [ -n "${FREE:-}" ]; then
    echo "$(date): GPU ${FREE} idle (<2000 MiB) -> launching recon_channel cross-corpus"
    export CUDA_VISIBLE_DEVICES="${FREE}"
    exec "$PY" -m methods.metric_implementer.recon_channel \
      --tasks math_se,peer_review,news_homepages,reddit_humor,patents,notice_and_comment \
      --mode free --out outputs/metric_implementer_scale/recon_crosscorpus.json
  fi
  echo "$(date): no idle GPU (all >2000 MiB used); sleeping 120s"
  sleep 120
done
