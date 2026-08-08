#!/bin/bash
# stall-resistant HF download: 15-min timeout per attempt, resume-on-retry (NAT64 long-stream
# stalls; morning 2026-07-09). Usage: robust_dl.sh <repo> <logname>
set -u
export HOME=/lfs/skampere3/0/alexspan HF_HOME=/lfs/skampere3/0/shared_hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_ETAG_TIMEOUT=30
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
for i in $(seq 1 60); do
  timeout 900 $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$1', max_workers=4)" && { echo "DL-DONE $1 attempt $i $(date)"; exit 0; }
  echo "DL-RETRY $1 attempt $i rc=$? $(date)"
  sleep 20
done
echo "DL-GAVE-UP $1 $(date)"; exit 1
