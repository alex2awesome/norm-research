#!/bin/bash
# PUPA legs retry (teardown-lag fix): wait for GPU0 truly free before each leg.
export HOME=/lfs/skampere1/0/alexspan
export ZAI_KEY_FILE=$HOME/.z-ai-api-key-live.txt
H=$HOME
L=$H/norm-research/datasets/prompt-optimality-test/logs/pupa_chain.log
waitfree() {
  for i in $(seq 1 60); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    [ "$U" -lt 2000 ] && return 0
    sleep 20
  done
  return 1
}
echo "$(date -u +%FT%TZ) pupa chain start" >> $L
waitfree || { echo "$(date -u +%FT%TZ) GPU0 never freed" >> $L; exit 1; }
bash $H/arm_lane_sk1.sh 0 8218 pupa official_merge 600 mergesk1
echo "$(date -u +%FT%TZ) pupa merge leg done" >> $L
waitfree
bash $H/arm_lane_sk1.sh 0 8218 pupa mipro 2400 miprov2sk1
echo "$(date -u +%FT%TZ) pupa mipro leg done" >> $L
echo "$(date -u +%FT%TZ) PUPA CHAIN COMPLETE" >> $L
