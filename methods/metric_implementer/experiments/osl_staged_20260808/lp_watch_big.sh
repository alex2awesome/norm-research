#!/bin/bash
# Waits for a GPU with >=155 GiB free, then runs the LP frontier legs (llama70b,
# qwen25-72b) on it via lp2_zxa.sh. Skips GPU2 (fam lane holds it for its own queue).
# Single instance; polls every 10 min for up to 24h.
set -u
B=/lfs/skampere3/0/alexspan; O=$B/outputs/osl_multi
for i in $(seq 1 144); do
  for want in llama70b qwen25-72b; do :; done
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' '$1 != 2 && $2 > 158000 {print $1; exit}')
  if [ -n "${GPU:-}" ]; then
    NEED=0
    for EX in llama70b qwen25-72b; do
      [ -s $O/mbar_zxaLP_humor_${EX}.npz ] || NEED=1
    done
    [ $NEED -eq 0 ] && break
    echo "LPWATCH claiming GPU$GPU $(date)" >> $O/FLEET_STATUS
    bash $O/lp2_zxa.sh $GPU 0.90 llama70b qwen25-72b
  fi
  sleep 600
done
echo "LPWATCH exit $(date)" >> $O/FLEET_STATUS
