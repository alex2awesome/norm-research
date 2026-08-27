#!/bin/bash
# first-free-GPU waiter for the flip queue (allowed GPUs 3 5 6 7; double-check claim)
set -u
export HOME=/lfs/skampere3/0/alexspan
O=/lfs/skampere3/0/alexspan/outputs/osl_multi
echo "FLIPQ WAITER ALIVE pid $$ $(date)" >> $O/FLEET_STATUS
while true; do
  for g in 3 5 6 7; do
    U1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$U1" -ge 2000 ] && continue
    sleep 60
    U2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$U2" -ge 2000 ] && continue
    echo "FLIPQ CLAIM gpu$g $(date)" >> $O/FLEET_STATUS
    exec bash $O/flipqueue_lane.sh $g
  done
  sleep 300
done
