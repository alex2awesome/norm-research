#!/bin/bash
# CPU-only waiter: launch fam_zxa.sh on the first GPU free (<5GB) for 3 consecutive
# 120s polls. PRECONDITIONS added 2026-07-10: wait for or_author_fam.py to exit and
# for the fam freezes to exist (avoids authoring-file clobber race with the OR lane).
set -u
O=/lfs/skampere3/0/alexspan/outputs/osl_multi
L=$O/logs/fam_waiter.log
declare -A consec
echo "$(date) waiter start (precondition-aware)" >> $L
while pgrep -f "or_author_fam.py" >/dev/null; do sleep 120; done
while [ ! -s $O/freeze_zxa_humor_fam_v1.json ]; do
  pgrep -f "or_scope.sh" >/dev/null || { echo "$(date) or_scope gone and no freeze; waiter exits (fix authoring first)" >> $L; exit 1; }
  sleep 120
done
echo "$(date) preconditions met (authoring done, freezes exist)" >> $L
while true; do
  if pgrep -f "bash $O/fam_zxa.sh" >/dev/null || pgrep -f "author_fam_arms.py" >/dev/null; then
    echo "$(date) fam lane already running; waiter exits" >> $L; exit 0
  fi
  while read -r idx mem; do
    if [ "$mem" -lt 5000 ]; then consec[$idx]=$(( ${consec[$idx]:-0} + 1 )); else consec[$idx]=0; fi
    if [ "${consec[$idx]}" -ge 3 ]; then
      echo "$(date) GPU$idx free 3 consecutive polls -> launching fam_zxa.sh $idx" >> $L
      cd $O && nohup bash fam_zxa.sh $idx >> $O/fam_zxa_gpu${idx}.log 2>&1 &
      echo "$(date) launched pid=$!" >> $L
      exit 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ',')
  sleep 120
done
