#!/bin/bash
# Detached V_new refit chain: waits for the RUNNING certification job (bqc43msj2,
# not relaunched) to emit cert_<cell>.json per cell, then refits, then sentinel.
cd "$(dirname "$0")"
LOG=vnew_out/chain_refit.log
echo "chain_refit start $(date -u +%FT%TZ) ppid=$PPID" >> $LOG
for c in jokes_community press_verdict nc_responded; do
  n=0
  until [ -f vnew_out/cert_${c}.json ]; do
    sleep 60; n=$((n+1))
    if [ $n -gt 90 ]; then echo "TIMEOUT waiting cert_${c}" >> $LOG; echo "TIMEOUT cert_${c}" > vnew_out/REFIT_DONE; exit 1; fi
  done
  sleep 5
  if [ ! -f vnew_out/results_${c}.json ]; then
    echo "refit $c start $(date -u +%FT%TZ)" >> $LOG
    python3 pilot.py refit $c >> $LOG 2>&1
    echo "refit $c rc=$? $(date -u +%FT%TZ)" >> $LOG
  fi
done
echo "ALL_REFITS_DONE $(date -u +%FT%TZ)" > vnew_out/REFIT_DONE
echo "chain_refit done $(date -u +%FT%TZ)" >> $LOG
