#!/bin/bash
# Homepage-curation completion chain: rebuilt A bank -> T0 -> layer 1.
# ONE card, stages run in sequence on it. Caller supplies GPU (the launcher claims it).
#
#   GPU=3 bash methods/dense/run_homepage_v2_chain.sh
#
# Stage 0  legacy triage  : per-criterion coherence for the 14 census criteria, through
#                           the ORIGINAL system prompt, so "which criteria were entity
#                           detectors" is MEASURED and the salvage decision is evidenced.
# Stage 1  pilot          : 300 items x 29 criteria + K>=50 batteries on the v2 bank.
# Stage 2  smoke gate     : label-blind coherence/distribution gate. FAIL => exit 7 and
#                           the criteria get revised (the GEPA iteration for this bank).
# Stage 3  full scoring   : 12,998 x 29 = 376,942 judge calls, shard-checkpointed.
# Stage 4  T0             : rows builder (CPU) then base-Llama-3.1-8B zero-shot scoring.
# Stage 5  layer 1        : CPU ledger (also runnable off-box).
#
# Every stage is resumable: stage 3 skips shards whose npz exists, stage 4 skips a cell
# whose score file exists, so a retry after a lost card resumes rather than restarts.
set -u
export HOME=/lfs/skampere3/0/alexspan
NR=$HOME/norm-research
PY=$HOME/envs/gemma4/bin/python
LOGS=$NR/logs/homepage_v2
mkdir -p "$LOGS"
cd "$NR" || exit 1

export CUDA_VISIBLE_DEVICES=${GPU:?GPU must be set}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NR_REPO=$NR
export VA_OUT_HP2=$NR/outputs/va_gemma_banks_homepage_v2
mkdir -p "$VA_OUT_HP2"

SCORER=$NR/datasets/va_gemma_banks/score_homepage_v2_bank.py
ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }
say () { echo "[chain $(ts)] $*"; }

# ---------------------------------------------------------------- stage 0 ---
if [ ! -f "$VA_OUT_HP2/.stage0_done" ]; then
  say "stage 0: legacy 14-criterion coherence triage"
  $PY "$SCORER" --tasks homepage_legacy_triage --battery-only --battery 50 \
      --auto-util --util 0.93 --min-gib 80 > "$LOGS/stage0_triage.log" 2>&1
  rc=$?; say "stage 0 rc=$rc"
  [ $rc -ne 0 ] && exit $rc
  touch "$VA_OUT_HP2/.stage0_done"
else
  say "stage 0 already done, skipping"
fi

# ---------------------------------------------------------------- stage 1 ---
if [ ! -f "$VA_OUT_HP2/.stage1_done" ]; then
  say "stage 1: v2 pilot (300 items) + K=50 batteries"
  $PY "$SCORER" --tasks homepage_v2 --smoke 300 --battery 50 \
      --auto-util --util 0.93 --min-gib 80 > "$LOGS/stage1_pilot.log" 2>&1
  rc=$?; say "stage 1 rc=$rc"
  [ $rc -ne 0 ] && exit $rc
  touch "$VA_OUT_HP2/.stage1_done"
else
  say "stage 1 already done, skipping"
fi

# ---------------------------------------------------------------- stage 2 ---
say "stage 2: smoke gate"
$PY "$NR/methods/taste_decomposition/check_homepage_smoke_gate.py" \
    2>&1 | tee "$LOGS/stage2_gate.log"
grc=${PIPESTATUS[0]}
say "stage 2 gate rc=$grc"
if [ "$grc" -ne 0 ]; then
  say "GATE FAILED -- stopping before the full run. Revise rubrics_v2.jsonl, delete"
  say "  $VA_OUT_HP2/.stage1_done, and re-launch (stage 0 stays done)."
  exit 7
fi

# ---------------------------------------------------------------- stage 3 ---
if [ ! -f "$VA_OUT_HP2/.stage3_done" ]; then
  say "stage 3: full v2 scoring (12,998 x 29)"
  $PY "$SCORER" --tasks homepage_v2 --battery 50 \
      --auto-util --util 0.93 --min-gib 80 > "$LOGS/stage3_score.log" 2>&1
  rc=$?; say "stage 3 rc=$rc"
  [ $rc -ne 0 ] && exit $rc
  grep -q HOMEPAGE_V2_SCORE_DONE "$LOGS/stage3_score.log" || { say "no DONE sentinel"; exit 3; }
  touch "$VA_OUT_HP2/.stage3_done"
else
  say "stage 3 already done, skipping"
fi

# ---------------------------------------------------------------- stage 4 ---
if [ ! -f "$NR/methods/taste_decomposition/fusion/t0_scores/homepage_curation_storygrouped.jsonl.gz" ]; then
  say "stage 4a: T0 rows"
  $PY "$NR/methods/taste_decomposition/fusion/t0_build_rows_homepage.py" \
      > "$LOGS/stage4a_t0_rows.log" 2>&1
  rc=$?; say "stage 4a rc=$rc"
  [ $rc -ne 0 ] && exit $rc
  say "stage 4b: T0 scoring (base Llama-3.1-8B, frozen template)"
  $PY "$NR/methods/taste_decomposition/fusion/t0_score_vllm.py" \
      --cell homepage_curation_storygrouped --gpu-frac 0.45 \
      > "$LOGS/stage4b_t0_score.log" 2>&1
  rc=$?; say "stage 4b rc=$rc"
  [ $rc -ne 0 ] && exit $rc
else
  say "stage 4 already done, skipping"
fi

# ---------------------------------------------------------------- stage 5 ---
say "stage 5: layer 1 (CPU)"
$PY "$NR/methods/taste_decomposition/homepage_v2_layer1.py" \
    > "$LOGS/stage5_layer1.log" 2>&1
rc=$?; say "stage 5 rc=$rc"
[ $rc -ne 0 ] && exit $rc

say "HOMEPAGE_V2_CHAIN_DONE"
exit 0
