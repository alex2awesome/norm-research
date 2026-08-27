#!/bin/bash
# Poll for a genuinely FREE GPU (0 MiB / 0% util, no un-released ledger claim),
# claim it in gpu_ledger.txt, then launch the CW-expert rebuild chain on it.
# Both cells stack on that ONE card (they are small). Releases on exit.
#
#   HOME=/lfs/skampere3/0/alexspan nohup bash scripts/tools/cw_expert_gpu_claim_and_launch.sh &
set -u
export HOME=/lfs/skampere3/0/alexspan
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt
AGENT=claude-cw-expert-rebuild
JOB=cw_expert_chain
LOGS=$NR/logs/cw_expert
mkdir -p "$LOGS"

ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }

# A GPU is claimable if nvidia-smi shows 0 MiB used AND 0% util AND the ledger has
# no CLAIM for it that is not followed by a later RELEASE for the same GPU.
ledger_free () {
  local g=$1
  awk -v g="$g" '
    $0 ~ ("GPU=" g "[^0-9]") || $0 ~ ("GPU=" g "$") {
      if ($0 ~ /RELEASE/) claimed = 0;
      else if ($0 ~ /CLAIM/) claimed = 1;
    }
    END { exit (claimed ? 1 : 0) }
  ' "$LEDGER"
}

# DEFAULT (strict, as instructed): only a card at 0 MiB / 0% util with no un-released
# ledger claim is claimable.
#
# OPT-IN: ALLOW_STACK=1 additionally accepts a ledger-free card with at least
# STACK_MIN_FREE_MIB free (default 92160 = 90 GiB, enough for Gemma-4-31B bf16 plus
# headroom), i.e. the CLAIM-STACKED pattern the rest of this box already uses. It is
# OFF by default and must be turned on deliberately -- on 2026-08-08 the box had no
# card at 0 MiB at all (GPU4 held 115 GiB by another user's 12-day parked VLLM::
# EngineCore, GPU6 52 GiB by another agent's 7.5 h job), so the strict rule can be
# unsatisfiable for hours. Co-tenants are NEVER touched either way.
ALLOW_STACK=${ALLOW_STACK:-0}
STACK_MIN_FREE_MIB=${STACK_MIN_FREE_MIB:-92160}

pick_gpu () {
  local total free
  while read -r idx used util; do
    idx=${idx%,}; used=${used%,}
    used=$(echo "$used" | tr -dc '0-9'); util=$(echo "$util" | tr -dc '0-9')
    [ -z "$used" ] && continue
    ledger_free "$idx" || continue
    if [ "$used" -eq 0 ] && [ "$util" -eq 0 ]; then
      echo "$idx"; return 0
    fi
    if [ "$ALLOW_STACK" = "1" ]; then
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$idx" | tr -dc '0-9')
      free=$(( total - used ))
      if [ "$free" -ge "$STACK_MIN_FREE_MIB" ]; then
        echo "$idx"; return 0
      fi
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)
  return 1
}

ATTEMPTS=${ATTEMPTS:-12}
for attempt in $(seq 1 $ATTEMPTS); do
echo "[attempt $attempt/$ATTEMPTS] $(ts)"
echo "[poll] waiting for a free GPU $(ts)"
GPU=""
# 10 s cadence + a short 6 s settle: the box is heavily contended and other agents
# CLAIM-STACKED within seconds of a release, so a long settle loses the card. Still
# strictly "wait for a FREE card" -- 0 MiB / 0% util / no un-released ledger claim --
# never crowding a card someone else is holding.
for i in $(seq 1 8640); do          # up to ~24 h at 10 s
  if cand=$(pick_gpu); then
    sleep 6                         # settle: re-verify the card is still idle
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$cand" | tr -dc '0-9')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$cand" | tr -dc '0-9')
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$cand" | tr -dc '0-9')
    free=$(( total - ${used:-total} ))
    if ledger_free "$cand" && { { [ "${used:-1}" -eq 0 ] && [ "${util:-1}" -eq 0 ]; } \
         || { [ "$ALLOW_STACK" = "1" ] && [ "$free" -ge "$STACK_MIN_FREE_MIB" ]; }; }; then
      GPU=$cand; break
    fi
    echo "[poll] GPU$cand taken during settle (used=$used util=$util), keep waiting $(ts)"
  fi
  sleep 10
done

if [ -z "$GPU" ]; then echo "[poll] no free GPU within budget, giving up $(ts)"; exit 2; fi

echo "$(ts) | cell=cw_royalroad_verdict + cw_wigleaf_curation (CW expert rebuild: mature GEPA/Gemma-4-31B A bank + FIRST-EVER dense T) | GPU=$GPU | agent=$AGENT | job=$JOB (both cells stacked on one card: gemma scoring -> 6 LoRA runs -> eval pass) | CLAIM (mode=$([ "$ALLOW_STACK" = 1 ] && echo STACKED-ok || echo strict-0MiB); free=${free:-?} MiB immediately before claim; co-tenants NEVER touched)" >> "$LEDGER"
echo "[claim] GPU=$GPU claimed $(ts)"

GPU=$GPU bash "$NR/methods/dense/run_cw_expert_chain.sh" > "$LOGS/chain.log" 2>&1
rc=$?
echo "[chain] EXIT $rc $(ts)"

# release: only after confirming none of OUR processes remain on the card
mine=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" | tr -d ' ' | tr '\n' ' ')
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=$JOB | RELEASE rc=$rc (remaining compute PIDs on card: ${mine:-none} -- any listed belong to other agents/users and were never touched)" >> "$LEDGER"
echo "[release] GPU=$GPU released rc=$rc $(ts)"

if [ $rc -eq 0 ]; then
  echo "CW_EXPERT_LAUNCHER_DONE rc=0"
  exit 0
fi
# Lost the card to a co-tenant that stacked during vLLM init, or a transient
# failure. Stage 1 is shard-checkpointed and stage 2 is RUN_DONE-sentinel
# resumable, so re-polling for a freer card resumes rather than restarts.
echo "[retry] rc=$rc — re-polling for another free GPU $(ts)"
sleep 60
done

echo "CW_EXPERT_LAUNCHER_DONE rc=exhausted after $ATTEMPTS attempts"
exit 1
