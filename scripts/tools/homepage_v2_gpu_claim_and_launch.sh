#!/bin/bash
# Poll for a claimable GPU, claim it in gpu_ledger.txt, run the homepage-curation
# completion chain on it, release on exit. Same discipline as
# scripts/tools/cw_expert_gpu_claim_and_launch.sh (that file is the precedent; this one
# differs only in the job it launches and in the GPU allow/deny list).
#
#   HOME=/lfs/skampere3/0/alexspan nohup bash scripts/tools/homepage_v2_gpu_claim_and_launch.sh &
#
# LANE PROTECTION: GPUs 5, 6 and 7 are held by lanes A/B/C and are NEVER candidates,
# even when nvidia-smi shows them momentarily idle. Only the cards in ALLOW_GPUS are
# ever considered (default 0,1,2,3 -- the patents cards, which free up as those jobs
# finish). GPU4 is excluded too: another user's VLLM::EngineCore has been parked on it
# for days.
set -u
export HOME=/lfs/skampere3/0/alexspan
NR=/lfs/skampere3/0/alexspan/norm-research
LEDGER=$NR/gpu_ledger.txt
AGENT=claude-homepage-curation-completion
JOB=homepage_v2_chain
LOGS=$NR/logs/homepage_v2
mkdir -p "$LOGS"

ALLOW_GPUS=${ALLOW_GPUS:-0,1,2,3}
# strict by default: 0 MiB / 0% util and no un-released ledger CLAIM.
ALLOW_STACK=${ALLOW_STACK:-0}
STACK_MIN_FREE_MIB=${STACK_MIN_FREE_MIB:-96256}   # 94 GiB: Gemma-4-31B bf16 + headroom
ATTEMPTS=${ATTEMPTS:-12}
POLL_TICKS=${POLL_TICKS:-8640}                    # ~24 h at 10 s

ts () { date -u +%Y-%m-%dT%H:%M:%SZ; }

allowed () { case ",$ALLOW_GPUS," in *",$1,"*) return 0;; *) return 1;; esac; }

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

pick_gpu () {
  local total free
  while read -r idx used util; do
    idx=${idx%,}; used=${used%,}
    used=$(echo "$used" | tr -dc '0-9'); util=$(echo "$util" | tr -dc '0-9')
    [ -z "$used" ] && continue
    allowed "$idx" || continue
    ledger_free "$idx" || continue
    if [ "$used" -eq 0 ] && [ "$util" -eq 0 ]; then echo "$idx"; return 0; fi
    if [ "$ALLOW_STACK" = "1" ]; then
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$idx" | tr -dc '0-9')
      free=$(( total - used ))
      if [ "$free" -ge "$STACK_MIN_FREE_MIB" ]; then echo "$idx"; return 0; fi
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)
  return 1
}

for attempt in $(seq 1 $ATTEMPTS); do
echo "[attempt $attempt/$ATTEMPTS] $(ts)"
GPU=""
for i in $(seq 1 $POLL_TICKS); do
  if cand=$(pick_gpu); then
    sleep 6
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$cand" | tr -dc '0-9')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$cand" | tr -dc '0-9')
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$cand" | tr -dc '0-9')
    free=$(( total - ${used:-total} ))
    if ledger_free "$cand" && { { [ "${used:-1}" -eq 0 ] && [ "${util:-1}" -eq 0 ]; } \
         || { [ "$ALLOW_STACK" = "1" ] && [ "$free" -ge "$STACK_MIN_FREE_MIB" ]; }; }; then
      GPU=$cand; break
    fi
    echo "[poll] GPU$cand taken during settle (used=$used util=$util) $(ts)"
  fi
  sleep 10
done

if [ -z "$GPU" ]; then echo "[poll] no claimable GPU within budget $(ts)"; exit 2; fi

echo "$(ts) | cell=homepage_curation (journalism CURATION completion: rebuilt GEPA/Gemma-4-31B A bank after the census bank's coherence failure, + T0 column) | GPU=$GPU | agent=$AGENT | job=$JOB | CLAIM (mode=$([ "$ALLOW_STACK" = 1 ] && echo STACKED-ok || echo strict-0MiB); free=${free:-?} MiB immediately before claim; lanes A/B/C GPUs 5/6/7 excluded by policy; co-tenants NEVER touched)" >> "$LEDGER"
echo "[claim] GPU=$GPU claimed $(ts)"

GPU=$GPU bash "$NR/methods/dense/run_homepage_v2_chain.sh" > "$LOGS/chain.log" 2>&1
rc=$?
echo "[chain] EXIT $rc $(ts)"

mine=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" | tr -d ' ' | tr '\n' ' ')
echo "$(ts) | GPU=$GPU | agent=$AGENT | job=$JOB | RELEASE rc=$rc (remaining compute PIDs on card: ${mine:-none} -- any listed belong to other agents/users and were never touched)" >> "$LEDGER"
echo "[release] GPU=$GPU released rc=$rc $(ts)"

if [ $rc -eq 0 ]; then echo "HOMEPAGE_V2_LAUNCHER_DONE rc=0"; exit 0; fi
if [ $rc -eq 7 ]; then
  echo "HOMEPAGE_V2_LAUNCHER_STOP rc=7 (smoke gate failed -- criteria need revision, not a retry)"
  exit 7
fi
echo "[retry] rc=$rc -- re-polling $(ts)"
sleep 60
done

echo "HOMEPAGE_V2_LAUNCHER_DONE rc=exhausted after $ATTEMPTS attempts"
exit 1
