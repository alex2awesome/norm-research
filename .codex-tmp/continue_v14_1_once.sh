#!/usr/bin/env bash
set -euo pipefail

# One-shot handoff only: this script never polls SSH.
repo=/Users/spangher/Projects/stanford-research/norm-research
local_root=$repo/.codex-tmp/cr3-v14.1-roadmap
remote_root=/lfs/skampere3/0/alexspan/cr3-v14.1-roadmap
mkdir -p "$local_root"

# One SSH connection: require the remote readiness marker and fetch the frozen
# ladder design. Exit 75 when it is not ready; a caller may invoke this later.
if ! ssh -o BatchMode=yes -o ConnectTimeout=20 sk3 \
  "HOME=/lfs/skampere3/0/alexspan bash --noprofile --norc -c 'test -f \"$remote_root/READY_FOR_LOCAL_REFERENCE\" && tar -C \"$remote_root/outputs\" -czf - ladder'" \
  | tar -C "$local_root" -xzf -; then
  echo "v14.1 Phase A is not ready (or the single SSH handoff failed); no work launched" >&2
  exit 75
fi

cd "$repo"
scripts/run_ceiling_reference_local.sh "$local_root/ladder"

# One SSH connection: transfer only the independent reference and launch the
# ordered native-audit -> constructor -> executor -> aggregate chain. The
# remote code was already frozen at commit 3ee59ef.
remote_launch=$(cat <<'REMOTE'
set -euo pipefail
root=/lfs/skampere3/0/alexspan/cr3-v14.1-roadmap
out=$root/outputs/ladder
mkdir -p "$out" "$root/logs"
tar -C "$out" -xzf -
export HOME=/lfs/skampere3/0/alexspan
nohup bash -c '
  set -euo pipefail
  root=/lfs/skampere3/0/alexspan/cr3-v14.1-roadmap
  out=$root/outputs/ladder
  v13=/lfs/skampere3/0/alexspan/cr3-v13.1
  "$root/code/scripts/run_ceiling_ladder_sk3.sh" native-audit "$out" \
    --metrics-manifest "$v13/manifests/tier_b.json" \
    --v13-root "$v13/outputs/tier_b/lanes" --n-permutations 10000
  "$root/code/scripts/run_ceiling_ladder_sk3.sh" constructor "$out" 5 \
    --constructor-model meta-llama/Llama-3.3-70B-Instruct
  "$root/code/scripts/run_ceiling_ladder_sk3.sh" executor "$out" 7
  "$root/code/scripts/run_ceiling_ladder_sk3.sh" aggregate "$out"
' >"$root/logs/ordered_A_B.log" 2>&1 &
printf '%s\n' "$!" >"$root/run.pid"
REMOTE
)
tar -C "$local_root/ladder" -czf - reference \
  | ssh -o BatchMode=yes -o ConnectTimeout=20 sk3 \
      "HOME=/lfs/skampere3/0/alexspan bash --noprofile --norc -c $(printf '%q' "$remote_launch")"

echo "independent reference complete; ordered remote A/B chain launched"
