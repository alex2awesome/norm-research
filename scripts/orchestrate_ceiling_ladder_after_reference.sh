#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=$(cd -- "$script_dir/.." && pwd)
local_root=${CEILING_LOCAL_ROOT:-$code_root/.codex-tmp/cr3-v14.1-roadmap}
remote_root=${CEILING_REMOTE_ROOT:-/lfs/skampere3/0/alexspan/cr3-v14.1-roadmap}
v13_root=${V13_ROOT:-/lfs/skampere3/0/alexspan/cr3-v13.1}
manifest=$local_root/reference/reference_manifest.json
watch_log=$local_root/orchestrator.log

exec 219>"$local_root/orchestrator.lock"
if ! flock -n 219; then
  echo "a ceiling-ladder reference orchestrator is already active" >&2
  exit 75
fi

cd "$code_root"
manifest_complete() {
  [[ -f "$manifest" ]] || return 1
  python - "$manifest" <<'PY' >/dev/null 2>&1
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if p.get("n_decided_metrics") == 35 else 1)
PY
}

while ! manifest_complete; do
  alive=0
  for file in "$local_root"/reference_logs/final_part_*.pid; do
    [[ -f "$file" ]] || continue
    pid=$(<"$file")
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  if [[ $alive -eq 0 ]]; then
    # Allow the parent shell to assemble the final manifest after both workers checkpoint.
    sleep 30
    if ! manifest_complete; then
      echo "reference workers exited without a complete manifest" | tee -a "$watch_log" >&2
      exit 1
    fi
    break
  fi
  sleep 60
done

python - "$manifest" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("n_decided_metrics") != 35:
    raise SystemExit(f"incomplete independent reference manifest: {payload}")
if payload.get("n_valid_metrics", 0) <= 0:
    raise SystemExit("no independently referenced metric survived")
if payload.get("n_items") != payload.get("n_valid_metrics") * 390:
    raise SystemExit(f"reference item count does not match 390 probes/valid metric: {payload}")
if not payload.get("reference_is_independent_of_executor_outputs"):
    raise SystemExit("reference manifest is not independent")
PY

rsync -azR \
  ./methods/metric_implementer/experiments/ceiling_ladder.py \
  ./methods/metric_implementer/experiments/v14_probe_extension.py \
  ./scripts/run_ceiling_ladder_sk3.sh \
  "sk3:$remote_root/code/"
rsync -az "$local_root/reference/" "sk3:$remote_root/outputs/reference/"

remote_command=$(cat <<EOF
set -euo pipefail
cd "$remote_root"
mkdir -p logs
export HOME=/lfs/skampere3/0/alexspan
nohup bash -c '
  set -euo pipefail
  "$remote_root/code/scripts/run_ceiling_ladder_sk3.sh" native-audit "$remote_root/outputs" \
    --metrics-manifest "$v13_root/manifests/tier_b.json" \
    --v13-root "$v13_root/outputs/tier_b/lanes" --n-permutations 10000
  "$remote_root/code/scripts/run_ceiling_ladder_sk3.sh" constructor "$remote_root/outputs" 5 \
    --constructor-model meta-llama/Llama-3.3-70B-Instruct
  "$remote_root/code/scripts/run_ceiling_ladder_sk3.sh" executor "$remote_root/outputs" 7
  "$remote_root/code/scripts/run_ceiling_ladder_sk3.sh" aggregate "$remote_root/outputs"
' >"$remote_root/logs/ordered_A_B.log" 2>&1 &
echo \$! >"$remote_root/run.pid"
echo "launched ordered A/B chain as PID \$(cat "$remote_root/run.pid")"
EOF
)
ssh -o BatchMode=yes -o ConnectTimeout=20 sk3 "$remote_command"
echo "reference transferred and ordered remote A/B chain launched" | tee -a "$watch_log"
