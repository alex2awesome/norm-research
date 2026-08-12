#!/usr/bin/env bash
# launch_ecert_slice_sk3.sh — E-CERT VERDICT-SLICE campaign (2026-08-12, prereg frozen in
# notes/2026-08-05__direction12-battery-plan.md before launch).
# Mines alpha-probe *_sigs.npz banks for the 185 slice metrics (3 verdict cells) that are
# not already in the 272-metric cr3-v12 bank. Sequential per-task sweeps, ONE GPU.
#
# Usage:  ./launch_ecert_slice_sk3.sh            # full chain, background
#         SMOKE=1 ./launch_ecert_slice_sk3.sh    # foreground, first 2 humor metrics only
# Input:  $HOME/outputs/ecert_slice_v1/gilists.json  (shipped from laptop
#         outputs/analyses/ecert_slice_gilists_v1.json; task -> [{gi,name,cell}])
# Safety: gi->merged_name verified per task BEFORE mining (abort on any mismatch);
#         GPU pick restricted to sk3 physical 0,5,6,7 (1-2 user-forbidden, 3-4 policy).
set -uo pipefail
REPO=/lfs/skampere3/0/alexspan/norm-research
export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=/lfs/skampere3/0/shared_hf_cache
export HUGGINGFACE_HUB_CACHE=/lfs/skampere3/0/shared_hf_cache/hub
export HF_HUB_OFFLINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_MOE_FP8=0
export TOKENIZERS_PARALLELISM=false
export VLLM_GPU_MEM_UTIL=0.93
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY="${PY:-/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# Live GLM key (the resolver's default preference is the DEAD alexander- account -> silent 429)
export ZAI_KEY_FILE="$HOME/.z-ai-api-key.txt"
# Proposer families: CANONICAL 4-family default (glm/qwen/llama/haiku) restored 2026-08-12
# after the user topped up OpenRouter (new key deployed to ~/.openrouter-api-key.txt on sk3;
# old key kept as .bak-20260812). The interim 2-family (glm+gpt-4.1-mini) humor banks are
# quarantined in $OUT/twofam_v1/ as a family-set-sensitivity arm — instrument must be ONE
# fixed family set across all cells.
OUT="$HOME/outputs/ecert_slice_v1"
GILISTS="$OUT/gilists.json"
cd "$REPO"
[ -f "$GILISTS" ] || { echo "missing $GILISTS"; exit 1; }

# GPU: first idle among the ALLOWED set only
FREE=""
for g in 0 5 6 7; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v g="$g" '$1==g {print $2}')
  if [ -n "$used" ] && [ "$used" -lt 2000 ]; then FREE=$g; break; fi
done
[ -n "$FREE" ] && export CUDA_VISIBLE_DEVICES="$FREE" || { echo "no allowed idle GPU (0/5/6/7)"; exit 1; }

export PY MODEL OUT GILISTS FREE REPO
TASKS="${TASKS:-humor creative-writing news-homepages math-stackexchange peer-review}"
export TASKS

run_chain() {
  cd "$REPO"
  for TASK in $TASKS; do
    GIS=$("$PY" - "$TASK" "$GILISTS" <<'PYEOF'
import json, re, sys
task, path = sys.argv[1], sys.argv[2]
rows = json.load(open(path)).get(task, [])
if "SMOKE" in __import__("os").environ:
    rows = rows[:2]
if not rows:
    print(""); sys.exit(0)
from methods.metric_implementer.experiments import mine_clusters as mc
groups = mc.r2_groups(task, "general")
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
for r in rows:
    got = norm(groups[r["gi"]].get("merged_name", ""))
    if got != norm(r["name"]):
        sys.stderr.write(f"GI MISMATCH {task} gi={r['gi']}: want {r['name']!r} got {got!r}\n")
        sys.exit(3)
print(",".join(str(r["gi"]) for r in rows))
PYEOF
    ) || { echo "$(date): $TASK gi verification FAILED — task skipped, chain aborted"; return 3; }
    [ -z "$GIS" ] && { echo "$(date): $TASK — no slice metrics, skipping"; continue; }
    N=$(awk -F, '{print NF}' <<< "$GIS")
    echo "$(date): $TASK — $N metrics, gi-list verified, GPU $FREE"
    "$PY" -m methods.metric_implementer.experiments.run_alpha_probe \
      --task "$TASK" --r2-bucket general --level R2 --target-model "$MODEL" \
      --gi-list "$GIS" --n-metrics 0 --M-freegen 60 --n-probes 300 --gepa-reserve 60 \
      --skip-existing --out-dir "$OUT" \
      || echo "$(date): $TASK sweep exited nonzero — continuing chain (resume = re-run script)"
  done
  echo "$(date): E-CERT SLICE CHAIN DONE"
}

mkdir -p "$OUT" "$HOME/logs"
if [ "${SMOKE:-0}" = "1" ]; then
  export SMOKE
  TASKS="humor"
  echo "SMOKE MODE: humor first-2 only, foreground"
  run_chain
else
  LOG="$HOME/logs/ecert_slice_v1_gpu${FREE}.log"
  nohup bash -c "$(declare -f run_chain); run_chain" > "$LOG" 2>&1 &
  echo "$(date): LAUNCHED wrapper pid=$! log=$LOG"
  echo "  kill discipline: kill wrapper shell FIRST, then python child, then EngineCore"
fi
