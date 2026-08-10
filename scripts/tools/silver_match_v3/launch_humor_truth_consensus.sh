#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

root="outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2"
out="$root/truth_consensus_v1"
mkdir -p "$out"
printf '%s\n' "$$" > "$out/WATCHER.pid"

exec python -u -m scripts.tools.silver_match_v3.watch_exact_truth_consensus \
  --source-pack "$root/truth_collection_v1" \
  --initial-pass pass_a="$root/labelpacks_v1/pass_a" \
  --initial-pass pass_b="$root/labelpacks_v1/pass_b_reslate_seed20260714" \
  --output-root "$out" \
  --task humor \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --annotator codex-gpt-5.6-sol-high \
  --label-source-prefix humor_ce_v2_independent_exact_full_bank \
  --concurrency "${HUMOR_RESOLVER_CONCURRENCY:-4}" \
  --timeout-seconds 900 \
  --chunk-attempts 2 \
  --chunk-size 20 \
  --poll-seconds 30 \
  --max-passes 6 \
  --resolver-seed-base 2026071400 \
  --training-truth-output "$out/training_truth_v1" \
  --ce-truth-output-root "$out/ce_truth_v1" \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r1_precision.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r2_precision.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r3_exact_object.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r4_speech_act_and_audio_owner.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r5_criterion_nucleus.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r6_falsification_and_abstention.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r7_fullbank_resolver_train_only.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r8_named_outcome_and_owner_train_only.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r9_truth_structure_and_freshness_train_only.txt
