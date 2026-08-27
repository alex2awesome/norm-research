#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "a" && "$1" != "b" ) ]]; then
  echo "usage: $0 a|b" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
root="outputs/silver_match_v3/humor/remediation_v3/model_improvement_v2/labelpacks_v1"
if [[ "$1" == "a" ]]; then
  pack="$root/pass_a"
  pass_name="humor-ce-v2-pass-a"
else
  pack="$root/pass_b_reslate_seed20260714"
  pass_name="humor-ce-v2-pass-b"
fi

exec python -u -m scripts.tools.silver_match_v3.run_codex_pack_labels \
  --pack-root "$pack" \
  --task humor \
  --pass-name "$pass_name" \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --concurrency "${HUMOR_TRUTH_CONCURRENCY:-4}" \
  --timeout-seconds 900 \
  --chunk-attempts 2 \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r1_precision.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r2_precision.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r3_exact_object.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r4_speech_act_and_audio_owner.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r5_criterion_nucleus.txt \
  --boundary-guide scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r6_falsification_and_abstention.txt
