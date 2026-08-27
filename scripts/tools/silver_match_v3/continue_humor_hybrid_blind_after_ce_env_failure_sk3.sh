#!/usr/bin/env bash
set -euo pipefail

CODE=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/repo_snapshot
ROOT=/lfs/skampere3/0/alexspan/runtime/humor_typed_llama_compact_v1
RUN=$ROOT/blind_hybrid_v1/strict_exp34944_v2
FREEZE=$RUN/freeze/FREEZE.json
TYPED=$RUN/typed_inference
CE=$RUN/ce_inference
DEVCE=$ROOT/dev_hybrid_v1/nemotron_dev_stage_v2
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python

test "$(sha256sum "$CODE/scripts/tools/silver_match_v3/run_hybrid_ce_typed_blind_eval.py" | cut -d' ' -f1)" = \
  926cf8394e27afa1ad12c5175cbe264b003793b28896dc2c66442b32fa495940
test "$(sha256sum "$FREEZE" | cut -d' ' -f1)" = \
  8378551ea0461cb048d6604e22530c859ec2bccf24deede2f8fa2658e487146c
test "$(sha256sum "$TYPED/typed.original.jsonl" | cut -d' ' -f1)" = \
  7ce4eecf2caa28198fdac77bcb22d0cdd949e1eee0dd171075ea1b296fe92839
test "$(sha256sum "$TYPED/typed.reordered.jsonl" | cut -d' ' -f1)" = \
  7013dca40d1f784d92ca7e2070648f47e68ad77cc296860b195aba9707960027
test -f "$TYPED/INFERENCE_META.json"
test ! -e "$CE/blind.scores.jsonl"
test ! -e "$CE/blind.scores.jsonl.meta.json"
test ! -e "$RUN/SCORE.json"

export HOME=/lfs/skampere3/0/alexspan
export HF_HOME=$HOME/.cache/huggingface
export XDG_CACHE_HOME=$RUN/cache/ce_retry/xdg
export TORCH_EXTENSIONS_DIR=$RUN/cache/ce_retry/torch_extensions
export TRITON_CACHE_DIR=$RUN/cache/ce_retry/triton
export TORCHINDUCTOR_CACHE_DIR=$RUN/cache/ce_retry/torchinductor
export TMPDIR=$RUN/tmp
export CUDA_VISIBLE_DEVICES=5
unset PYTHONPATH
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR"
cd "$CODE"

"$PY" -u -m scripts.tools.silver_match_v3.run_nemotron_ce score \
  --input-pairs "$RUN/freeze/ce.pairs.truth_blind.jsonl" \
  --output "$CE/blind.scores.jsonl" \
  --model "$DEVCE/source/models/llama-embed-nemotron-8b-aa3b43a495a9b280d1bdb716da37c54bb495d630-mirror-v1" \
  --base-manifest "$DEVCE/BASE_MODEL_MANIFEST.relocated.json" \
  --base-manifest-sha256 d1a13c104772dbf82cf95c08fc52dd88f93e9a48284aa5d8ba81f1c52ae406c8 \
  --checkpoint "$DEVCE/source/runtime/humor_ce_binary_v1/runs/final-joined-recipe-v1/seed-2026071502/checkpoints/exposure-000000100000" \
  --training-report "$DEVCE/training_report.relocated.json" \
  --training-report-sha256 31be7932392295fbb909c2dee0730f210165942fc884211916a7d3a6428b6c59 \
  --batch-size 8 --max-length 1024 --device 0 --attention eager

test -f "$CE/blind.scores.jsonl.meta.json"
# This is still the only invocation receiving the blind gold-bearing source,
# and it occurs strictly after both typed and CE inference metadata are sealed.
"$PY" -u -m scripts.tools.silver_match_v3.run_hybrid_ce_typed_blind_eval score \
  --freeze "$FREEZE" \
  --typed-meta "$TYPED/INFERENCE_META.json" \
  --ce-scores "$CE/blind.scores.jsonl" \
  --ce-meta "$CE/blind.scores.jsonl.meta.json" \
  --blind-gold-source /lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1/final_stack_handoff_v1/gemma/dataset/blind.jsonl \
  --output "$RUN/SCORE.json"

echo "COMPLETE $RUN/SCORE.json"
