#!/usr/bin/env bash
set -euo pipefail

# CPU-only sk3 continuation.  The caller must first publish the byte-exact
# relocated consensus directory at CONSENSUS.  This script never starts a
# trainer, scorer, vLLM server, torchrun, or GPU process.

ROOT=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1
REPO=$ROOT/repo_snapshot
STATIC=$ROOT/static
CONSENSUS=$ROOT/consensus_relocated_v1
GATE=$ROOT/untouched_dev_k200_gate_v1
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
RETRIEVAL=$DATA/production_v2/humor/retrieval_diverse_v1/coverage_preserving
PAIR_ROOT=$DATA/production_v2/humor/ce_production_pairs_k200_v1
PYTHON=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python

cd "$REPO"
test -f "$CONSENSUS/MANIFEST.json"
test -f "$CONSENSUS/truth.all.jsonl"
test -f "$CONSENSUS/truth.dev.jsonl"
test -f "$CONSENSUS/CE_REPORT.json"
test -f "$CONSENSUS/source.validation.json"
test ! -e "$GATE"

"$PYTHON" -u -m scripts.tools.silver_match_v3.gate_humor_k200_consensus_dev \
  --bank "$DATA/banks/humor.json" \
  --consensus-dev-truth "$CONSENSUS/truth.dev.jsonl" \
  --consensus-manifest "$CONSENSUS/MANIFEST.json" \
  --k200-candidates "$RETRIEVAL/humor_multi.three-lane.component50.k200.jsonl" \
  --fullbank-candidates "$RETRIEVAL/humor_multi.three-lane.component50.full285.jsonl" \
  --prior-candidate-bundle "$STATIC/capture_sequence.dev-selection.k50.v1.json" \
  --dev-match-labels "$GATE/dev.matches.jsonl" \
  --capture-report "$GATE/k200.capture.json" \
  --rescue-misses "$GATE/fullbank285.rescue-misses.jsonl" \
  --gate-report "$GATE/GATE_REPORT.json" \
  --candidate-bundle-output "$GATE/final.candidate-bundle.json"

GATE_STATUS=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$GATE/GATE_REPORT.json")
if [[ "$GATE_STATUS" != K200_UNTOUCHED_DEV_GATE_PASSED ]]; then
  exit 42
fi

"$PYTHON" -u -m scripts.tools.silver_match_v3.freeze_humor_consensus_completion_queue freeze \
  --manifest "$DATA/manifest.json" \
  --bank "$DATA/banks/humor.json" \
  --hierarchy "$STATIC/humor_general_r3_expanded.json" \
  --existing-truth "$STATIC/truth.canonical.jsonl" \
  --existing-truth-report "$STATIC/truth.canonical.report.json" \
  --consensus-truth "$CONSENSUS/truth.all.jsonl" \
  --consensus-truth-manifest "$CONSENSUS/MANIFEST.json" \
  --candidate-capture-freeze "$GATE/final.candidate-bundle.json" \
  --pilot-selection "$STATIC/pilot_recipe_relocated_v1/PILOT_SELECTION.json" \
  --ce-model /lfs/skampere3/0/alexspan/.cache/huggingface/hub/models--nvidia--llama-embed-nemotron-8b/snapshots/aa3b43a495a9b280d1bdb716da37c54bb495d630 \
  --gemma-model /lfs/skampere3/0/alexspan/models/gemma-4-31b-it \
  --independent-labeling-guide "$REPO/scripts/tools/silver_match_v3/INDEPENDENT_LABELING_GUIDE.md" \
  --python "$PYTHON" \
  --ce-trainer "$REPO/scripts/tools/silver_match_v3/train_nemotron_cross_encoder.py" \
  --ce-scorer "$REPO/scripts/tools/silver_match_v3/run_nemotron_ce.py" \
  --gemma-trainer "$REPO/scripts/tools/silver_match_v3/train_gemma4_typed_lora.py" \
  --runtime-root "$ROOT/final_training_runtime_v1" \
  --output-root "$ROOT/final_stack_handoff_v1" \
  --ce-seed 2026071501 \
  --ce-seed 2026071502 \
  --gepa-rule R1="$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r1_precision.txt" \
  --gepa-rule R2="$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r2_precision.txt" \
  --gepa-rule R3="$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r3_exact_object.txt" \
  --gepa-rule R4="$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r4_speech_act_and_audio_owner.txt" \
  --gepa-rule R5="$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r5_criterion_nucleus.txt" \
  --gepa-rule R6="$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r6_falsification_and_abstention.txt" \
  --gepa-rule R7="$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r7_fullbank_resolver_train_only.txt" \
  --gepa-rule R8="$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r8_named_outcome_and_owner_train_only.txt" \
  --gepa-rule R9="$REPO/scripts/tools/silver_match_v3/prompts/verify_humor_gepa_r9_truth_structure_and_freshness_train_only.txt" \
  --gepa-train-only-audit R7="$STATIC/RESOLVER_R7_TRAIN_ONLY_JUDGE_AUDIT.json" \
  --gepa-train-only-audit R8="$STATIC/RESOLVER_R8_TRAIN_ONLY_JUDGE_AUDIT.json" \
  --gepa-train-only-audit R9="$STATIC/RESOLVER_R9_TRAIN_ONLY_JUDGE_AUDIT.json" \
  --consensus-source-validation "$CONSENSUS/source.validation.json" \
  --consensus-relocation-report "$CONSENSUS/RELOCATION_REPORT.json" \
  --ce-truth-report "$CONSENSUS/CE_REPORT.json" \
  --production-candidate humor_multi="$RETRIEVAL/humor_multi.three-lane.component50.k200.jsonl" \
  --production-candidate-audit humor_multi="$RETRIEVAL/humor_multi.three-lane.component50.k200.audit.json" \
  --production-rescue-candidate humor_multi="$RETRIEVAL/humor_multi.three-lane.component50.full285.jsonl" \
  --production-rescue-candidate-audit humor_multi="$RETRIEVAL/humor_multi.three-lane.component50.full285.audit.json" \
  --production-train-capture-diagnostic "$RETRIEVAL/audits/k200_train_double_agree_progressive_component_v3.json" \
  --production-dev-capture-gate "$GATE/k200.capture.json" \
  --production-dev-policy-gate "$GATE/GATE_REPORT.json" \
  --production-pairs "$PAIR_ROOT/humor.k200.pairs.jsonl" \
  --production-norm-universe "$PAIR_ROOT/humor.universe.jsonl" \
  --production-k 200 \
  --production-context-chars 1400 \
  --receipt-directory "$ROOT/completion_receipts_v1" \
  --repo-root "$REPO" \
  --poll-seconds 30 \
  --queue-output "$ROOT/CONSENSUS_COMPLETION_QUEUE.json"

"$PYTHON" -u -m scripts.tools.silver_match_v3.freeze_humor_consensus_completion_queue run \
  --queue "$ROOT/CONSENSUS_COMPLETION_QUEUE.json"
