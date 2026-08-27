#!/usr/bin/env bash
set -euo pipefail

# Resume the Humor training-only handoff after the untouched-dev K200 gate
# fails closed.  This preserves the failed gate, binds full-bank rescue, and
# freezes CE/Gemma training inputs without promoting K200 for production.

ROOT=/lfs/skampere3/0/alexspan/runtime/humor_final_handoff_v1
REPO=$ROOT/repo_snapshot
STATIC=$ROOT/static
CONSENSUS=$ROOT/consensus_relocated_v1
GATE=$ROOT/untouched_dev_k200_gate_v1
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
RETRIEVAL=$DATA/production_v2/humor/retrieval_diverse_v1/coverage_preserving
PYTHON=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
OUTPUT=$ROOT/final_stack_handoff_v1

cd "$REPO"
test ! -e "$OUTPUT"

"$PYTHON" - "$GATE/GATE_REPORT.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["status"] == "K200_UNTOUCHED_DEV_GATE_FAILED_FULLBANK_RESCUE_REQUIRED"
assert payload["gate"]["passed"] is False
assert payload["rescue"]["misses_not_found_in_fullbank"] == 0
assert payload["rescue"]["misses_routed"] > 0
PY

"$PYTHON" - "$STATIC/capture_sequence.dev-selection.k50.v1.json" \
  "$RETRIEVAL/humor_multi.three-lane.component50.k200.jsonl" \
  "$GATE/final.candidate-bundle.json" "$GATE/GATE_REPORT.json" <<'PY'
import json
import sys
from pathlib import Path

from scripts.tools.silver_match_v3.gate_humor_k200_consensus_dev import (
    _freeze_candidate_bundle,
)

output = Path(sys.argv[3])
if output.exists():
    payload = json.loads(output.read_text(encoding="utf-8"))
else:
    payload = _freeze_candidate_bundle(
        prior_path=Path(sys.argv[1]),
        k200_path=Path(sys.argv[2]),
        output_path=output,
        gate_report_path=Path(sys.argv[4]),
    )
assert payload["k200_primary_promoted"] is False
assert payload["fullbank_required_for_production"] is True
PY

"$PYTHON" -u -m scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff \
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
  --output-root "$OUTPUT" \
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
  --gepa-train-only-audit R9="$STATIC/RESOLVER_R9_TRAIN_ONLY_JUDGE_AUDIT.json"

"$PYTHON" - "$OUTPUT/FINAL_STACK_QUEUE.json" "$GATE/GATE_REPORT.json" <<'PY'
import hashlib
import json
import sys

queue_path, gate_path = sys.argv[1:]
queue = json.load(open(queue_path, encoding="utf-8"))
gate = json.load(open(gate_path, encoding="utf-8"))
assert len(queue["ce"]["runs"]) == 2
assert {run["seed"] for run in queue["ce"]["runs"]} == {2026071501, 2026071502}
assert gate["gate"]["passed"] is False
print(json.dumps({
    "status": "TRAINING_QUEUE_FROZEN_K200_FAILED_FULLBANK_REQUIRED",
    "queue": queue_path,
    "queue_sha256": hashlib.sha256(open(queue_path, "rb").read()).hexdigest(),
    "gate": gate_path,
    "gate_sha256": hashlib.sha256(open(gate_path, "rb").read()).hexdigest(),
    "ce_runs": queue["ce"]["runs"],
}, sort_keys=True))
PY
