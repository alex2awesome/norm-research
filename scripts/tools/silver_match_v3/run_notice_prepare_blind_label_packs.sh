#!/usr/bin/env bash
set -euo pipefail

REPO=/lfs/skampere3/0/alexspan/norm-research
DATA=/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful
RESCUE="$DATA/production_v1/rescue/notice-and-comment"
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
PREPARE=scripts/tools/silver_match_v3/prepare_final_decision_label_pack.py
PINNED_PREPARE_SHA=288d4cdd18d6875aa199e0f106c4488c69b387d12c312c1db81709e049e4a89b

cd "$REPO"
while test ! -f "$RESCUE/blind_audit_match/sample_report.json" \
  || test ! -f "$RESCUE/blind_audit_abstention/sample_report.json"; do
  sleep 30
done
test "$(sha256sum "$PREPARE" | cut -d' ' -f1)" = "$PINNED_PREPARE_SHA"

if test ! -f "$RESCUE/blind_audit_match/task_label_pack/validation.json"; then
  "$PY" -m scripts.tools.silver_match_v3.prepare_final_decision_label_pack \
    --sample-report "$RESCUE/blind_audit_match/sample_report.json" \
    --scope task:notice-and-comment \
    --output-root "$RESCUE/blind_audit_match/task_label_pack" \
    --chunk-size 25 \
    --seed 57721
fi
test "$(sha256sum "$PREPARE" | cut -d' ' -f1)" = "$PINNED_PREPARE_SHA"

if test ! -f "$RESCUE/blind_audit_abstention/task_label_pack/validation.json"; then
  "$PY" -m scripts.tools.silver_match_v3.prepare_final_decision_label_pack \
    --sample-report "$RESCUE/blind_audit_abstention/sample_report.json" \
    --scope task:notice-and-comment \
    --output-root "$RESCUE/blind_audit_abstention/task_label_pack" \
    --chunk-size 25 \
    --seed 91577
fi
test "$(sha256sum "$PREPARE" | cut -d' ' -f1)" = "$PINNED_PREPARE_SHA"
sha256sum \
  "$RESCUE/blind_audit_match/task_label_pack/validation.json" \
  "$RESCUE/blind_audit_abstention/task_label_pack/validation.json"
