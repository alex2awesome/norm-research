#!/usr/bin/env bash
#
# GEPA-optimized clustering across all tasks
# Runs via Claude Code orchestration (this script is a plan, not autonomous)
#
# Strategy:
# 1. For each task (peer-review, creative-writing, code-review, etc.):
#    - Load 300-form biased sample + kNN batches + v6 pairs
#    - Cluster each batch via Opus Max subagent
#    - Score vs v6 labels
#    - GEPA-revise low-recall batches (<70%)
#    - Reconcile via union-find
# 2. Save results to outputs/analyses/gepa_clusters_{task}.json
# 3. Run prompt-optimization pipeline on R1 clusters (children + GEPA → large Ω)

TASKS="peer-review creative-writing code-review humor press-releases"

echo "=== GEPA Clustering Plan ==="
echo "Tasks: $TASKS"
echo "Phase 1: Cluster + Score + Revise (per task)"
echo "Phase 2: Prompt-optimization pipeline on R1 clusters"
echo ""
echo "This script is a PLAN — Claude Code will orchestrate via subagents"
echo "DO NOT RUN DIRECTLY"
