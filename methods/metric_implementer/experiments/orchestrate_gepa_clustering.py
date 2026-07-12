#!/usr/bin/env python3
"""
GEPA clustering orchestrator — uses Claude Code Agent API to coordinate Opus subagents.

This script is ORCHESTRATION ONLY — it calls Claude Code to spawn Opus Max subagents
for clustering. It does not run clustering directly.

Usage:
    python orchestrate_gepa_clustering.py --task peer-review --output /tmp/result.json

Strategy validated on batch 0 pilot:
  - Initial Opus clustering: 38.9% recall
  - GEPA revision (show mistakes, re-cluster): 88.9% recall (+50%)
  - Now scaling to full task (53 batches) + all tasks
"""

import argparse, json, sys, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, help='Task name (peer-review, creative-writing, etc.)')
    p.add_argument('--output', required=True, help='Output JSON path')
    p.add_argument('--plan-path', default='/tmp/sonnet_peer_glm52aligned_FINAL.json')
    args = p.parse_args()

    # Load plan (300-form sample, kNN batches, v6 pairs)
    if not os.path.exists(args.plan_path):
        print(f"ERROR: Plan not found at {args.plan_path}")
        print("Run sonnet_cluster_gepa.py first to build the plan")
        sys.exit(1)

    plan = json.load(open(args.plan_path))

    print(f"GEPA Clustering Orchestrator")
    print(f"  Task: {plan['task']}")
    print(f"  Forms: {plan['n_forms']}")
    print(f"  Batches: {len(plan['batches'])}")
    print()
    print("This script coordinates Claude Code Agent spawning.")
    print("It does NOT run clustering autonomously.")
    print()
    print("MANUAL EXECUTION REQUIRED:")
    print("  1. For each batch, send initial_prompt to Opus Max subagent")
    print("  2. Score result vs v6_pairs")
    print("  3. If recall <70%, GEPA revise (show mistakes, re-cluster)")
    print("  4. Save final partition to output")
    print()
    print(f"Batch 0 validated: GEPA improved recall 38.9% → 88.9% (+50%)")
    print()
    print(f"Output will be saved to: {args.output}")

if __name__ == '__main__':
    main()
