#!/usr/bin/env python3
"""
Orchestrator for Sonnet-based clustering with GEPA iteration.

Loads plan from sonnet_cluster_gepa.py, then Claude Code executes it via Max subagents.
"""

import sys, json

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_sonnet_cluster.py <plan.json>")
        sys.exit(1)

    plan_path = sys.argv[1]
    plan = json.load(open(plan_path))

    print(f"Loaded plan: {plan['task']}, {len(plan['batches'])} batches, {plan['n_forms']} forms")
    print(f"Ready for Claude Code orchestration via Max subagents + GEPA")
    print(f"Registry: {plan['registry_dir']}")
    print()
    print("ORCHESTRATION PROTOCOL:")
    print("  1. For each batch, send initial_prompt to Max subagent")
    print("  2. Parse JSON groups from response")
    print("  3. Score against v6_pairs (score-2 = same, score-0 = different)")
    print("  4. If mistakes, GEPA revise: show errors + ask for corrected groups")
    print("  5. Iterate up to max_rounds")
    print("  6. Reconcile all batch results via union-find + min_votes")
    print("  7. Compare to existing partition + v6 labels")
    print()
    print(f"First batch sample:")
    b0 = plan['batches'][0]
    print(f"  batch_id: {b0['batch_id']}, {len(b0['items'])} items")
    print(f"  initial_prompt preview: {b0['initial_prompt'][:200]}...")
