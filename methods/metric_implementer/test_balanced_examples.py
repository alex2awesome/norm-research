#!/usr/bin/env python3
"""A/B test: balanced k=30 vs hi_lo k=4 examples on skewed metric (g125 capitalization).

Tests whether showing balanced YES/NO examples (oversampling minority class) helps GLM
recover the true metric better than the legacy hi_lo approach on skewed metrics.

Target: peer-review g125 "General capitalization standards" (93% YES skewed case.
Expected: balanced k=30 should induce rules mentioning "capitalization" more often
than hi_lo k=4 (which collapses to generic "novelty/quality" prior).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .config import ImplementerConfig, apply_task_preset
from .manifest import load_metrics, full_manifest
from .backends import LLMBackend
from .recon_channel import _balanced_examples, _hi_lo_examples, induce_free, _pyes


def load_pool(pool_path: str) -> list:
    """Load pool from jsonl.gz."""
    import gzip
    data = []
    with gzip.open(pool_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data


def count_metric_mentions(text: str, metric_keywords: list) -> int:
    """Count how many metric-related keywords appear in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in metric_keywords if kw.lower() in text_lower)


def main():
    # Paths
    pool_path = Path(__file__).parent / "trial" / "pool_peer_review.jsonl.gz"

    if not pool_path.exists():
        print(f"ERROR: Pool not found at {pool_path}")
        sys.exit(1)

    # Load pool
    print(f"Loading pool from {pool_path}...")
    pool = load_pool(str(pool_path))
    print(f"  Loaded {len(pool)} items")

    # Extract texts
    texts = [item["text"] for item in pool]
    print(f"  Extracted {len(texts)} texts")

    # Load g125 from mine_clusters (the "General capitalization standards" metric)
    print("\nLoading g125 from mine_clusters...")
    from .experiments.mine_clusters import r2_groups
    groups = r2_groups("peer-review", "specific")
    g125_group = next((g for g in groups if g.get("group_idx") == 125), None)

    if not g125_group:
        print("ERROR: Could not find group_idx 125 (capitalization)")
        sys.exit(1)

    print(f"  Found group_idx 125: {g125_group['merged_name']}")
    print(f"  Description: {g125_group['merged_description']}")
    print(f"  Total leaves: {g125_group['total_leaf_rubrics']}")

    # For testing, we'll use the merged_description as the rubric
    # This is a reasonable approximation of the M_ω criterion
    rubric_body = g125_group['merged_description']
    print(f"  Using merged description as rubric (length: {len(rubric_body)} chars)")

    # Configure backend for scoring (use a small model for P(YES) scoring)
    # Use GLM-5 via zai_anthropic for both scoring and reconstruction
    cfg = ImplementerConfig()
    cfg.backend = "zai_anthropic"

    print("\nInitializing GLM-5 via zai_anthropic...")
    backend = LLMBackend("glm-5", "executor", cfg)

    # Score M_ω on the pool (SAMPLED YES/NO)
    print("\nScoring M_ω on pool (SAMPLED YES/NO, R=3)...")
    cfg_entry = ImplementerConfig()
    apply_task_preset(cfg_entry, "peer-review")
    max_chars = getattr(cfg_entry, "max_text_chars", 4000)
    noun = getattr(cfg_entry, "item_noun", "review")

    # Use sampled binary (not logprob) for M_ω scoring
    # We'll use R=3 to get a more stable estimate
    R_score = 3
    temp_score = 0.7

    # For sampled binary, we need to call _sampled_binary (but it's not imported)
    # Let's implement it inline here
    from .recon_channel import _sampled_binary

    score_runs = []
    for r in range(R_score):
        print(f"  Scoring run {r+1}/{R_score}...")
        scores = _sampled_binary(backend, rubric_body, texts, max_chars, temp_score, r+1)
        score_runs.append(scores)

    # Average across runs to get stable P(YES)
    score_mat = np.column_stack(score_runs)
    pyes = np.nanmean(score_mat, axis=1)

    # Check skew
    binary = (pyes > 0.5).astype(int)
    n_yes = int((binary == 1).sum())
    n_no = int((binary == 0).sum())
    frac_yes = n_yes / len(binary)

    print(f"\nMetric skew:")
    print(f"  YES: {n_yes} ({frac_yes:.1%})")
    print(f"  NO: {n_no} ({1-frac_yes:.1%})")
    print(f"  Total: {len(binary)}")

    # Split into train (for inducing rules)
    n_train = 30
    rng = np.random.default_rng(42)
    order = rng.permutation(len(texts))
    train_idx, held_idx = order[:n_train], order[n_train:]
    train_texts = [texts[i] for i in train_idx]
    train_pyes = pyes[train_idx]

    print(f"\nTrain set: {n_train} items")
    train_binary = (train_pyes > 0.5).astype(int)
    print(f"  YES: {int((train_binary == 1).sum())}, NO: {int((train_binary == 0).sum())}")

    # A/B test: hi_lo k=4 vs balanced k=30
    print("\n" + "="*70)
    print("A/B TEST: hi_lo k=4 (OLD) vs balanced k=30 (NEW)")
    print("="*70)

    # Condition A: hi_lo k=4 (legacy)
    print("\n--- Condition A: hi_lo k=4 (legacy) ---")
    examples_hi_lo = _hi_lo_examples(train_texts, train_pyes, k=4, max_chars=600)
    print(f"Examples preview (first 400 chars):\n{examples_hi_lo[:400]}...")

    rules_hi_lo = []
    for r in range(3):
        print(f"  Inducing rule {r+1}/3...")
        result = induce_free(backend, noun, examples_hi_lo, seed=r, max_tokens=450)
        rule = result.get("rule", "")
        rules_hi_lo.append(rule)
        print(f"    Rule preview: {rule[:150]}...")

    # Condition B: balanced k=30
    print("\n--- Condition B: balanced k=30 (NEW) ---")
    examples_balanced = _balanced_examples(train_texts, train_pyes, k=30, max_chars=600)
    print(f"Examples preview (first 400 chars):\n{examples_balanced[:400]}...")

    rules_balanced = []
    for r in range(3):
        print(f"  Inducing rule {r+1}/3...")
        result = induce_free(backend, noun, examples_balanced, seed=r, max_tokens=450)
        rule = result.get("rule", "")
        rules_balanced.append(rule)
        print(f"    Rule preview: {rule[:150]}...")

    # Analyze: count "capitalization" mentions
    capitalization_keywords = ["capital", "uppercase", "lowercase", "case", "letter"]
    generic_keywords = ["novelty", "quality", "good", "interesting", "clear"]

    print("\n" + "="*70)
    print("RESULTS: metric mentions in induced rules")
    print("="*70)

    cap_counts_hi_lo = [count_metric_mentions(r, capitalization_keywords) for r in rules_hi_lo]
    gen_counts_hi_lo = [count_metric_mentions(r, generic_keywords) for r in rules_hi_lo]

    cap_counts_balanced = [count_metric_mentions(r, capitalization_keywords) for r in rules_balanced]
    gen_counts_balanced = [count_metric_mentions(r, generic_keywords) for r in rules_balanced]

    print(f"\nCondition A (hi_lo k=4):")
    print(f"  Capitalization mentions per rule: {cap_counts_hi_lo} (avg: {np.mean(cap_counts_hi_lo):.1f})")
    print(f"  Generic prior mentions per rule: {gen_counts_hi_lo} (avg: {np.mean(gen_counts_hi_lo):.1f})")

    print(f"\nCondition B (balanced k=30):")
    print(f"  Capitalization mentions per rule: {cap_counts_balanced} (avg: {np.mean(cap_counts_balanced):.1f})")
    print(f"  Generic prior mentions per rule: {gen_counts_balanced} (avg: {np.mean(gen_counts_balanced):.1f})")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if np.mean(cap_counts_balanced) > np.mean(cap_counts_hi_lo):
        print("✓ SUCCESS: balanced k=30 shows MORE capitalization mentions")
        print(f"  Improvement: {np.mean(cap_counts_balanced) - np.mean(cap_counts_hi_lo):.1f} mentions/rule")
    elif np.mean(cap_counts_balanced) == np.mean(cap_counts_hi_lo):
        print("⚠ TIE: balanced k=30 shows EQUAL capitalization mentions")
    else:
        print("✗ FAILURE: balanced k=30 shows FEWER capitalization mentions")
        print(f"  Difference: {np.mean(cap_counts_balanced) - np.mean(cap_counts_hi_lo):.1f} mentions/rule")

    # Check minority class size ceiling
    n_minority_train = min(int((train_binary == 1).sum()), int((train_binary == 0).sum()))
    print(f"\nMinority class in train set: {n_minority_train} items")
    if n_minority_train < 15:
        print(f"  ⚠ CEILING: minority class < k//2 = {30//2}, balanced mode showing all {n_minority_train} minority items")

    # Show example rules
    print("\n" + "="*70)
    print("EXAMPLE RULES (full text)")
    print("="*70)

    for i, (rule_h, rule_b) in enumerate(zip(rules_hi_lo, rules_balanced)):
        print(f"\n--- Seed {i} ---")
        print(f"hi_lo k=4:\n{rule_h}")
        print(f"\nbalanced k=30:\n{rule_b}")

    print("\n" + "="*70)
    print("NOTE: Using merged description from g125 group as the rubric")
    print(f"Rubric: {rubric_body[:200]}...")
    print("="*70)

    # Save results
    out = {
        "metric_id": "g125",
        "metric_name": g125_group['merged_name'],
        "metric_description": g125_group['merged_description'],
        "rubric_body": rubric_body,
        "n_train": n_train,
        "train_yes": int((train_binary == 1).sum()),
        "train_no": int((train_binary == 0).sum()),
        "results": {
            "hi_lo_k4": {
                "rules": rules_hi_lo,
                "capitalization_counts": cap_counts_hi_lo,
                "generic_counts": gen_counts_hi_lo,
                "avg_capitalization": float(np.mean(cap_counts_hi_lo)),
                "avg_generic": float(np.mean(gen_counts_hi_lo)),
            },
            "balanced_k30": {
                "rules": rules_balanced,
                "capitalization_counts": cap_counts_balanced,
                "generic_counts": gen_counts_balanced,
                "avg_capitalization": float(np.mean(cap_counts_balanced)),
                "avg_generic": float(np.mean(gen_counts_balanced)),
            },
        },
        "verdict": "success" if np.mean(cap_counts_balanced) > np.mean(cap_counts_hi_lo) else "failure",
    }

    out_path = Path(__file__).parent / "trial" / "balanced_examples_ab_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
