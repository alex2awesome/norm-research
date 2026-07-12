#!/usr/bin/env python3
"""Test script to A/B induce_free vs induce_reasoning on peer-review g125 metric."""

import gzip
import json
import os
import sys
from types import SimpleNamespace

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

import numpy as np

from methods.metric_implementer.backends import LLMBackend
from methods.metric_implementer.config import apply_task_preset, ImplementerConfig
from methods.metric_implementer.recon_channel import (_pyes, _hi_lo_examples_wide,
                                                        _hi_lo_examples, induce_free,
                                                        induce_reasoning, _sampled_binary)
from methods.metric_implementer.experiments.mine_clusters import r2_groups


def load_peer_review_pool(n_items=60):
    """Load peer-review texts from the trial pool."""
    pool_path = "methods/metric_implementer/trial/pool_peer_review.jsonl.gz"
    texts = []
    with gzip.open(pool_path, 'rt') as f:
        for i, line in enumerate(f):
            if i >= n_items:
                break
            obj = json.loads(line)
            texts.append(obj.get('text', ''))
    return texts


def main():
    print("=" * 80)
    print("A/B TEST: induce_free vs induce_reasoning on peer-review g125")
    print("=" * 80)

    # Load configuration
    cfg = apply_task_preset(ImplementerConfig(), "peer-review")
    texts = load_peer_review_pool(n_items=60)
    print(f"Loaded {len(texts)} peer-review texts")

    # Get g125 metric (General capitalization standards)
    groups = r2_groups("peer-review", "specific")
    g125 = None
    for g in groups:
        if g["group_idx"] == 125:
            g125 = g
            break

    if not g125:
        print("ERROR: g125 not found in peer-review/specific groups")
        print("Available groups:", [g["group_idx"] for g in groups[:10]])
        return

    print(f"\n=== g125: {g125['merged_name']} ===")
    print(f"Description: {g125['merged_description'][:200]}...")

    # Setup GLM backend (zai_anthropic)
    rcfg = ImplementerConfig()
    rcfg.backend = "zai_anthropic"
    recon = LLMBackend("glm-4.7", "reconstructor", rcfg)
    print(f"\nReconstructor: glm-4.7 via zai_anthropic")

    # Get M_ω labels using sampled binary (since GLM can't do logprob scoring)
    # We'll use a simple approach: generate YES/NO and parse
    rubric = g125['merged_description'] or g125['merged_name']
    print(f"\nScoring M_ω with rubric: {rubric[:100]}...")

    # Use _sampled_binary to get labels (this works with GLM)
    pyes_samples = []
    for r in range(3):  # 3 passes to get stable estimate
        binary = _sampled_binary(recon, rubric, texts, cfg.max_text_chars,
                                temperature=0.7, seed=r)
        pyes_samples.append(binary)

    # Aggregate: take majority vote or mean
    pyes_aggregated = np.nanmean(np.array(pyes_samples), axis=0)
    m_std = float(np.nanstd(pyes_aggregated))
    m_mean = float(np.nanmean(pyes_aggregated))

    print(f"M_ω stats: mean={m_mean:.3f}, std={m_std:.3f}")
    print(f"  (Note: base_rate is mean of soft scores, not binary split)")

    if m_std < 0.05:
        print("WARNING: M_ω has very low discrimination - may be degenerate")

    # Build examples (k=4 for free, n=30 for reasoning)
    examples_free = _hi_lo_examples(texts[:30], pyes_aggregated[:30], k=4, max_chars=600)
    examples_reasoning = _hi_lo_examples_wide(texts[:30], pyes_aggregated[:30], n_examples=30, max_chars=600)

    print(f"\n{'=' * 80}")
    print("METHOD A: induce_free (4 examples)")
    print("=" * 80)

    free_rules = []
    for r in range(3):
        result = induce_free(recon, cfg.item_noun, examples_free, seed=r, max_tokens=450)
        rule = result.get("rule", "")
        free_rules.append(rule)
        print(f"\n[Reconstruction {r+1}]")
        print(rule[:300])
        if len(rule) > 300:
            print("...")

    print(f"\n{'=' * 80}")
    print("METHOD B: induce_reasoning (30 examples, 3-round critique loop)")
    print("=" * 80)

    reasoning_rules = []
    for r in range(3):
        result = induce_reasoning(recon, cfg.item_noun, examples_reasoning, seed=r,
                                  max_tokens=450, rounds=3)
        rule = result.get("rule", "")
        reasoning_rules.append(rule)
        print(f"\n[Reconstruction {r+1}]")
        print(rule[:300])
        if len(rule) > 300:
            print("...")

    # Analyze results
    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print("=" * 80)

    # Check for "capitalization" mentions
    capitalization_free = sum(1 for r in free_rules if "capital" in r.lower())
    capitalization_reasoning = sum(1 for r in reasoning_rules if "capital" in r.lower())

    # Check for generic priors
    generic_terms = ["novelty", "problem formulation", "innovation", "quality", "clear"]
    generic_free = sum(1 for r in free_rules if any(t in r.lower() for t in generic_terms))
    generic_reasoning = sum(1 for r in reasoning_rules if any(t in r.lower() for t in generic_terms))

    print(f"\nMentions 'capitalization':")
    print(f"  induce_free: {capitalization_free}/3")
    print(f"  induce_reasoning: {capitalization_reasoning}/3")

    print(f"\nUses generic prior terms ({', '.join(generic_terms)}):")
    print(f"  induce_free: {generic_free}/3")
    print(f"  induce_reasoning: {generic_reasoning}/3")

    print(f"\n{'=' * 80}")
    print("SUCCESS CRITERIA")
    print("=" * 80)

    if capitalization_reasoning > capitalization_free:
        print("✓ reasoning mode MORE likely to mention capitalization")
    else:
        print("✗ reasoning mode did NOT improve capitalization detection")

    if generic_reasoning < generic_free:
        print("✓ reasoning mode LESS likely to use generic priors")
    else:
        print("✗ reasoning mode did NOT reduce generic priors")

    print(f"\nM_ω skew caveat: base_rate={m_mean:.3f}, std={m_std:.3f}")
    if m_mean > 0.8 or m_mean < 0.2:
        print("  ⚠ M_ω is skewed (high YES rate) - may be easier/harder to recover")
    elif m_std < 0.12:
        print("  ⚠ M_ω has low discrimination (<0.12 std) - near-constant verdicts")


if __name__ == "__main__":
    main()
