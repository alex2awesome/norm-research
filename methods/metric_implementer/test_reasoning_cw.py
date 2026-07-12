#!/usr/bin/env python3
"""Test script to A/B induce_free vs induce_reasoning on creative-writing metrics."""

import gzip
import json
import os
import sys

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


def load_creative_writing_pool(n_items=60):
    """Load creative-writing texts from the trial pool."""
    pool_path = "methods/metric_implementer/trial/pool_creative_writing.jsonl.gz"
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
    print("A/B TEST: induce_free vs induce_reasoning on creative-writing")
    print("=" * 80)

    # Load configuration
    cfg = apply_task_preset(ImplementerConfig(), "creative-writing")
    texts = load_creative_writing_pool(n_items=60)
    print(f"Loaded {len(texts)} creative-writing texts")

    # Get a few creative-writing metrics
    groups = r2_groups("creative-writing", "general")
    if not groups:
        print("ERROR: No creative-writing/general groups found")
        return

    # Test first 2 metrics
    test_metrics = groups[:2]

    # Setup GLM backend (zai_anthropic)
    rcfg = ImplementerConfig()
    rcfg.backend = "zai_anthropic"
    recon = LLMBackend("glm-4.7", "reconstructor", rcfg)
    print(f"\nReconstructor: glm-4.7 via zai_anthropic")

    for metric in test_metrics:
        gidx = metric["group_idx"]
        print(f"\n{'=' * 80}")
        print(f"METRIC g{gidx}: {metric['merged_name']}")
        print("=" * 80)
        print(f"Description: {metric['merged_description'][:200]}...")

        rubric = metric['merged_description'] or metric['merged_name']

        # Get M_ω labels using sampled binary
        pyes_samples = []
        for r in range(3):
            binary = _sampled_binary(recon, rubric, texts, cfg.max_text_chars,
                                    temperature=0.7, seed=r)
            pyes_samples.append(binary)

        pyes_aggregated = np.nanmean(np.array(pyes_samples), axis=0)
        m_std = float(np.nanstd(pyes_aggregated))
        m_mean = float(np.nanmean(pyes_aggregated))

        print(f"M_ω stats: mean={m_mean:.3f}, std={m_std:.3f}")

        if m_std < 0.05:
            print("  WARNING: M_ω has very low discrimination - skipping")
            continue

        # Build examples
        examples_free = _hi_lo_examples(texts[:30], pyes_aggregated[:30], k=4, max_chars=600)
        examples_reasoning = _hi_lo_examples_wide(texts[:30], pyes_aggregated[:30],
                                                   n_examples=30, max_chars=600)

        print(f"\nMETHOD A: induce_free (4 examples)")
        free_rules = []
        for r in range(2):  # Only 2 reconstructions to save quota
            result = induce_free(recon, cfg.item_noun, examples_free, seed=r, max_tokens=450)
            rule = result.get("rule", "")
            free_rules.append(rule)
            print(f"\n  [Reconstruction {r+1}] {rule[:200]}...")

        print(f"\nMETHOD B: induce_reasoning (30 examples)")
        reasoning_rules = []
        for r in range(2):
            result = induce_reasoning(recon, cfg.item_noun, examples_reasoning, seed=r,
                                      max_tokens=450, rounds=3)
            rule = result.get("rule", "")
            reasoning_rules.append(rule)
            print(f"\n  [Reconstruction {r+1}] {rule[:200]}...")

        # Simple analysis: check rule specificity
        avg_len_free = np.mean([len(r) for r in free_rules])
        avg_len_reasoning = np.mean([len(r) for r in reasoning_rules])

        print(f"\n  Analysis:")
        print(f"    Avg rule length: free={avg_len_free:.0f}, reasoning={avg_len_reasoning:.0f}")
        print(f"    M_ω skew: mean={m_mean:.3f}, std={m_std:.3f}")


if __name__ == "__main__":
    main()
