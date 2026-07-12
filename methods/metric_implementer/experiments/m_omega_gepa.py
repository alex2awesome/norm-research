"""GEPA-M_ω: optimize the metric prompt for discriminative M_ω (2026-06-25).

Problem: M_ω = the executor X's verdict on the metric prompt. Currently we score the raw R2
`merged_description`, which is often skewed or near-constant (capitalization → 93% YES; PRISMA →
100% NO). A near-constant M_ω caps recovery at ~0 — there's nothing to reconstruct.

Solution: Use GEPA to GENERATE a discriminative M_ω — optimize the metric prompt so the executor's
verdict actually varies across items (balanced base-rate ≈ 0.5, high std). This is the proof's
`OPT = sup_p R(p)` search over prompts, with transmission/discrimination as the objective.

Algorithm:
  1. Seed prompt = seed_body (the R2 merged_description).
  2. GEPA loop per round:
     - Reviser (GLM) proposes n_mutations variant prompts (paraphrases / sharpenings /
       scope-narrowings — keep it the SAME underlying criterion, just worded so X discriminates).
     - Executor (X) scores each candidate's M_ω on items via SAMPLED YES/NO (generate "YES"/"NO",
       parse — mirror recon_channel._sampled_binary).
     - Objective = discrimination: score by balance+spread, e.g. std(pyes) - 0.5*abs(mean(pyes)-0.5)
       (high std, base-rate near 0.5).
     - Keep best; feed failures (items where it's still near-0.5 / ambiguous) back to reviser.
  3. Return optimized prompt, M_ω (pyes vector), mean, std, base_rate, per-round trajectory.

Test harness: Compare raw merged_description M_ω vs GEPA-optimized M_ω on creative-writing metrics.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.metric_implementer.backends import LLMBackend
from methods.metric_implementer.batch_scoring import _YESNO_TEMPLATE
from methods.metric_implementer.config import ImplementerConfig, apply_task_preset, REPO_ROOT
from methods.metric_implementer.experiments.mine_clusters import r2_groups


@dataclass
class GEPAStats:
    """Statistics for one candidate prompt."""
    prompt: str
    pyes: np.ndarray  # binary vector (0/1, NaN for parse failures)
    mean: float
    std: float
    base_rate: float
    discrimination: float  # objective = std - 0.5*abs(mean - 0.5)


def _score_binary_sampled(backend: LLMBackend, rubric: str, texts: List[str],
                           max_chars: int, temperature: float = 0.7,
                           seed: int = 0) -> np.ndarray:
    """Sampled YES/NO verdict per item (mirror recon_channel._sampled_binary)."""
    prompts = [_YESNO_TEMPLATE.format(rubric=rubric, text=t[:max_chars]) for t in texts]
    outs = backend.generate_batch(prompts, system=None, max_tokens=4,
                                  temperature=temperature, seed=seed)
    v = np.full(len(texts), np.nan)
    for i, o in enumerate(outs):
        t = (o or "").strip().lower()
        if t.startswith("yes"):
            v[i] = 1.0
        elif t.startswith("no"):
            v[i] = 0.0
    return v


def _discrimination_score(pyes: np.ndarray) -> float:
    """Objective = high std + base-rate near 0.5. Returns std - 0.5*abs(mean - 0.5)."""
    if np.all(np.isnan(pyes)):
        return -np.inf
    valid = pyes[~np.isnan(pyes)]
    if len(valid) == 0:
        return -np.inf
    mean_val = float(np.mean(valid))
    std_val = float(np.std(valid))
    # Reward high std, penalize deviation from 0.5 base-rate
    return std_val - 0.5 * abs(mean_val - 0.5)


def _compute_stats(pyes: np.ndarray, prompt: str) -> GEPAStats:
    """Compute full statistics for a candidate prompt."""
    valid = pyes[~np.isnan(pyes)]
    mean_val = float(np.mean(valid)) if len(valid) > 0 else 0.0
    std_val = float(np.std(valid)) if len(valid) > 0 else 0.0
    return GEPAStats(
        prompt=prompt,
        pyes=pyes,
        mean=mean_val,
        std=std_val,
        base_rate=mean_val,
        discrimination=_discrimination_score(pyes)
    )


def _mutation_prompt(noun: str, current_prompt: str, round_idx: int,
                    failure_examples: str) -> str:
    """Generate the reviser's mutation request."""
    return f"""You are refining an evaluation criterion for {noun} excerpts.

Your task: PARAPHRASE, SHARPEN, or NARROW THE SCOPE of the criterion below so that a fresh
evaluator would give MORE VARIED verdicts across different {noun} examples (mix of YES and NO,
NOT near-constant). Keep the SAME underlying property — just reword it to help the evaluator
discriminate better.

Current criterion (round {round_idx}):
{current_prompt}

Examples where the current criterion struggles (near-ambiguous / wrong base-rate):
{failure_examples}

Propose ONE reworded criterion. Reply with ONLY the criterion text (no preamble, no JSON,
no explanation)."""


def _fewshot_block(yes_examples, no_examples, max_chars: int = 400) -> str:
    """A labeled-examples block APPENDED to a reworded criterion (the few-shot operator). Concrete
    YES/NO anchors often calibrate a weak executor's base-rate and sharpen discrimination — an
    orthogonal lever to rewording. Code-appended (not model-generated) so the A/B isolates the effect:
    same rewording ± this block. `yes_examples`/`no_examples` are item-text lists (HELD OUT from the
    scored set, so they don't inflate discrimination by memorization).

    FEW-SHOT SELECTION DESIGN (locked 2026-06-25, the A+B+C hybrid the user specified — NOT yet wired;
    currently the caller passes code-selected top/bottom-P(YES) items):
      A. CODE surfaces candidates — top-k + bottom-k P(YES) items (executor-confident extremes).
      B. LLM PICKS by DIVERSITY — the reviser chooses the most prototypical/diverse YES & NO from A
         (not just the extremes — clarifies the criterion rather than reinforcing executor bias).
      C. LLM SYNTHESIZES/CONDENSES if needed — merge or fabricate a compact epitome when A/B are weak.
    CAVEAT (A/B 2026-06-25): few-shot with A-only (code extremes) HURT discrimination (0.448→0.292) —
    it reinforced the executor's skew. So few-shot is ALLOWED (mutation_mode fewshot/mixed) but NOT
    forced (default reword); when used, it should go through A+B+C, not A alone."""
    yes_blk = "\n".join(f"- YES (satisfies): {t[:max_chars]}" for t in (yes_examples or []))
    no_blk = "\n".join(f"- NO (does not satisfy): {t[:max_chars]}" for t in (no_examples or []))
    return ("Examples — judge each NEW excerpt the SAME way as these:\n" + yes_blk + "\n" + no_blk).strip()


def _select_failures(texts: List[str], pyes: np.ndarray, k: int = 10,
                    max_chars: int = 600) -> str:
    """Select k items where M_ω is near-ambiguous (pyes near 0.5) for reviser feedback."""
    if np.all(np.isnan(pyes)):
        return "(no valid scores to select from)"

    # Items with scores closest to 0.5 (most ambiguous)
    valid_mask = ~np.isnan(pyes)
    if not np.any(valid_mask):
        return "(no valid scores)"

    valid_pyes = pyes[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Sort by distance from 0.5
    distances = np.abs(valid_pyes - 0.5)
    sorted_order = np.argsort(distances)
    k_select = min(k, len(sorted_order))
    selected_indices = valid_indices[sorted_order[:k_select]]

    examples = []
    for idx in selected_indices:
        score = pyes[idx]
        examples.append(f"[score={score:.2f}]\n```\n{texts[idx][:max_chars]}\n```")

    return "\n\n".join(examples) if examples else "(no failure examples)"


def gepa_discriminative_m_omega(executor: LLMBackend, reviser: LLMBackend,
                                seed_body: str, texts: List[str], noun: str,
                                *, rounds: int = 3, n_mutations: int = 4,
                                max_chars: int = 600,
                                mutation_mode: str = "reword",
                                fewshot_examples=None) -> dict:
    """GEPA loop to find a discriminative M_ω prompt.

    Args:
        executor: LLMBackend for scoring (X, e.g., glm-4.7)
        reviser: LLMBackend for prompt mutations (e.g., glm-5.2)
        seed_body: Initial prompt (R2 merged_description)
        texts: List of item texts to score
        noun: Item noun (e.g., "story", "paper")
        rounds: Number of GEPA rounds (default 3)
        n_mutations: Number of mutations per round (default 4)
        max_chars: Max chars per text (default 600)

    Returns:
        dict with:
            - 'optimized_prompt': best prompt found
            - 'pyes': binary verdict vector (n_items)
            - 'mean': mean of pyes
            - 'std': std of pyes
            - 'base_rate': mean of pyes
            - 'discrimination': discrimination score
            - 'trajectory': list of (round, best_prompt, best_std, best_base_rate,
              best_discrimination)
    """
    n_items = len(texts)
    trajectory = []

    # Score seed prompt
    print(f"Round 0: Scoring seed prompt...")
    seed_pyes = _score_binary_sampled(executor, seed_body, texts, max_chars)
    seed_stats = _compute_stats(seed_pyes, seed_body)
    trajectory.append((0, seed_body, seed_stats.std, seed_stats.base_rate,
                      seed_stats.discrimination))
    print(f"  Seed: std={seed_stats.std:.3f}, base_rate={seed_stats.base_rate:.3f}, "
          f"discrimination={seed_stats.discrimination:.3f}")

    best_prompt = seed_body
    best_pyes = seed_pyes
    best_stats = seed_stats

    # GEPA rounds
    for r in range(1, rounds + 1):
        print(f"\nRound {r}: Generating {n_mutations} mutations...")

        # Generate mutation prompts
        failures = _select_failures(texts, best_pyes, k=10, max_chars=max_chars)
        mutation_request = _mutation_prompt(noun, best_prompt, r, failures)

        mutation_prompts = reviser.generate_batch(
            [mutation_request] * n_mutations,
            system=None,
            max_tokens=200,
            temperature=0.9,
            seed=r  # vary seed per round for diversity
        )

        # few-shot operator: append a labeled-examples block to (some) rewordings (isolates the effect)
        fs_block = None
        if mutation_mode in ("fewshot", "mixed") and fewshot_examples:
            ys, ns = fewshot_examples
            if ys or ns:
                fs_block = _fewshot_block(ys, ns, max_chars)

        # Score each mutation
        candidates = []
        for i, mut_prompt in enumerate(mutation_prompts):
            if not mut_prompt or len(mut_prompt.strip()) < 10:
                continue  # skip empty/too-short
            p = mut_prompt.strip()
            # fewshot: append to ALL; mixed: append to every other (A/B within one round)
            if fs_block and (mutation_mode == "fewshot" or (mutation_mode == "mixed" and i % 2 == 1)):
                p = p + "\n\n" + fs_block
            pyes = _score_binary_sampled(executor, p, texts, max_chars)
            stats = _compute_stats(pyes, p)
            candidates.append(stats)

            if i == 0 or (i + 1) % 5 == 0:
                print(f"  Mutation {i+1}/{n_mutations}: "
                      f"std={stats.std:.3f}, base_rate={stats.base_rate:.3f}, "
                      f"disc={stats.discrimination:.3f}")

        # Find best mutation
        round_best = max(candidates, key=lambda s: s.discrimination) if candidates else best_stats

        # Update global best
        if round_best.discrimination > best_stats.discrimination:
            best_prompt = round_best.prompt
            best_pyes = round_best.pyes
            best_stats = round_best
            print(f"  *** NEW BEST: std={best_stats.std:.3f}, "
                  f"base_rate={best_stats.base_rate:.3f}, "
                  f"disc={best_stats.discrimination:.3f}")
        else:
            print(f"  No improvement (best disc={best_stats.discrimination:.3f})")

        trajectory.append((r, best_prompt, best_stats.std, best_stats.base_rate,
                          best_stats.discrimination))

    return {
        "optimized_prompt": best_prompt,
        "pyes": best_pyes,
        "mean": best_stats.mean,
        "std": best_stats.std,
        "base_rate": best_stats.base_rate,
        "discrimination": best_stats.discrimination,
        "trajectory": trajectory
    }


def _load_creative_writing_pool() -> tuple[List[str], List[str]]:
    """Load creative-writing pool from methods/metric_implementer/trial/pool_creative_writing.jsonl.gz

    Returns:
        (texts, ids) tuples
    """
    pool_path = REPO_ROOT / "methods/metric_implementer/trial/pool_creative_writing.jsonl.gz"
    texts, ids = [], []

    with gzip.open(pool_path, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line)
                texts.append(obj["text"])
                ids.append(obj.get("candidate_id", f"idx_{len(texts)}"))
            except Exception:
                continue

    return texts, ids


def main():
    """Test harness: compare raw M_ω vs GEPA-optimized M_ω on creative-writing metrics."""
    parser = argparse.ArgumentParser(
        description="GEPA-M_ω: optimize metric prompts for discriminative M_ω")
    parser.add_argument("--task", default="creative-writing",
                       choices=["creative-writing", "peer-review"],
                       help="Task to test on")
    parser.add_argument("--bucket", default="general",
                       choices=["general", "specific", "hyper_specific"],
                       help="R2 bucket to sample metrics from")
    parser.add_argument("--n-metrics", type=int, default=3,
                       help="Number of metrics to test")
    parser.add_argument("--rounds", type=int, default=3,
                       help="GEPA rounds per metric")
    parser.add_argument("--n-mutations", type=int, default=4,
                       help="Mutations per GEPA round")
    parser.add_argument("--executor-model", default="glm-4.7",
                       help="Executor model (X)")
    parser.add_argument("--reviser-model", default="glm-5.2",
                       help="Reviser model (GLM)")
    parser.add_argument("--backend", default="zai_anthropic",
                       choices=["zai_anthropic", "openrouter"],
                       help="Backend for API calls")
    parser.add_argument("--max-text-chars", type=int, default=600,
                       help="Max characters per text item")
    parser.add_argument("--n-items", type=int, default=60,
                       help="Number of items to score (subset of pool)")
    args = parser.parse_args()

    print("=" * 80)
    print("GEPA-M_ω: Discriminative Metric Prompt Optimization")
    print("=" * 80)

    # Load config
    cfg = ImplementerConfig()
    cfg.backend = args.backend
    apply_task_preset(cfg, args.task)

    # Create backends
    executor = LLMBackend(args.executor_model, "executor", cfg,
                         temperature=cfg.judge_temperature)
    reviser = LLMBackend(args.reviser_model, "reviser", cfg,
                        temperature=cfg.other_temperature)

    print(f"\nBackend: {args.backend}")
    print(f"Executor: {args.executor_model}")
    print(f"Reviser: {args.reviser_model}")
    print(f"Task: {args.task}, Bucket: {args.bucket}")

    # Load pool
    print(f"\nLoading {args.task} pool...")
    if args.task == "creative-writing":
        texts, ids = _load_creative_writing_pool()
    else:
        # For peer-review or other tasks, add path here
        pool_path = REPO_ROOT / f"methods/metric_implementer/trial/pool_{args.task.replace('-', '_')}.jsonl.gz"
        texts, ids = [], []
        if pool_path.exists():
            with gzip.open(pool_path, 'rt') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        texts.append(obj["text"])
                        ids.append(obj.get("candidate_id", f"idx_{len(texts)}"))
                    except Exception:
                        continue
        else:
            print(f"  ERROR: Pool not found at {pool_path}")
            return

    # Subsample if needed
    n_items = min(args.n_items, len(texts))
    if n_items < len(texts):
        import random
        idx = random.sample(range(len(texts)), n_items)
        texts = [texts[i] for i in idx]
        ids = [ids[i] for i in idx]

    print(f"  Loaded {len(texts)} items (max {args.max_text_chars} chars each)")

    # Load R2 metrics
    print(f"\nLoading R2 metrics for {args.task}/{args.bucket}...")
    r2_metrics = r2_groups(args.task, args.bucket)
    print(f"  Found {len(r2_metrics)} R2 metrics")

    # Select top metrics by leaf count
    r2_metrics_sorted = sorted(r2_metrics, key=lambda m: m["total_leaf_rubrics"], reverse=True)
    n_metrics = min(args.n_metrics, len(r2_metrics_sorted))
    selected = r2_metrics_sorted[:n_metrics]

    print(f"\nSelected {n_metrics} metrics to test:")
    for i, m in enumerate(selected):
        print(f"  {i+1}. [{m['group_idx']}] {m['merged_name']} "
              f"({m['total_leaf_rubrics']} leaves)")

    # Run GEPA on each metric
    print("\n" + "=" * 80)
    print("Running GEPA optimization...")
    print("=" * 80)

    results = []
    for i, metric in enumerate(selected):
        print(f"\n{'=' * 20} Metric {i+1}/{n_metrics}: {metric['merged_name']} {'=' * 20}")
        print(f"Seed description:\n  {metric['merged_description'][:200]}...")

        try:
            # Raw M_ω (seed)
            print("\n--- RAW M_ω (seed prompt) ---")
            raw_pyes = _score_binary_sampled(executor, metric['merged_description'],
                                            texts, args.max_text_chars)
            raw_stats = _compute_stats(raw_pyes, metric['merged_description'])

            print(f"Raw: std={raw_stats.std:.3f}, base_rate={raw_stats.base_rate:.3f}, "
                  f"disc={raw_stats.discrimination:.3f}")

            # GEPA-optimized M_ω
            print("\n--- GEPA-OPTIMIZED M_ω ---")
            gepa_result = gepa_discriminative_m_omega(
                executor, reviser, metric['merged_description'],
                texts, cfg.item_noun,
                rounds=args.rounds,
                n_mutations=args.n_mutations,
                max_chars=args.max_text_chars
            )

            print(f"\nGEPA: std={gepa_result['std']:.3f}, "
                  f"base_rate={gepa_result['base_rate']:.3f}, "
                  f"disc={gepa_result['discrimination']:.3f}")

            # Summary
            delta_std = gepa_result['std'] - raw_stats.std
            delta_br = abs(gepa_result['base_rate'] - 0.5) - abs(raw_stats.base_rate - 0.5)

            print(f"\nDelta: Δstd={delta_std:+.3f}, Δ|base_rate-0.5|={delta_br:+.3f}")

            results.append({
                "metric_name": metric['merged_name'],
                "metric_idx": metric['group_idx'],
                "raw_std": raw_stats.std,
                "raw_base_rate": raw_stats.base_rate,
                "raw_disc": raw_stats.discrimination,
                "gepa_std": gepa_result['std'],
                "gepa_base_rate": gepa_result['base_rate'],
                "gepa_disc": gepa_result['discrimination'],
                "delta_std": delta_std,
                "delta_br": delta_br,
                "optimized_prompt": gepa_result['optimized_prompt'],
                "trajectory": gepa_result['trajectory']
            })

        except Exception as e:
            print(f"\nERROR on metric {metric['merged_name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Raw vs GEPA-optimized M_ω")
    print("=" * 80)
    print(f"{'Metric':<40} | {'Raw std':>8} | {'Raw BR':>8} | {'GEPA std':>8} | "
          f"{'GEPA BR':>8} | {'Δstd':>7} | {'Δ|BR-0.5|':>9}")
    print("-" * 80)

    for r in results:
        metric_short = r['metric_name'][:37] + "..." if len(r['metric_name']) > 40 else r['metric_name']
        print(f"{metric_short:<40} | {r['raw_std']:8.3f} | {r['raw_base_rate']:8.3f} | "
              f"{r['gepa_std']:8.3f} | {r['gepa_base_rate']:8.3f} | "
              f"{r['delta_std']:+7.3f} | {r['delta_br']:+9.3f}")

    # Overall statistics
    n_improved_std = sum(1 for r in results if r['delta_std'] > 0)
    n_improved_br = sum(1 for r in results if r['delta_br'] < 0)  # negative = closer to 0.5

    print("-" * 80)
    print(f"Improved std: {n_improved_std}/{len(results)}")
    print(f"Improved base-rate (closer to 0.5): {n_improved_br}/{len(results)}")

    # Show optimized prompts for inspection
    print("\n" + "=" * 80)
    print("OPTIMIZED PROMPTS (for manual inspection)")
    print("=" * 80)

    for i, r in enumerate(results):
        print(f"\n--- Metric {i+1}: {r['metric_name']} ---")
        print(f"Raw: {selected[i]['merged_description'][:200]}...")
        print(f"\nGEPA-optimized:\n{r['optimized_prompt']}")
        print(f"\nStats: std={r['gepa_std']:.3f}, base_rate={r['gepa_base_rate']:.3f}")

    print("\n" + "=" * 80)
    print("Done. Cost summary:")
    print(f"  Executor ({args.executor_model}): {executor.stats.as_dict()}")
    print(f"  Reviser ({args.reviser_model}): {reviser.stats.as_dict()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
