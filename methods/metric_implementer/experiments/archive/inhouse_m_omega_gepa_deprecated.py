"""DEPRECATED 2026-07-19 (user decision): official github GEPA is the only sanctioned optimizer
for reconstruction experiments. Same-pool comparison on the sibling v14 decoder search: official
best pooled -0.014 / best admissible -0.052 vs in-house -0.078 (see outputs/fast/development/tuning/
on sk2).

This module holds the VERBATIM in-house GEPA search loop that previously lived in
``methods/metric_implementer/experiments/m_omega_gepa.py`` (``gepa_discriminative_m_omega`` and its
loop-only helper ``_fewshot_block``). It is the SECOND in-house loop retired under directive D1 (the
first was the v14 decoder-template search, ``archive/inhouse_gepa_deprecated.py``). Retained for
provenance and reproducibility of the pre-migration numbers ONLY.

The live ``m_omega_gepa.gepa_discriminative_m_omega`` now drives official ``gepa.optimize`` through a
``GEPAAdapter`` while preserving the estimand exactly (same executor primitive
``_score_binary_sampled``, same canonical pool objective ``_discrimination_score`` for the report; the
per-instance search signal is the mean-absolute-deviation decomposition, which is rank-equivalent to
the canonical objective for binary verdicts — see that function's docstring). The public name and
return contract are unchanged, so ``run_r2_recovery.py --gepa-m-omega`` needs no edit.

Unlike the v14 decoder shims, the public name here is NOT converted to a raising stub: it keeps
working, on official GEPA. This file is the frozen record of what it used to do.

The scoring/reporting primitives (``GEPAStats``, ``_score_binary_sampled``, ``_discrimination_score``,
``_compute_stats``, ``_select_failures``, ``_mutation_prompt``) stay live in ``m_omega_gepa`` — the
official path reuses every one of them — and are imported here so this record stays faithful rather
than forking a second copy.

CAVEAT carried forward (memory ``project_a_bank_degeneracy_audit``): "variance-revival !=
information-revival". A discrimination-maximizing objective can manufacture high-variance but
UNINFORMATIVE criteria; mined banks ran 54-68% degenerate. The migration preserves this pre-existing
objective and does NOT endorse it.

Do NOT wire this into any new experiment.
"""
from __future__ import annotations

from typing import List

from ...backends import LLMBackend
from ..m_omega_gepa import (
    _compute_stats,
    _mutation_prompt,
    _score_binary_sampled,
    _select_failures,
)

__all__ = ["gepa_discriminative_m_omega", "_fewshot_block"]


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
