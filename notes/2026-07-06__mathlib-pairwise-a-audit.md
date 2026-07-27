# Mathlib pairwise-A audit — "becoming merge-ready" is articulable

**Date:** 2026-07-06
**Scope:** Audit of the within-PR pairwise experiment on MathlibPR (Xie/Liu/Zhang, UVA,
arXiv:2605.07147) — the one place A carries signal on mathlib. Covers the pairwise-A judge
result and its V companion (the revision-direction footprint), the Y definition, confound
checks, and how it relates to the static nulls.
**Artifacts (sk3):** `datasets/math/mathlib/mathlibpr/{pairwise_a_judge.py, pairwise_a_cache.jsonl (n=300), pairwise_a.log, pairwise_vpp.py, pairwise_rows.json, diffs_cache.jsonl, reconstruct_diffs.py}`. Standing memory: `project_mathlibpr_replication_2026_06_29.md`.

---

## 1. Headline

An LLM judge (GLM-4.7) **cannot** score a single mathlib diff as merge-ready (static-A is null
everywhere), but **can** pick which of two versions of the same PR is the revised/accepted one
(pairwise-A = **0.723**). The reframing:

> **"Being merge-ready" is not articulable; "becoming merge-ready" is.**

This does not overturn the static nulls — it adds a caveat: mathlib-A is null in *absolute* mode
but recovers in *contrast* mode.

## 2. Result table

All numbers are within-PR (earlier vs final snapshots of the **same** merged PR), so topic /
author / difficulty / scope are held fixed — the only thing that varies is the revision.

| Readout | Metric | Value | vs chance |
|---|---|---|---|
| **Pairwise-A** (GLM judge, blind, order-randomized) | accuracy picks the FINAL snapshot | **0.723** (95% CI [0.673, 0.774]; n=300) | p ≈ 2.8e-15 |
| **V** (mechanical, revision *direction*) | AUROC of the within-pair feature delta | **0.661** | — |
| V, size-only | AUROC | 0.540 | (size is not the source) |
| V, non-size | AUROC | 0.656 | (signal is tactic-idiom, not size) |
| Static-A (all prior mathlib runs) | AUROC vs `target_merged` | ~0.50 / redundant with V | null |

Pairwise-A **beats V's mechanical 0.661**, and is calibrated (conf-5 → acc 0.765, n=136;
conf-4 → 0.685, n=162). No A/B position bias: `final_is` balanced 162/138, predictions 155/145.

## 3. How Y is measured (the crux)

Both Y's derive from real MathlibPR git history of **merged** PRs. A PR has many commits over
time; two snapshots are selected from each:

- **earlier** = `first_build_success_snapshot` (e.g. source_commit_seq 41) — the first commit that
  *compiled*, but review sent it back. Label **`not_merge_ready`**.
- **final** = `final_snapshot` (e.g. seq 62) — the commit actually **merged into master**. Label
  **`merge_ready`**.

From the benchmark doc: *"The label is attached to the selected snapshot, not to the PR as a
whole."* So the label is not "good PR vs bad PR" — it is "this *version* was merge-ready vs this
*version* wasn't yet."

### The level vs. difference distinction

- **"Being merge-ready" (level, null):** per-snapshot binary `target_merged` on the 15,895-snapshot
  static benchmark (AUROC). Requires knowing the absolute merge-ready bar from a single diff → not
  legible → null (V″/A/retrieval all ≈ chance).
- **"Becoming merge-ready" (difference, the signal):** given the PAIR in random order, which one is
  the `final_snapshot` (accuracy). Everything is held fixed across the pair except the revision, so
  the within-PR difference is what both V and A read.

### Honest caveat on the ground truth

`merge_ready` is not a human quality rating — it is *"this is literally the commit that got
merged."* The earlier snapshot is `not_merge_ready` *because it was revised before acceptance*;
the label is defined by the revision event itself. So "becoming merge-ready" is, precisely,
"can you detect the direction of the edit the maintainers induced." That is exactly what V's
revision-direction result measures mechanically — so pairwise-A and V are reading the same
event from two angles (judged vs. featurized), which is why both succeed and why both are
honest rather than circular.

## 4. Pairwise-A design

Script `pairwise_a_judge.py`. Per pair:

1. Reconstruct BOTH cumulative diffs `git diff base..snapshot -- '*.lean'` (clean per-snapshot PR
   content; median 2 / mean ~9 files — see §6 confound).
2. Truncate each to 9,000 chars (middle-preserving).
3. **Order-randomize** per pair (deterministic `md5(pair_id) % 2`): the judge sees SNAPSHOT_A /
   SNAPSHOT_B and is *not* told which is earlier. `final_is` records the true position.
4. GLM-4.7 via the **free** z.ai Anthropic endpoint (`/api/anthropic/v1/messages`, `x-api-key`
   header). Qwen is deliberately **not** used for eval (per `feedback_qwen_not_for_eval`).
5. Blind: system prompt names only SNAPSHOT_A/B, no label, no PR id, no author, no merge outcome.
6. Forced-JSON response: `{more_merge_ready: A|B, confidence: 1-5, what_improved: ...}`.
7. Resumable JSONL cache; stops cleanly on real quota exhaustion (HTTP 429 code 1310).

**Blindness/leakage checks passed:** the judge sees only author-stripped Lean diff text; no
metadata (PR number, author, dates, CI, comments) is in the prompt. Order is randomized and
balanced, so any position bias would show in the `pred` balance — it does not (155/145).

## 5. What the judge names (not fished)

Correct picks cite real mathlib review concerns:

- *refactoring / consolidation* — "consolidates `reorderAttr` and `relevantArgAttr` into a single
  `argInfoAttr`."
- *idiomatic naming* — "`partialSups_const_mul` / `partialSups_mul_const` vs the less idiomatic
  `partialSups_mulLeft`."
- *simp / to_additive idiom* — "adds `@[to_additive (attr := simp, norm_cast)]` … standard and
  highly useful for simplification and casting."
- *docs / justification* — "includes explanatory comments justifying the addition of the
  instances (avoiding reliance on order theory for algebraic results)."
- *structural cleanup* — "correctly closes the `section iSup` scope with `end iSup`."

Wrong picks (chose the earlier snapshot) are *principled judgment calls*, not noise — e.g.
preferring fuller variable declarations in the signature, or explicit proofs over `simp`/`grind`
expansion.

## 6. Confound checks

- **snap..snap base drift (CAUGHT & DISCARDED).** A natural "control" is to featurize the direct
  edit `git diff earlier_snap..final_snap`. Diagnosis on pair pr7172: that diff is **1,645 files /
  28K lines**, ≈99.9% **master drift** (the two snapshots sit at different base commits;
  `earlier_base..final_base` alone is 1,644 files). It is *not* the review edit. The real review
  edit is tiny (15→14 insertions on pr7172). The cumulative `base..snap` diffs are the clean PR
  content, so both the V delta and the pairwise-A judge use those. The snap..snap control was
  discarded (and was pathologically slow precisely because of the drift).
- **Position bias:** ruled out by randomized order + balanced `pred` distribution.
- **Judge collapse / all-same-score:** ruled out — conf is split 5/4 with both buckets populated
  and accuracy rising with confidence.
- **Label leakage:** none — judge input is author-stripped Lean text only.

## 7. Relation to the static nulls

The static accept/reject decomposition on MathlibPR (15,895 build-passing snapshots) is null on
all three arms and replicates our own mathlib null:

| Arm | Our data | MathlibPR |
|---|---|---|
| V″ (deterministic) | 0.68 | 0.604 |
| Retrieval (duplication) incremental | +0.009 | +0.015 (null) |
| Style linters incremental | −0.007 | −0.002 (null) |
| A-judge incremental | ~null | −0.009 (null) |

Those are all *static/level* reads (single snapshot → `target_merged`). The pairwise result lives
on the *difference* axis and does not contradict them — it specifies *where* A can succeed on
mathlib (contrast) when it cannot succeed absolutely.

## 8. Why this matters (the mechanism)

Pairwise contrast is the **same mechanism that makes Math.SE-A work** (sibling answers to one
question give the judge a reference). It dodges the scope-incomparability that killed static
mathlib-A: there, a 6-line tactical PR and a 500-line theory PR are both "is this merge-ready?"
with no shared yardstick, so the judge hedges; in a within-PR pair the yardstick is implicit
(the earlier version), so "what improved?" becomes answerable.

Net for the paper: mathlib goes from "flat leg, A buys nothing" → "flat leg, but **A ≈ V on the
revision axis** — both live in the contrast, not the artifact." Sharper than a plain triple-null.

## 9. Limitations / open

- **n = 300** of 3,520 reconstructable pairs. The CI [0.673, 0.774] clears chance and V comfortably,
  but scaling to the full ~3,500 would tighten it (a natural follow-up; resumable cache).
- This is a **within-PR rank** (accuracy), directly comparable to V's 0.661 (also within-PR
  direction). It is **not** comparable to the static accept/reject AUROCs (different task /
  population).
- Single judge (GLM-4.7). A cross-judge replication (Claude / Codex) would harden the "articulable"
  claim — Qwen excluded by the eval-trust rule.
- Pairwise-A on the static accept/reject task (not just within-PR) is untried; would test whether
  contrast helps there too.
