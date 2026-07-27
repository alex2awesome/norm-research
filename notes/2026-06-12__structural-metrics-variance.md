# Honest variance for structural metrics (resampling fix)

Date: 2026-06-12. Script: `scripts/bootstrap_structural_metrics.py`.
Outputs: `outputs/analyses/structural_metrics/bootstrap_v1/` (per-task JSON,
`subtask_types_<task>.txt` inspection files, `summary.md`).

## The two bugs this fixes

1. **Wrong independence unit.** Forms within a source page are correlated
   (one guide emits dozens of rubrics). Resampling forms understates variance
   of every structural statistic. Fixed: resample **pages**.
2. **With-replacement bootstrap is biased for richness statistics.**
   A twice-drawn page turns its singleton clusters into size-2 clusters;
   bootstrap CIs for singleton rate / compression / Zipf slope excluded the
   point estimate entirely (e.g. code-review singleton 0.787, bootstrap CI
   [0.24, 0.47]). Fixed: duplication-free schemes — half-subsampling without
   replacement (Politis–Romano sqrt(m/n) scaling) + leave-one-subtask-type-out
   grouped jackknife.

## Subtask strata

`pages.parquet` has per-page `subtask_short/keywords/breadth/orientation`, but
`subtask_short` is near-unique free text (53K values). Derived ~25–50 coarse
subtask types per task via TF-IDF + k-means, recursively splitting any type
holding >20% of pages (single-pass k-means left a 51% catch-all on
peer-review). Types inspected manually: coherent (STROBE observational
studies, CHI reviewing, JOSS software review, Python PEPs, GoF patterns…).

## Results (singleton rate example; full tables in summary.md)

- page-level sd is **1.1–2.3×** the form-level sd (the old understatement).
- **Between-subtask variance dominates: 3–7× the page-level sd.** Total sd on
  singleton rate is ±0.015–0.043; on Zipf slope ±0.024–0.066.
- Zipf slope 95% CIs are now wide enough that many task pairs overlap —
  cross-task slope comparisons (and the T2 exponent check) must use these
  CIs, not point estimates.
- Reporting rule going forward: any statistic computed on the rubric corpus
  gets `total sd = sqrt(strat² + btype²)` from this script; CI = point ±
  1.96·total.

## Caveats

- Subsample sd assumes ~1/n variance scaling (fine for fractions/slopes;
  do not use for extreme-value stats like max cluster size).
- Grouped jackknife with unequal type sizes is approximate; types come from
  k-means on LLM-extracted free text, so btype is an estimate of
  composition sensitivity, not a clean sampling design.
- Point estimates are unchanged; only uncertainty is corrected. The realized
  under-merge issue (45–68% FN vs v6, see 2026-06-12 deep-dive discussion)
  is a separate, additive concern: report tail-sensitive stats as
  [as-clustered, after-judge-merge] intervals on top of these CIs.
