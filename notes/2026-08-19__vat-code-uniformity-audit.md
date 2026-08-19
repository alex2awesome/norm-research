# VAT code uniformity audit (2026-08-19, user-requested)

Trigger: transition_full_ladder.py initially re-implemented OOF/within-repo/fusion
locally. Audit question: is the same statistical machinery uniform across cells?

## Canonical modules (the ONLY things new cell scripts should use)
- `methods/taste_decomposition/layer1_gemma_cells.py` — outer_folds, linear_oof_family1,
  gbm_oof_family1 (per-fold median+indicator imputation, inner GroupKFold(3) grid over
  max_leaf_nodes {15,31}, lr .06, max_iter 400), GBM_SEEDS; VA_nl = mean over GBM_SEEDS.
- `methods/taste_decomposition/so_votes_layer1.py` — within_group_auc (PAIR-WEIGHTED,
  skips pure groups, returns dict), modal_share + COLLAPSE_MODAL_MAX collapse gate.
- `methods/taste_decomposition/unified_fused_stack.py` — VAT V3 = GroupKFold(5)
  logistic (StandardScaler+LR) stack on [VA_nl OOF, dense seed-mean], same-rows.

## Findings
1. within_group_auc: ~8 files carry LOCAL COPIES (si_v2, tweets_community, bbc_mostread,
   patents_fwdcites, nc_cosigning_layer1, nc_cosigning_attach_T, testexec_ladder) — all
   verified SEMANTICALLY IDENTICAL (pair-weighted n_pos*n_neg, pure-group skip). Drift
   risk only, no current error.
2. `layer2_robustness.py` within_group_auc is DIFFERENT BY DESIGN: group-SIZE-weighted
   mean with min_n filter (the D2 appendix's declared statistic). NEVER quote its
   within-group numbers interchangeably with pair-weighted ones.
3. GBM family1: per-cell pre-refactor scripts (press_verdict, code_competitions,
   layer1_stack, patents_verdict, mathlib_verdict, nc_*) carry the SAME GRID + inner-CV
   convention inline; linear-gate imputation varies (median+indicator vs constant-0.5),
   documented in each file, affects the linear gate only (VA_nl unaffected).
4. FIXED today: transition_full_ladder.py rewritten to import L/SV/stack pattern
   (first version used ad-hoc logistic+HistGB(3-seed fixed-lr) and dense-as-feature-column
   fusion — killed mid-run, no numbers were recorded from it).
5. SUPERSEDED: testexec_ladder.py (37-row A join; local but consistent implementations)
   — kept for the record, header note added; its published numbers stand under its own
   conventions.

## Rule going forward (BEST-PRACTICES candidate)
New cell scripts import the three canonical modules; local re-implementations of
folds/OOF/within-group/fusion are a review-blocking defect. When a deliberately
different statistic is needed (layer2 style), name it differently in the output JSON.
