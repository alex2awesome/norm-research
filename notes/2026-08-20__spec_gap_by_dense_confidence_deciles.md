# SPEC — localize the articulability gap by dense-confidence strata

Date: 2026-08-20. Requested by user ("sure, I'm interested in this test if it
shows us anything"). Charge for the ladder/VAT lane (claude-main); CPU-only,
no GPU, no new judging. Author of spec: claude-fig lane.

## Motivation

Fig 5's ceiling audit found raw gap ~ ceiling r=.45 (p=.034) across the 22
cells: part of the raw gap is mechanical headroom. The user's sharper worry:
**some datasets may contain subpopulations that are easy to classify but hard
to capture with written criteria** — dense's edge would then be a property of
an identifiable slice, not of the whole population. This test localizes the
gap instead of arguing about it: compute the dense-vs-bank gap *within
confidence strata*. Two possible outcomes, both informative:
- gap concentrated in dense-confident strata where the bank is ~chance →
  tacit subpopulation localized (then: inspect those rows; are they a
  nameable stratum → route to Track B; or genuinely undescribed signal →
  the phenomenon, now with receipts);
- gap roughly uniform across strata → the tacit signal is diffuse, and the
  easy-subpopulation account is wrong for that cell.

## Procedure (per cell)

Inputs: the SAME held-out rows used for the cell's published gap — per-row
dense OOF/held-out score `p_T(x)` and bank score `p_VA(x)` (grouped-OOF where
the cell is grouped), plus y. NEVER refit anything; this is a readout.

1. Confidence = |p_T(x) − .5| after per-cell rank-normalizing p_T to [0,1]
   (rank first: dense scores are not calibrated). Deciles of confidence over
   the held-out rows; for grouped cells, deciles WITHIN group then pooled
   (composition guard — same reason the code cell reads within-repo).
2. Within each decile: AUC_T, AUC_VA on those rows (skip deciles with <30
   rows or a single class; report skipped mass). Per-decile gap = AUC_T −
   AUC_VA. Also the cumulative curve: gap on the top-k% most-confident rows,
   k = 10..100.
3. **Symmetric arm (mandatory):** repeat with strata defined by bank
   confidence |p_VA − .5|. Conditioning on a model's own score
   range-restricts THAT model's within-stratum AUC (mechanically deflating
   it), so each arm is conservative against the model doing the stratifying;
   read the two arms together, never one alone.
4. Readout per cell: 10-row table (decile, n, AUC_T, AUC_VA, gap) × 2 arms +
   the two cumulative curves. Threshold-free; no significance theater — this
   is descriptive localization (pre-kill checklist applies before any claim).

## Cells (run in this order)

Large-gap cells with per-row artifacts already on disk, plus one null control:
| cell | gap | artifacts |
|---|---|---|
| peer_revealed (citations) | .160 | results/peer_revealed_va_nl_oof_mean3.npy + dense same-rows preds (samerows_T_peer_cells.json lineage) |
| cw_community (story upvotes) | .127 | results/cw_community_* + dense preds |
| peer_verdict (accept/reject) | .109 | results/peer_verdict_* |
| math.SE votes | .048 | results/mathse_vote_score_* |
| CONTROL: competitions or homepage (gap ≈ 0) | .000/.011 | comp_fourplatform / homepage_fused_stack lineage |

The control cell must come back flat/null or the instrument is suspect.

## Guards / disciplines
- Same rows for both models in every stratum (apples-to-apples rule).
- Grouped cells: within-group ranks + grouped deciles; report both pooled and
  within-group versions if they disagree.
- If a dense-confident/bank-chance stratum appears: BEFORE claiming a tacit
  subpopulation, sweep the named-nuisance columns (F2 block; length, era,
  identity, topic) on that stratum — if a named channel separates it, it is
  Track-B material, not tacit.
- Descriptive note in notes/, numbers in a results json next to the cell's
  ledger; no paper text until the user reads it.

Pointer added to the run registry. Related: FIGURES.md 2026-08-20 round 5-6
(ceiling audit), fig:auc-ceilings appendix caption.
