# N&C agree — cell brief for the map-focused Layer-3 round

Date: 2026-08-06. Prepared as job 4 of the gap-closer batch (parent task D6, full-grid
drive plan notes/2026-08-05__taste-decomposition-design.md §10). Status per the freeze
declaration (notes/2026-08-05__layer3-closure-prereg.md, "FREEZE DECLARATION 2026-08-06"):
N&C agree is on the **map-focused roster** (Track-B emphasis, Track A still run), NOT the
full dual-track roster — Δ_beyond does not clear the >.02 gate decisively (see below).
**This brief does not run the round.** The maps-batch agent picks it up from here.

## Verdict: inputs ARE sufficient for a map-focused round, with three caveats to carry

- `methods/taste_decomposition/results/nc_agree_layer1.json` — Layer-1 gate PASSED to
  machine precision (VA published .5509982 vs reproduced .5509985635, diff 3.15e-7; V/A
  exact). Ledger present: V_lin .6002 / A_lin .5236 / VA_lin .5510; V_nl .5906 / A_nl .5682
  / VA_nl .5844 (mean of seeds 0/1/2, spread .0093); Delta_interact +.0335 [docket-bootstrap
  CI −.011,+.097, p(>0)=.88 — NOT significant at the group level despite the row-level
  bootstrap reading positive, p=1.0]. OOF prediction arrays saved
  (`nc_agree_va_nl_oof_{seed0,mean3}.npy`).
- `methods/taste_decomposition/closure/samerows_preds/nc_agree_dense_preds_slim.csv`
  (5,047 rows incl. header) — per-row dense probability + `dense_split` (train/eval/test)
  + `in_dense_train` flag, joined on `doc_id`/`docket`. This is the same-rows T rescore
  (freeze #2) artifact: `results/samerows_T_nc_agree.json` reports dense-held-out n=1,009,
  AUC on held-out = **.6034** — sits between the divergent eval (.566) and test (.639)
  registry numbers, train-overlap fraction .800 (matches the other cells' ~80% grouped-split
  overlap by construction).
- v4 matrices: `datasets/notice-and-comment/v4/nc_scores_shard{0..4}.npz` (pre-GEPA, 198-rubric
  Gemma-4-31B A-bank, the same matrix nc_agree_layer1.json's A block was built from) are
  present locally and confirmed used by the Layer-1 run. `datasets/notice-and-comment/v4/`
  also has `nc_multiy_results.json` (the original 3-y aggregate this Layer-1 run's gate
  target came from) and the raw field/text jsonls (`nc_fields*.jsonl`) needed for texts.
- **Gap (not a blocker, but the map-round builder needs it):** `maps_batch1/cells.py`'s
  `LOADERS` dict currently covers `{peer_curation, peer_revealed, cap_crowd, cap_finalist,
  nc_outcome}` only — nc_agree has NO loader entry yet. Adding one is mechanical, following
  the existing `_load_nc_outcome()` pattern one function up in the same file: swap
  `data.valid_out`/`y_out_by_id`/`docket_m` for `NCData`'s agree-side attributes
  (`methods/taste_decomposition/nc_layer1_stack.py`: `valid_agr`, `y_agr_by_id`, same
  `docket_m`/`X_m`/`text_m`; label = majority `accepted`/`agree` vs `disagree` response_type,
  built by `--cell agree` in that script) and point the dense-preds join at
  `nc_agree_dense_preds_slim.csv` instead of `nc_outcome_dense_preds_slim.csv`. No new
  scoring or population-building needed — every input the loader would read already exists.

## Population

n=5,046 matched-labeled comment/response pairs (of 7,439 scored-matched total; agree/
disagree label only defined on the matched-with-a-labeled-response subset), pos rate .528
(agree/accepted vs disagree). Grouping unit: **docket** (944 distinct groups) — the
program-wide canonical N&C grouping. A-bank: 198 pre-GEPA Gemma-4-31B rubrics (same FROZEN
GKF-design matrix as N&C responded/outcome); V: 27 deterministic features.

## Splits

No FIT+MINE/MONITOR split has been built yet for this cell specifically (unlike nc_outcome,
which already has `maps_batch1/nc_outcome_splits.json`) — that construction is part of the
map round itself, not a prerequisite; `maps_batch1/build_splits.py` generalizes over cells
via `cells.py`, so once the loader entry above lands the same stable-hash-on-group,
MONITOR-inside-dense-held-out splitter applies unchanged. Note the dense-held-out pool this
splitter draws MONITOR from is n=1,009 (per samerows_T_nc_agree.json) — noticeably smaller
than nc_outcome's 1,417 or nc_responded's 1,904, so expect a tighter MONITOR set and wider
closure-curve CIs than those cells.

## T caveats — read before quoting anything from this cell

1. **Unstable-y eval/test divergence (registry-flagged, reconfirmed by same-rows rescore).**
   Registry (notes/2026-07-27__vat-run-registry.md, DENSE CHAIN table): N&C agree is the
   only cell where eval (.566) and test (.639) do not agree — "DIVERGES, agree-y instability
   again (n_eval 505, docket-skewed) ... report agree with BOTH numbers, never one." The
   same-rows rescore reproduces this on the *identical* held-out population, split apart:
   eval-only AUC .5660 (n=505) vs test-only AUC .6411 (n=504) vs the honest pooled
   dense-held-out AUC .6034 (n=1,009) — a .075 point gap between two halves of what should
   be one homogeneous held-out set. **Any Δ/Δ_beyond number for this cell must report the
   eval/test split explicitly (or use the pooled .6034 held-out number and say so), never a
   single number presented as "T".** Layer-1's own ledger already does this correctly
   (Delta_beyond_eval −.0184 vs Delta_beyond_test +.0546 — opposite signs).
2. **Docket-identity is a near-chance-collapsing confound within-group.** Layer 2
   (`results/layer2_nc_agree.json`, part_a_grouped_transfer): VA_nl pooled AUC .5863, but
   **within-docket AUC = .4934 (CHANCE)** on the 20 qualifying groups (≥both classes,
   n=1,539 rows), while **docket-identity ALONE gets AUC .8616**. This is the sharpest
   version of the program-wide finding (registry, LAYER 2 COMPLETE note): "N&C docket-identity
   leak severe ... N&C AGREE within-docket = .493 (CHANCE) — its entire edge is cross-docket."
   Read every VA_nl/T number on this cell as substantially docket-composition-driven, not
   comment-content-driven; a map-round proposer chasing "what predicts agree/disagree" is, to
   first order, chasing which docket a comment sits in.
3. **Δ_beyond does not clear the confirmatory gate decisively.** Layer-1 ledger:
   Delta_beyond_eval = −.0184 (VA_nl beats T-eval), Delta_beyond_test = +.0546 (T-test beats
   VA_nl); same-rows honest (.6034 held-out) vs VA_nl .5844 gives +.019 — all three readings
   sit at or below the .02 confirmatory-entry threshold used elsewhere in the campaign. This
   is exactly why the freeze declaration routes nc_agree to the **map-focused** track
   (Track-B/nuisance-mining emphasis, Track A still run for completeness) rather than the
   full closure-curve treatment given to CW community (+.176) / N&C responded (+.092).

## Recommended framing for the map round

Given (2) above, the map round's most informative contribution is probably NOT "does mining
close the residual" (the residual itself is unstable/near-zero and confound-dominated) but
"what, if anything, in the text tracks agree/disagree ACROSS dockets once docket identity is
accounted for" — i.e. lean into Track B (nuisance/upstream-factor mode, FREEZE ADDENDUM 2)
to characterize the docket-identity channel's textual fingerprints, alongside a standard but
lightly-weighted Track A pass. Flag this framing to the maps-batch agent; it is a suggestion,
not a re-scoping of the frozen protocol.
