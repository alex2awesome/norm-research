# Chandra cells — per-subreddit ladders (Task A, user-ordered 2026-08-24)

**FRAME (binding): v1 populations; era channel open; v2 rescore will supersede.**
(Era-compression flaw on the kept side is known; kept_v2 recollection in flight
elsewhere — nothing here touches kept_v2.)

Design (mirrors the pooled quotes exactly):
- **V / A / VA** = layer-1 REFIT within the sub's rows only
  (`chandra_layer1_persub.py`, canonical layer1_gemma_cells machinery +
  so_votes collapse gate recomputed within-sub), GBM 3-seed-mean OOF AUC.
  Grouping DECLARED: the removal log carries **no timestamps** (era undated),
  so created-month bins are unavailable → **10 stable-row-hash pseudo-groups**.
- **T** = **pooled-trained dense, sub-restricted readout** (the pooled
  `dense_standard_chandra_*` seed-mean `preds_test.csv` restricted to the sub;
  zero retraining). Test leg quoted, eval in the jsons.
- **VAT** = per-sub bake-off (`chandra_vat_bakeoff_persub.py`): variants
  selected on the sub's eval leg among {rank_mean, eval_weighted_rank},
  reported on the sub's test leg (parents = baselines; logistic_evalfit and
  fused_stack recorded descriptively in the jsons).

## chandra_humor (pos rate .50 every sub; tifu n=544 skipped)

| sub | n | V | A | VA | T (pooled-dense, sub-restricted, test) | VAT (bakeoff, test) |
|---|---|---|---|---|---|---|
| funny | 23,240 | .589 | .731 | .744 | .844 | .834 |
| Showerthoughts | 17,992 | .625 | .786 | .797 | .853 | **.857** |
| nottheonion | 12,448 | .597 | .734 | .744 | .888 | .871 |
| me_irl | 7,348 | .596 | .742 | .749 | .781 | **.804** |
| POOLED (ref) | 61,572 | .551 | .689 | .694 | .849 | .829 |

## chandra_cw

| sub | n | V | A | VA | T (pooled-dense, sub-restricted, test) | VAT (bakeoff, test) |
|---|---|---|---|---|---|---|
| nosleep | 29,370 | .596 | .776 | .784 | .917 | .905 |
| books | 14,632 | .654 | .819 | .831 | .930 | .917 |
| asoiaf | 10,202 | .590 | .770 | .774 | .896 | **.902** |
| gameofthrones | 8,412 | .556 | .692 | .697 | .867 | .863 |
| POOLED (ref) | 62,616 | .543 | .583 | .579 | .911 | .906 |

Winner by eval = eval_weighted_rank in all 8 subs.

## Readout (descriptive)

1. **Within-sub refits articulate MORE, everywhere.** All 8 subs beat the
   pooled V, A, and VA. The effect is dramatic on chandra_cw: pooled A .583 /
   VA .579 vs within-sub A .69–.82 / VA .70–.83. The pooled cw layer-1 was
   fighting cross-subreddit norm heterogeneity (nosleep horror-craft vs books
   discussion vs ASOIAF theorycrafting); one criteria model per community
   recovers .78–.83 VA. Humor shows the same direction, milder (+.05 to +.10).
2. **The articulated–dense gap narrows within-sub but does not close.**
   Pooled cw gap T−VA ≈ .33; within-sub it is .10–.17. Humor pooled gap .155;
   within-sub .056–.14 (Showerthoughts .056).
3. **T is heterogeneous across subs** (.78 me_irl → .89 nottheonion; .87
   gameofthrones → .93 books): the pooled T .849/.911 averages over genuinely
   different per-community separability.
4. **VAT beats T in 3/8 subs** (me_irl +.023, asoiaf +.006, Showerthoughts
   +.004) — exactly the subs where VA is closest to T; elsewhere the rank
   blend dilutes a dominant T, as in the pooled cells.
5. Small print: per-sub VA uses pseudo-group folds (no sub-identity channel
   possible within one sub, but era/author channels remain undeclared — same
   v1 caveat as the pooled cell).

## Artifacts (sk3)

- Per-sub ledgers/OOF/bakeoffs:
  `methods/taste_decomposition/results/chandra_{humor,cw}_persub_<sub>_{ledger.json,va_oof.npz,vat_bakeoff.json}`
- Scripts (repo, committed): `methods/taste_decomposition/chandra_layer1_persub.py`,
  `chandra_vat_bakeoff_persub.py`, `chandra_persub_harvest.py`
- Logs: `logs/chandra_{humor,cw}_persub_{l1,vat}.log`
