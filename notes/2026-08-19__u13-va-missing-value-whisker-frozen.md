# U13 — VA missing-value upper whisker: FROZEN DEFINITION (2026-08-19, before computation)

## Estimator
For each cell with a completed closure campaign:

  whisker_top = VA_bar + max(0, VA_new − VA) × M̂_A / (1 − M̂_A)

- M̂_A = Track-A (real-criteria) Good-Turing missing mass f1/N from the LAST
  completed round's species file (closure/<cell>/<cell>_r<K>_species.json).
- (VA_new − VA) = the cell's observed total mining gain (master-ladder frame).
- The multiplier M̂/(1−M̂) scales observed gain-per-discovered-mass to the
  remaining mass, LINEARLY.

## Why this is an UPPER bound, not an estimate
Proposers surface salient (high-value) criteria first, so value-per-unit-mass is
non-increasing in discovery order; under that concavity the linear extrapolation
bounds the remaining gain from above. Where mining is immature (high mass, large
observed gain) the bound is wide — that is honest, not a bug (it says "mining
has not converged", exactly the articulated-share lower-bound caveat).

## Exclusions / display
- No campaign OR no VA_new column ⇒ NO whisker; bar marked "n/e" (no estimate).
- Whisker display-capped at .99 AUC; the uncapped value goes in the provenance
  comment of the figure source.
- Track-B (spurious) Z values are NEVER used here (wrong side — they bound
  confound strength, not articulable value).

## Inputs (harvested 2026-08-19, results/u13_trackA_harvest.json)
masses: aops .558 | bbc .492 | cap_crowd .387 | cap_finalist .507 | hashtagwars
.733 | jokes_comm .564 | mathse_acc .325 | mathse_vote .283 | nc_agree .450 |
nc_outcome .500 | peer_curation .778 | peer_revealed .456 | press_verdict .589 |
style_inv .733
