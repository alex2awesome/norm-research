# SPUR-BATTERY — evidence-based spurious-vs-real routing (frozen spec v1)
User directive 2026-08-18: "be really careful deciding which are actually spurious
vs real... establish spurious-feature tests... psychometric tests... multiple LLMs."
Written BEFORE the pilot runs; the pilot set is the 13 BBC word-probe channels.

## The four empirical tests (per candidate channel, from its Gemma scores)
1. TRANSPORT (measurement invariance): AUC(channel, y) per era band and per beat
   stratum. Report the stability spread (max-min across contexts, grouped-boot CI).
   Context-bound collapse = evidence of spuriousness; stability = evidence of merit.
2. WITHIN-VS-BETWEEN: pooled AUC vs within-stratum pair-weighted AUC (strata =
   topic/beat clusters). Between-only signal = composition channel.
3. DISCRIMINANT LOADING (MTMM logic): correlation of the channel with (a) the
   quality-criterion factor (first PC of the A bank) vs (b) the nuisance factor
   (first PC of [length, position/ordinal, era, known-B channels]). Report the
   loading pair.
4. PARAPHRASE INTERVENTION (contested channels only; GPU): meaning-preserving
   rewrite that strips surface fingerprints; score shift under paraphrase =
   surface channel. Precedent: metric-lexicon register/paraphrase result.

## The panel
3-family LLM panel (codex sol + GLM + Claude-family judge), each shown the
evidence card (tests 1-3 tables, test 4 where run) + the channel definition, ruling
quality-relevant / spurious / MIXED. Majority routes; disagreement -> MIXED.
Planted probes continue (known-real and known-spurious anchors seeded per batch).

## Caveats (carried into every readout)
- Instability separates CONTEXT-BOUND from STABLE, which correlates with but does
  not equal spurious-vs-real; genuinely merit-relevant era-local norms exist ->
  MIXED, never auto-verdict. The battery produces EVIDENCE for the panel.
- Tests 1-3 reuse the channel's corpus scores (cheap, always run); test 4 is
  reserved for panel-contested channels.

## Pilot
Set: the 13 BBC word-probe channels (8 self-tagged MIXED — deliberately hard).
Success criterion (declared): the battery separates the entity/era-cohort channels
(predicted: transport-unstable) from the house-style channels (predicted:
transport-stable, nuisance-loading) with panel agreement >= 2/3 on non-MIXED calls.
