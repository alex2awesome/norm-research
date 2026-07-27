# Hierarchy execution status

Frozen stop rule: a level is accepted when corrected LLM-truth recall **>.50** and uniform
predicted-positive LLM-audited precision **>.50**. Each task gets one fixed-width primary build and
at most one targeted correction. Every group over 30 receives a special whole-group LLM audit; size
alone never forces a split.

| Task | R1 truth | Primary R1 build | Next bounded action |
|---|---|---|---|
| Humor | Corrected 3-judge final: 45/900 SAME | 9k verify net: 25/30 shards complete; shards 010–019 showed confirmed judge drift | Finish tail, then fresh LLM re-audit of all 995 unique proposed edges from drifted shards before apply |
| Creative writing | Corrected 3-judge final: 53/900 SAME | Corrected 12k / 40-shard verify net ready (ceiling .792); first 20 assigned | Calibrated verifier pass; audit R/P |
| News | Corrected 3-judge final: 22/900 SAME | Corrected 6k verify net: 14/20 shards complete; live anchors pass | Complete calibrated verifier; audit R/P |
| Peer review | Corrected 3-judge final: 22/900 SAME | Corrected 6k / 20-shard verify net ready (ceiling .773) | Calibrated verifier; audit R/P |
| Grant | Corrected 3-judge final: 58/900 SAME | Corrected 9k / 30-shard verify net ready (ceiling .828) | Calibrated verifier; audit R/P |
| Legal outcome | Corrected 3-judge final: 24/900 SAME | Corrected 6k / 20-shard verify net ready (ceiling .833) | Calibrated verifier; audit R/P |
| Notice | Corrected 3-judge final: 30/900 SAME | Corrected 6k / 20-shard verify net ready (ceiling .767) | Calibrated verifier; audit R/P |
| Math | Existing strict truth retained | Conservative 2-LLM recovery rejected (R=.138); broader R=.507 candidate rejected by blind global precision audit: 18/300=.060, CI [.038,.093] | Prefer a frozen strong-LLM veto pass over 11,016 unique edges proposed by the completed 30k over-permissive verifier; 37 shards ready |
| Patents | Existing strict truth retained | Existing 30k verifier proposed 15,406 build edges and produced global precision .20 after certification; resolution sweeps cannot put both binary metrics >.50 | Frozen strong-LLM veto pass over 15,409 unique proposed pairs; 52 shards ready; preserve every original non-edge |
| Press releases | Existing strict truth retained | R1 verify payloads staged | Complete fixed-width build |
| Code review | Corrected 3-judge final promoted: 64/900 SAME (A=39, B=104, C adjudicated all 169 disagreements) | Corrected 40k / 100-shard verifier staged; routing ceiling .656, with a preplanned 10k tail extension if recall misses .50 | Start calibrated verifier after the current NEWS gate; audit R/P |

## 23:58 PDT execution checkpoint

- **HUMOR verifier repair completed.** A fresh LLM re-audited every unique pair called SAME in the
  drifted 010--019 tranche: 232/995 remained 2, 761 became 1, and 2 became 0. The corrected hard-edge
  graph has 525 SAME edges over 3,363 nodes and only .067 held-out-truth recall, demonstrating a
  structural sparsity failure rather than a Louvain-resolution failure.
- A fail-closed weighted-evidence option was added to `apply_pairwise`. Its default remains exactly
  the old hard graph (`related_weight=0`). The selected HUMOR correction uses score-1 LLM judgments
  only as weak weight .2 evidence at Louvain resolution .5; score-2 edges retain weight 1. It reaches
  corrected recall .578 (chance-corrected .553), 621 groups, and 18 groups over 30. The fixed-mixture
  precision .187 is diagnostic only, not a global estimate. Parameters were selected on a
  deterministic half of the frozen eval (dev SAME recall .65); the untouched hash half retains
  SAME recall .52, so the >.50 recovery is not solely an in-sample resolution sweep artifact.
- Those 18 oversized groups contain 2,694 nodes and **99.96% of all predicted-positive pair mass**.
  They are now staged for two independent full-group LLM repartitions. A group survives intact only
  if both judges certify it; two differing partitions are combined by their common refinement so
  neither requested split is erased. Every resulting >30 subgroup is recursively certified.
- HUMOR anchor exclusions used only for gate arithmetic: `29da862e9ec689b8` and
  `7c75bb654d3d6a7e`. Both remain build-edge blocked. The first is a genuinely contested 3--2 panel
  SAME decision; the second is a 3--2 RELATED decision in the targeted five-judge check. The one
  unambiguous positive and all three negative anchors score 1.0 under the surviving gate.
- **NEWS:** the original 6,100-row verifier is frozen. Its negative-anchor failure triggered a fresh
  blind re-audit of all 1,878 unique proposed score-2 edges: 631 remained SAME, 1,218 became RELATED,
  and 29 DIFFERENT. The corrected stream passes both anchor classes at 1.0. Its hard graph has recall
  .182; transferring the HUMOR-selected weight .2/resolution .5 rule without NEWS tuning raises
  corrected recall to .591 (chance-corrected .550). Twelve >30 groups contain 99.96% of positive
  pair mass and are queued for the same two-judge whole-group common-refinement certification.
- **CREATIVE WRITING:** corrected verifier shards 000--007 are sealed; shard 008 is undergoing a
  full semantic recheck after an unusually high provisional score-2 share.
- **CODE REVIEW:** corrected truth is promoted; verifier shard 000/099 is sealed. The judge will
  pause after 005 to provide the independent HUMOR oversized-group replicate, then resume at 006.
- Full suite after the weighted-evidence, replicated-certificate, and frozen-source-relocation
  changes: **218 passed**.

No R2/R3 build starts from an R1 that has not crossed the stop rule.

The semantic-recovery confirmer now receives every and only first-LLM score-2 pair.  The screen
still covers all candidates, confirmation is frozen and exact, and a merge still requires two
independent score-2 judgments.  This preserves the semantic gate while avoiding a second pass over
pairs that cannot be admitted under the rule.

## Corrected R1 truth panel checkpoint

All seven suspect banks now have two complete blind 900-pair passes plus third-LLM adjudication of
every A/B disagreement.

| task | final SAME / 900 | A/B exact ordinal agreement | A/B binary SAME agreement | binary kappa |
|---|---:|---:|---:|---:|
| Humor | 45 | .707 | .947 | .375 |
| Creative writing | 53 | .587 | .846 | .241 |
| News | 22 | .751 | .964 | .093 |
| Peer review | 22 | .722 | .982 | .521 |
| Grant | 58 | .660 | .937 | .362 |
| Legal outcome | 24 | .761 | .981 | .576 |
| Notice and comment | 30 | .734 | .976 | .488 |

Code review completed the same replicated audit after its live verifier called all three old
positive anchors non-SAME; one anchor (maximum return statements versus maximum positional
arguments) is plainly a related-but-distinct pair under the frozen R1 definition.  Fresh judges A/B
had exact ordinal agreement .812, binary SAME agreement .919, and binary kappa .455.  An independent
third GPT-5 judge adjudicated all 169 ordinal disagreements; the ordinal-median truth contains
64 SAME pairs and is now promoted, with the old 156-SAME truth archived.

The ambiguous-pair reconsideration path is fail-closed: the original screen and source manifest
are hashed, two fresh vote directories must cover the frozen selection exactly, and only pairs
scored 2 by both new LLM judges can add an edge to an already authorized base partition.  The full
codability test suite passes (198 tests at the latest full run).  A separate selective-re-audit
path freezes complete verifier shards and permits a proposed edge to survive only after a fresh
LLM score-2 judgment; this is being used to correct the HUMOR 010–019 calibration discontinuity.
