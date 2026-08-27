# Undermerge repair program — fix L0, relabel, propagate; arbiter-grounded recall ladder for R1/R2/R3

*2026-07-06 late. User directive: TOP PRIORITY. (a) fix undermerging at the leaf layer, relabel,
propagate upwards; (b) replicate the 6/24 experiment (GEPA, Sonnet/Opus arbiters) to get per-level
recall using each level's own definition. Resources authorized: many Sonnet subagents.*

## 0. Method decision (user delegated: pairwise vs grouping vs spectral)

**Grouping as the merge ENGINE; pairwise as the VERIFICATION and MEASUREMENT layers. No spectral.**

- Engine = GLM-4.7 group-tuned prompt (the 6/24 GEPA lineage: batch recall .544→.982; partition
  .76–.81 recall / .88 precision vs both arbiters) over ~30-statement neighborhood batches,
  min_votes=1 reconcile.
- Drift containment (the pairwise worry): group proposals are never applied transitively —
  every proposed CROSS-cluster merge is verified at PAIR level (3-vote) before application;
  trusted score-0 edges block; chain-proof star application; size caps + audit samples.
- Measurement rigor (the grouping worry): partition→pair scoring against a FROZEN,
  arbiter-labeled eval set (opus_rescore.py convention) — the partition method never grades
  itself.
- Spectral: rejected — new threshold/geometry assumptions, and embeddings can't appear in
  reported numbers anyway (S⊥R).

## 1. Level definitions (from the build prompts themselves — user veto welcome)

| level | relation | operational definition (verbatim lineage) |
|---|---|---|
| L0c (leaf cluster) | SAME CRITERION | v6 SYSTEM: same verdict on essentially ALL realistic work; wording irrelevant (0/1/2, same=2) |
| R1/R2/R3 `merged_groups` | SAME CONSTRUCT | build prompt: "a human evaluator would score them identically on the same piece of work" — same relation as L0c, applied at coarser representatives |
| parents/grandparents | SUBSUMPTION | "related but measure DISTINCT constructs; parent captures their shared axis" — NOT equivalence |

Consequences: (i) ONE arbiter-labeled same-construct pair set powers recall at every level —
recall@level ℓ = P(arbiter-SAME pair co-noded at ℓ), quoted with per-level coverage;
(ii) parent/grandparent nodes get a CONTAINMENT certificate (belongs-under + planted anchors,
already staged as `containment_payload_<task>.jsonl`), never a sameness recall.

## 2. Arbiter framework (replicating 6/24, family-clean)

- Engine family: GLM (z.ai). Arbiters: **Sonnet subagent panel** (volume) + **GLM-5.2**
  (conservative second family-line) → disagreements adjudicated by **Opus subagents**
  (hard unchecked judgment calls → Opus, per tiering rule). Truth = between the arbiters,
  disagreement rate reported as the boundary band (6/24: Opus 27% SAME vs GLM-5.2 19%,
  agreement .836).
- Blinded anchors in every arbiter batch (trusted unanimous positives + score-0 negatives).
- Also run the 6/24 leftover: Opus adjudication of the 484 disputed peer-review pairs
  (`opus_adjudicate_disagreement.py` brief) — pins the boundary on the pilot task.

## 3. Phases

**P0 — frozen eval sets (tonight, CPU).** Per task (humor+CW pilot; peer reuses 6/24 spectrum
assets): stratified pair sample ≈1.5K — TF-IDF cosine-spectrum bins (frozen, embedding-free
pre-filter) + shared-rare-name stratum + random + within-cluster (precision arm) + cross-cluster
T3 sample. Stable-hash seeded; FROZEN before any repair runs (same eval grades every round).

**P1 — arbiter labeling at scale (tonight, MANY Sonnet subagents).** ~100 pairs/agent × ~30
agents/wave-set, v6 0/1/2 rubric verbatim; GLM-5.2 pass on identical pairs (queued behind the
running CW fresh-judge — GLM stays serial); Sonnet×GLM-5.2 disagreements → Opus panel.
Output: `arbiter_labels_<task>.jsonl` = the ladder's ground truth.

**P2 — repair rounds (loop-until-dry).** Candidates per round: frozen-embedding kNN (pre-filter
only) ∪ lexeme-overlap net (from bulk extraction, queued on GLM tonight) ∪ fresh-confirmed
cross-cluster edges (running) ∪ group-engine discoveries. Group engine proposes → pairwise
3-vote verifies → chain-proof apply (score-0 blockers, size caps). After each round: score vs
P1 frozen eval. STOP at recall plateau or precision < .95. Target: **recall ≥ .80 at precision
≥ .95 vs arbiters** (beats 6/24's .78/.88 because of the verification layer).

**P3 — relabel + propagate upwards.** Rebuild (not repair) the tree on the fixed L0c partition —
old R1's .05–.08 recall (batch-local visibility) makes rebuild cheaper than repair: group engine
over cluster representatives with the SAME same-construct relation → R1; over R1 reps → R2; over
R2 reps → R3; parent axes proposed per build-prompt PARENT rule. Every round's batches get
overlap + reshuffle passes (kills the batch-locality failure). Versioned new artifacts
(append-only; old tree untouched).

**P4 — the recall ladder deliverable.** For OLD tree (baseline) and NEW tree: recall@L0c/R1/R2/R3
vs Sonnet-arbiter and vs GLM-5.2 (both quoted, truth-in-between), precision, coverage per level,
containment coherence for parent nodes; per task. GEPA re-tune of the grouping prompt only if
per-batch recall < .9 on the P1 dev slice (few rounds, GLM quota discipline).

## 4. Resource map

- Sonnet subagents (Max): P1 arbiter panel + P4 rescoring audits + spot-audits of GLM verdicts.
- Opus subagents: disagreement adjudication only.
- GLM-4.7 (subscription, serial queue): CW fresh-judge (running) → bulk extraction humor+CW
  (lexeme net + census substrate) → group-engine rounds → GLM-5.2 arbiter pass.
- CPU: sampling, reconcile, scoring, certificates. GPU: none (sk3 day_runner untouched).

Definitions frozen in this note before P1 labels are read. All ladder numbers quote: ruler
(which arbiter), denominator (in-net vs beyond-net), coverage.

---

## R0 — arbiter-panel harvest (2026-07-06 night; outputs/lexicon/round0_arbiter_baseline.json)

Panel: 26 Sonnet agents, 2,465/2,465 pairs, anchors pos 21/24 + neg 24/24.

| task | partition | recall vs Sonnet | precision vs Sonnet |
|---|---|---|---|
| humor | tau-base | .670 | .826 |
| humor | repaired (r0) | **.734** | **.771** |
| CW | tau-base | .774 | .755 |
| CW | repaired (r0) | **.821** | **.708** |

**Finding 1 — the problem is now PRECISION as much as recall.** Sonnet disputes 25–30% of
within-cluster pairs (SAME-rate .63–.70 in the within_cluster stratum) — vs .999 precision
under v6-trusted labels. The trusted repairs bought recall (+.05–.06) at precision cost
(−.05). Mirrors the 6/24 arbiter-calibration gap (boundary genuinely contested at high-sim);
Sonnet sits conservative of the v6 lineage. ⇒ the .95-precision target must be defined vs
ADJUDICATED truth (Sonnet ∧ GLM-5.2 agree; disagreements → Opus), not vs any single arbiter.

**Finding 2 — beyond-net mass quantified: 36% (humor 186/518) and 29% (CW 138/470) of
arbiter-SAME pairs were NEVER labeled by the v6 kNN candidate net.** The old in-net recall
numbers lived in a universe missing roughly a third of true sames in these strata (TF-IDF +
shared-name nets surface pairs the embedding net didn't). Candidate generation in P2 must keep
multiple nets.

Sonnet spectrum SAME-rate .48–.50 is NOT comparable to 6/24's 19–27% (our spectrum stratum is
kNN-top10-derived, high-sim-heavy by construction).

Next: GLM-5.2 arbiter pass on the identical 2,465 pairs (queued behind CW fresh-judge, at
2000/2142) → Sonnet×GLM-5.2 disagreements → Opus adjudication → adjudicated-truth targets for
the repair rounds.

## P2 engine revision (user directive, 2026-07-06 night): CE-calibrated escalation ladder

User: scaling merging was always the hard part; CHECK any scorer THOROUGHLY against adjudication
labels; escalate uncertain pairs to bigger models. And: the existing CEs (ModernBERT-base
per task + LoRA-bge, trained 5/18 on v6 labels) inherit the v6 boundary → likely need retraining
on the new Sonnet/GLM-5.2 gold.

Revised engine:
1. **Old CE = router-candidate only**, pending calibration on the frozen eval vs
   Sonnet/GLM-5.2/adjudicated labels (reliability + AUC + band selection). Fails ⇒ round 1 runs
   CE-less (nets ranked by TF-IDF/lexeme overlap; GLM tier as first filter). CE never truth,
   never in reported numbers (5/19 lesson: CE over-reaches on shared-vocab/different-concept).
2. **Escalation ladder**: net/CE score (free) → GLM-4.7 3-vote (cheap) → Sonnet panel (split
   votes, router-judge disagreements) → Opus (Sonnet×GLM conflicts). Per-tier confusion matrix
   vs adjudicated truth measured on the frozen eval BEFORE the tier is trusted at scale; blinded
   anchors at every tier.
3. **CE-v2 retraining on ladder-generated gold** after round 1 (~10-20K labeled candidate pairs
   expected, disjoint from eval by construction): v6-soft-label pretrain + gold fine-tune, per
   task; recalibrate bands every round. EVAL HYGIENE: the frozen 2,465 eval pairs are NEVER
   trained on.
4. Group-prompt kept as one discovery net (cheap batch calls) feeding the same ladder — no
   longer the engine.
