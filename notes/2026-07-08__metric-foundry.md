# Metric Foundry: proposal reasoning + articulation quality as the binding constraints (2026-07-08)

## Why this exists (the diagnosis chain, evening of 07-08)

1. **CW stage-2 came back BLANKET NULL** (12/12 candidates rep_bits ≈ 0, p=.3-.9 on strictly
   fresh within-genre samples) — including the convergent, coherent families (prompt-integration
   ×2, earnestness-over-irony ×2).
2. Before accepting "all winner's curse": **a protocol confound was found in the stage-2
   script** — stage-1 measures every candidate scored SOLO (loop._score_one → one rubric per
   prompt); my stage-2 scored candidates inside a 45-rubric PANEL with the bank. Panel context
   plausibly attenuates subtle rubrics (attention dilution). Confound test queued
   (solo_vs_panel_test.py: same fresh items, same candidate, both contexts → score-vector rho
   + gain comparison).
3. User direction (2026-07-08): the shortcoming is likely (a) not enough reasoning/example-
   selection at PROPOSAL time, and (b) not enough attention to ARTICULATION/implementation so
   induced metrics can actually be scored well. Pull from all streams: GEPA, articulation/
   decompression-rungs, metric-seam/coding.

## The foundry pipeline (design)

PROPOSE → ARTICULATE → OPERATIONALIZE → EXECUTE → GATE → REPLICATE

- **PROPOSE** (round 2): strong reasoner; example-SELECTION step (proposer sees 20 contrast
  candidates, picks the 6 pairs it finds most instructive before proposing — self-curated
  batteries); extended reasoning budget; GEPA-style reflective mutation of the proposer prompt
  per community (feed back what died and WHY per the annotated drops).
- **ARTICULATE** (round 1, BUILT): decompression-rungs applied to induced candidates — expand
  one-sentence rubrics to rung-3 dense rubrics (DEFINITION / anchored GUIDANCE for 1.0-0.5-0.0 /
  POSITIVE+NEGATIVE SKETCH / BOUNDARY NOTE; ≤250 words; label-free).
  `articulate_candidates.py` → dense_{cw,humor,math}.json.
- **OPERATIONALIZE** (round 2): GEPA on the rubric TEXT against label-free objectives
  (test-retest reliability + applicability spread) — never against labels (reconstruction-only
  discipline; the gate owns label contact). Metric-seam: compile checkable sub-clauses into
  code/judge_extract steps (deep_metrics DSL) where the seam battery says the channel is typed.
- **EXECUTE**: SOLO scoring always (panel only for the fixed bank); protocol must match between
  stage-1 and stage-2 (measurement invariance).
- **GATE/REPLICATE**: unchanged (stage-1 conservative; stage-2 Bonferroni over replicated set).

## Round 1 (queued as foundry_round1.sh, end of GPU chain)
(a) solo-vs-panel confound test (hell-deal + marriage legs);
(b) articulation ladder over ALL stage-1 candidates (3 domains, ~30 GLM calls);
(c) stage-2 REDONE: solo protocol × dense rubrics, all 3 domains
    (outputs/ctree/stage2/*-solo-dense).
Readout matrix: {panel-sparse (done, null for CW), solo-sparse (from confound test), solo-dense}
→ separates protocol artifact / articulation ceiling / genuine winner's curse per candidate.

## Round 1 results (partial, 2026-07-08 19:58)

**Solo-vs-panel confound test (hell-deal + marriage candidates, strictly fresh):**
- rho(solo, panel) = **0.15–0.70** — far below the ~0.9 same-context retest: the panel and solo
  contexts ARE different instruments (measurement non-invariance CONFIRMED). Worst:
  absence_of_melodramatic_inner_monologue rho 0.15.
- BUT solo does not rescue the candidates: solo strictly-fresh gains −0.0008…+0.0046, all ns
  (best: power-dynamics-subversion +0.0046 @ p=0.19). Punchline Omission INVERTS (panel
  +0.0039/p=.009 → solo −0.0006) — its panel survival was itself context luck.
- Reading: stage-1 tails were predominantly selection noise measured through a context-fragile
  instrument. Both failure modes real; neither candidate family survives either context.

**Ops:** articulation crashed mid-run on z.ai 529-overloaded (7/13 CW candidates articulated;
no retry, end-of-run save) → dense_math/cw incomplete → solo-dense legs crashed on missing
files (supervisor's rc=0 masked it — rc taken after the echo). FIXED: retry-with-backoff ×4 +
incremental save + resume in articulate_candidates.py; foundry_round1b.sh queued (articulation
redo + all three solo-dense legs).

## ★★★ Round 1b results (2026-07-08 21:51): FIRST FORMAL STAGE-2 KEEP — articulation was the bottleneck

Solo × DENSE × strictly-fresh (the corrected instrument), all three domains:

- **KEPT: `provides_broad_contextualization_or_counterexample` (math/topology): +0.0200 bits,
  p_auc=1.14e-06, p_bits=2.29e-05** — clears Bonferroni/8 by ~3 orders of magnitude. The SAME
  candidate on the SAME fresh sample measured **0.000 under panel-sparse**. The measurement
  stack (solo protocol + rung-3 dense rubric) unlocked it.
- Near-keeps resurrected from panel-sparse zeros:
  math Open-Ended-Theoretical-Inquiry +0.0095 (p=.0011/.0095 — missed the bar on p_bits only);
  humor power-dynamics-subversion +0.0073 (p=.044/.026), narrative-misdirection +0.0070
  (p=.029/.030); CW Thematic-Narrative-Anchoring +0.0052 (p=.053), escalating_scenario_logic
  +0.0045 (p=.039), dramatized_interactive_comedy +0.0044 (p=.024), Prompt-Integration +0.0027
  (p=.033).
- Full matrix (strictly-fresh): panel-sparse ≈ all zero; solo-sparse small/ns; **solo-dense =
  1 formal keep + 8 near-keeps p<0.1 across all 3 domains.** Articulation density is worth
  roughly the entire effect (0 → +0.005-0.020 bits); the panel context was destroying it.

**Sample-3 confirmation round QUEUED** (foundry_confirm_round.sh: third disjoint sample per
community, solo × dense, --only-from the solo-dense near-keeps, Bonferroni over the
confirmation set) — near-keeps that hold at ~+0.005-0.010 bits on sample 3 become formal keeps.

## Sample-3 confirmation round (2026-07-08 22:34)

- **math/topology keep: pool EXHAUSTED** (3,530 of 3,597 groups consumed by stage-1 + rep2) —
  no third sample possible; the formal keep stands on its sample-2 evidence (p=1.1e-06,
  3 orders under the bar). Leg-isolation gap found (topology's crash killed the probability
  confirm; rerun launched for probability solo).
- **Two-fresh-sample replicated-direction candidates** (independent samples, descriptive
  Fisher-combined p):
  * dramatized_interactive_comedy (CW hell-deal): +0.0044 (p=.024) & +0.0055 (p=.011, n=994)
    → Fisher p≈0.004
  * power-dynamics-subversion (humor marriage): +0.0073 (p=.044) & +0.0049 (p=.038, n=2400)
    → Fisher p≈0.012
- Killed by sample-3: escalating_scenario_logic (sign flip). Weakened to ns: Prompt-to-Story-
  Integration (+0.0031), narrative-misdirection (+0.0027).
- Pool accounting is now binding: hell-deal/wakeup-mystery/topology are consumed; further
  rounds need bigger communities (marriage/bar-jokes/family/doctor have room; math tags
  real-analysis/calculus/linear-algebra have room).

## FINAL PROGRAM TALLY (2026-07-08 23:00, all replication rounds complete)

| verdict | metric | community | evidence |
|---|---|---|---|
| **FORMAL KEEP** | provides_broad_contextualization_or_counterexample | math/topology | sample-2 strictly-fresh +0.0200 bits, p_auc=1.1e-06 (pool exhausted for sample-3) |
| replicated-direction | Open-Ended Theoretical Inquiry | math/probability | +0.0095 (p=.0011) & +0.0072 (ns @ n=1184); Fisher p≈.003 |
| replicated-direction | dramatized_interactive_comedy | CW/hell-deal | +0.0044 (p=.024) & +0.0055 (p=.011 @ n=994); Fisher p≈.004 |
| replicated-direction | Subversion of established power dynamics | humor/marriage | +0.0073 (p=.044) & +0.0049 (p=.038); Fisher p≈.012 |
| killed | escalating_scenario_logic (sign flip), Prompt-Integration, misdirection, + all panel-era candidates | | |

All four survivors are COMMUNITY-LOCAL content/practice norms in three different domains, none
expressible by reweighting the general banks, all invisible under the panel-sparse instrument,
all measured through: GLM proposer → within-community contrast → rung-3 dense articulation →
solo scoring → multi-sample disjoint replication. The methodological arc (user's three
hypotheses: subtask conditioning → proposer strength → articulation/implementation quality)
was confirmed at each step by instrument changes that recovered signal.

## Arm attribution (user question: is metric_tree better BECAUSE it conditions on subsets?)

Across all GLM-proposer community legs: residual 159 proposals → 16 stage-1 tails (10.1%) →
**1 replication survivor**; metric_tree 154 proposals → 10 tails (6.5%) → **3 survivors**.
metric_tree's tails replicate at ~5× residual's rate (3/10 vs 1/16). Reading: residual's
whole-community contrast surfaces more *apparent* signal (more winner's curse); metric_tree's
partition-conditioning (contrasts within nodes INSIDE the community = double conditioning)
yields fewer but far more robust candidates. Confirms the user's conjecture; wave-3 keeps both
arms to grow this comparison (n is small: 4 survivors total).

## WAVE-3 LAUNCHED (2026-07-08 ~23:30): 18 new communities, full validated recipe

6 CW genres (aliens .69 base, villain, soulmate, ai, time-travel, meta-experimental .33) +
6 humor topics (political-classroom, police, chicken-crossing, everyday-observational,
absurd-wordplay, topical-corona; offense-centered clusters deferred) + 6 remaining math tags
(calculus, abstract-algebra, algebra-precalculus, sequences-and-series, complex-analysis,
integration). Chain per wave3_communities.sh: 18 stage-1 legs (GLM, residual+metric_tree,
dedup+annotated-feedback ON) → articulation (resume-capable) → stage-2 solo×dense rep2 →
sample-3 confirm for p<.05. ~overnight on GPU 5; ~500 GLM calls.

## OPERATIONALIZE stage BUILT (2026-07-09, user directive: GEPA/MI-recovery for every metric)

Answer to "is GEPA/MI-recovery built into metric_tree?": MI-recovery WAS already in the shared
loop (reconstruction_accuracy — blind re-derivation + held-out agreement, stamped on every
proposal, all arms) but DIAGNOSTIC-ONLY; nothing iterated the rubric. Now built:
- `methods/metrics_tree_infilling/operationalize.py`: score calibration slice → diagnose
  (retest, MI-recovery agreement + what the blind reader THOUGHT the metric was, distribution
  collapse) → GLM rewrite fed the diagnostics → keep best variant by (retest+recovery)/2.
  Max 2 rewrites; good rubrics pass through untouched. ALL objectives label-free.
- Wired into global_infill (cfg.operationalize_proposals; ledger op_iterations/op_retest/
  op_recovery; run_arm_comparison --operationalize) — EVERY proposal, EVERY arm, for all
  future legs. 73/73 tests (3 new).
- Wave-3 launched pre-flag, so its candidates get the GEPA RE-PASS post-hoc
  (`wave3_gepa_repass.sh`, queued behind wave-3): optimize all w3 stage-2 hot candidates'
  dense rubrics → rep4 confirm (disjoint from rep2+rep3) with optimized rubrics.
Cost note: in-loop operationalization ≈ +600-800 judge calls & 2-3 GLM calls per proposal
(≈ +35 min/leg, ~60 GLM calls/leg) — the price of instrument-grade metrics.

## WAVE-3 STAGE-1 HARVEST (2026-07-09 06:18, all 18 legs rc=0, articulation done)

406 GLM proposals over 18 new communities (dedup + drop-annotated feedback ON, first wave
with both). Hot tails (stage-2-eligible: kept OR confirm p_auc<.05 with positive bits):

| arm | proposals | hot tails | rate |
|---|---|---|---|
| metric_tree | 198 | 15 | 7.6% |
| residual | 208 | 4 | 1.9% |

- **Arm attribution sharpens:** metric_tree now out-proposes residual 4:1 at the tail stage
  (wave-2 was the reverse: residual 10.1% vs metric_tree 6.5% tail rate, but metric_tree
  replicated 5x better). Residual's tail rate COLLAPSED 10.1%->1.9% under the annotated-drop
  feedback — consistent with its wave-2 tails having been re-proposal churn that the feedback
  now suppresses. Combined tally for "does partition-conditioning find more robust metrics":
  metric_tree 25 tails/352 proposals vs residual 20/367 across both waves, with all wave-2
  replication survivors' arm-ratio 3:1.
- Tails span 12/18 communities; humor-topic-police is hottest (5 tails).
- Top tails by stage-1 p: Mathematical Proof Refutation (abstract-algebra, +0.0085, p=.0031,
  residual); punchline_payoff_completeness (police, +0.0156, p=.0038); "Resolves the specific
  mathematical error or contradiction" (integration, +0.0170, p=.0082); meta_premise_integration
  (meta-experimental, p=.0124); alien_perspective_irony (aliens, p=.0205). Content again =
  community-local practice/content norms, not surface.
- Gate health: 298/406 dropped at auc<.005, 30 at bits<.003, 15 surface-guard, 5 dup, 3
  viability, 1 redundant — the funnel is doing its job.
- **Chain gap found + fixed:** wave3_communities.sh gave math no confirm3 and the GEPA re-pass
  covered CW/humor only. Queued `wave3_math_tail.sh` (sk3 PID 321589) strictly behind
  "GEPA REPASS COMPLETE": per math tag with stage-2 hot candidates -> rep3 confirm -> GEPA
  optimize -> rep4; skips tags with 0 hot candidates before engine boot.
- Caveat for the final tally: w3-math-old re-scores the SAME rep2 sample as foundry_round1b's
  math solo-dense leg (no new salt) — it is a redo in w3 ledger format, NOT fresh evidence.

## ★★★ WAVE-3 STAGE-2 + CONFIRM3 (2026-07-09 08:28): TWO NEW FORMAL KEEPS

Strictly-fresh rep2 (solo x dense; Bonferroni over each output dir's candidate set). Old-leg
rows in w3-cw/w3-humor are CACHE-IDENTICAL redos of foundry round-1b (same default rep2 salt)
— not new evidence; only NEW communities counted below.

- **FORMAL KEEP #2: `meta_prompt_subverts_expected_tone_or_medium` (CW/meta-experimental):
  rep2 +0.0128 bits, p_auc=.0007, p_bits=.0001, n=2400** (bar .05/17=.0029). Stage-1 p=.028.
- **FORMAL KEEP #3: `Multilayered_Resolution_in_the_Pivot` (humor/everyday-observational):
  rep3 +0.0122 bits, p_auc<1e-4, p_bits=.0013, n=2400** (bar .05/7=.0071) after rep2 +0.0097
  (p=.0125/.0156) — THREE independent positive samples incl. stage-1; formally clears on the
  third. Strongest humor result of the program.
- Near-keeps / replicated-direction (new communities): meta_premise_integration (CW/meta-exp,
  rep2 +.0086, p=.0026/.0049 — missed p_bits bar only); punchline_cognitive_completeness
  (absurd-wordplay: rep2 p=.0049, rep3 p=.0035/.0433 — 3 positive samples);
  punchline_payoff_completeness (police: st1 .0038 / rep2 .0357 / rep3 .0163);
  logical_foundation_for_absurdity (police: .0174/.0133/.0725). Weakened: Covert Semantic
  Pivot (corona rep3 .098). Single-sample nominal: alien_perspective_irony (aliens, p=.031,
  pool now EXHAUSTED — no confirm possible).
- Math: **Question Answered by Alternative Method (sequences-and-series) KEPT at rep2**
  (+0.0140, p_auc=.0264, p_bits=.0069, n=1519) — but per-tag dirs make the Bonferroni bar
  m=1 (.05), much looser than CW/humor pooled dirs; and its pool is EXHAUSTED (rep2 already
  short at 1519) so no rep3 exists. Report as kept-under-per-tag-bar, single fresh sample.
  Mathematical Proof Refutation (abstract-algebra) went DEGENERATE at rep2 (dense rubric
  collapsed on fresh sample: appl<30% or std<.05) — the exact failure operationalize targets;
  GEPA rescue + rep4 queued (~1.5k rows of pool room). complex-analysis/integration tails ns.
- **Verdict inversion vs wave-2 arms note:** both new formal keeps are metric_tree proposals;
  running survivor tally metric_tree 5+ vs residual 1.

### Ops corrections this round
- CW confirm3 CRASHED on aliens' empty pool (n=0 -> sklearn ValueError) and took the
  remaining CW legs with it (meta-experimental rep3 lost). PATCHED replicate_candidates.py:
  pool-exhaustion guard (n<50 or single-class -> status pool-exhausted, leg skipped) +
  per-candidate try/except. Also affects the queued GEPA rep4 CW leg (aliens in glob) —
  patch synced BEFORE it runs.
- **rc=$? echo bug (2nd occurrence):** `echo "[$(date)] rc=$?"` always prints rc=0 — $(date)
  resets $? before expansion. All wave3 supervisor rc lines are meaningless; ground truth =
  ledgers. Rule: capture rc=$? on its own line before any date-echo.
- gepa_optimize_rubrics + replicate_candidates now handle degenerate candidates
  (--include-degenerate carries them into a GEPA-rubric rep4).
- wave3_math_tail.sh (sk3, chained behind GEPA REPASS COMPLETE) now: math rep3+GEPA+rep4
  (skipping exhausted sequences-and-series) + CW meta-experimental rep3 (~490 fresh groups,
  out w3-cw-meta-confirm3).

## GEPA RE-PASS RESULTS (2026-07-09 09:12, verified by artifacts not rc)

**Operationalize diagnostics (label-free; 13 hot candidates):** 7 passed through untouched
(iters=0, retest .84-1.00, recovery .65-.87 — punchline_payoff .86 / logical_foundation .87
are the most recoverable rubrics measured yet); 6 rewritten (1-2 iters), all reaching retest
>= .91 (alien_perspective .97, power-dynamics 1.00). GEPA touches weak instruments, leaves
strong ones alone — exactly the designed behavior.

**Humor rep4 (FOURTH independent fresh sample per candidate; iters=0 candidates = same
instrument, clean 4th replication):**
- Multilayered_Resolution_in_the_Pivot: **+0.0102, p_auc=.0016, p_bits=.0139 — 4/4 positive
  fresh samples** (.0125 / <1e-4 / .0016 p_auc sequence), two formally clearing. Rock solid.
- punchline_payoff_completeness (police): +0.0093, p=.0042/.0101 — strongest of its 4 samples;
  4/4 positive.
- logical_foundation_for_absurdity (police): +0.0040 (.083/.032) — 4/4 positive direction.
- Marriage rewrites (new instrument at rep4): power-dynamics +0.0027 (p_auc=.031),
  misdirection +0.0024 (.038/.033) — direction holds through the rubric rewrite.
- Weakening confirmed: Covert Semantic Pivot ns again; punchline_cognitive mixed
  (p_auc=.0018, p_bits=.20, n=1206 pool thinning).

**CW rep4: pool-exhaustion guard worked** — 5 legs skipped as pool-exhausted (incl. both
meta-experimental candidates: rep2+rep3-salt exclusions consume the genre), no crash,
abstract-premise scored (Prompt-to-Story -0.0009 ns, stays dead). The queued CW-meta rep3
(math-tail supervisor) is the LAST fresh sample available for meta-experimental.

## Round 2 (planned, pending round-1 readout)
- Proposal-side: example-selection + reasoning budget + per-community GEPA of proposer prompts.
- Seam compilation of surviving candidates into typed programs (depth premium per candidate).
- New sibling sets: law_se, stackoverflow_python, code-review top repos, patents IPC classes,
  peer-review venues (F1000 vs ICLR) — after the protocol is trusted.
