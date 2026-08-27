# Tacit-learning MECHANISM catalog — how tacit knowledge is ACQUIRED

Date: 2026-07-23. The acquisition-side counterpart to the operationalization catalog
(notes/2026-07-22__tacit-knowledge-operationalization-catalog.md). Purpose: complete menu of
acquisition/installation mechanisms from the literature, each mapped to an implementable LLM
training channel + predicted battery signature. The two catalogs multiply: final object =
mechanisms × operationalizations ("which acquisition routes produce knowledge that is tacit
in which senses").

Status: cogsci harvest COMPLETE (§M-A); apprenticeship/social + ML harvests pending.
Current channel menu graded against: {articulation, exemplars-in-context, soft-label
distillation, GEPA-articulation, ordinal reward, distal reward, curriculum}.

## M-A. Cognitive psychology / skill acquisition (complete; 16 mechanisms + 3 audits)

**Tier A (cheap — selection/prompt/schedule changes only):**
- **M1 near-miss contrastive pairs (Winston):** teacher generates a MINIMAL edit breaking
  exactly one construct mechanism; contrastive/ranking loss on (positive, near-miss). MISSING
  from menu. Predicts sharpest gains at the decision boundary first. Gate: the edit must
  change exactly one mechanism (sloppy generators = noise).
- **M2 paired/contrastive exemplars, judge-verified alignment (Gentner analogical encoding):**
  side-by-side alignable cases + "what structural difference separates these" — ~3× transfer
  vs separate study (Loewenstein/Gentner). Refines exemplars-in-context. Predicts CROSS-
  CONSTRUCT transfer more than in-distribution gains. Gate: pairs by judge-verified mechanism
  alignment, NEVER embedding similarity (the literature's flagged failure mode).
- **M3 dimension-isolating variation sets (Marton):** vary ONLY the target dimension, hold
  nuisance dims fixed; null-control = nuisance-varying sets must show nothing.
- **M4 train-time articulation, FADED at inference (Schooler deployment rule):** articulate
  during training, never force verbalize-then-decide at inference on holistic constructs.
  Discipline change; pairs with our sign-inversion/overshadowing story.
- **M5 interleaved (vs blocked) batches (Bjork; Shea & Morgan contextual interference):**
  benefit concentrates on CONFUSABLE construct pairs; can reverse on well-separated ones.

**Tier B (one extra pass / scheduler):**
- **M6 self-generated-rationale-then-repair distillation (Chi self-explanation):** executor
  drafts its rationale BEFORE seeing teacher's; train on (item, self-draft, teacher-repair,
  label). Distinct from articulation AND distillation. Risk: paraphrase-not-inference.
- **M7 competence-gated fading of articulation detail (Sweller/Kalyuga expertise reversal):**
  full worked rationale while weak → hints → bare exemplars → nothing, gated on per-construct
  accuracy. Predicts over-explaining a mastered construct HURTS (articulation value is
  time-varying).
- **M8 holistic metaphor articulation (Liao & Masters analogy instruction):** GEPA searches
  for ONE holistic metaphor instead of rule-lists; predicts dual-task-level pressure
  robustness at a fraction of compute.
- **M9 errorless/low-error-gated escalation (Masters/Maxwell):** escalate difficulty only
  while error stays low; suppresses explicit hypothesis-testing; predicts pressure-robustness.
- **M10 spacing schedules** (d≈.46 procedural, shrinks with task complexity — discount for
  gestalt constructs). **M11 intermittent/variable reward (PREE):** predicts forgetting-
  resistance after later unrelated fine-tuning (cheap to run, slow to verify).

**Tier C (real infrastructure):**
- **M12 decomposed subskills + late fusion, backward-chaining variant (Wightman & Lintern):**
  predicted to FAIL on gestalt constructs (whole-task usually wins; backward chaining the
  exception) — mechanistic account of our gestalt-undershoot.
- **M13 distributional/quantile reward critic (Dabney dopamine-distributional-RL):** for
  noisy subjective judges; predicts variance reduction on Taste constructs.
- **M14 reward-staleness ablation (COVIS delay, CORRECTED: 2.5–10s disrupts implicit
  learning vs 0.5s baseline; RB insensitive):** analog = credit-assignment staleness in
  async RL; predicts Taste constructs degrade with stale reward, verifiable ones don't.
- **M15 dual-task/secondary-load distillation (Masters reinvestment):** blocks explicit-rule
  accumulation → pressure-robust; REQUIRES a new battery axis (robustness-under-pressure:
  low-token-budget / no-CoT / distractor-loaded inference).
- **M16 unlabeled curated-contrast exposure (Gibson differentiation; Saffran statistical
  learning):** the ONLY fully label-free channel; moves the PRE-label prior, fast-saturating,
  test at LOW data scale. Menu has no unlabeled channel at all.

**Validated-as-is:** distillation = ACT-R compilation (with Heathcote correction: fit
per-construct EXPONENTIAL approach-to-asymptote, not pooled power law — pooling is the
averaging artifact); scalar reward = TD-error. **Explicitly excluded:** sleep/consolidation
(no analog; replay ≠ consolidation — do not add).

**Audit items (gates, not channels):** mere-exposure confound (swap-ablate reused exemplar
sets); teacher rationale-fidelity pre-check before trusting articulation on a construct
(does the target's own stated rationale predict its own scores?); generation-effect double-
dissociation check (self-generation channels may raise conceptual probes while LOWERING
data-driven/pressure probes — candidate mechanism for the N&C GEPA fidelity↑/AUC↓ result).

**Cross-cutting syntheses:**
1. **"Harder training" is NOT one axis:** desirable-difficulties optimizes transfer/retention;
   errorless optimizes pressure-robustness — different DVs, opposite prescriptions; an
   UNRECONCILED gap between literatures (no citation bridges them) → track both DVs.
2. Verbal overshadowing ≡ expertise reversal (same shape, different literatures) → fade
   schedules everywhere, never fixed articulation.
3. Contrastive channels (M1/M2/M3) are gated on judge-verified alignment quality.

## M-B. Apprenticeship / social transmission (complete; 33 mechanisms, 10 clusters)

**The headline revision:** the single best-evidenced gap is **M42 contingent scaffolding +
fading** (Wood & Middleton 1975 shift-rule; van de Pol 2010 decade-review; independently
corroborated) — support recalculated after EVERY learner response, driven to ZERO on a
competence trigger. Static curriculum is the OPEN-LOOP CONTROL this literature shows losing.
"If the team can only build one new channel, build this one."

**Missing channels (strongly evidenced):**
- **M42 contingent shift + fade + transfer-of-responsibility** (B4/B5): closed-loop hint/
  support controller keyed to live per-batch success, decay-to-zero on competence trigger,
  hint-free verification step. Caveat: needs a trustworthy per-turn success signal (hard for
  tacit targets — the trigger becomes a gameable proxy).
- **M43 coaching / on-policy correction of the learner's OWN attempt** (B2; converges with
  M-C's M19 DAgger/GKD + motor-learning guidance-hypothesis + surgical TESA data): multi-turn
  correct-THIS-attempt transcripts → SFT on corrected trajectory or DPO(original, corrected).
- **M44 interactional-expertise immersion dialogue** (F1, Collins): long varied multi-turn
  conversation WITH the target (not one-shot extraction). Caveat the literature itself
  supplies: installs fluent TALK, not necessarily judgment — evaluate on the
  talk-vs-judgment dissociation (ties to catalog op #30 Imitation Game).
- **M45 war-story/narrative transmission** (H3, Orr): first-person post-mortem anecdotes
  (situation → false lead → actual cause) as the transmitted object; targets long-tail
  exceptions that averaged labels destroy.
- **M46 ambient consequence-exposure** (A2, Lave & Wenger): the learner SEES downstream
  outcomes of its own past judgments as context — no scalar, no gradient, no correction.
  Distinct from distal reward (which optimizes). Real apprenticeship corrects far less
  verbally than assumed; consequences do the work.
- **M47 learner self-explanation-for-critique** (B6) + **M48 learner self-comparison/
  reflection** (B7): the SMALL model articulates/diagnoses; teacher critiques the
  justification's validity. NOTE the three-way "articulation" naming collision (static
  extraction ≠ GEPA prompt-rearticulation ≠ learner self-explanation) — only the third is
  missing; rename in the channel registry.
- **M49 self-consistency practice** (E1, Sennett tool-resistance): train against ITSELF on
  perturbed variants of the same item, zero teacher calls; consistency-not-truth caveat.
- **M50 ZPD live co-solving** (C1): teacher continues the rollout from the learner's exact
  failure point; joint trajectory = training data.
- **M51 uncurated high-volume streaming imitation** (E2, Polanyi co-presence) · **M52
  community-panel norm formation** (H2; panel debate transcripts, not just consensus labels)
  · **M53 collective enculturation** (F2: train on the COMMUNITY's discourse, not one
  teacher) · **M54 persistent-mentor conditioning** (H4 serial socialization) · **M55 job
  rotation** (H5: cyclic cross-domain alternation on ONE construct — directly relevant to
  GTK/transfer) · **M56 prestige/track-record-weighted multi-teacher trust** (I3) · **M57
  ostensive generic-vs-episodic marking** (I4) · **M58 ratchet lock-in** (I5: freeze gains
  before further exploration) · **M59 guided participation via embedded multi-agent logs**
  (D1: a data SOURCE, not a loss).

**Framing-level findings (not channels):**
- **M60 THE SEQUENCING LAW (F4, MacKenzie & Spinardi cookbook-insufficiency):** in every
  documented full-blueprint transfer (Fuchs→USSR, UK, Pakistan, Iraq), a complete correct
  explicit spec was NEVER sufficient — years of own-attempt practice always followed.
  Distillation is a required FIRST STAGE, not one alternative among seven: the menu is an
  ORDER (distill → corrected practice → fade), not a buffet.
- **M61 distributed cognition (H7, Hutchins):** challenges the premise that the target is a
  self-sufficient model — competence can live in the pipeline (small model as one node +
  verifier + checklist). The in-pipeline-vs-alone gap is itself a measurement. Discussion-
  section material for the capstone.
- **M62 somatic substrate ceiling (F5):** some teacher-specific components (architecture/
  RLHF-history-bound) may be rebuildable-not-transferable — budget a non-closing residual.
- Diagnostics: Bandura four-gate lens (attend/retain/express/use — "can recite but doesn't
  spontaneously use" failure); strategic-concealment fidelity gap on hedge-prone topics (F6);
  psychosocial mentorship flagged as probable category error (H6).

## SYNTHESIS — the three harvests converge (74 raw items → ~25 distinct mechanisms)

**Four mechanisms are independently named by ALL THREE literatures** (strongest possible
prior for our setting):
1. **Closed-loop support-calibration + fade-to-zero** (van de Pol contingency ≡ Kalyuga
   expertise-reversal fading ≡ errorless gating ≡ scaffold-and-fade) — M42/M7/M9.
2. **On-policy correction of the learner's own attempts** (CBN coaching ≡ guidance
   hypothesis ≡ COVIS own-response feedback ≡ DAgger/GKD) — M43/M19; directly indicated by
   the M17 data-side verdict (fidelity dies exactly off-support, where the student's own
   rollouts live).
3. **Structured contrast/variation** (Gentner aligned pairs ≡ Marton variation ≡ Winston
   near-miss ≡ interleaved confusables ≡ contrastive losses) — M1/M2/M3/M5/M23.
4. **Explicit-first-then-practice sequencing** (ACT-R compilation ≡ worked-examples-then-
   fading ≡ cookbook-insufficiency) — M60: the pipeline is distill → corrected practice →
   fade, in that order.

**v2 design implication (the apprenticeship pipeline):** v1b (distill, N=512, checkpoints)
→ one M43/M19 on-policy correction round → M42 contingent-fade of any in-context support →
hint-free verification. Exploratory riders: M20 KTO, M46 consequence-exposure, M45
war-stories arm. The battery profiles every stage (mechanisms × operationalizations).

**New battery axes the mechanism review demands:** robustness-under-pressure (M15);
talk-vs-judgment dissociation (M44/Imitation-Game); in-pipeline-vs-alone gap (M61);
spontaneous-use vs on-demand (Bandura gate 4).

## THE TACIT-SPECIFICITY FILTER (user challenge, 2026-07-23: "these sound like general
learning techniques, not anything focused on tacit knowledge itself")

Correction accepted: the four convergent mechanisms are GENERAL PEDAGOGY (they'd optimize
calculus or tennis; several are saturated with explicit content). A mechanism is
tacit-SPECIFIC iff: (a) nothing in the loop ever DESCRIBES the content; (b) explicit
orientation actively HURTS acquisition; (c) the channel carries what the TEACHER cannot
state; (d) carriers have no semantic relation to the content.

**Survivors (the tacit-specific core, 7):**
1. Incidental statistical exposure (M16 + Reber 1976: rule-search instruction IMPAIRS complex-
   grammar learning — criterion b, the diagnostic);
2. Suppression training (M15 dual-task / M9 errorless — anti-explicit by design);
3. Osmosis/uncurated co-presence (M51/E2 — Polanyi: transmits rules "not explicitly known to
   the master HIMSELF"; any curation step re-routes content through description);
4. Ambient consequence exposure (M46 — criterion a);
5. **Distal-outcome selection = OUR §5.1** (metric never named, installed from its footprint —
   the user's original behaviorist hypothesis IS one of the few tacit-specific mechanisms;
   the harvest under-ranked it among general techniques);
6. Subliminal learning (M40 — semantics-free carriers, criterion d; same-base-model constraint
   ≈ Collins' somatic/substrate limit);
7. Metaphor/motto carriers (M8 — explicit token, non-decomposed content; = the boundary-object
   thread from note §3b).

**Shared skeleton: the learning signal is never a REPRESENTATION of the content — selection,
not instruction** (frequencies, consequences, suppression, co-occurrence, inert carriers).
General mechanisms transmit via descriptions; tacit-specific ones via effects.

**ROUTE-SIGNATURE HYPOTHESIS (new headline candidate):** knowledge acquired through
tacit-specific routes carries a more tacit PROFILE (unstatable, token-free, pressure-robust,
exclusion-resistant) than same-accuracy knowledge acquired through explicit routes. Null:
profile is fixed by content; route washes out. The battery measures the profile; the channel
menu now splits explicit-route vs tacit-route arms; the mechanisms×operationalizations grid's
central cell. No prior literature could run this test (none had channels + battery in one
apparatus). Honest caveat: the tacit-specific core is thin partly because the literature
could not instruct-what-cannot-be-instructed — lab isolation exists only for implicit-
learning and motor domains.
## M-C. ML policy-transfer mechanisms (complete; 25 items, citations VERIFIED per-item)

**P0 — run before choosing any channel:**
- **M17 fidelity-vs-accuracy diagnostic (Stanton et al. 2106.05945):** does the distilled
  student reproduce the teacher's EXACT probabilities on near-training held-out items? Low
  fidelity even near-support → optimization/capacity limit (no data-side fix helps); high
  fidelity + low criterion-agreement → student faithfully learned the teacher's idiosyncratic
  errors. Determines whether P1's +.093 needs data channels or representation channels. FREE.
- **M18 temperature sweep on soft labels (Hinton):** (c)'s own knob, unswept; caveat: binary
  YES/NO has little dark knowledge by construction.

**P1 — cheapest high-expected-value channels (all confirmed MISSING):**
- **M19 on-policy distillation / DAgger (Ross 1011.0686; GKD 2306.13649 retitled "On-Policy
  Distillation of LMs: Learning from Self-Generated Mistakes"; Thinking Machines 2025):**
  student rolls out, teacher relabels the student's own visited items, aggregate 2-4 rounds.
  Fixes compounding error/covariate shift. Predicts gains concentrated where the base run went
  wrong; shrinking train/held-out gap. THE clearest gap; direct LLM precedent.
- **M20 KTO on existing YES/NO labels (2402.01306):** near-free loss-format change — same
  data as (c), prospect-theoretic unpaired loss; isolates signal-FORMAT vs sample-count as
  the bottleneck.
- **M21 active/learner-driven querying (Settles; Muldrew 2402.08114):** student picks which
  items get expensive teacher labels (uncertainty/disagreement); at matched 128-query budget,
  beats random draw or arrives at +.093 with fewer queries.

**P2:** M22 DPO/IPO proper (pairwise over generated candidates; likelihood-displacement
caveat) · M23 contrastive judgment-space (InfoNCE/CPL/NCA 2402.05369 — works from scalar
scores; O(N²) pairs free; boundary-mined negatives; predicts rank gains > raw-agreement
gains) · M24 IRL two-stage infer-then-install (fit rφ, pseudo-label a large pool, distill
against THAT; non-identifiability caveat) · M25 process/step-level reward (Lightman
2305.20050; CoT-faithfulness risk acute for tacit constructs) · M26 born-again second round
(free; null = information limit evidence) · M27 RLVR pass@k + spurious-reward audits (Yue
2504.13837 sharpening-vs-teaching; Shao 2506.10947 random-reward gains — REQUIRED validity
layer on any reward channel; no external verifier exists for tacit constructs) · M28
reward-model Goodhart curve (Gao 2210.10760; early turnover expected at N=128) · M29
reasoning-trace-augmented distillation (Lampinen 2505.00661's own fix for FT narrowness) ·
M30 dynamic per-item nearest-neighbor retrieval (item-conditioned exemplars vs static
few-shot) · M31 self-paced ordering + RANDOM-ORDER CONTROL (Wu 2012.03107: curricula only
help under tight budget or label noise — N=128 is plausibly the favorable regime, but random
control is mandatory).

**P3-P4:** M32 influence/LESS data selection (2402.04333) · M33 representation distillation
(FitNets/CRD; cross-architecture adapter needed) · M34 task/function-vector activation
injection (Hendel 2310.15916, Todd 2310.15213 — a literal THIRD modality: neither context nor
gradients; cross-model transfer untested) · M35 reversal-curse/self-patching diagnostic
(2607.08393's own method: criterion present-but-unrouted vs never-encoded) · M36 forgetting
audit + EWC/KL anchor (Luo 2308.08747: worsens with scale; Qi 2310.03693) · M37 induction-head
probing (Olsson 2209.11895; predicts (a)/(b) gains, dissociates from (c)) · M38 MAML/Reptile
multi-construct meta-init (Raghu ANIL caveat: feature reuse not rapid learning; our own
34/7/0% crossing rates predict pooling risk) · M39 MetaICL-style meta-tuning (format-similar
tasks only) · M40 exposure-only/continued pretraining (subliminal-learning 2507.14805 needs
SAME base model; Gudibande: style not substance; likely low-yield at our scale but the only
unsupervised weight channel) · M41 dataset distillation (P5 — solves the opposite problem).

**Interpretation lenses for the P1 result (not channels):** LIMA/superficial-alignment
(gains should correlate with PRE-training latent agreement — stratify!), fine-tuning-vs-RAG
(familiar-content concentration; epochs→overconfident-wrong), reversal-curse (gains
concentrate on training-phrased items).

**Sequencing recommendation (reviewer's, adopted):** M17+M18 first (free, reframe everything)
→ M19/M20/M21 in parallel → P2 tier only as diagnostics indicate → P3-P4 only if evidence
points representational.
