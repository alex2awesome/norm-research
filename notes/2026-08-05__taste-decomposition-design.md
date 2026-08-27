# Taste-residual decomposition — three-layer design (uniform across tasks)

Date: 2026-08-05. Status: DESIGN + PILOT (exploratory). Confirmatory runs on the
remaining tasks require the prereg freeze in §6 first (per the prereg-before-
confirmatory rule). User sign-off: 2026-08-05 ("run Layer 1 for every V and V+A
calculation we have"; Layer 2 (a)+(b) → appendix robustness; Layer 3 on two
tracks — real-feature mining to saturation + explicit spurious-feature mining
with score discounting).

Motivating problem (user, 2026-08-05): Δ = T − (V+A) cannot currently be
distinguished from (i) nonlinear interactions of already-articulated criteria,
(ii) spurious/shortcut features exploited by the dense model, (iii) genuine
tacit signal. Δ is an UPPER bound on taste; these layers tighten it and name
the parts.

## 0. Quantity ledger (fixed names, used in all tables)

| symbol | definition |
|---|---|
| `VA_lin` | current linear aggregation of the per-example V+A score matrix (task's existing protocol, unchanged) |
| `VA_nl` | nonlinear (GBM) aggregation of the SAME matrix, same grouped OOF protocol — no text access |
| `T` | dense-standard clean-eval AUC (Llama-8B LoRA recipe) |
| `Δ_total` | T − VA_lin (the number we currently call the residual) |
| `Δ_interact` | VA_nl − VA_lin — interactions **of articulated criteria**; articulable components, tacit combination rule; NOT taste |
| `Δ_beyond` | T − VA_nl — only this part is eligible to be called taste |
| `Δ_r` | closure curve: Δ_beyond after r rounds of Layer-3A mining |
| `T_adj`, `VA_adj`, `Δ_adj` | spurious-discounted values after Layer-3B (stratified/matched readout) |

Claim discipline: never quote raw Δ_total as "taste" once a task has Layer-1
numbers; the paper-quotable taste bound is Δ_beyond (pre-closure) or the Δ_r
plateau (post-closure), spurious-discounted where Layer 3B ran.

## 1. Layer 1 — nonlinear stack (run on EVERY V and V+A cell)

Input: per-example score matrix X (all V features + all A criterion scores),
label y, grouping unit (task's canonical one). No text, no embeddings.

Protocol invariants (uniform across tasks):
- Mirror the task's existing linear aggregation EXACTLY for `VA_lin` (replicate
  the published number first as a gate; if it doesn't reproduce, stop and flag).
- `VA_nl`: sklearn HistGradientBoostingClassifier, small frozen grid
  (max_leaf_nodes ∈ {15, 31}, learning_rate .06, max_iter 400 + early stopping),
  grid selected by inner grouped CV **within train folds only**; same grouped
  OOF folds as the linear run; seed 0.
- Report: VA_lin, VA_nl, Δ_interact, Δ_beyond, and (descriptive only) top-10
  SHAP interaction pairs.
- Also run V-only matrices → `V_lin`, `V_nl` (user: "every V and V+A
  calculation").

Cost: CPU-only, minutes per cell. No new judging.

## 2. Layer 2 — robustness appendix (a: grouped transfer, b: nuisance strata)

(a) Grouped-transfer table, per cell: pooled AUC vs within-group AUC vs
group-identity-alone AUC, for both T and VA_nl. (Code cell is the exemplar:
repo-alone .802/.854, within-repo ~.58.) Mostly already computed — this
formalizes it into one uniform appendix table.

(b) Nuisance-stratified readouts (threshold-free, per the stratified/rank-stat
rule — no residual-regression AUCs). Uniform declared nuisance set:
1. char length (+ log token count)
2. formatting density (linebreak rate, markdown/code-fence/list-marker rate)
3. topic: k=20 k-means on bge-large embeddings (base model — NOT the collapsed
   code_review finetune), fit on train only
4. date/era where the corpus has it
Report per cell: nuisance-alone AUC (each + jointly), then decile/cluster-
stratified AUC of T and VA_nl (mean of within-stratum AUCs, n-weighted).
Survival = stratified AUC stays within .02 of pooled.

Cost: CPU + one embedding pass per corpus. No new judging.

## 3. Layer 3 — articulation-closure (two tracks, kept separate end-to-end)

Shared machinery: mining loop on TRAIN only; criteria scored corpus-wide by
Gemma-4-31B offline-batch vLLM on sk3 (A-bank rule: GEPA-iterated +
Gemma-4-31b judge; blinded anchor battery EVERY batch); bank grows; VA_lin/
VA_nl recomputed each round with the frozen Layer-1 protocol.

**Label-blindness (critical):** the mining target is the DENSE SCORE, never y.
Round r inspects train examples with the largest |dense_prob − VA_nl_pred|
disagreement and proposes criteria that explain what the dense model perceives.
Metrics stay label-blind (reconstruction-only rule); y is touched only by the
frozen evaluation readout.

Split discipline: train = fit + mine; eval = closure-curve monitoring Δ_r;
test = quoted once, at the declared stopping round only. No selection on eval.

### Track A — real-feature mining to saturation
- Round r: propose k=15 new candidate quality-relevant criteria from the
  disagreement slice (GEPA-style proposer; existing infra:
  methods/metric_implementer mining + methods/autometrics loops).
- Score, extend bank, recompute VA_nl^(r), plot Δ_r on eval.
- Stopping rule (to be frozen in prereg after pilot): saturation = 2
  consecutive rounds with eval VA_nl gain < ε (pilot ε=.005), max B=5 rounds.
- Deliverable: closure curve Δ_r vs cumulative criteria; plateau value = the
  defensible taste bound. Prior evidence saturation is real: value-census
  saturation (15/15 pools ≥80% by 10% draws) and capture-recapture (mining
  moves the bound's LEVEL; audit its WIDTH) — expect fast flattening.

### Track B — explicit spurious-feature mining + discounting
- Separate proposer, explicitly instructed to mine SUSPECTED-SPURIOUS
  predictive features: length proxies, formatting/boilerplate, venue/community
  style markers, topic markers, temporal tells — "predictive but not quality."
- Same scoring machinery, but the resulting features are DECLARED NUISANCES,
  never added to the A bank.
- Blind classification audit: pool Track-A and Track-B proposals, strip
  provenance, and have a Sonnet-or-better judge classify each as
  quality-relevant vs incidental; disagreements adjudicated by frontier
  arbiter. Guards against real features hiding in B or spurious in A.
- Discounting (existing tooling: hierarchical/topic-stratified balancing builds
  + length-balanced precedents): report T_adj and VA_adj as
  spurious-stratified AUCs (strata from mined-spurious feature deciles/
  clusters), and spurious-alone AUC. Δ_adj = T_adj − VA_adj.

Cost per round ≈ 15 criteria × n_union prompts (peer: ~430K prompts ≈ one
Gemma batch, few GPU-hours). Full pilot (5A + 2B rounds) ≈ 1–2 GPU-days on one
sk3 GPU.

## 4. Uniform application — task matrix (priority order)

| # | cell | X matrix | T preds | n_eval | Δ_total | layers |
|---|---|---|---|---|---|---|
| 1 | peer verdict (PILOT) | vat_3y/union_scores.npz (154A+V) | dense chain rm_out (sk3) | ~2.8K | +.06 | 1, 2, 3A+3B |
| 2 | N&C responded | v4/nc_scores_shard*.npz | dense chain | — | +.17 | 1, 2, 3A+3B |
| 3 | CW community | va_gemma_banks npz (45 crit) | r2 pass | 9,573 | +.15 | 1, 2, 3A |
| 4 | math.SE accepted | gemma rescore matrices | sweep grouped .794 | — | +.09 | 1, 2 |
| 5 | humor caption crowd | 364-bank caption matrices | r2 pass .5631 | 1,098 | −.09 | 1, 2 (negative-Δ control) |
| 6 | peer revealed / N&C outcome / patents / journalism | as available | — | — | mixed | 1, 2 |
| 7 | code PR-merge (v3 enriched, when landed) | pending A-rescore on enriched input | v3 rm_out | ~5.9K | TBD | 1, 2 |

Rules: negative-Δ cells still get Layer 1 (Δ_interact can be nonzero even when
Δ_total ≤ 0 — VA_nl may EXCEED T, sharpening "no residual"). Layer 3 only where
Δ_beyond > .02 after Layer 1.

## 4b. FULL ROLLOUT QUEUE (2026-08-05, user: "expand to all other tasks")

Wave = one delegated batch; a cell advances to Layer 2 once it has VA_nl, and to
Layer 3 only if Δ_beyond > .02 after Layer 1 + prereg freeze.

**Wave 1 — DONE/RUNNING:** peer verdict ✓, CW community ✓, HashtagWars ✓,
Style Inv ✓, cap crowd ✓, cap finalist ✓, N&C responded/outcome/agree (running;
outcome doubles as the curation change-made cell).

**Wave 2 — same-matrix extensions (LAUNCHED):** peer curation (oral/spotlight,
gate V .550/A .563/VA .567, T .593) and peer revealed (citation-pct, gate
V .705/A .751/VA .761, T .871) — both reuse vat_3y/union_scores.npz + the pilot
code; title-grouped; group-level bootstrap per freeze #3.

**Wave 3 — discovery + run (LAUNCHED):** cells whose matrices need locating
before the gate: math.SE community (accepted∧upvoted; V .565/VA .673/T .794),
mathlib verdict (V .684/VA .680; T .770 split-unverified — Layer 1 runs on
scores alone, no T claim), patents verdict claim-fell (V .601/VA .626, no T),
press/journalism verdict editorial-pickup (V .628, A-only .648 — VA stack NEVER
FIT: Layer 1's linear leg IS the first fit; gate on V and A-only components,
mark VA_lin "first-fit" — this also closes the old V4 press-stack gap).
Rule: matrix not found or gate fails → skip cell + report, never force.

**Wave 4 — blocked/conditional:**
- code PR-merge: after v3 T lands AND A-bank rescored on enriched input
  (same-input rule); then full Layer 1 on the enriched matrix.
- journalism homepage curation: weak-flag instrument — needs user decision
  whether to invest before decomposing.
- peer best-papers: blocked on V10 A-bank build.
- legal cells: optional (off-grid, App. A); run Layer 1 only if the paper
  quotes a legal Δ.
- CW curation (Wigleaf): mined-A null → V-only Layer 1 has little to decompose; skip.
- claim-matching: CLOSED, excluded.

**Cross-cutting GPU batch (queue after v3 training frees GPU1):** same-rows T
rescores (freeze #2) for every cell quoting Δ_beyond: peer verdict/curation/
revealed, N&C ×3, CW community, cap ×2, math.SE. One chained job, one GPU.

**Layer 2 build:** implement layer2_robustness.py after N&C lands (so the
driver generalizes over all wave-1 cells at once); nuisance features need raw
text access per corpus — config.py gains a text_path per cell.

**Layer 3:** draft prereg after wave 2-3 land (final Δ_beyond>.02 list known);
freeze ε/B/k/audit; pilots = peer verdict + CW community (+ N&C responded if
it survives Layer 1).

## 5. Code layout

`methods/taste_decomposition/` (new): `config.py` (per-cell registry: matrix
path, T-preds path, group column, nuisance availability), `layer1_stack.py`,
`layer2_robustness.py`, `layer3_closure/` (proposer prompts A/B, scoring
driver, curve + audit). One driver, one JSON result schema per cell.

## 6. Prereg items to freeze AFTER the pilot, BEFORE confirmatory
ε and B (saturation rule); GBM grid; nuisance list; k per round; the blind
classification audit protocol; which cells are confirmatory. Pilot (peer
verdict) is declared exploratory.

### Pilot outcomes to carry into the freeze (2026-08-05, peer verdict)
- Gate reproduced to machine precision; protocol template works
  (methods/taste_decomposition/layer1_stack.py + 7 protocol notes in
  notes/2026-08-05__layer1_peer_verdict_pilot.md).
- **Δ_interact = −.002 (95% CI [−.009, +.005]) — NULL.** The +.06 peer residual
  is NOT tacit combination rules over articulated criteria; SHAP interactions
  are dominated by instrument redundancy (v_kw_code × availability-statement
  rubric), not synergy. Δ_beyond = +.0654 → clears the Layer-3 gate.
- FREEZE CHANGE 1: VA_nl := mean over seeds {0,1,2} with spread reported
  (single-seed spread .0099 ≈ 5× |Δ_interact|).
- FREEZE CHANGE 2: Δ_beyond requires SAME-ROWS T — rescore the dense
  best_model on the exact A/V-scored population per cell (pilot inherited the
  registry convention: T eval rows 5,408 ≠ 6,030 scored rows). Small GPU job
  per cell; queue before confirmatory quotes.
- FREEZE CHANGE 3 (from Gemma-cell rollout): Δ_interact CIs must use
  GROUP-LEVEL bootstrap (resample groups, not rows) — coarse-grouped cells
  (40 hashtags / 316 weeks / ~225 contests) otherwise overstate precision.
- Rollout observation to encode: when Δ_interact > 0, decompose via the V-only
  interaction gain — a large V_nl−V_lin alongside length-feature-dominated
  SHAP pairs (Style Inv pattern) marks SURFACE nonlinearity → route to
  Layer 2(b)/Track B, not to a tacit-combination claim; only A-side synergy
  with small V-only gain (cap_finalist pattern) is a candidate genuine
  interaction.

## 7. Program updates 2026-08-06 (user directives)

1. **Missing-mass estimation joins the discovery loop** (user: "good-turing or missing
   mass estimation... inline with our prompt scaling work"). Retrospective prototype on
   the pilot running (notes/2026-08-06__closure-swap-and-missing-mass.md when it lands);
   prospective design for confirmatory: P independent proposers per round (pilot rounds
   were sequential/conditioned — no independence), species = deduped concepts, Good-
   Turing missing mass = singletons/total, remaining-AUC bound = mass × marginal species
   value, reported each round next to Δ_r as an additional stopping diagnostic. Kinship:
   capture-recapture discipline from prompt-optimality (mining moves the bound's LEVEL;
   audit its WIDTH).
2. **Proposer model = GLM-5.2 via subscription API** for confirmatory Track A/B rounds
   (user: biggest model we have access to). Roles unchanged elsewhere: scoring judge =
   Gemma-4-31B; blind auditor = Sonnet-class; arbiter = frontier Claude. Constraints:
   Lite plan 87M tokens/week — batch proposal calls, smoke-test trace lengths before
   any think-heavy round (GLM quota rule); GLM NEVER judges (judges-Sonnet-or-better
   rule stands).
3. **Confirmatory roster BROADENED** (user: "be a bit broader on datasets"):
   - Tier 1 (ready once same-rows T lands): CW community (+.159), N&C responded
     (+.084), peer revealed (+.104, needs topic-strat robustness first).
   - Tier 2 (needs a dense arm first — these are the remaining V4 dense-standard runs):
     HashtagWars verdict, Style Invitational, patents claim-fell. Dense chain queued.
   - Tier 3 (instrument work first): code v3 (enriched A-rescore in flight; joins with
     a fresh, provenance-clean bank); press (gate check; NOTE VA_nl>T there → no
     closure needed, Layer-2 only).
   - Out: math.SE (user call — published numbers stand, cell excluded from
     decomposition), caption cells (no residual), mathlib (T unverified), N&C agree
     (unstable y).
4. **Code cell baseline discipline**: published V .576/VA .592 has BROKEN provenance
   (coded backend/no judge, spliced protocols, fingerprint aspects, not reproducible —
   registry 2026-08-06). The enriched Gemma rescore builds the cell's first
   provenance-clean articulated baseline; never difference it against .592.

## 8. Robustified missing-mass program (user directive 2026-08-06)

Question formalized: "what is the likelihood that further mining discovers new criteria
that fill the remaining gap" — now with its own validation battery.

**M1 — Multi-proposer fleet (heterogeneous, sealed).** P≥5 independent proposers per
round drawn from ≥3 model families: GLM-5.2 (thinking; smoke-test trace lengths first,
87M/wk budget), gpt-5.6-luna via Codex companion (escalate to gpt-5.6-sol only if luna
proposals are weak — user authorization 2026-08-06), Claude Sonnet/Opus. Same sealed
slice, no sight of bank or each other. Cross-family recapture rates feed the species
estimator; family-specific discovery profiles reported (a family that proposes what
others miss raises the richness estimate).

**M2 — RETRO-1: estimator backtesting (predict-the-discoveries).** Fit the
missing-mass/decay estimator on rounds 0..r, predict round r+1's new-species count and
AUC gain, score against actuals — on the pilot's 4 rounds now, and on every confirmatory
round as it accrues. The estimator earns quotation rights by out-of-sample prediction,
not by fit.

**M3 — RETRO-2: leave-out recovery (the positive control).** From the peer bank's 54
distinct concepts, hold out sets of K=8 (3 replicates, stratified by alone-AUC:
high/medium/low), recompute the depleted VA_nl (CPU), regenerate the disagreement slice,
run the sealed fleet, and measure: (a) REDISCOVERY RATE — fraction of held-out concepts
semantically matched (τ calibrated on planted probes, .78-.80) by ≥1 proposer; (b) AUC
RECOVERY — re-add the ORIGINAL score columns of matched concepts (zero new judging) and
measure how much of the depletion loss returns; (c) sensitivity curve — rediscovery rate
vs held-out criterion's alone-AUC. Interpretation contract: the confirmatory closure
verdict "no more articulable signal" is quotable only at a measured rediscovery
sensitivity ≥ some floor (target ≥70% on high-value holdouts); below that, the plateau
is "not discoverable BY THIS MINER", a weaker claim, and must be labeled so.

Cost note: RETRO-2 needs proposer calls only — matched concepts are measured with their
existing score columns. New-species proposals from these runs may be banked for later
scoring but are not scored within M3.

## 9. Spurious-influence control instruments — decisions (user 2026-08-06)

- ADOPTED: stacked-increment readout + matched sampling (both in the freeze, running).
- **REJECTED: counterfactual text editing / paraphrase canonicalization** — user: rewrites
  cannot be trusted to perturb only the intended nuisance channel; they may remove
  taste-bearing content itself (form≈content in CW/humor especially). Do not build; do
  not propose again without new evidence of surgical-edit fidelity.
- CONDITIONAL: **adversarial representation debiasing (gradient-reversal)** — approved to
  pilot ONLY behind a planted-check validation battery (user: "we'd need planted checks
  and tests to make sure our machinery is correct"). No text editing involved: a second
  head tries to predict the named nuisance scores from the dense model's internal
  representation and its gradient is REVERSED, penalizing the representation for
  encoding those channels. Named-channels-only limitation acknowledged.
  Validation battery (all must pass before any real-cell number is quoted):
  V1 exploit check: plant a synthetic token correlated with y; vanilla dense must
     exploit it (AUC jump vs no-plant baseline).
  V2 removal check: debias with the planted channel named → the jump must vanish AND a
     post-hoc probe head must fail to recover the plant from the representation (probe
     AUC ≈ chance).
  V3 specificity check: plant a REAL (content) signal; debiasing the named nuisances
     must NOT remove it; and debiasing must not damage AUC on a cell where the named
     channel is verifiably unused.
  V4 consistency check: debias a known-strong real channel (length on N&C responded)
     and compare the implied influence against the stacking/matched-sampling estimate —
     instruments must agree in sign and rough magnitude.
  Pilot cell: N&C responded (strong known nuisances, big n). Recipe: dense standard +
  GRL head on nuisance-score vector; report head weight sweep; planted tokens are
  mechanical appends (verifiable, minimal edit — not model rewrites).

## 10. FULL-GRID DRIVE PLAN (user 2026-08-06: all tasks except law, all y variables)

| field | y | status in the decomposition program |
|---|---|---|
| Peer review | verdict | DONE (pilot, plateau +.081) |
| | curation (oral/spot) | maps batch RUNNING |
| | revealed (citation-pct) | maps batch RUNNING |
| | best-papers | BLOCKED: A-bank build (V10 #52) |
| Regulatory (N&C) | responded | full campaign RUNNING |
| | outcome/change-made | maps batch RUNNING |
| | agree | map-only, unstable-y caveat → maps batch 2 |
| | co-signing | BLOCKED: no build (V8 #50) |
| Creative writing | community (upvotes) | full campaign RUNNING |
| | curation (Wigleaf) | DOCUMENTED NULL BANK (craft-bank 0/150 kept, retest .90) — skip, cite the null |
| | fiction verdict (RoyalRoad) | same null-bank status — skip, cite |
| Humor | caption crowd + finalist | maps batch RUNNING |
| | HashtagWars verdict | dense arm landing → auto-join full track |
| | Style Invitational | dense arm → auto-join |
| | reddit jokes community | GAP: needs mature A-bank + dense (queued) |
| Math | mathlib verdict | GAP-CLOSER: T split-verify → then maps |
| | math.SE community | GAP: V2 rebuild (#44) + Gemma bank — the one math cell needing new measurement |
| | AoPS curation | GAP: no A-bank stack; queue behind jokes |
| Code | PR-merge v3 | A-bank rescore landing → auto-join |
| | competitions curation | GAP-CLOSER: locate matrices (bank .731 > dense .690 — the strongest bank>dense exemplar deserves its map) |
| | SO votes | BLOCKED: no build (V6 #48) |
| Journalism/press | verdict (editorial pickup) | GAP-CLOSER: gate check → then maps |
| | homepage curation | GAP: stack integration first (weak-flag instrument; user earlier deferred — include per all-tasks directive, flag quality) |
| | tweets community | BLOCKED: unlabeled (V9 #51) |
| Patents | claim-fell verdict | dense arm in chain → auto-join |
| | forward citations | BLOCKED: no build (V7 #49) |
| Law | all | EXCLUDED (user) |

GAP-CLOSER batch (one agent, small jobs): press gate check; mathlib T split-verify;
code-competitions matrix discovery + Layer 1; N&C agree map add-on.
QUEUED BUILDS (after campaign peak): reddit-jokes mature bank + dense; math.SE V2
rebuild + Gemma bank; AoPS bank; homepage stack.

## 11. STANDING RULE (user 2026-08-09): fused system must beat the bank — auto-audit otherwise

Expectation: the best fused arm (V+A+T stack or V3 criteria-in-prompt) beats VA_nl on
every cell. ANY final ledger where max(fused arms) ≤ VA_nl AUTO-TRIGGERS a Fable audit
of that cell (why is fusion failing: data starvation, feature injection format, stack
leakage rules too strict, bank overfit to eval, label noise ceiling?). Current
triggers: cap_crowd (bank .6217 > fused .6204/.6190) — COVERED by the running V3
optimization audit; press — pending its honest T (audit fires if bank ≥ fused after
scale-up A lands). Logging discipline (same directive): every landing goes to the
registry + strict list before anything else is launched; no test repeated without a
written breakage reason (reuse-before-rebuild).

## 12. PRODUCTION V+A+T PIPELINE with decorrelated training (user-approved 2026-08-10)

The user's four steps, solidified:
1. FIT BASELINES: dense T (class-weighted where imbalanced — mathlib lesson) + the
   V/A bank stacks (VA_lin, VA_nl) per the frozen Layer-1 spec.
2. MINE DUAL-TRACK: sealed multi-family fleet proposes real (Track A) and spurious
   (Track B) candidates; TELLING THEM APART = the blind routing audit (provenance-
   stripped classification + planted probe pairs each round + arbiter on disputes +
   MIXED-channel decomposition into real/surface components). Validated: 0-4%
   misrouting, probes 10/10 rounds.
3. PHRASE + SCORE: GEPA-style fidelity-optimized label-blind phrasing on accepted
   criteria; Gemma-4-31B scoring with anchors K≥50 + collapse gates.
4. USE ASYMMETRICALLY:
   - Real criteria → A bank → VA stacks + V3 (criteria-in-prompt dense).
   - Spurious criteria → declared nuisance set → (a) readout controls (stacked
     increment; matched sampling past .65) AND (b) **DECORRELATED TRAINING**: importance-
     REWEIGHT the dense training distribution so y ⊥ joint-nuisance-score (no text
     edits, no row deletion — weights only), retrain → T_decor; optionally V3_decor.
     The goal is removing the INCENTIVE to learn shortcuts (so capacity goes to real
     features), not removing decodability — hence the gate below tests RELIANCE, not
     probes.
VALIDATION GATE for (b), planted battery (reuses debias infra; runs BEFORE any scaled
decor arm): V1 vanilla exploits the plant; V2' reweighted training - the planted-vs-
unplanted jump vanishes AND ablation reliance ≈ 0 AND task AUC held within seed band;
V3' a planted REAL signal survives reweighting; V4' the implied length-channel influence
agrees with stacking/matching in sign and ~2× magnitude.
FULL-GRID LAUNCH (user: all 7 tasks × 3 preference variables, all free sk3 GPUs): every
strict-list cell with instruments gets the fused ledger — VA_lin/VA_nl, T, VAT stack,
V3 arm (current-best config until the V3 audit's production recipe lands), and T_decor
where a mined nuisance set exists (gated on the battery). REUSE: harvest the running
VAT-stack agent's outputs; never re-run a delivered arm.
HUMOR CANONICAL STRUCTURE (user 2026-08-10): verdict = HashtagWars; curation = caption
finalist (primary) with Style Invitational as appendix robustness; community = caption
crowd (primary, same-item contrast with finalist) with reddit-jokes as replication;
Newsjack/SNL = documented non-candidates. Never pool across humor sources.

## 13. PRODUCTION DECORRELATION STACK — FREEZE-READY (Fable decision, 2026-08-08)

Decision-maker: Fable (user directive: "make a decision on the right stack to do
proper decorrelation with spurious variables"). Evidence base: GRL (gradient-reversal
adversarial debiasing) definitive negative on both architectures
(notes/2026-08-07__debias_audit_fable.md); LEACE (LEAst-squares Concept Erasure)
pilot adopted linear-scope (notes/2026-08-10__leace_pilot.md); readout-instrument
track record across all campaigns (notes/2026-07-27__vat-run-registry.md); field
survey (notes/2026-08-10__litreview_spurious_debiasing.md). The decorrelated-training
battery (notes/2026-08-10__decorrelated_training_battery.md) was IN FLIGHT at decision
time (gradcheck PASS, V1 PASS from stored artifacts, V2'-V4' pending); its leg below
is written as an explicit conditional on its own declared, BINDING gates — no peeking,
no re-litigation when it lands: the branch executes itself.

### 13.0 The stack in one table

| need | instrument(s) | role |
|---|---|---|
| (a) influence measurement | stacked increment (PRIMARY) + matched sampling (SECONDARY, sign/consistency only) + LEACE erase-and-refit (THIRD LEG, intervention scope) | how much of an instrument's performance rides on named channels |
| (b) training-time protection | decorrelated reweighting → T_decor (CONDITIONAL on battery V2'-V4'; robustness arm, never a T replacement) | remove the incentive to over-allocate capacity to shortcuts |
| (c) removal certification | LEACE, linear scope only | certify a LINEAR score path cannot use channel X |
| retired/banned | GRL (any variant); decile stratification as a discount instrument at spurious-alone > .65; counterfactual text editing (standing user rejection, now lit-backed) | — |

"Spurious-alone" below = grouped-OOF AUC of the joint nuisance model (Track-B/declared
channels) predicting y. All instrument readouts obey the threshold-free rule (AUC/rank
stats) and the reconstruction-only rule (metrics never label-aware; y touched only by
frozen evaluation readouts).

### 13.1 (a) Influence measurement — three legs, and when each governs

**Leg 1, GOVERNS BY DEFAULT — stacked increment** (instrument-over-nuisance-model
ΔAUC; the partial-input-baseline pedigree: Gururangan 2018 / Poliak 2018). Runs on
EVERY cell with a nuisance set, both for T and VA_nl. Quoting rules:
- POSITIVE increments carry the Westfall & Yarkoni (2016) incremental-validity caveat:
  our nuisance scores are single noisy LLM-judged indicators, and unreliability + large
  n biases TOWARD declaring signal beyond the nuisance. Confirmatory positive-increment
  quotes ship with a reliability sensitivity band (lit-review adoption #2): recompute
  the increment with attenuation-simulated nuisance reliability r ∈ {.5, .7, .9}
  (noise-inject the nuisance scores, refit the nuisance model, grouped OOF) and quote
  the band, not the point. CPU-only.
- NULL/NEGATIVE increments are robust to that critique (the asymmetry cuts the other
  way) and may be quoted directly.
- Standing limitation sentence (Feng et al. 2019), verbatim in any confirmatory use:
  a clean stacked increment rules out only the channels we scored — not their
  interactions, not channels never named. Maps are lower bounds on the channel set
  (B-side missing mass > A-side, registry 2026-08-07).

**Leg 2, SECONDARY, SIGN/CONSISTENCY ONLY — matched sampling** on the nuisance score
(blocking-style matching on the covariate itself — the form King & Nielsen 2019
recommends; the critique transfers only to matching on a LEARNED joint propensity,
which we therefore never do for matching — joint scores are for weighting, §13.2).
Track record demands demotion from magnitude duty: on the one channel where every
instrument ran (length, N&C responded), matched read 2-7× smaller than stacked/
stratified/LEACE under one protocol (calipers .01/.02/.05 → .0162/.0079/.0044) and
flipped SIGN under a near-identical one (pre-registered R00 references: matched(.02)
−.0089 eval / −.0212 evaltest vs stacked +.0070/+.0182). Quoting rules:
- Always the full caliper sweep {.01, .02, .05}, never one caliper.
- |estimate| < .01 = inside its demonstrated protocol-sensitivity band → quote as
  "≈0/indeterminate", never as sign evidence.
- Its job is CONSISTENCY: same-sign agreement with Leg 1 strengthens a claim;
  divergence is recorded (a property of the cell) and triggers Leg 3, never averaged.
- Past spurious-alone .65 its match-support thins (width problems) — it stays
  reported but LEACE becomes the second quantitative leg there.

**Leg 3, INTERVENTION SCOPE — LEACE erase-and-refit joins the stack** (decision:
YES, as third leg, not as co-primary). Erase channel X from frozen h, refit the
linear head, read the implied influence = AUC(raw) − AUC(erased) MINUS the y-tax
(below). Validated: V4 agreement with stratified 0.93× and stacked 1.21× on length.
It runs when ANY of:
1. Legs 1-2 disagree in sign, or by >2× where the conclusion depends on magnitude;
2. a HEADLINE per-channel influence number is quoted (confirmatory cells);
3. spurious-alone > .65 (the regime where stratification is banned and matched
   magnitude is untrustworthy — LEACE is an intervention, immune to the
   conditioning-on-label failure).
Quoting rules (from the pilot verdict, now frozen): report the MLP nonlinear residue
next to ANY erasure number (plant .96, length .92 — the residue is generic); report
the y-matched placebo and quote CHANNEL-SPECIFIC cost = total erasure cost − y-tax
(erased AUCs understate the channel-absent counterfactual by the tax); eval-row
transfer is ≈.55 not .50 — never "provably zero" off fit rows; CONTINUOUS channels
are erased as one-hot bins (tercile or median; lit-review live-issue fix — the strong
any-convex-loss guarantee is proven for categorical Z only; erase the binned version,
optionally report the continuous-Z erasure beside it).

**Decile stratification: RETIRED as a discount instrument.** At spurious-alone > .65,
stratifying on the nuisance model ≈ conditioning on the label (twice-documented:
Δ_adj +.1146 at .712, +.110 at .713 — both NEVER-QUOTE). It survives only as the
Layer-2(b) descriptive appendix readout at spurious-alone ≤ .65, and never yields a
quotable Δ_adj point estimate at any level: Δ_adj is quoted as a band, or in the
negative form ("discounting does not shrink the residual") which is the form the
track record shows is robust.

### 13.2 (b) Training-time protection — decorrelated reweighting, battery-gated

CONDITIONAL ADOPTION: the recipe below freezes IF AND ONLY IF the in-flight battery
passes its declared gates (V2' chain-gate = ablation-reliance |Δ| ≤ .005 AND task AUC
≥ vanilla 3-seed band min; V3' specificity both halves; V4' consistency sign + ratio
[.5, 2] or INDETERMINATE-non-blocking). V1 already PASS; gradcheck already PASS.

**Frozen weight recipe (on pass):** stabilized inverse-propensity
w_i = P̂(y_i)/P̂(y_i|s_i); P̂(y|s) = StandardScaler + logistic (C=1), 5-fold
GroupKFold on the cell's grouping unit, OOF, TRAIN rows only; p floored 1e-3; clipped
at the 99th percentile; renormalized to mean 1; applied as per-example LOSS weights
(never WeightedRandomSampler); eval/test always unweighted, select-on-eval unweighted;
trainer otherwise = the frozen dense standard. Per-cell reporting REQUIRED: n_eff =
(Σw)²/Σw², clip rate, weighted AUC(ŝ,y) (target ≈.50, residual documented), weighted
vs unweighted positive rate. FLOOR: n_eff/n ≥ .70 or the decor arm is not run (the
unstabilized form's 32-43% n_eff cost is the cautionary case; Cortes 2010
second-moment bounds are the theory). Byrd & Lipton discipline: the 2-epoch +
select-on-eval recipe is the early-stopping regime where weighting can bite; any
recipe change that lengthens training re-opens the inertness question and requires a
battery rerun.

**When mandatory vs optional (per cell):**
- MANDATORY: spurious-alone > .65 AND the cell quotes a confirmatory Δ_beyond
  (> .02, L3 gate) — the regime where shortcut incentive is materially present and a
  taste claim is at stake.
- OPTIONAL robustness arm: spurious-alone ∈ [.55, .65] on confirmatory cells, at
  GPU-budget discretion.
- SKIP: spurious-alone < .55 (weights ≈ 1 by construction — the D09 near-no-op), or
  no confirmatory claim (nothing to protect).

**T_decor's ledger role: ROBUSTNESS ARM, never a replacement for T.** T remains the
headline dense standard; Δ_beyond is quoted against T. T_decor appears beside it as a
shortcut-sensitivity readout: Δ_decor = T_decor − VA_nl and the ablation-reliance
delta. Rationale: (i) T_decor is design-conditional (trained on a reweighted
distribution — quote it WITH the named design, per the design-conditional rule);
(ii) T_decor ≈ T is ambiguous between "no shortcut reliance" and "weights inert"
(Byrd & Lipton), so it can corroborate but not headline; (iii) never mix T and
T_decor scorings in one figure or staircase. Reading: Δ_beyond stable under decor =
the residual does not ride the named shortcuts; Δ_beyond collapsing under decor =
the T-side residual was shortcut-fed — report both, the pair IS the finding.

**Fail branch (executes without re-litigation):** if V2' fails, scaled T_decor arms
are cancelled; training-time protection is declared OPEN; the two designated successor
candidates — named now so nobody re-derives them — are (1) AFR-style nuisance-balanced
last-layer refit (cheapest; our frozen-backbone + small-head architecture is literally
the DFR precondition, and the nuisance-only model we already build supplies the
weights) and (2) ODIN-style additive output-decorrelation penalty (no min-max game to
defeat — the GRL root cause does not apply). NEITHER is adopted here: each requires
its own planted battery (with the Bastings tic/op difficulty ladder, §13.5) and user
sign-off first (check-before-new-approach). A V2' fail on the plant does NOT condemn
decorrelation on weaker real channels (the lit review found no stress test of
reweighting at near-deterministic shortcut strength — the battery's plant is the
hardest case); the fail branch verdict is scoped accordingly.

### 13.3 (c) Removal certification — LEACE, linear scope, rare by design

Needed ONLY when a claim has the form "this score path cannot use channel X":
(1) defending a headline Δ_beyond against a named-channel objection (erase X, refit,
show the residual persists); (2) certifying a frozen scoring artifact before external
release; (3) as Leg 3 of §13.1. It is NOT a routine per-cell instrument.

Certificate language, verbatim and frozen: "the score, being a linear functional of
the erased representation, cannot use the channel" — never "the channel is gone."
Scope conditions that VOID the certificate: any retraining of backbone or adapters
after erasure (re-erase + refit + re-probe on the new model); any nonlinear consumer
of the representation; any multi-bin readout of the score (log-linear guardedness
does not compose, Ravfogel 2023 Thm 3.4). All §13.1-Leg-3 quoting rules apply (MLP
residue, y-tax placebo, ≈.55 eval transfer, one-hot binning). INLP is never used
(dominated: no closed form, more collateral damage). Nonlinear/kernel erasure (KRaM,
Obliviator) = watch-list only; our score path is linear, so linear scope suffices —
that architectural fact is load-bearing and the dense standard's linear head is
hereby part of the freeze (a nonlinear head would orphan every certificate).

### 13.4 Interactions between instruments

- **LEACE-then-refit vs decorrelated-retrain are COMPLEMENTARY SCOPES, not
  substitutes**: LEACE certifies non-USE by the linear score path, post hoc, frozen
  model, closed-form, cheap; decor prevents over-ALLOCATION during training, changes
  the model, costs a training run. Use BOTH on any cell that is mandatory-decor
  (§13.2): quote Δ_beyond raw, Δ under decor, and Δ under erase-and-refit — the
  triangulated residual is the paper-grade number.
- **The cross-instrument consistency signature (new, adopt):** for a model that does
  not rely on channel X, the LEACE erasure cost of X ≈ the y-tax placebo alone
  (channel-specific cost ≈ 0). So run the y-matched placebo + erasure on T_decor's
  reps: decor WORKED on channel X iff T_decor's channel-specific erasure cost is ≈ 0
  while vanilla T's is > 0. This is the quantitative reconciliation of the reliance
  scope (decor) with the intervention scope (LEACE), and it needs no new machinery.
- Probes NEVER certify removal anywhere in the stack (Kumar et al. 2022; our own
  GRL false-PASS near-miss): reliance (ablation) and erasure-cost readouts gate;
  probe AUCs are scope notes.
- Never re-quote one instrument's magnitude through another's protocol: each leg's
  number is quoted under its own name (never mix scorings in one figure).

### 13.5 What the lit review adds (adopted / rejected)

ADOPTED: (i) reliability sensitivity band on positive stacked increments (§13.1,
Westfall & Yarkoni fix); (ii) one-hot binning of continuous channels for LEACE
(§13.1/13.3 — closes the continuous-Z guarantee gap, a live correctness issue);
(iii) Bastings et al. 2022 tic (token-in-context) / op (ordered-pair) planted-shortcut
types as the difficulty ladder for any FUTURE planted battery (existing batteries are
not rerun); (iv) ESS/second-moment reporting + the Byrd-Lipton early-stopping
discipline for decor (already §13.2); (v) citation posture: the battery methodology
is Bastings et al.'s protocol in our modality; the ablation-reliance gate is amnesic
probing (Elazar 2021); the stacked increment is a partial-input baseline; the readout
form has the AlpacaEval-LC / Arena-style-control pedigree.
REJECTED / PARKED: ODIN and AFR/DFR (successor candidates only, §13.2 fail branch —
no third training-time instrument while one is mid-battery); RAZOR and all rewrite-
based debiasing (standing user rejection, now formally backed by Joshi & He 2022,
Kaushik 2021, Chandra Mouli 2022); RRM-style pairing-permutation counterfactuals
(edit-free, so not forbidden — parked, needs user sign-off); RM-Bench style-robustness
screen (nice-to-have, not stack); GRL in any dress including the ICML-2026-style
causal-factor variants (retired, definitive).

### 13.6 Per-cell decision procedure (the flowchart, frozen)

Given a cell: s = spurious-alone AUC (joint nuisance model, grouped OOF), and
Δ_beyond from Layer 1.
1. ALWAYS: nuisance-alone readout + stacked increment (T and VA_nl over the nuisance
   model). Cheap, universal.
2. Δ_beyond ≤ .02 (no L3 claim): STOP after step 1 — influence is reporting-only; no
   matched, no decor, no LEACE, regardless of s.
3. s < .55: stacked increment alone suffices; stratified readout optional descriptive.
4. .55 ≤ s ≤ .65: + matched sampling (full caliper sweep, sign/consistency duty);
   stratified allowed as appendix descriptive; decor arm OPTIONAL; LEACE only on
   Leg-1/Leg-2 disagreement (sign, or >2× where it matters).
5. s > .65: stratification BANNED as discount; matched reported sign-only; + LEACE
   erase-and-refit (mandatory second quantitative leg); if the cell is confirmatory,
   + T_decor (mandatory, battery-gated) + the §13.4 cross-check (channel-specific
   erasure cost on T vs T_decor).
6. Headline quoting for any confirmatory Δ_beyond: the instrument triplet under their
   own names + reliability band + the Feng limitation sentence; Δ_adj only as a band
   or in negative form; T_decor beside T, never instead of it.

## 14. PRE-KILL CHECKLIST (user 2026-08-08, binding on every agent)
No cell or instrument may be declared dead/terminal-failure without ALL FIVE recorded:
(1) absolute minority-class count in train (not just the rate); (2) a simple baseline
(TF-IDF/logistic) on the same split — baseline>chance while the big model is at chance
= training-run failure, not cell failure; (3) registry search for historic working runs;
(4) the verdict names WHICH DESIGN failed (grouping/transfer demand, k of groups);
(5) seed spread vs the claimed effect. Origin: the mathlib (94% pos, ~360 train
negatives, TF-IDF .786 on the same split) and homepage (k=8 outlet-held-out) retractions.

### §8 addendum (2026-08-08, from the A-side recovery audit)
Two-tier rule for all future proposer fleets: the SEALED independent fleet is the only
input to missing-mass estimators (Good-Turing/Chao1-odds); taxonomy-DIRECTED sweeps
(Addendum-4) are a coverage instrument only and never feed the estimator (non-
independent draws). Report sensitivity WITH its retained-control (zero lift is the
static-prior mechanism's signature, not a defect), and quote the dose-response
(P(rediscover) rises .30→.98 over alone-AUC .52→.607) whenever the .333 headline
sensitivity is quoted. Registry entry 2026-08-08 has the full anatomy.

### §13 addendum (2026-08-08, from the decor battery's V4' leg)
Measured limitation of the production readout stack: on nc_responded at n≈2K the two
adopted instruments DISAGREE IN SIGN (matched-sampling −.0117 vs stacked-increment
+.0255). "Read jointly" therefore means: when the two disagree in sign and both
magnitudes are <.03 at n≲2K, the debias readout is UNREADABLE at that resolution —
report the disagreement, never pick one. Also adopted from the battery: any
importance-weighted retraining proposal must report CLASS-CONDITIONAL n_eff (gate:
minority class ≥ .95); pooled n_eff hides the collapse that caused V2''s failure.
