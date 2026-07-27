# Intro/abstract support audit + new-experiment plan (2026-07-26)

User directive: "How much is the current introduction supported by the experiments we have, and
what do we need to get there? Don't change the intro, just plan new experiments." Plus: "I want
a set of insights for our scaling behaviors... maybe run some of Tatsu's 2024 OSL experiments"
(= Ruan, Maddison & Hashimoto, *Observational Scaling Laws*, 2024: use MANY existing models
across families as observational scale points; extract low-dim capability measures; fit scaling
curves in capability space rather than parameter count).

Verdict up front: **the comparison claims are fully supported; the bound claims are half
supported; the scaling, isomorphism, and reconstruction claims are currently unsupported by
this campaign's artifacts and each needs one designed experiment.** One intro claim is now
*measured false as written* (C4a below) — good news: it failed for a fixable estimator reason.

## Part A — claim-by-claim audit

| # | claim (abstract/intro) | support today | gap |
|---|---|---|---|
| C1 | units = "minimal perturbations that induce **causal** behavioral changes" | PARTIAL — marginals + independent draw-level regressions (HB114) are correlational-at-scale; causal evidence is 2 ablations (hover clause null; hotpot boxed-LaTeX autopsy) | no systematic per-unit causal battery; **no minimality evidence at all** |
| C2 | ε-certifiability gives "**information-theoretic upper bounds**" | ESTIMATE, not bound: Chao1 conservativeness runs the wrong way (HB104); valid recapture exists on ifbench only; UCB machinery built (HB105) but unvalidated | valid K-replicate mining (z.ai-blocked); UCB assembly; **backtest that B̂ actually upper-bounds** |
| C3 | "ties or beats GEPA (and intro adds MIPROv2) by a wide margin (XX–YY)" | **SUPPORTED**: hotpot +.220 (p<1e-13, content-audited), hover +.036..+.051 (stable), never-worse elsewhere | MIPROv2 exists on hover only; intro names it → one MIPRO run on hotpot closes it |
| C4a | intro: "at search step-5 we can reliably predict missing mass at step 8 [to-check]" | **CHECKED TODAY — FALSE AS WRITTEN** (E1 backtest): concave fit on the first ~25% of prefix points mispredicts later value by ±.05–.10; on aime the fitted asymptote sits BELOW the realized max (anti-conservative, the HB104 failure made concrete); livebench final-horizon error −.005 (good), hotpot +.096 (bad) | needs a better estimator + honest bands (E1-full), or softened wording |
| C4b | abstract: "upper bounds that **predict prompt-scaling behavior**" | UNSUPPORTED — 1.7B–32B ladders are truncation-confounded (8k), scaling was cut from this paper | the OSL battery (E5) |
| C5 | reconstruction bottleneck, "correlations .4–.8 with AUC on human anchors" | lives in the metric-seam line, not this campaign; memory suggests the range is real (mention-AUC .36–.71 stratified; within-metric ρ≈.39) but no assembled artifact table | inventory + possibly 2–3 encoder/decoder runs (E8) |
| C6 | "prompt isomorphism across LLM sizes makes explicit **relational-tacit** knowledge" | UNSUPPORTED here (absorption data is 8k-confounded and retracted for scaling use) | cross-scale unit-value matrix (E5's core readout) |
| C7 | "prompt→code isomorphism shows **thin/thick** rule differences" | PARTIAL — HB111 taxonomy (format/strategy/evidence tiers with per-tier marginals) is hand-classified and prompt-side only; no CODE side exists | unit→code implementation experiment (E7) |
| C8 | intro: "certifiably optimal prompts … equivalent to a V-information optimal prompt" | theory footnote, no artifact; the vacuity theorem constrains how "certifiably optimal" may be phrased (pool-relative only) | write the formal statement; no experiment |
| C9 | intro: "as capacities → infinity, the inherent articulability ceiling of a concept" | aspirational; nothing measured | OSL extrapolation gives the only honest empirical handle (E5-ext) |

## Part B — the experiment plan

### E1-full — Predictive-ceiling backtest with honest bands (CPU only; START IMMEDIATELY)
The naive version ran today and failed informatively. Full design: (a) replace the greedy-prefix
fit with the **draw-based value curve** (the 40-draw regressions give V(k) samples far less
noisy than one greedy path); (b) fit concave (saturating-exp AND power-law, report both);
(c) wrap in a **conformal band** calibrated by leave-future-out cross-validation within each
bench; (d) the paper's claim becomes "the band covers realized value at 2× horizon in X/4
benches", whatever X is. Success = honest calibrated statement replacing "[to-check]".
Artifacts: existing proposals.jsonl + the two 40-draw grids. ~2h of analysis, no GPU.

### E2 — Per-unit causal battery (1 GPU-day; supports C1 "causal")
For the top-10 chosen units on hotpot and hover: same-session paired ablation (candidate vs
candidate-minus-unit, k3, cache off, one invocation per bench). Turns the correlational
marginals into per-unit causal deltas with CIs. Also directly powers the Fig-of-unit-anatomy.
Protocol: one session per bench, 11 arms each (full + 10 ablations), n=300 items.

### E3 — Minimality probe (piggybacks on E2, +2h)
Split each top unit into its natural sub-clauses; measure sub-clause ablations in the same
session. Units are "minimal" iff sub-perturbations lose the effect. Directly licenses the word
"minimal" in the abstract's definition. If units turn out decomposable, the honest fix is
"small" not "minimal" — either way the claim becomes supported.

### E4 — Valid capture-recapture + UCB ceilings (BLOCKED on z.ai top-up; machinery ready)
K=3 cache-off re-mining per bench (mine_k_replicates.py, fixed), Chao1 + Chao-CI + Good-Turing
coverage, ε̂_UCB = UCB(N̂−S) × q95(singleton marginals) capped by the concave extrapolation.
Then E1's backtest applied to the UCB version. This is what lets "upper bound" survive review
(or the wording falls back to "estimated ceiling" per HB114's decision).

### E5 — ★ The OSL articulation battery (the scaling-insights centerpiece; 2–4 GPU-days, phased)
Tatsu-style observational design: no training, many existing models as scale points.
**Models (all locally cached):** Qwen3 1.7B/4B/8B/14B/32B (+ optional cross-family points:
Llama-70B BF16 recipe, Gemma-4-31B env on sk3 — cross-family is what makes it OSL rather than
a single-family ladder; capability scores per model from public benchmark aggregates per Ruan
et al., or our own 100-item probe).
**Per (model, bench∈{hotpot, hover}) at 24k, frozen pools, all single-session:**
1. init score (GEPA-official candidate, fixed across scales);
2. N=40 random recombination draws → pool-value distribution (mean, sd, max);
3. transfer of the 8B winner p*₍₈B₎ → isomorphism readout (does the same articulation work?);
4. per-unit draw-regression marginals → the **unit-value-by-scale matrix**.
**The insights this buys (each a paper sentence):**
- *Absorption vs capability*: pool lift (draw mean − init) as a function of capability score —
  the clean replacement for the retracted 8k absorption claims.
- *Relational-tacit onset (C6/Collins)*: units valuable only above a capability threshold =
  knowledge the small model cannot use even when told — the name-sufficiency dissociation
  pattern, now on task units. Classify onset curves: always-on / comes-online / never-on.
- *Thin vs thick scaling (C7/Daston)*: do format (thin) units flatten with scale while strategy
  (thick) units grow? Prediction from HB111: thin rules saturate early, thick rules ride
  capability.
- *Ceiling-predicts-scaling (C4b)*: fit B̂ at each scale from that scale's draws; test whether
  B̂(capability) fitted on small models predicts realized best at the next scale up — the
  observational version of "bounds predict scaling", and the honest cash-out of C9's
  "capability → ∞" gesture via extrapolation in capability space.
**Phasing:** P1 = Qwen 1.7/4/8/14 on hotpot (1 GPU-day, one box). P2 = +32B + hover. P3 =
cross-family (Llama/Gemma) for true OSL. Decision gates between phases.

### E6 — Thin/thick classification at n=all-units (blocked on z.ai for the judge; interim:
hand-coded taxonomy for the 2 headline pools, judge-validated post top-up.)

### E7 — Prompt→code isomorphism pilot (1 GPU-day + local model; supports C7's arrow)
For the top-20 units per headline pool: a code-writer LLM (local Qwen, not a judge) attempts a
Python implementation of each unit as a post-processor/verifier; units where code reproduces
the unit's behavioral delta (paired test on items) = **thin/mechanizable** (Collins'
"mechanization" criterion, literally); units where no faithful implementation exists = thick.
Deliverable: fraction mechanizable per tier + equivalence deltas — the prompt→code isomorphism
made operational. (Repo precedent: existing_metrics_runner/coded machinery.)

### E8 — Reconstruction-claim inventory (CPU now; runs only if the table is thin)
Assemble the .4–.8 evidence from the metric-seam artifacts (mention-AUC stratified readouts,
MI-vs-silver correlations) into one table: task × channel (behavioral I(m,m′) vs MCQ) ×
correlation-with-human-anchor-AUC. If fewer than 3 tasks × both channels survive provenance
checks, plan 2–3 encoder/decoder runs on the v2 labeled datasets (local vLLM, no z.ai needed
for the encoder/decoder; anchor AUC from existing human labels).

### E9 — MIPROv2 on hotpot (half GPU-day; closes the intro's name-check)

## Priority order under deadline
1. **E1-full** (CPU, today) — fixes a claim that is currently measured-false.
2. **E2+E3** (1 GPU-day) — the abstract's *definition* of units rests on "causal"+"minimal".
3. **E5-P1** (1 GPU-day) — unblocks C4b, C6, and the scaling-insights section; the user's OSL ask.
4. **E9** (0.5 day) — cheap intro-consistency.
5. **E8 inventory** (CPU) — determines whether the reconstruction claim needs new runs.
6. **E7** (1 day) — highest novelty-per-hour after the above.
7. **E4 + E6** on z.ai top-up.
Total GPU: ~3.5-4 days for items 2-4+6; boxes are currently idle.

## Claims that need no experiment, only wording care
- "information-theoretic" (C2): HB114 already ruled — reserve "bound" for the reachability cap
  and rank certificates unless E4 lands; the intro's "certifiably optimal prompts" must carry
  the pool-relative scope (vacuity theorem) or it will be attacked with our own Section.
- Collins' four propositions in the intro currently listed as "(4) something else" — the actual
  fourth criterion in Collins (2010) is **explanation (scientific description)**; the four are
  elaboration / transformation / mechanization / explanation. No experiment, just the citation.
