# Experimental plan: the mechanization floor as a measurement instrument (2026-06-11)

**The contribution is an instrument.** Prior work (design doc §8) either observes that some
criteria resist weak judges (GER-Eval), argues the floor exists for humans (Sadler, Lumley),
proxies codifiability by survey/annotation (Zander & Kogut → ALM → Schaal), or measures the
dual quantity with outcome labels (Alur et al.). Nobody *measures a criterion's tacitness by
attempting its articulation across executor capabilities and locating the floor*. Every
experiment below is positioned against that map: E0 validates the instrument on known
answers, E1 is the core measurement, E2–E3 decompose and predict it, E4 defends it as
measurement, E5 applies it to the norms program.

All infrastructure exists (`registry.py`, `optimizer.py`, `measures.py`, `run_trial.py
scaling --judge-models`); experiments are configurations + analysis, not new systems.

---

## E0 — Known-answer validation: does the instrument recover planted thickness?

*An instrument paper must show the instrument reads correctly where truth is known. No prior
work validates a tacitness measure against constructs with by-construction thickness.*

**Design.** Extend the `test_scenario` world-generator into a **thickness ladder** of
synthetic criteria over generated solution-like texts, with ground-truth ordering:

| Rung | Planted criterion | True floor (by construction) |
|---|---|---|
| T0 | single surface keyword present | code (regex) |
| T1 | syntactic property (e.g. guard-before-use) | code (AST) |
| T2 | 3-clause conjunction of surface properties | code, larger program |
| T3 | paraphrase-robust semantic property (stated many ways) | small LLM |
| T4 | property requiring multi-step composition over the text | mid LLM |
| T5 | property defined by family-resemblance to an exemplar set, no compact rule | strong LLM / anchor-only |

**Procedure.** Run the full frontier descent (E1 procedure) on all six. **Pass criterion:**
recovered floors reproduce the planted ordering (Kendall τ ≥ 0.8 across seeds); T0–T2 floor
at code-kind; T5 right-censored at the strongest weak tier.

**Falsifiable failure modes this catches:** optimizer too weak (everything floors high),
judge-noise dominating (ordering scrambles), anchor leakage (T5 spuriously mechanizes).

Cost: ~$20–40 OpenRouter. **Run this before trusting anything from E1.**

---

## E1 — The core measurement: frontier descent over a real criteria bank

**Hypotheses.**
- **H1 (frontier exists):** best-achievable agreement-with-anchor is monotone in judge tier
  after per-tier optimization, and per-tier optimization beats the fixed seed rubric at
  every tier (this alone separates us from GER-Eval's fixed-rubric observation).
- **H2 (heterogeneity):** floors differ substantially across criteria — the construct
  property has variance worth measuring.
- **H3 (it's a property of the criterion):** floor is stable across optimizer seeds,
  counterfactual-generator families, and corpus subsamples. (If H3 fails, there is no
  instrument — this is the kill-switch hypothesis, stated upfront.)

**Design.**
- **Bank:** 24 criteria for competitive code: the 3 trial metrics + ~21 drawn from
  `datasets/code-review/online-rubrics` + LeetCode editorial dimensions, hand-picked to
  span hypothesized thin ("states time complexity", "compiles/has tests") → thick
  ("elegant", "idiomatic", "instructive approach"). Each registered with a seed rubric;
  code-kind seeds where plausible.
- **Tiers** (the x-axis, all via OpenRouter slugs; code-kind = tier 0 "interpreter"):
  code → Llama-3.2-1B → Llama-3.1-8B → Llama-3.3-70B. **Anchor:** Sonnet-4.5 + rich
  rubric, itself CF-validated before use.
- **Per (criterion × tier):** GEPA descent, objective dominated by agreement-with-anchor
  (oracle repointed at anchor scores), 3 seeds, fixed generous budget caps, acceptance on
  cross-family frozen holdout. ~290 runs. *(E7 amendment: run at three nested
  instruction-token caps instead of one generous cap, so each criterion yields an L × tier
  frontier grid — see E7.)*
- **τ:** report floors at τ ∈ {0.6, 0.7, 0.8} disattenuated agreement — the KM curves at
  three thresholds show conclusions aren't a τ artifact.

**Readouts.**
1. Per-criterion frontier curves (the paper's main figure).
2. **Floor B\*(m)** per criterion, right-censored at code/1B ends.
3. **Kaplan–Meier survival curve over the bank** — "fraction of this community's criteria
   articulable at executor capability X." First curve of its kind in any literature.
4. Criterion classification: {code-mechanizable, 1B-, 8B-, 70B-, anchor-only(censored)}.

**Cost:** ~$100–150 fully on OpenRouter; ~$30 if bulk judge scoring moves to sk3 vLLM
offline batch (recommended; reviser/oracle stay on OpenRouter). Pilot first: 6 criteria ×
3 tiers × 1 seed ≈ $10 to calibrate run-cost and τ before the full grid.

---

## E2 — Words vs. reader: the 2×3 definition ablation

*SILICON names guideline- vs model-induced error as confounds; Codebook-LLMs observed
name-following at one tier. Nobody derives the coordinate.*

**Design.** {1B, 8B, Sonnet} × {name-only stub, seed rubric, E1-optimized rubric for that
tier}. All cells score the same frozen 100-item sample; reference = anchor scores.

**Derived quantities per criterion:** words effect (stub→full gain, per tier), reader
effect (tier gain at fixed definition), interaction (words only work for capable readers =
Daston's thick-rule-written-down), and the **A/dense-boundary diagnostic**:
corr(Sonnet+full, Sonnet+stub) ≈ 1 ⟹ the rubric is a pointer into the reader's prior.

**Validation hypothesis (links E2 to E1):** the interaction term predicts B\*(m). If it
does, the cheap 2×3 (≈600 judge calls/criterion) estimates the floor without running the
full descent — the instrument gets a fast mode.

Cost: ~$10 for the bank. Shares E1's anchor scores.

---

## E3 — Intrinsic predictors: can thickness be read off the definition?

**Design.** For every artifact in every E1 descent sequence (all in the registry with
lineage + complexity):
1. **Clause-decidability profile:** LLM-tag each rubric clause on the 6-level ladder
   (string-match < syntactic < executable < shallow-semantic < deep-semantic <
   community-membership); hand-validate ~50 clauses (per `feedback_validate_before_scaling`).
2. **Structural complexity trajectory:** instruction tokens, clause count, AST size along
   each descent — does mechanization *measurably* shift profiles down-ladder?
3. **Prediction:** regress B\*(m) on the SEED rubric's profile (Cox model, censored).

**Deliverable if it works:** a **static thickness estimator** — paste in a rubric, get a
predicted minimum judge tier. Immediately useful to anyone provisioning judge models, and
the practical payoff that makes the paper more than a measurement curiosity.

Cost: ~$5 (tagging) + zero new runs (pure registry analysis).

---

## E4 — The validity battery for the instrument itself

*This is what makes it a measurement paper rather than a benchmark paper.*

| Validity type | Test |
|---|---|
| Known-answer | E0 (planted ladder recovered) |
| Reliability | E1-H3: floor invariance across seeds / generator families / corpus halves |
| Convergent | floor vs. (i) blinded human tacitness ranking of the 24 criteria (you + 2 colleagues, ~30 min each); (ii) inter-implementation agreement (scorecard measure 7); (iii) judge−code gap (B−A) from existing v2 codegen programs where available |
| Discriminant | floor must NOT be explained by rubric length, topic frequency in pretraining (proxy: term frequency), or applicability rate — regression with these as competitors |
| Anchor-robustness | re-run 6 criteria with a different anchor family (e.g. GPT-class via OpenRouter); floors should move ≤ 1 tier |

The discriminant tests are the reviewer-proofing: "your floor just measures rubric length /
concept frequency" must be answerable with a number.

Cost: ~$30 + ~1.5 expert-hours.

---

## E5 — Application: the thin/thick composition of communities' norm systems

Run the E1+E2 pipeline on 2–3 additional tasks (peer-review, creative-writing — reusing
cells-DB judge scores where possible) and compare **floor distributions across
communities**: is peer review's criteria bank thicker than competitive code's? Do aesthetic
criteria floor uniformly high across communities while procedural ones floor at code? This
is the bridge back to the norms paper: the KM curve per community = the measured
articulable fraction; its shape = the community's position on the rules-vs-standards
spectrum ([[project_thin_thick_rules_philosophy]], [[project_tacitness_two_layers]]).

Cost: ~2× E1 per added task; gate on E1 results.

---

## E6 — The downstream variable: payoff test + Goodhart gap

*y (judgement/acceptance/upvotes) is the dependent variable of the science, never the
objective of an instrument. Four channels: PROPOSE (y allowed — residuals/contrasts say
where metrics are missing), CALIBRATE (y forbidden — the invariant), SELECT/WEIGHT (y
allowed over frozen instruments, holdout discipline), EVALUATE (confirm split only).
"Optimize the lens y-free; aim the telescope with y." Foundation: construct validity vs
empirical keying (Cronbach & Meehl 1955; criterion contamination); empirically, our y's
are corrupted criteria (news_homepages = layout not engagement; homepage confounds ~0.2
AUC; LitBench taste-laundering) — optimizing instruments against them bakes corruption in.*

**E6a — Payoff test (registered, falsifiable bet of the whole fidelity program).**
H: fidelity-optimized metrics (zero label access) predict y BETTER than their seeds on the
confirm split, individually (per-metric AUC delta) and as a bank (joint model). Systematic
failure ⇒ the fidelity battery optimizes something the community doesn't use — find out
early. Uses E1's accepted artifacts + existing labeled datasets; cost ≈ scoring passes
only.

**E6b — Goodhart gap (deliberate y-optimization as a quarantined measurement).**
For each criterion, optimize a SECOND rubric copy directly against y (same GEPA loop,
objective = label fit; no fidelity terms). Compare tracks on confirm:
1. **Confound share** = AUC(y-track) − AUC(fidelity-track) — how much of the criterion's
   apparent predictive power is carried by things other than the stated construct.
2. Run the full fidelity battery on the y-track copies: they should FAIL counterfactuals
   and drift under reconstruction — and **reconstruction names what they drifted toward**:
   label-fitting becomes a discovery procedure for unstated norms/confounds, articulated
   by our own instrument.
3. Promotion path: a y-track discovery that passes the fidelity battery as a NEW construct
   enters the bank via the propose channel (front door) — y-optimization as proposal,
   never calibration.
Quarantine rules: y-track artifacts live under a separate registry namespace
(`metrics/<id>/ytrack/`), are never merged into HEAD, and never feed the reviser of the
fidelity track. Cost: ≈ one extra E1-scale descent per criterion at ONE tier (8B), ~$10–20
bank-wide with the sk3/subagent architecture.

Novelty note: rubric-reward work (Rubrics-as-Rewards etc.) optimizes rubrics against
outcomes as the GOAL; using the fidelity↔label-fit divergence as a per-criterion
confound measurement, with reconstruction articulating the drift, is not in the prior-art
sweep's findings.

---

## E7 — Bounded articulability: the sandwich + the Chinchilla null fit (added 2026-06-12)

*Formalization §6 / T7. The design question: given that (i) no executor-free bound exists,
(ii) fitted asymptotes are indefensible as bounds, (iii) the optimizer only certifies one
side, and (iv) judge noise attenuates everything — what is the BEST bounded estimate of
articulability we can construct? Answer: a measured two-sided bracket per (criterion ×
tier × difficulty stratum), with a Chinchilla-style separable fit used to structure the
interior of the bracket, never to extend it.*

**Estimand.** For each criterion m, tier E, hard/backbone stratum s: the frontier
fidelity\*(m; E, s) is reported as an **interval [F_lo, F_hi]**, not a point:

- **F_lo — constructive lower bound.** Best exhibited articulation, *re-evaluated on a
  fresh frozen holdout it never touched* (kills the T5 winner's curse; the artifact is
  exhibited in the registry, so the bound is a certificate, not an estimate).
- **F_hi — scaling-free upper bound.** The disattenuation ceiling at tier E: no
  articulation can agree with the anchor construct beyond what (a) the tier's own
  test–retest reliability and (b) the anchor's inter-implementation agreement permit.
  F_hi = √(ρ_E,panel · ρ_anchor) under the D-study-chosen panel allocation (twin-ceiling
  logic; both factors measured, label-free, no scaling involved). When F_lo ≈ F_hi the
  criterion is articulated at that tier *and the instrument says it cannot do better* —
  the strongest closure statement available without an executor-free floor.

**Grid (the Chinchilla N×D analog).** Nested instruction-token caps L ∈ {64, 256, 1024}
× the 4 E1 tiers × 3 optimizer seeds; frontier (best accepted) per cell; difficulty as a
conditioning stratum per the triad layer (frontier estimated on the hard stratum and the
backbone separately — easy-item agreement is where lawful-looking-but-wrong rubrics hide).
x-axes: L = tokens actually used (≤ cap); C = per-criterion empirical capability of the
tier (Relative Scaling Laws rule — never nominal parameter count).

**Discipline imported from the scaling literature's failure record:**
1. **Isotone projection first (T1).** Project each measured frontier onto monotone-in-L;
   the projection residual is the free estimate of GEPA's optimizer slack, reported per
   cell (instrument calibration at zero cost).
2. **Optimizer-budget convergence.** Double rounds/mutations on a subsample; F_lo must
   plateau or we are measuring the optimizer, not the criterion (06-11 amendment, now a
   gate for E7 cells).
3. **Multi-form fits.** Fit power, exponential, and logistic decays to every frontier
   curve; any conclusion that survives under only one functional form is flagged.
   Power-vs-exponential is itself the T2 tail diagnostic.
4. **The fit is interior-only.** The separable null
   `fidelity\*(m; L, C) = (1 − τ_m) − A_m·L^(−α_m) − B_m·C^(−β_m) − I_m(L, C)`
   is fit by nonlinear least squares with paired-bootstrap CIs over eval items, with the
   fitted surface **constrained to lie inside the measured brackets**; τ̂_m is reported as
   a descriptive column with CI in every scorecard table and is never cited as a bound.

**Three estimators, required to agree (the Chinchilla three-approaches echo).** Chinchilla
defended its result by triangulating three estimation approaches; ours: (1) the bracket
endpoints at maximal tested budgets, (2) the parametric τ̂_m, (3) the KM/censoring
classification {articulated, climbing, resistant} from §3 survival analysis. The
per-criterion verdict is issued only where all three agree; disagreement is itself
reported (it localizes which assumption failed — optimizer slack, form sensitivity, or
censoring).

**Readouts & hypotheses (T7a–c).**
1. **T7a (thin = law, thick = residual):** lack-of-fit I_m (residual deviance of the
   separable fit vs. the saturated cell-means model, sign-checked against the E2
   interaction) rank-correlates with independent thickness measures (words_share from the
   triad G-study; floor B\*(m)). Kill condition: no association ⇒ the fit is demoted to
   descriptive.
2. **T7b (fast mode):** I_m predicts B\*(m) at least as well as the E2 2×3 interaction —
   if yes, the grid fit becomes the backbone of the cheap floor estimator.
3. **T7c (shared mechanism):** fitted α_m agrees with the parent task's norm-frequency
   Zipf tail exponent within CI (T2's three-way check at metric level; norm-cluster sizes
   already extracted).
4. **Exchange rate, where licensed:** for criteria with I_m ≈ 0, the words↔capability
   substitution rate and iso-fidelity contours (design §3's "articulation-optimal budget
   allocation") — reported ONLY for separable criteria; for thick criteria the contour
   does not exist, and saying so is the result.

**Classification rule (replaces any asymptote argument).** A criterion is declared
*resistant at tier E* iff F_hi (not the fitted asymptote) sits significantly below τ AND
the bracket midpoint's slope in every axis is ≈ 0 at max budget; *articulated* iff
F_lo ≥ τ on the fresh holdout; otherwise *climbing* (right-censored). Tacitness language
attaches only to resistant-everywhere + dense-learnable criteria, as before.

**Cost.** Pure analysis over the E1 registry once E1 runs the 3-cap grid (~3× E1 run
count; ~$90–200 OpenRouter full-bank, ~$30–60 with sk3 offline-batch judging). If cost
binds, densify L only on the 12 criteria at the thin/thick extremes of the E2 quick
screen and run the rest at one cap.

**Gate:** E7 verdicts are issued only for cells passing the reliability floor and the
optimizer-convergence check; cells failing either are reported as instrument-limited,
never as thick.

### E7 integration amendments (2026-06-13) — observational scaling + annotation lit

*(Formalization §7, U1–U6. Operationalizes the scaling-paper + 171-paper annotation-sweep findings
into the E7 procedure. Reference: `2026-06-13__llm-annotation-litreview.md`.)*

1. **Continuous fidelity readout (U3 / Schaeffer 2406.04391).** Replace the hard 0/0.5/1 agreement
   in the frontier objective with the judge's **logprob / probability-over-label** (or a Brier-style
   continuous agreement): the hard label is the maximally-decorrelating node of the metric chain, so
   hard-label frontiers look noisier than the construct warrants. Keep the **applicability 0.5/NA
   channel separate** (= Schaeffer's incorrect-choice mass). Estimate frontiers **per difficulty
   stratum, never pooled** (nonlinearity does not average out).

2. **Null = model-selection contest (U2).** The "multi-form fits" discipline (E7 item 3) now
   explicitly pits the additive separable null against the **Montgomery multiplicative-complementary
   form** `P=[1−e^{−A(L/Lᶜ)^α}]·[1−e^{−B(C/Cᶜ)^β}]·σ(L−L_ctx)` (2510.14919). T7a refined: thin ⇔
   additive fits (`I_m≈0`); thick ⇔ residual collapses onto the multiplicative cross-term. **IFScale
   (Jaroslawicz 2025)** acc≈(instr-acc)^n is the empirical instruction-budget precedent.

3. **L-axis confound controls — MANDATORY before any length claim.** (a) **Pipal et al. 2026 (arXiv:2604.03684):** hold the
   *number of distinct coding schemes/criteria* fixed while varying token count — and vary L *within
   a fixed criterion set* (more words for the SAME construct), never by adding criteria, else we
   measure their scheme-count k, not budget — the length effect
   must survive a length-matched, scheme-count-matched control, else it is multi-scheme load, not
   budget. (b) **Liu 2023 (lost-in-middle):** randomize clause *order* within the rubric across
   passes and report position-robust frontiers (mid-rubric criteria are under-used). (c) **U6:**
   report L in **tokenizer-invariant units** (clauses/quanta or bytes), not raw judge tokens, so
   `α_m` compares across tiers.

4. **Capability axis (U1).** C stays per-criterion *empirical* capability (Relative-Scaling
   amendment); now additionally fittable as IRT ability θ from the long table (item 5). **Engels
   2025** Double-ReLU = importable capability-gap shape; **Krumdick 2025** = articulability is
   upper-bounded by judge competence ⇒ report it as a function of capability, with a written rubric
   as a capability substitute.

5. **IRSL estimator option (U4) — OPEN, do not overstate.** The §9 triad/D-study ceiling *could* gain
   an IRT sibling: Beta-IRT over `(judge × item × rubric-version × pass)` → θ (capability), z
   (difficulty), + a **rubric-informativeness** factor κ_r. `O(M+N+R)`, ~50 items/cell. **But κ_r is
   OURS, not in any cited paper:** IRSL (2606.07616) is unidimensional with no rubric/judge parameter;
   Choi (2602.00521) makes θ the *judged item's* quality, fits each judge independently, and treats
   scale only descriptively — so **judge-ability-vs-scale is still open after Choi.** κ_r↔z
   identifiability is now **verified as textbook** — the d_j·κ_r product is the 2PL scale/unit
   indeterminacy (Noventa 2024; San Martín 2015), identifiable only by anchoring a reference condition.
   Note the fork: τ_r (severity) = squarely standard (Many-Facet Rasch, Linacre 1989) but only a
   leniency shift; κ_r (discrimination) = the articulability-relevant one but niche (nonuniform DIF /
   Jin & Eckes 2022) and harder. Gate on a long-table prototype under anchoring before relying on this.

6. **Phrasing-distribution requirement (U5 + Baumann threat).** T5's winner's curse becomes a hard
   rule: report articulability as a **distribution/interval over rubric phrasings** (Sclar
   FormatSpread; Polo PromptEval), not one optimized prompt; correct any E6 downstream y-estimate with
   **PPI (Angelopoulos 2023) / DSL (Egami 2023)**. **Baumann 2025** (LLM-hacking: ~31% of conclusions
   wrong, prompt KIND <1% of conclusion-correctness variance, 100 human > 100K LLM) is the
   reviewer-proofing citation. Calibration target: the seed-to-frontier gap vs capability should trace
   the inverse-scaling curve (PRIME / black-box-opt).

**Must-cite collisions (add to related work):** Ruan 2405.10938; Schaeffer 2406.04391; Truong/Koyejo
IRSL 2606.07616; Montgomery 2510.14919; Jaroslawicz IFScale 2025; Engels 2025; Krumdick 2025
("No Free Labels"); Guerdan 2025 (judge validation under rating indeterminacy — gold mis-ranks judges
≤34%); Choi 2026 (judge IRT); Baumann 2025 (LLM hacking).

## Amendments (2026-06-11, after GEPA review + Relative Scaling Laws)

**Optimizer semantics — one-sided certificates.** Any optimizer lower-bounds the true
frontier (sup over articulations), so measured floors are biased TOWARD thickness. State
explicitly: "thin" verdicts are constructive certificates (the artifact is exhibited);
"thick" verdicts are defeasible (censored at the tested search budget — which includes
optimizer effort). New check in E0+E1: **optimizer-budget convergence** — double
rounds/mutations on a subsample; floors must plateau, else we are measuring the optimizer.

**GEPA fidelity.** Our loop is GEPA-style (reflective mutation, rich textual failure
artifacts), not full GEPA (arXiv 2507.19457). Adopt **per-instance Pareto candidate
retention** for E1 (keeps candidates best on different items alive — the missing feature
behind the observed MECHANIZE monoculture). MECHANIZE/DECOMPOSE routing is OUR layer, not
GEPA's — GEPA has no mechanization or weak-executor concept; the thin/thick semantics are
entirely this method's contribution and must be validated here (E0).

**X-axis under heterogeneous scaling** (Relative Scaling Laws, arXiv 2510.24626: scaling
is not a universal equalizer; per-distribution gaps converge/persist/diverge). Plot
frontiers against **per-criterion empirical capability** (each tier's agreement-with-anchor
under the RICH rubric) rather than nominal size; nominal-tier plots as secondary. Crossing
frontier curves across criteria are expected findings, not anomalies. H1 monotonicity is
tested, never assumed. Drop any asymptote-extrapolation defense ("no stronger judge would
close this") — gap claims rest on the label-free sandwich (dense = achieved lower bound;
twin-ceiling = scaling-free upper bound).

**Cost architecture (closed-source enters at O(bank), not O(grid)):** anchor scores cached
once per criterion (Sonnet, ~$5–10 for the bank); bulk judge calls on sk3 vLLM offline
batch (~$0) or cheap open tiers; **reviser/reflection via Max-plan Claude subagents
($0 marginal, ~50 calls/run, quality-sensitive — the right place for Claude)**; cross-family
acceptance in cents. Revised total for E0–E4: **~$20–50**. Closed judge tiers (Haiku/Sonnet
as x-axis points): optional, one frontier point per criterion, single seed — bounded
~$10–20.

## Order, gates, and risks

```
Pilot E1 (6×3×1, ~$10)  →  E0 known-answer (~$30)  →  full E1 w/ 3-cap L grid (~$90–200)
        →  E2 + E3 + E7 fit (share the E1 grid, ~$15)  →  E4 battery (~$30)  →  E5 (gated)
```

- **Gate 1:** pilot shows per-tier optimization beats seed at ≥2 tiers (else the optimizer
  is the bottleneck — fix it before measuring anything).
- **Gate 2:** E0 ordering recovered (else the instrument is broken; do not proceed).
- **Gate 3:** E1-H3 floor stability (else no instrument claim; salvage = report frontier
  curves as engineering result only).
- **Known risks:** ρ per tier must clear the reliability floor or "agreement" is
  noise-matching (already enforced); anchor quirk-overfitting (frozen holdout + fresh CF,
  already wired); 1B tier may be too unreliable to optimize at temp 0.7 — fall back to
  k=3 ensembled scoring at that tier and report the Spearman–Brown-adjusted floor.
- **Total budget, all of E0–E4:** ≈ $100–250 OpenRouter (less with sk3 vLLM for bulk
  judging), ~2 expert-hours, no GPUs required beyond optional sk3 batch scoring.

## What is genuinely new, in one table

| Experiment | Nearest prior art | The delta |
|---|---|---|
| E0 | — (no tacitness instrument has known-answer validation) | planted-thickness ladder |
| E1 | GER-Eval (fixed rubric, transfer fails); Alur (label-needed residual) | per-tier *optimized* frontier, label-free floor, KM over a bank |
| E2 | SILICON (confounds to remove); Codebook-LLMs (one tier) | the words/reader/interaction coordinate, tied to the floor |
| E3 | holistic/analytic/atomic taxonomy (defined, never measured) | decidability profile *predicting* the measured floor |
| E4 | — | the validity battery itself |
| E5 | econ codifiability-by-annotation | codifiability-by-attempt, compared across communities |
| E7 | Chinchilla/Henighan irreducible-loss fits (model-free entropy floor) | no executor-free floor claimed: two-sided measured bracket; separability lack-of-fit as a *construct property* (thickness); asymptote demoted to descriptive |
| E7·null (2026-06-13) | Chinchilla additive law; Montgomery 2510.14919 multiplicative context law; IFScale (instr-budget on verifiable tasks) | thin/thick as additive-vs-multiplicative *model selection*; the only controlled instruction-budget articulability curve on no-gold subjective tasks, scheme-/position-confound-controlled (Pipal, Liu) |
| E7·estimand (2026-06-13) | PPI/DSL (correct downstream params); Choi 2026 / Guerdan 2025 (validate judge choice) | a per-metric articulability coefficient with a valid no-gold (PPI) CI reported as a distribution over phrasings; capability-controlled per-construct prompt-budget exponent α_m |


## References (auto-verified BibTeX, 2026-06-15)

> Citations below were extracted from this document and web-verified by an automated fact-check pass (search → fetch → retrieve resolvable id), with the attributed claim checked against the located paper. 32 entries; 1 also passed an independent second-pass audit (the rest were verified once — the audit pass was cut off by a quota limit, not by a failure). Entries are real located works; do not treat as hand-checked. See "needs manual review" below for 0 citations whose attributed claim the source paper appears to **contradict** and 1 unlocatable shorthands.

```bibtex
@misc{agrawal2025gepa,
  title        = {GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author       = {Agrawal, Lakshya A. and Tan, Shangyin and Soylu, Dilara and Ziems, Noah and Khare, Rishi and Opsahl-Ong, Krista and Singhvi, Arnav and Shandilya, Herumb and Ryan, Michael J. and Jiang, Meng and Potts, Christopher and Sen, Koushik and Dimakis, Alexandros G. and Stoica, Ion and Klein, Dan and Zaharia, Matei and Khattab, Omar},
  year         = {2025},
  eprint       = {2507.19457},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2507.19457}
}

@inproceedings{alur2024human,
  title     = {Human Expertise in Algorithmic Prediction},
  author    = {Alur, Rohan and Raghavan, Manish and Shah, Devavrat},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS 2024)},
  year      = {2024},
  eprint    = {2402.00793},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2402.00793}
}

@article{angelopoulos2023prediction,
  title={Prediction-powered inference},
  author={Angelopoulos, Anastasios N. and Bates, Stephen and Fannjiang, Clara and Jordan, Michael I. and Zrnic, Tijana},
  journal={Science},
  volume={382},
  number={6671},
  pages={669--674},
  year={2023},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/science.adi6000}
}

@misc{baumann2025large,
  title={Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation},
  author={Baumann, Joachim and R{\"o}ttger, Paul and Urman, Aleksandra and Wendsj{\"o}, Albert and Plaza-del-Arco, Flor Miriam and Gruber, Johannes B. and Hovy, Dirk},
  year={2025},
  eprint={2509.08825},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2509.08825}
}

@misc{cheng2024err,
  title={To Err Is Human; To Annotate, SILICON? Toward Robust Reproducibility in LLM Annotation},
  author={Cheng, Xiang and Mayya, Raveesh and Sedoc, Jo\~{a}o},
  year={2024},
  eprint={2412.14461},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  howpublished={\url{https://arxiv.org/abs/2412.14461}}
}

@misc{choi2026diagnosing,
  title        = {Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory},
  author       = {Choi, Junhyuk and Park, Sohhyung and Cho, Chanhee and Park, Hyeonchu and Kim, Bugeun},
  year         = {2026},
  eprint       = {2602.00521},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2602.00521}
}

@inproceedings{choi2026diagnosinga,
  title={Diagnosing the Reliability of {LLM}-as-a-Judge via Item Response Theory},
  author={Choi, Junhyuk and Park, Sohhyung and Cho, Chanhee and Park, Hyeonchu and Kim, Bugeun},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026},
  note={arXiv:2602.00521},
  eprint={2602.00521},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{cronbach1955construct,
  author  = {Cronbach, Lee J. and Meehl, Paul E.},
  title   = {Construct Validity in Psychological Tests},
  journal = {Psychological Bulletin},
  year    = {1955},
  volume  = {52},
  number  = {4},
  pages   = {281--302},
  doi     = {10.1037/h0040957}
}

@book{daston2022rules,
  author    = {Daston, Lorraine},
  title     = {Rules: A Short History of What We Live By},
  publisher = {Princeton University Press},
  series    = {The Lawrence Stone Lectures},
  year      = {2022},
  isbn      = {9780691156989},
  address   = {Princeton, NJ}
}

@inproceedings{egami2023using,
  title={Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models},
  author={Egami, Naoki and Hinck, Musashi and Stewart, Brandon M. and Wei, Hanying},
  booktitle={Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year={2023},
  eprint={2306.04746},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{engels2025scaling,
  title={Scaling Laws For Scalable Oversight},
  author={Engels, Joshua and Baek, David D. and Kantamneni, Subhash and Tegmark, Max},
  year={2025},
  eprint={2504.18530},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2504.18530}
}

@inproceedings{guerdan2025validating,
  title     = {Validating {LLM}-as-a-Judge Systems under Rating Indeterminacy},
  author    = {Guerdan, Luke and Barocas, Solon and Holstein, Kenneth and Wallach, Hanna and Wu, Zhiwei Steven and Chouldechova, Alexandra},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  eprint    = {2503.05965},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2503.05965}
}

@misc{gunjal2025rubrics,
  title={Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains},
  author={Gunjal, Anisha and Wang, Anthony and Lau, Elaine and Nath, Vaskar and He, Yunzhong and Liu, Bing and Hendryx, Sean},
  year={2025},
  eprint={2507.17746},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.17746}
}

@article{halterman2026codebook,
  title={Codebook {LLM}s: Evaluating {LLM}s as Measurement Tools for Political Science Concepts},
  author={Halterman, Andrew and Keith, Katherine A.},
  journal={Political Analysis},
  volume={34},
  number={2},
  pages={188--204},
  year={2026},
  publisher={Cambridge University Press},
  doi={10.1017/pan.2025.10017}
}

@misc{held2025relative,
  title        = {Relative Scaling Laws for LLMs},
  author       = {Held, William and Hall, David and Liang, Percy and Yang, Diyi},
  year         = {2025},
  eprint       = {2510.24626},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2510.24626}
}

@misc{jaroslawicz2025how,
  title        = {How Many Instructions Can LLMs Follow at Once?},
  author       = {Jaroslawicz, Daniel and Whiting, Brendan and Shah, Parth and Maamari, Karime},
  year         = {2025},
  eprint       = {2507.11538},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2507.11538}
}

@article{jin2022detecting,
  title={Detecting Differential Rater Functioning in Severity and Centrality: The Dual DRF Facets Model},
  author={Jin, Kuan-Yu and Eckes, Thomas},
  journal={Educational and Psychological Measurement},
  volume={82},
  number={4},
  pages={757--781},
  year={2022},
  doi={10.1177/00131644211043207},
  publisher={SAGE Publications}
}

@misc{krumdick2025no,
  title        = {No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding},
  author       = {Krumdick, Michael and Lovering, Charles and Reddy, Varshini and Ebner, Seth and Tanner, Chris},
  year         = {2025},
  eprint       = {2503.05061},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2503.05061}
}

@book{linacre1989many,
  author    = {Linacre, John Michael},
  title     = {Many-Facet Rasch Measurement},
  publisher = {MESA Press},
  address   = {Chicago},
  year      = {1989},
  isbn      = {0-941938-02-6}
}

@article{liu2023lost,
  author    = {Nelson F. Liu and Kevin Lin and John Hewitt and Ashwin Paranjape and Michele Bevilacqua and Fabio Petroni and Percy Liang},
  title     = {Lost in the Middle: How Language Models Use Long Contexts},
  journal   = {Transactions of the Association for Computational Linguistics},
  volume    = {12},
  pages     = {157--173},
  year      = {2024},
  doi       = {10.1162/tacl_a_00638},
  note      = {arXiv:2307.03172}
}

@article{lumley2002assessment,
  title   = {Assessment criteria in a large-scale writing test: what do they really mean to the raters?},
  author  = {Lumley, Tom},
  journal = {Language Testing},
  volume  = {19},
  number  = {3},
  pages   = {246--276},
  year    = {2002},
  doi     = {10.1191/0265532202lt230oa}
}

@misc{montgomery2025predicting,
  title        = {Predicting Task Performance with Context-aware Scaling Laws},
  author       = {Montgomery, Kyle and Park, David and Tu, Jianhong and Bendersky, Michael and Gunel, Beliz and Song, Dawn and Wang, Chenguang},
  year         = {2025},
  eprint       = {2510.14919},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.48550/arXiv.2510.14919},
  url          = {https://arxiv.org/abs/2510.14919}
}

@article{noventa2024identifiability,
  title   = {On the Identifiability of 3- and 4-Parameter Item Response Theory Models From the Perspective of Knowledge Space Theory},
  author  = {Noventa, Stefano and Ye, Sangbeak and Kelava, Augustin and Spoto, Andrea},
  journal = {Psychometrika},
  volume  = {89},
  number  = {2},
  pages   = {486--516},
  year    = {2024},
  doi     = {10.1007/s11336-024-09950-z}
}

@misc{pipal2026researchers,
  title={Researchers waste 80\% of LLM annotation costs by classifying one text at a time},
  author={Pipal, Christian and Vogel, Eva-Maria and Wack, Morgan and Esser, Frank},
  year={2026},
  eprint={2604.03684},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2604.03684},
  url={https://arxiv.org/abs/2604.03684}
}

@inproceedings{polo2024efficient,
  title={Efficient multi-prompt evaluation of {LLM}s},
  author={Maia Polo, Felipe and Xu, Ronald and Weber, Lucas and Silva, M{\'i}rian and Bhardwaj, Onkar and Choshen, Leshem and de Oliveira, Allysson Flavio Melo and Sun, Yuekai and Yurochkin, Mikhail},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  eprint={2405.17202},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2405.17202}
}

@inproceedings{ruan2024observational,
  title={Observational Scaling Laws and the Predictability of Language Model Performance},
  author={Ruan, Yangjun and Maddison, Chris J. and Hashimoto, Tatsunori},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  eprint={2405.10938},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2405.10938}
}

@article{sanmartin2015on,
  author  = {San Mart{\'i}n, Ernesto and Gonz{\'a}lez, Jorge and Tuerlinckx, Francis},
  title   = {On the Unidentifiability of the Fixed-Effects {3PL} Model},
  journal = {Psychometrika},
  year    = {2015},
  volume  = {80},
  number  = {2},
  pages   = {450--467},
  doi     = {10.1007/s11336-014-9404-2}
}

@misc{schaeffer2024why,
  title={Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?},
  author={Rylan Schaeffer and Hailey Schoelkopf and Brando Miranda and Gabriel Mukobi and Varun Madan and Adam Ibrahim and Herbie Bradley and Stella Biderman and Sanmi Koyejo},
  year={2024},
  eprint={2406.04391},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2406.04391}
}

@inproceedings{sclar2024quantifying,
  title={Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting},
  author={Sclar, Melanie and Choi, Yejin and Tsvetkov, Yulia and Suhr, Alane},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  eprint={2310.11324},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@misc{siro2026learning,
  title        = {Learning to Judge: LLMs Designing and Applying Evaluation Rubrics},
  author       = {Siro, Clemencia and Aliannejadi, Pourya and Aliannejadi, Mohammad},
  year         = {2026},
  eprint       = {2602.08672},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  note         = {Introduces GER-Eval (Generating Evaluation Rubrics for Evaluation)},
  url          = {https://arxiv.org/abs/2602.08672}
}

@misc{truong2026item,
  title        = {Item Response Scaling Laws: A Measurement Theory Approach for Efficient and Generalizable Neural Scaling Estimation},
  author       = {Truong, Sang and Tu, Yuheng and Schaeffer, Rylan and Koyejo, Sanmi},
  year         = {2026},
  eprint       = {2606.07616},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2606.07616}
}

@article{zander1995knowledge,
  author  = {Zander, Udo and Kogut, Bruce},
  title   = {Knowledge and the Speed of the Transfer and Imitation of Organizational Capabilities: An Empirical Test},
  journal = {Organization Science},
  year    = {1995},
  volume  = {6},
  number  = {1},
  pages   = {76--92},
  doi     = {10.1287/orsc.6.1.76},
  publisher = {INFORMS}
}

```

### Citations needing manual review

**Could not be located (1)** — verify the citation exists or correct the shorthand:

- Schaal — Could not locate any published work authored by "Schaal" on codifiability annotation/measurement after a genuine multi-angle search. Searches covered:

**Partial claim-match (12)** — paper located, attributed claim only partly supported; spot-check before relying on the exact number/wording:

- `alur2024human`; `angelopoulos2023prediction`; `choi2026diagnosing`; `choi2026diagnosinga`; `cronbach1955construct`; `guerdan2025validating`; `held2025relative`; `jaroslawicz2025how`; `jin2022detecting`; `liu2023lost`; `noventa2024identifiability`; `siro2026learning`
