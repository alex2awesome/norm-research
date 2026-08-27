# Paper split decision + OSL preregistration plan

*2026-07-20. Decision record from the Koyejo/Ho feedback session. Status: recommendation
made, awaiting Alex's final call.*

## The two proposed splits

**Alex's split**
- Paper #1: Preference articulability gaps (baseline methods) + heterogeneity
- Paper #2: Prompt optimization

**Sanmi's split**
- Paper #1: Prompt optimization (+ OSL for prompt articulability)
- Paper #2: Preference articulability with optimized prompts

## Recommendation: Sanmi's ordering, with two modifications

### Why (the decisive argument is our own data)

The N&C GEPA result is the load-bearing evidence. Fidelity-optimizing the rubric bank
*lowered* predictive AUC (outcome .592 -> .578; agree within-docket .558 -> .501) because part
of the baseline bank's predictive power was **unarticulated judge residue**, not the stated
constructs (see `project_nc_vat_run`).

Consequence: a gap measured with baseline prompts is not merely noisier, it is biased in a
**known direction** — baseline-A overstates the articulable share, so the articulability gap
is **understated**. Alex's Paper #1 would therefore headline a quantity we already know to be
biased, and the correction for that bias *is* the Paper #2 machinery.

Second argument: the standard referee question is "how do you know the gap isn't just a bad
prompt?" Our only real answers are the certified prompt-space bounds (DPI fixed-target cap /
T_soft bracket, CR-3 missing-mass intervals) and the GEPA null/negative results — all of which
sit in Paper #2 under Alex's split. So Alex's Paper #1 must either import half of Paper #2
defensively (blurring the split) or stand exposed.

Third: Sanmi's ordering matches his own standing instruction that every claim be relativized to
the (executor, rubric) pair. Paper #1 *is* the study of that pair — prompt axis (PO
certificates) x executor axis (OSL). The certification angle is also the niche neither Hypobench
nor Autometrics occupies, which resolves his smaller "2a in-or-out" question in favour of **in,
framed as certification**.

Fourth (Dan's advice): Paper #1 as "a set of discoveries about instrument behaviour" is
TMLR-shaped; Paper #2 is the broad-audience substantive paper and benefits from going second
with a hardened instrument.

### Modification 1 — serialize the submissions, NOT the work

The risk of Sanmi's ordering is calendar risk: the PO line is the most retraction-prone part of
the program (MCQ instrument dead, live loop v1 retracted, v13/v14 still in flight). Gating the
substantive paper on it is dangerous.

Mitigation: because the baseline->optimized delta on the A rows is empirically small-or-negative,
Paper #2's analyses can run **now** on baseline banks, with the fidelity-optimized arm added as
a robustness column when it lands. Draft both in parallel; Paper #2 cites Paper #1's bounds.

### Modification 2 — heterogeneity stays in Paper #2

Heterogeneity is a Y-side measurement question (rater models, noise ceilings; Sanmi items 5/6,
Dan's IRT point), orthogonal to the prompt and executor axes. It pairs naturally with the gaps:
the same rater model that measures heterogeneity supplies the per-domain noise ceiling that
disattenuates the gap. Alex's split had this pairing right — carry it into Sanmi's ordering.

### Caveat to state to both mentors

Under the standing reconstruction-only rule, "optimized prompts" in Paper #2 can only ever mean
**fidelity-optimized (label-blind)**, never AUC-optimized. Paper #2's framing must therefore be
"gaps under construct-faithful instruments, plus certified bounds over all prompts" — NOT "gaps
under best-possible prompts," which is unattainable without label leakage.

## How Tatsu's OSL paper preregistered (and what we copy)

Source: Ruan, Maddison & Hashimoto, *Observational Scaling Laws and the Predictability of
Language Model Performance*, NeurIPS 2024 (arXiv:2405.10938; code github.com/ryoungj/ObsScaling).
Read from the camera-ready PDF, sections 4 / D.1.1 / E.9.

Mechanism is lightweight — no OSF registry, just public commitment via arXiv versioning:

1. **Freeze the fitted forms publicly at v1.** Quote: "In the initial release of our paper
   (May 2024), we have preregistered our scaling predictions for future models (see
   preregistered functional forms in Appx. E.9) and committed to updating the manuscript on
   ArXiv with our prediction results after 4 months." Appendix E.9 / Table E.2 publishes the
   **literal numeric coefficients** of every fitted law, e.g.
   `phi^-1(Y,1.00) = 6.74*PC1 - 3.22*PC2 - 1.37*PC3 - 4.93`, so nothing can be refit post hoc.
2. **Dated commitment window.** Four months. They then collected every qualifying new release
   (20 models, 8 families, May -> Sept 1 2024, incl. Llama-3.1-405B and Qwen2-72B) as an
   untouchable test set and reported how the frozen curves did (Fig. 4, Appx. E.3).
3. **Backed by in-paper holdout discipline.** FLOPs cutoff at Llama-2-7B (47 train / 30 test
   models); all preprocessing (PCA imputation) fit on train only; MSE on normalized targets;
   robustness checks over alternative holdout strategies.

### Mapping onto our OSL / z x a line

The internal discipline already exists — the 70B name-sufficiency prereg (frozen sha 62e4b3f0)
was exactly this pattern, done privately. The move is to make it public.

- **Freeze at submission**: the declared capability axis (anchor-battery definition + sha), the
  frozen metric slates, and the fitted per-(metric x family) forms with literal coefficients —
  beta-hat, z*-hat, sigmoid parameters — in a Table-E.2-style appendix.
- **What to predict**: articulability onsets (z*-hat, N*) at named not-yet-run rungs and named
  future model releases. **Family-conditional only** — the beta sign-flip result (compressed-quotable:
  Llama metric-B beta>+.25 vs Qwen metric-A beta=-.74) means a pooled forecast is falsified before
  it is made; Koyejo pathology 5.3 forbids fitted infinite-ceilings regardless. Forecast crossing
  points, never asymptotes.
- **Readout to preregister**: ordinal / rank forms, not literal persistence. Our own internal dry
  run says which shape survives contact with new rungs: persistence 34/51 FALSIFIED while the
  frozen deficit-*ranking* stayed predictive (rank-AUC .689, perm p=.008). Preregister the claim
  shape the dry run says is robust.
- **Two disciplines we learned independently**: their train-only preprocessing == our
  freeze-before-eval mandate; their clean release-date cutoff == the cw/humor contamination
  incident (70B grids predating the Jul-5 freeze). The public prereg needs an explicit
  "no target model was scored before date X" attestation.
- **Timing synergy**: the ~4-month resolution window is about one conference cycle, which fits
  the recommended ordering — Paper #1 v1 posts with frozen forecasts, the validated-forecast
  update lands around Paper #2 submission, so Paper #2 cites a *validated* instrument rather
  than a promised one.

## Open decisions for Alex

1. Final call on the split (recommendation above).
2. Does the measurement-model-of-Y workstream (peer-review IRT + patents art units) count as a
   new measurement target needing formal sign-off? My read: yes — it introduces IRT-latent y's.
3. Family budget for the cross-family recovery matrix (GLM quota is the binding constraint).
4. Per-subgroup norm recovery: mine fresh banks (honest, expensive) vs reweight existing.

Related: `project_paper_macro_structure`, `project_osl_executor_scaling`,
`project_momega_audit_bracket`, `project_nc_vat_run`, `project_mi_vs_silver_norms`.

---

## 2026-07-21 — FOUR-PAPER SPLIT DECIDED (user) + Paper-4 (VAT) design section

**Decision: 4 papers.** (1) codability [ACL] — introduces the metric collection, clustering/
grouping, the L0→R3 hierarchy AS ARTIFACT, metric-space operations. (2) prompt-bounds
[ICLR/NeurIPS] — introduces the text corpora (+ silver labels IF load-bearing there; the
MI-vs-silver external-validity use stays in paper 1 — allocation pass pending); supervised
beat-GEPA + upper-bound framework + reconstruction-AS-OBJECTIVE (optimality theorems, identity
dilution, Goodhart). (3) tacit knowledge [NeurIPS/ICML] — no new datasets; reconstruction-AS-
INSTRUMENT measurements over paper 1's hierarchy (ladders, seam, decompression, enculturation)
+ NEW interventional experiment: which operations ADD tacit knowledge (fine-tuning vs RL vs
knowledge patching; seed evidence = TASTE online 1B→3B, CRAFT flat, MECH never). (4) VAT [ACL/
EMNLP] — introduces the preference variables. Discipline: papers self-sufficient; cross-refs
enrichment not dependencies; instrument-vs-objective division keeps 2/3 disjoint.

### Paper 4 (VAT) design requirements (user + Dan Ho comments, 2026-07-21)

**Seam y-prediction as baseline/control.** The metric-seam thin/coded extractors (title_vii,
patents slices exist) = the "codable surface features" control arm: V/A/T rows are reported as
lift OVER the seam baseline, separating judgment-predictable from feature-predictable y.

**The y-triangle: expert VERDICTS / expert REVEALED / community PREFERENCES.** Headline
question: which y's agree, and is the agreeing part the articulable part? Current inventory
(pairs exist, triples mostly don't):
| domain | verdict | revealed | community | missing leg |
|---|---|---|---|---|
| peer-review | decisions | scores (stated; revealed≈?) | — | community (public reviews/citations?) |
| patents | grants | examiner-LOO leniency | — | community |
| math-SE | accepted answer (asker) | — | votes | expert revealed |
| N&C | adoption (agree-vs-disagree y) | — | comment support? | revealed |
Main new data work: complete the triple in ≥3 domains; report the full cross-y agreement
matrix (threshold-free AUC) alongside per-y V/A/T rows.

**Subgroup-heterogeneity protocol (Dan Ho).** Our own program pre-validated the concern 3×:
peer-review pooled .77 = venue confound (retracted; ICLR-only .61); sibling-lattice pooled
wants = base-rate confound; Collins RTK = case-volume not org size. Protocol therefore:
(a) WITHIN-SUBGROUP estimation as default (within-docket / within-venue / examiner-LOO /
within-community), pooled-vs-within gap reported as a HEADLINE quantity (where does preference
signal live), never silently pooled; (b) heterogeneity-as-result: per-subgroup V/A/T + variance
decomposition + minimum-n gates (cf. sub-community #60: 1 formal keep + 3 replicated community
norms); (c) all grouping-level claims must clear the random-grouping null (silver-matching
lesson); (d) balance via stratified eval panels, not post-hoc reweighting alone.

### Paper concept anchors (user, 2026-07-21)
1. **Codability** — anchor TBD; user tentative: "Explicit metrics and norms"; Fable
   recommendation: the census question — "How many norms are there?" / "the norm lexicon"
   (saturating top of hierarchy, open leaf tail, dialects over a shared register, missing-mass
   for the uncollected remainder). The hierarchy is the map; the count is the headline.
2. **Articulation upper bounds** — names the construct, not the method; GEPA/M_ω are instances
   of articulation-mediated optimization; vacuity/EVT/missing-mass = bounds on what can be said.
3. **Tacit knowledge** — keep. ⚠ rename the TASTE/CRAFT/MECH category axis (collision with
   paper 4's construct; paper 4 owns the word "taste").
4. **"Taste is an articulability gap"** — thesis title; operationalized by Outcome = V + A +
   Taste: taste := residual between revealed/dense ceiling and articulated-criteria capture;
   y-triangle + subgroup heterogeneity = evidence the residual is real and structured.
Arc to state in each intro: 1 = the space of explicit norms; 2 = limits of using articulation;
3 = the gap on the KNOWLEDGE axis; 4 = the same gap on the PREFERENCE axis.
