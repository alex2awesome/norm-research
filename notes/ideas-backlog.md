# Research Ideas Backlog ("ideas-to-run")

Standing list of proposed experiments / directions that are **not yet executed**.
Living document, not a dated session note — append to the relevant section as ideas
arrive, and graduate them out when run.

## How to use
- **Status tags:** `[ ]` not started · `[~]` partial / machinery exists · `[?]` possibly-done-since the source note — **verify before promoting** · `[P]` parked (deliberate future-paper / out-of-scope) · `[G]` methodological guard/lesson, not an experiment.
- Each idea is self-contained enough to pick up cold: *what · cheapest decisive step (where known) · source (`file` or `memory-slug`) · status.*
- When an idea graduates to a real run, replace its body with a one-liner pointing at the dated note that supersedes it.
- This is **not** an overnight run-queue (that's `2026-05-28__overnight-queue.md`); it's the backlog of things we've thought about but not built.
- **Provenance:** §1–§13 below were assembled 2026-06-25 by a four-way sweep over `memory/`, `notes/`, the four large theory docs, and `running-research-notes.md` + READMEs. `[?]` items were forward-looking when written but may have been run since — check the source before acting.

---

## ★ Active / priority (designed with the user, ready to run)

### A — Stratifying the dense-model gap `C − (V+A)` ("what is the dense model doing above the line?")
*Raised 2026-06-25. Status: `[?]`-design-done, decisive probe cheap and runnable now.*

**What.** The dense reward model reaches AUC `C`; the articulated rubric+judge pipeline reaches
`V+A < C`. Stop treating the gap as one substance ("taste") and **stratify it** — confounder-vs-tacit
is the top and bottom of a five-rung ladder, each rung with its own subtraction:

| Stratum | What it is | Subtraction |
|---|---|---|
| Confound | author/venue/topic/length leakage | group-split + residualize-on-known-confounds (press-release 0.71→0.58) |
| Articulable-unmined | nameable criteria GEPA didn't find | push search / ALPHA-PROBE coverage; what closes was never tacit |
| **Interaction** | right features, wrong *combination* rule | **flexible vs. additive head on the SAME criteria** ← decisive |
| Long-tail features | nameable cues too numerous to enumerate | contrastive-pair elicitation (STaR) |
| Tacit | intersubjective, non-verbalizable | inter-expert agreement on residual pairs |

**Decisive first step — features vs. combination.** Metrics aggregate criterion scores
`s(X)=(s₁…s_k)` **additively**. Fit two heads on the same cached criterion-score matrix:
```
  C_clean            dense, deconfounded held-out   = I(Y; X)
     ≥  I(Y; s(X))   ANY function of criteria       ← flexible head (GBT/MLP)
     ≥  I(Y; Σwᵢsᵢ)  additive — current practice    ← linear head
```
`I(Y;s(X)) − I(Y;additive)` = info **in the features but destroyed by additive aggregation** → tacit
*combination*, not tacit features. `C_clean − I(Y;s(X))` = features the criteria don't carry at all.
If `I(Y;s(X)) ≈ C_clean`: dense found no new features, only weighted them better — "taste" = secret
*weighting*, fix is a learned aggregator (paper-worthy). If `≪`: drop to elicitation, hunt features.

**Theory link.** Interaction gap `I(Y; s(X) | additive(s(X))) ≥ 0` is the **same object** as the
`η`-vs-`φ` sufficiency deficit (`2026-06-18__prompt-optimality-theory.md` §11.1), at the aggregation
layer → closable in principle, unlike the tacit floor. **Confounder-vs-tacit demarcation =
intersubjectivity** (Noah framing): residual pairs to independent experts → agree+can't-verbalize =
tacit floor; disagree = noise in `1−C`, not a ceiling. Caveat (§11.3a independence): overlapping-corpus
model "experts" make shared bias look like taste → want humans or decorrelated families.
**Generalized per-node version** lives in §5 (gap-triage ladder).

### B — Prompt specificity trajectory & the prompt↔code (V/A) seam
*Raised 2026-06-25. Status: `[~]` — full proposal written 2026-07-01: `notes/2026-07-01__metric-seam-proposal.md`
(typed-channel hybrid artifact, gated MIGRATE + WORKFLOW ops, evidence-vs-computation op taxonomy,
matroid-U₂ + tightening-lemma certificates, E-S0–E-S4 plan; related-work sweep confirms the
fidelity-objective + certificate cell is unoccupied). E-S0 (longitudinal mining of existing lineages) is free and unblocked.*

**What.** Two coupled studies over the GEPA trajectory: (a) how the prompt gets more specific across
implementations; (b) the division of labor between the natural-language prompt and the Python code.

**Why — the prompt/code split is the V/A boundary made physical.** A criterion that compiles to a
deterministic Python check (Tier 1–4 codegen) is *verifiable*; one that must stay LLM-judged is
*articulable-but-not-codable* (the `B−A` language-tacit layer). Tracking criteria migrating prompt→code
across iterations **measures the V/A frontier per task** (`project_refactoring_algorithm_idea`'s
"library=norms, main()=taste", as a trajectory). **Prediction:** as code coverage grows it offloads the
verifiable, so prompt specificity shifts *composition* toward the A-layer residual.

**Metrics over GEPA versions:** token count · rare-/domain-term density · concrete `if-then` rules vs.
abstract guidance · embedding drift from seed · **rate of new-criterion introduction** (rarefaction on
the *adaptive* stream — here the **right** instrument: search-saturation, the use §12.1a-D distinguishes
from breadth-stream α). Does specificity saturate (magic words found) or keep climbing, and does the
saturation point differ by task (creative-writing > news_homepages, the ASQA/QAMPARI intuition)?

```
   V        |        A         |      tacit
 code-able  |  prompt-only     |  dense-only
 ───────────┴──────────────────┴────────────
   Idea B measures THIS seam     Idea A measures THIS seam
```

---

## 1. Cross-cutting V/A/T theory & the gap

- `[ ]` **Task-level judge-scaling curve for `C−B`** — measure `C−B` at 8B→70B→122B→frontier-API; an asymptoting gap can't be dismissed as operationalization weakness, a still-shrinking one can. Descriptive only — do NOT extrapolate the asymptote (2510.24626). Dense = achieved lower bound; 1-NN/twin = upper bound. (`methods/metric_implementer/README.md` "Gap arithmetic"; `running-research-notes.md` 682–711)
- `[ ]` **Tacit knowledge as a cross-task constant** — expert-LLM − lay-LLM gap on task-specific factual probes, or normalized `C−B / (C−B + articulable)`. Tension with Noah's operationalization-dependence (C depends on who counts as replicable expert). (`project_tacit_knowledge_measurement`; `running-research-notes.md` §5.3)
- `[ ]` **Per-language-model tacit-knowledge profile — REVISED 2026-07-01 (was "B_E − T tacit reserve"; the old subtraction measured neither face).** Problems with the original framing: (a) it conflated recovery R with the transmission ceiling T (`feedback_T_lower_bound_Mstar_be_upper` — the exact recurring collapse); (b) axis mismatch — `be_report` gives B_E (fragile species COUNT: order ~2×, probe 10–25%, §12.6.5 capacity-artifact caveat) minus `rec_r2` (a Ridge R² whose 30 learned weights are themselves unarticulated — a linear head is not *telling*); (c) uncertified — can't split "pool never proposed it" from "reachable but inarticulable." Replace with TWO instruments, two faces of tacit knowledge:
  **Face 1 — certified residual (nearly free, post-hoc on the R3 rescore checkpoints):** `Δ(E) = lowerCI(ceiling) − [OPT_Ω(E) + ε(E)]` per tier via `value_certificate --scaling`; flat-vs-shrinking staircase verdict per §12.6.5 (no fitted asymptotes); keep `be_report` B_E/rec_r2 as a descriptive scan only.
  **Face 2 — sparse-decompression ("apply a rich, sparsely worded concept successfully"):** fix the MESSAGE, vary the READER — decompression curves `R_E(L)` over three channel rungs: **R3-name-only** (pointing; thick target = the cluster's member-metric consensus) / **L-capped rubric** (telling; `recon_channel` budget directives) / **rules+exemplars** (showing; the codability L3 channel), executed by the ladder + one WEAK reader (Llama-1B weak-judge machinery from articulation_star). Executor's tacit knowledge of a concept = **strong−weak agreement gap at the sparsest message**; cross-reader execution (W≠E) as the DEFAULT recovery kills writer private codes. Sparse-decompression-vs-size = the *enculturation* scaling law (the concept lives in the weights; the name is an address).
  **Planted controls (mandatory):** compressible rule → reader-flat curve; pointer-concept → strong/weak gap; private-code → cross-reader collapse. Direction-of-error: reader-tacit decompression INFLATES R → conservative for the incommunicability thesis, but it misclassifies L2/L3 as L1 in the codability map — the weak-reader arm is the fix. Open: does Δ(E) stay flat while sparse-decompression grows with size (the interesting dissociation)? does the reserve track the semantic-vs-behavioral merge gap? (`feedback_T_lower_bound_Mstar_be_upper`; theory §12.6.5–12.6.6; `methods/codability/`; cross-link §3)
- `[ ]` **Within-domain A→C noise-reduction prediction** — dense ceiling should rise / taste residual shrink expert(A)→crowd(C) as aggregation averages out idiosyncratic taste. Testable on existing dense sweeps. (`2026-06-12__taste-taxonomy.md` §1)
- `[ ]` **Rationale-supervised gains largest on A-labels with written justifications** (legal opinions, patent actions) vs B/C labels without. (`taste-taxonomy.md` §1)
- `[ ]` **Social-influence control in C-labels** — popularity cascades (Salganik 2006) are not taste; control via early-window labels or identity-masked pools; measure the effect inside C-cells. (`taste-taxonomy.md` §1)
- `[?]` **Credential-rise register→doctrine shift** — within C-cells, as crowd credential rises (lay Reddit → practitioner SE → expert citations) the articulable layer should shift from presentation register toward substantive/doctrinal norms (r/supremecourt votes vs judicial-citation-pct). Cells exist; the cross-cell comparison not run. (`taste-taxonomy.md` §6)
- `[ ]` **Thin/thick ↔ (α, γ, A) correlation** — measure curvature α, submodularity ratio γ, articulation gap A per metric; validate thin = α≈0 ∧ γ≈1 ∧ A≈0, thick = γ≪1 ∨ A≫0. Conjectural until a positive correlation shows. (`prompt-optimality-theory.md` §8)
- `[G]` **Apples-to-apples within-class dense vs baseline** — never claim `C > baseline` without SAME split + SAME input (mathlib cross-split/cross-input faked +0.10 that was really 0). (`feedback_apples_to_apples_dense_vs_baseline`)

## 2. Prompt-optimality: certificates, recovery, coverage

- `[~]` **Held-out recovery certificate (Q1)** — train/test item split, run GEPA on train, certify R/T/A/DPI on test (guards prompt overfit to the item pool). Wired ~2026-06-16; confirm it's in `run_real_test`. (`2026-06-24__clustering-audit-and-certificate-nextsteps.md` §6b #29; `running-research-notes.md` 1832)
- `[ ]` **Prompt-transfer generalization (Q2)** — disjoint GEPA/test split; does the optimized prompt transfer to held-out *metrics/population*, not just held-out items? (`clustering-audit` #30; `running-research-notes.md` item 2)
- `[ ]` **Channel-cleanliness gate** — keep a criterion in Ω iff `adversarial_saturation` ≈ 0 ∧ no PRUNE-help (leakage alarm) ∧ `counterfactual_validity` (tracks a planted direction not a confound). Clean channels → near-submodularity *emerges* → cheap γ guarantees as a bonus. Instruments exist (`orthogonalize.py`), not yet composed into one filter. (`clustering-audit` #31; `running-research-notes.md` 1816–1830)
- `[ ]` **Missing-impact on real data** — Phase B with `harvest_gepa_omega` + deep_k=25 + pool_max=60; claim missing-impact < δ iff certified_bound < 0.005 ∧ probes saturate. Fix wired 2026-06-23, application pending. (`clustering-audit` #32; `project_missing_impact_headline`)
- `[ ]` **Multi-task certificate sweep** — run Q1+Q2 on CW + peer-review + code-review + math to validate across heterogeneous tasks. (`running-research-notes.md` 1833)
- `[ ]` **Full `B_E-ATLAS` coverage run** — breadth coverage → Heaps exponent α → GO/NO-GO on exhaustive search. ALPHA-PROBE machinery (§12.1a) built 2026-06-25, dry-run green, **gated on GLM quota (resets ~06-30)**. Sub-pieces:
  - `[ ]` cross-family GEPA coverage — independent proposers (GLM/Sonnet/…); high collision = saturation, diverging single-family curves = shared blind spots. (`prompt-optimality-theory.md` §6.9, §11.3)
  - `[ ]` capture-recapture / Chao1 / Good-Turing for |B_E| — four axes (prompt space, strong-LM, data-slice, Ω-algo); ONE adversarial novelty-tilted proposer ≫ k correlated lists. (`§11.3a`; `project_prompt_optimality_capture_recapture`)
  - `[ ]` discovery scaling law `OPT_Ω(t)` vs t → extrapolate `OPT_∞ ≤ cap_f`; halt when tail-γ≈1 ∧ adversarial-saturation fires. We already log full GEPA lineage. (`§6.9`)
- `[ ]` **Form-axis sweep** — materialize Φ = {k∈0,1,2,4,8} × persona × format × order; certify joint optimum over {S⊆Ω}×Φ with bootstrap CI. Includes few-shot saturation curve, persona effectiveness, format A/B, and the permutation order-sensitivity test (σ²_subset vs σ²_perm). (`prompt-optimality-theory.md` §6.8)
- `[ ]` **Multi-executor certificates** — escape single-executor scope via consensus target M, average objective R_avg (submodular, brute-force transfers), or worst-case R_min; the scientifically interesting object is the E-dependence itself (capability↔articulation substitution). Use Spectral Meta-Learner (Parisi) for consensus M instead of mean-then-median. (`§6.7a`, `§10.5`; `2026-06-19__unsupervised-to-Y-accuracy-map.md` §5)
- `[ ]` **Objective-blend γ composition bound** — bound the submodularity ratio of the fidelity scalarization via a composition theorem, then Sviridenko-2004 knapsack for budget caps. (`§10.6`)
- `[ ]` **Soft-readout both-legs validation** — run recovery with (a) sampled binary M′ and (b) continuous P(YES); confirm sampled `R<T` strictly and only η(X) attains `R=T` (proves both-readouts necessary for tightness). (`§11.1`, scorecard R8)
- `[ ]` **Submodular-optimization tooling for |Ω|>15** — branch-and-bound over the relaxation `C ⊇ B_E`; continuous-greedy multilinear extension (CCPV 2011). Both inherit the non-monotone-R caveat. (`§3.1`, `§6.4`, `§6.7b`)
- `[ ]` **Behavioral-capacity ceiling via output geometry** — bound B_E using the output-embedding spectrum σ(W_out) / logit geometry; tighter than the distribution-free cap for K-ary readouts. (`§3.1`)
- `[ ]` **Criteria-based parseable GEPA pivot** — prose GEPA did not raise T_prose; switch from prose to parseable criteria to dodge the T_prose binding cap. (`project_upper_bound_heldout_wiring`)
- `[ ]` **Rich-Ω GEPA certificate** — run `real_gamma` + greedy + sampled γ + co-information on the GEPA-mined **34-criterion** Ω (vs the 10–12 hand-written) — the submodularity regime, not brute-force. Ω already built. (`2026-06-19__tvd-consistency-real-data.md` §6.7a)
- `[ ]` **Executor-bottleneck matched-f gap for creative metrics** — ideal `I_TVD(M;X_S)` vs executor `R=I_TVD(M;M̂_S)` in consistent units, with Ω-aligned re-runs (add behavioral dedup), for ap_english/abbott/andrew_stanton/aristotle. (`tvd-consistency` §6.7a′)

## 3. Articulability scaling laws

- `[ ]` **Full articulability scaling suite** — optimizer under budget caps → frontier `fidelity*(m;B)` per metric → per-metric budget-to-articulation B* with right-censoring → **Kaplan–Meier "fraction articulable at budget B"**; classify articulated/climbing/resistant. Axes: instruction tokens, few-shots, data budget, model tier (judge vs optimizer **separately**), inference compute, optimizer rounds, structural complexity (clause count), interaction order. Iso-fidelity (Chinchilla-style) contours. **MVP:** peer-review, ~20 metrics spanning coverage, cheap axes only, 3 budget × 3 seeds. E0 ladder already validated on competitive code. (`methods/metric_implementer/README.md`; `2026-06-10__design.md` §3)
- `[ ]` **Capability-axis observational scaling (IRT, not Chinchilla)** — place judges on a *measured* capability scalar (benchmark PCA, or latent truth via Cultural-Consensus / Dawid–Skene), fit IRT 4PL; upper asymptote `d<1` = articulability ceiling read off the curve (not extrapolated). Expand tier ladder (27B–80B). (`running-research-notes.md` 682–711)
- `[ ]` **E-axis** observational law `I(E)=I_∞·σ(a(S_E−b))` (only axis with a monotonicity theorem). (`2026-06-16__articulability-vinfo-formalism.md` §4, §7.2)
- `[ ]` **K-axis** few-shot saturation (inverted-U; Bayesian-ICL closed form Arora 2410.16531). Canonical inducers hardcode N=5, never ablated. (`§7.2`)
- `[ ]` **N-axis** induction-set size (Prompt-MII peaks ~N=20 then overloads) — novel x-axis for rubric induction. (`§7.2`)
- `[ ]` **M-axis** criterion count, with the holistic-same-rubric confound guarded (2603.28005) and expert-count-vs-length tested (MoP 2407.00256). (`§7.2`)
- `[ ]` **Show vs Tell** — `I(m;m̂)` via K demonstrations vs via written rubric on the same metric ("how much can be *said* vs only *shown*", in bits). Nobody has cast this as transmitting a metric. (`§7.4`)
- `[ ]` **Paraphrase-stability robustness** — variance of R across paraphrases of the same rubric (Koyejo V-info pathology); flag instability. (`§6`)

## 4. Trainable articulation: STaR & refactoring

- `[ ]` **articulation_STaR v3 + fallback-defense ladder** — CoT-strong + logprob-weak contrastive (let 122B think to recreate the strong-weak gap 3B/1B failed at) + dedicated distilled-classifier leakage probe. Defense escalations: logprob judge scoring (replaces verdict parsing), seed/hybrid metric-conditioned rationales, held-out cross-family eval judge, counterfactual-swap probe (style vs substance), rStar process reward on groundedness+specificity, richer-than-binary prediction target. v1/v2 done with findings. (`datasets/creative-writing/README.md` §8; `project_articulation_star_fallback_defenses`; `project_articulation_star_rstar_followup`)
- `[~]` **Refactoring algorithm (unified V/A discovery)** — per-example programs/rationales → refactor into a shared library; measure library size at convergence, stabilization rate, main() complexity (= taste residual), migration rate articulation→verification (= thin/thick boundary). Add an extraction schema classifying input-side thickness (regex/program/llm). Early peer-review (100 ex). (`project_refactoring_algorithm_idea`; `running-research-notes.md` §3.8)
- `[ ]` **Verification pipeline (Direction 2) — finish steps 4–6** — implement hierarchy building, test-set program evaluation, and annotation with program success rates; consolidate into a single production runner. v1 ensemble AUC=0.500 on peer-review motivated this. (`methods/verification_library/README.md`; `project_verification_pipeline_recipe`)
- `[ ]` **Abstraction-type taxonomy resolution** — 8 current + 8 candidate-missing types (causal/entailment, ASPIC+ attacks, deontic modalities, compensatory/repair, temporal/ordering, analogical, epistemic status, AGM belief dynamics); decide constrain-vs-discover and how each operationalizes into scoring. (`project_abstraction_types_research`; `running-research-notes.md` §5.6)
- `[ ]` **Haupt entailment-based hierarchy** — build z₊/z₋ hierarchy via an entailment DAG (LLM pairwise entailment + greedy set-cover level summarization + latent-parent discovery) instead of clustering. (`project_star_algorithm_brainstorm`)

## 5. Metric-tree / infilling / fidelity

- `[ ]` **Metrics-tree-infilling real-data run (Phase 0, peer-review)** — engine validated on synthetic (detection 0.81 / FP 0.0), never run with a real LLM proposer+judge on a real task. (`methods/metrics_tree_infilling/README.md`; `project_metrics_tree_infilling`)
- `[ ]` **Gap-triage ladder per node** (the generalized Idea A) — fit `GLM(X) < flex(GBM) < dense(text, Llama-8B) < 1-NN-twin Bayes ceiling`; flex−GLM = misspecification, dense−flex = missing feature, ceiling≈dense≈flex = taste/noise. (`project_metrics_tree_infilling` next-steps §1)
- `[ ]` **Partition-specific steering** — proposer makes generic metrics at all depths; the 4-phase restructuring (robust generation → ternary YES/NO/NA scoring → rebuild → gap-fill) is designed but not built. (`project_metric_specificity`; `project_restructuring_pipeline`)
- `[ ]` **Scope + semantic hierarchies** — leaf-activity containment DAG + LLM-entailment semantic hierarchy; cross-tab for imposed-vs-emergent signal. (`project_metrics_tree_infilling`)
- `[ ]` **Codegen distill to Tier-1.5 programs** after hierarchy; **judge-prompt GEPA** with a fidelity-only objective (ρ, CF-acc, recon-match — never predictive perf); **reconstruction validity loop** (reconstructor sees only (text, judge-label), composed with decorrelating counterfactuals + cross-family). (`project_metrics_tree_infilling` next-steps §2–3)
- `[ ]` **Rubric fidelity validation** — round-trip code reconstruction (programmatic rubrics) + metric rediscovery (normative rubrics). (`project_rubric_fidelity_validation`)
- `[?]` **Live-LLM scenario test** (`--live`) — offline oracle only tests plumbing. (`project_metrics_tree_infilling`)

## 6. Metric-implementer experimental plan (E0–E6) & scorecard

- `[ ]` **E0–E6 plan** (gate before scaling): **E0** known-answer thickness ladder T0–T5 (GATE) · **E1** frontier descent over the 24-criterion competitive-code bank × {code,1B,8B,70B} × 3 seeds → first KM curve ($30–150) · **E2** words/reader/interaction 2×3 ablation · **E3** clause-decidability profile predicts floor · **E4** validity battery (reliability/convergent/discriminant/anchor-robustness) · **E5** cross-community floor distributions · **E6a** payoff (fidelity-optimized beats seeds at predicting y with zero label access) + **E6b** Goodhart gap (y-track − fidelity-track = confound share). (`project_metric_implementer`)
- `[ ]` **Scorecard #7/#8** — inter-implementation agreement (K blind implementations across families+code → mean pairwise κ as label-free thickness) and code↔judge convergence (corr + disagreement-cell adjudication). (`methods/metric_implementer/README.md`)
- `[ ]` **Bank-level scorecard** — scraped→dedup count, source-diversity per cluster, applicability rate from cells DB, generic-vs-task-specific fraction ("could this criterion apply verbatim to ≥3 of 9 tasks?"). Defends the "rubrics fall short" claim. (`README.md`)
- `[ ]` **LLM-reasoner aggregator over the metric bank** — feed datapoint + bank, let it reason to a judgment; measures articulability of the *bank* (closer to the B-layer ceiling). (`project_metric_implementer`)
- `[ ]` **Triad v2 with pluggable functionals** — generalize WQS/UQS/AQS beyond scalar agreement to rank-agreement (fixes compression artifact), reconstructability, CF-accuracy. **Cross-family grader at acceptance** (grader is Llama-70B even at accept → correlated-roles gap). **Stratified GEPA push** (top decile + resistant, not only top 10%). **Anchor grounding** (E0 audit, per-criterion w_oracle, anchor-robustness, small human-anchor subsample). (`project_metric_implementer` queued directions)

## 7. Local explanations

- `[ ]` **Optuna sweep (GPU-gated, Llama-70B-FP8)** — reuses cached Step 1+2; sweep weight-matrix, K, shrinkage α, embeddings, predictors (L1/RF/GBM/LLM), silhouette search; + scaling-law subsample 1K→full; + two-stage hierarchical clustering. Operating point: tw=0/eps=0 + LLM dedup. (`project_local_explanations_hyperparam_sweep`; `running-research-notes.md` §3.5)
- `[ ]` **Long-tail diversity follow-ups** — two-pass "ADDITIONAL features" extraction, anti-pattern few-shots (DPO-in-context), lift×log(coverage) filtering, per-example topic-hint prompting. (`project_local_explanations_followups`)

## 8. Rubric corpus & taxonomy analyses

- `[ ]` **Temporal norm drift** — per cluster: emergence year, dominance year, stability score; new/disappearing rubrics by decade. (`2026-05-11__rubric-variance-analysis-plan.md` §3b)
- `[ ]` **Authority-weighted analysis** — per-page authority (source-tier × content-quality × independent-discovery); top-quartile filter for cleaner variance. (`rubric-variance-analysis-plan.md`)
- `[ ]` **Negative-example signal** — do `rej_*` rejection-corpus rubrics cluster as negations of the positives, or reveal additional norms? (`rubric-variance-analysis-plan.md` RQ#7)
- `[ ]` **Cross-encoder canonical linkage** — fine-tune, cluster connected components above threshold → per-task canonical rubric count + density. (`rubric-variance-analysis-plan.md` §1a)
- `[ ]` **Noun/verb thickness chain extraction** — variable-length input_noun→verb→…→output_noun chains per merged group; cluster into archetypes; per-task prevalence. Plus **procedural-vs-predicate** classification (input-holism / operation-irreducibility / concept-contestedness). (`2026-05-14__noun-verb-thickness.md`)
- `[ ]` **Rubric decomposition granularity** — atomic (whole rubric, one call) vs decomposed (multi-call components, combined): accuracy/consistency/coherence/cost/aggregation trade-offs. Deep-research report already run. (`project_rubric_decomposition_question`)
- `[ ]` **signal_texts → validity matrix** — rows = R2/R3 metrics, cols = signal clusters, cells = correlation across docs; empty column = gap (need new metric), dead row = no reviewer cares. (`2026-06-19__norm-extraction-session.md` §8)
- `[ ]` **E1–E6 validity experiments / faithfulness / G-theory** — recoverability (hide rubric, infer rule from labeled data, apply held-out), obfuscated-code interpretation (rename to a/b/c, strip comments, guess the rubric), G-theory σ² decomposition over ~30 rubrics × 50 dps × 4 judges × 3 code impls. (`2026-05-16__validity-experiments-plan.md`; `2026-05-22__metric-level-empirical-test-design.md`)

## 9. Judge quality & the 0.5 problem

- `[ ]` **0.5 filtering strategies** — ~69% of relaxed-applicability CW cells score 0.5; test hard-drop, confidence-weighting, treat-as-missing+indicator, one-hot {0,0.5,1}, MI pre-select, binary re-prompt. (`project_judge_0p5_noise_filtering`; `datasets/creative-writing/README.md` §8)
- `[ ]` **Judge score-distribution collapse detector** — structured/guided-JSON judges silently collapse to all-min (valid schema, 0 variance → meaningless AUC); auto-check spread, switch to free-form+parse if collapsed. Standing rule, no automated tool yet. (`feedback_check_judge_score_distribution`)
- `[ ]` **Re-score other tasks with the v2relax fix** — per-task RELAX/STRICT categorization (cross-task peer-review prompt bug + relaxed applicability); only CW done (0.541→0.636). (`datasets/creative-writing/README.md` §8)

## 10. Per-task V/A/T fills & dense ceilings

- **Code review** — `[ ]` per-aspect Python predict-programs (394 aspects) + Tier 1–3 deterministic ladder (metadata → diff parsing → radon/lizard/jscpd/pylint/ruff/mypy/bandit/semgrep) + diff-enriched re-score (cells are title+comments only) + final RF reporting AUC per tier. (`project_code_review_verifiability_plan`; `methods/verification_library/README.md`)
- **Patents** — `[ ]` re-score all cells with a **patent-specific judge** (cross-task bug → prior V/A/T was random, 0.507); `[ ]` §102 spec-chunk retrieval full index + limitation-level decomposer + OA-text ground truth; `[ ]` §112 indefiniteness track (PEDANTIC, 14k claims); `[ ]` app-level V with spec paragraphs as evidence; `[~]` v6.1/v6.2 retriever (10× pairs, stable-hash split) + distilled cross-encoder rerank + Phase-2 hard-negative dataset. (`datasets/patents/README.md` §10; `2026-06-10__patents-vat-status-and-plan.md`)
- **Legal** — `[ ]` SS-disability A-layer (20 CFR §404.1520 sequential rules, 16K-case slice); `[ ]` Title VII McDonnell-Douglas A-layer (EEOC manual corpus) [fallback]; `[ ]` CAVC full 9,140-pair re-score (current n=600); `[ ]` DOL party-name de-leak; `[ ]` patents temporal/applicant re-split (PatentsView assignee join); `[?]` trademark few-shot (K=24 = one design point); `[ ]` EXA dense ceiling on TTAB ex-parte; `[?]` TTAB ~40× expansion pending label-quality check. (`2026-06-06__legal-outcome-prediction-thread.md`; `2026-06-22__legal-vat-audit.md`; `running-research-notes.md`)
- **Math** — `[~]` mathlib review-friction full build (per-PR review fetch ~35K PRs); `[?]` AoPS wiki integration (answer keys = code-checkable, solutions = editorial-sim y, thanks = taste); `[?]` sympy V-layer scale-up (76.7% answers have ≥1 checkable claim); `[ ]` Math.SE elegance dimensions (Inglis-Aberdein elegance/profundity/clarity/precision; filter by proof tags). (`running-research-notes.md` §2.8; `project_math_elegance_research`; `project_aops_dataset_collection`)
- **Creative writing** — `[ ]` dense ceiling on clean 96K (`writingprompts_modeling_clean`, group-split prompt_id; current sweep still climbs at 1.0); `[ ]` editor-curated source diversity (does the norm library transfer beyond crowd-votes?); `[?]` RoyalRoad stubs (cw-A) richer recovery; `[ ]` Wigleaf Top-50 (cw-B) full-text recovery. (`datasets/creative-writing/README.md` §8; `running-research-notes.md`)
- **Press releases** — `[ ]` dense ceiling on the publisher-grouped **deconfounded** split (confirm 0.71→~0.58 for a non-linear model); `[ ]` editorial-byline-only V control; `[ ]` extraction improvement (38% empty bodies). (`2026-06-25__press-release-audit.md` §6; `project_press_release_results`)
- **Humor** — `[?]` AST contest corpus integration (40,026 files parsed); `[ ]` platform deconfound. (`running-research-notes.md`)
- **Academia cells** — `[ ]` best-papers leftovers (141 unqueried); `[?]` acad-C S2 standardization (29K papers); `[ ]` NeurIPS-2022 0.989-leak cell decide; `[~]` embedding-cluster stratification for high-floor C cells (worked on reddit-news → 0.569; try law-C). (`taste-taxonomy.md` §17h; `running-research-notes.md` 2026-06-13)
- **Grant funding** — `[ ]` obtain rejected proposal text (NIH CSR collaboration / FOIA / PI partnerships / Open Grants expansion); weak A0/A1-suffix proxy as fallback. No canonical dataset exists (RePORTER is funded-only). (`datasets/grant-funding/README.md` §10; `project_nih_a0_a1_investigation`)

## 11. Norm-extraction corpora & dataset acquisition

- `[ ]` **GEPA anchor-harvest, remaining priority corpora** — math_se, crse, code_review, peer_review (humor/press/legaladvice/wp_comments done/in-flight). (`running-research-notes.md` 2026-06-24)
- `[ ]` **SE technical-answer extraction** — Stack Overflow + dba.SE + codegolf answers (≈756K records, multi-week). (`2026-06-16__norm-extraction-pipeline-state.md`)
- `[ ]` **LeetCode/Codeforces discussion threads** — real (doc X, comment-on-X) for competitive programming (editorials are author self-narration). (`2026-06-19__norm-extraction-session.md` §9)
- `[ ]` **Comments.xml extractions** — Math.SE and Law.SE comment critiques; LegalAdviceUK (config + 50K subsample); r/supremecourt v2 bulk (58K balanced); WritingPrompts comments aggregation; CR.SE 80K expansion. (`2026-06-14__human-norms-extracted-inventory.md`)
- `[ ]` **v2 re-classification with rubric-vocab bootstrap** — re-run earlier extractions with the v2 vocabulary for consistency. (`norm-extraction-pipeline-state.md`)
- `[ ]` **N&C improvements** — per-comment slicing on "Response to Comment X" markers; comment-critique-vs-substantive classifier (~15–25% usable yield). (`norm-extraction-session.md` §7,§9; `project_nc_structural_mismatch`)

## 12. Creative-writing dataset candidates (source diversity beyond crowd-votes)

- `[ ]` **WebNovelBench** (4K+ Chinese web novels, 8 narrative dims, PCA percentile — ranked #1); **LitBench prizewinners** (100 prize + 100 bestseller, needs author email); **SF short-stories psychometric** (Otsuka 2025, Zenodo 10.5281/zenodo.15556522); **Bridport / Commonwealth shortlists** (~100–200 stories/yr, expert-curated); **Inglis-Aberdein replication** (30 critics × ~200 stories × 4-dim Likert, ~$15–20K gold standard). (`project_creative_writing_dataset_search`; `datasets/creative-writing/README.md` §8)

## 13. Parked: theory/framing for future papers

- `[P]` **Imposed vs emergent norms** — z₊/z₋ distributions before/after explicit policy changes (NeurIPS checklist, linting/CI rules); imposed = step functions, emergent = gradual. (`project_norm_emergence_future`)
- `[P]` **Faultless-disagreement grounding** — Kölbel/MacFarlane as the theoretical home for the taste residual `1−C`. (Noah's recommendation; `running-research-notes.md` §6)
- `[P]` **Verifiability cycle** — the V/A/T frontier is non-monotonic; paradigm shifts / Goodhart / community expansion revert verifiable items to taste. (`project_refactoring_algorithm_idea`)
- `[P]` **Partner-swap rebound** — if the pair-specific slice is large, the formalization needs a relational term (Brennan & Clark partner-specificity); test via partner swap. (`project_external_field_articulability_framings`)
- `[P]` **Synthetic gap-calibration battery** — known-answer tests (gap=0, >0, >0-for-known-reason) + pure-noise negative control (true gap=0, measured "gap" = pipeline bias) + gap(K) compression-budget curves as K→1. Reframed as instrument calibration, possible follow-up paper. (`project_synthetic_gap_calibration`)
- `[P]` **TVD-MI IRT / additivity paper** — additivity ↔ the modular α→0 regime (Robertson & Koyejo 2510.14966). (`prompt-optimality-theory.md` §7)

---

## In-flight data jobs — verify before relaunching (status checks, not ideas)

Background jobs flagged "running/launched" in source notes whose completion was not confirmed at sweep
time (2026-06-25). Check state before re-launching or treating as done: norm-extraction Qwen-122B resume ·
mathlib review fetch (~37K PRs) · BVA pre-2012 backfill · AoPS scrape (10-day) · RoyalRoad stubs run ·
full GLM-4.7 pairwise spectrum labeling · intact press-release clean CSV on sk3.
