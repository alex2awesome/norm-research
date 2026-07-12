# Formalizing the articulability gap — definitions, imports, and derived predictions

*(Notes from 2026-06-11/12/13 discussions. Companion to `2026-06-10__design.md`. Status: scaffolding
for the paper's theory section; the derived predictions T1–T6 in §4 plus T7 in §6, with §7 folding in
the 2026-06-13 observational-scaling + annotation-lit updates (U1–U6) and §8 unifying V with
articulability (V ⊆ A), are the actionable part.)*

## 1. Definitions

Items x with labels y, η(x) = E[y|x]. **Bayes risk R\*** = risk of the best possible predictor
*given the representation x* — the aleatoric noise floor. It is class-independent and sits *below*
all approximation error (approximation error = best-in-class − Bayes).

- An **articulation** is a string s in a language Λ.
- An **executor** E maps s to a scoring function E(s): code interpreter, 8B judge, 70B judge,
  Sonnet, human — a ladder of executors with increasing judgment-free primitives.
- **Frontier**: R_E(L) = best risk achievable by any articulation with |s| ≤ L words under E.
- **Articulability gap**: G_E(L) = R_E(L) − R_C, where C is the dense-model ceiling. *(Units: this is
  in **risk** — R = error, R_C the risk **floor** — so G ≥ 0 since R_E(L) ≥ R_C. In **fidelity** units
  (the §6/E7 convention, higher = better) the identical gap is `F_C − F_E(L)` = ceiling − achieved.
  Same quantity, opposite sign; "ceiling" is the fidelity word, "floor" the risk word.)*
- **Tacitness w.r.t. E**: τ(E) = lim_{L→∞} G_E(L) — the gap that survives unlimited words.
  Two numbers characterize "level of tacit knowledge": the **asymptote** τ(E) (depth — what no
  words reach) and the **knee** L\*(E) (cost — the sophistication: words needed to exhaust the
  reachable part).

> **Notation update (2026-06-15) — `τ` REASSIGNED, `a(E)` introduced.** Per the executor-ladder
> reconciliation (`2026-06-15__executor-ladder-tacit-articulability.md`). Writing the L→∞ rungs
> `A = F_code`, `B = F_E*` (best rubric on E; `F_E* = sup_L F_E(L)` at finite `L*` — the sup is
> achieved at the knee, NOT at `lim_{L→∞}`, because instruction-following degrades with rubric length
> — IFScale), `C = F_dense-direct` (the §1 "Role of the dense model" ceiling):
> - **`a(E) := C − B`** = the **articulability gap** — the residual no rubric closes even for E
>   (Polanyi / "I know it when I see it"). *This is exactly the object called `τ(E)` immediately above,
>   i.e. `a(E) ≡ lim_{L→∞} G_E(L)`* (with the `sup_L`/knee correction). The paper's headline gap;
>   **falls** with E. The knee `L*(E)` attaches to this curve.
> - **`τ(E) := B − A`** = the **tacit knowledge contained in E** — what E's tacit competence recovers
>   over a code executor on the same construct. **Rises** with E.
> - Identity: `C − A = τ(E) + a(E)` (verifiability gap = E's tacit knowledge + articulability gap).
> Everywhere below, read the old `τ(E)` / "tacitness w.r.t. E" as **`a(E)`**; the new `τ(E)` is the
> B−A resource. Also: `a(E) → 0` as `E → ∞` is an *empirical extrapolation, not a theorem* (Chinchilla:
> a positive floor survives infinite capacity; the channel bottleneck ≠ the capacity bottleneck).

**Articulability of a metric m — working definition (2026-06-13).** For a rater/executor E and a
datapoint x,

> articulability(m; E, x) = the **reliable, generalizing discrimination** of the construct —
> agreement with the **anchor** target construct (not ground truth), **net of E's own noise**,
> measured on **held-out / counterfactual** variants so it cannot be label-leakage — achieved by the
> **best finite description of m that E can read**, at x's **clarity stratum**.

Aggregate over the applicable-x distribution for the metric-level `articulability(m; E)`; trace it
over E for the articulability curve, summarized by its **ceiling** (depth: `1−τ(E)`, the residual no
description closes) and its **knee** (cost: `L*(E)`, the description length to reach it).

Crucially, **E is not a capability scalar but a profile — (intelligence / raw capability, tacit
knowledge, background / common ground).** This conditioning is load-bearing: the *same* description is
a *different* specification to different readers, because what the words achieve depends on what the
reader already brings. It is why articulability is executor-relative (no executor-free number — §6),
why "thick" means words pay off *only* for a capable, knowledgeable reader (the multiplicative /
complementary regime — §7/U2), and why a rubric that merely *points into* a knowledgeable reader's
prior (E2's `corr(full, stub) ≈ 1`) is articulating nothing new. The four bracketed qualifiers each
close a failure mode: **anchor target** (discrimination relative to *what*); **net-of-noise**
(disattenuate E's test–retest, else a noisy rater shows spurious discrimination); **non-leakage** (must
transfer, not memorize the label); **applicability + clarity stratum** (aggregate only over items where
m applies; "clarity" splits into the datapoint's *aleatoric ambiguity* vs its *processing difficulty*,
and conditions the estimate rather than being averaged over). By conditioning on "the best finite
description," this targets the **ceiling** of the response curve, not the slope. The verifiable case is
the `E = compiler`, `description = code` instance, where on amenable x the residual is exactly 0 and
E's ability drops out (V/A unification).

**Anchor vs. ground truth.** *Ground truth* = the true construct value / real outcome; for subjective
tasks it often does not exist as a single value, and where a real outcome `y` does exist it is
frequently confounded (e.g. news_homepages = layout, not engagement). The *anchor* is the
**operational reference** we actually score against: the highest-capability construct-scorer available
(a strong model with a rich rubric, or a human expert), CF-validated before use, treated as the
stand-in for the construct. It is a **fallible proxy, not the truth** — we measure its test–retest
reliability and inter-implementation agreement and divide them out, and since `R_anchor ≥ R*` the gaps
we report are conservative. It targets the *construct*, not the downstream outcome `y` (kept separate —
E6), and when judge and anchor are both LLMs they must be drawn from **different families** to avoid
self-preference leakage. One-liner: *the anchor is the strongest fallible judge we agree to measure
against because the truth is unavailable or corrupted — validated, bounded, treated as a ceiling, never
as a fact.* Corollary (V/A unification): a verifier on **uncorrupted, V-amenable** data is the one case
where the anchor coincides with ground truth; **corrupted-V is exactly the anchor-is-fallible case.**

**Granularity axis (k = datapoints covered per rule).** k=N: one global rule. 1<k<N: a *codebook*
— partition + rule per cell + **router whose description length counts toward the budget**
(otherwise tacit knowledge hides in case-routing). k=1: per-case explanation; trivially zero
without a transfer test; meaningful under transfer (CF perturbation + behavioral reconstruction
on neighbors) = local articulability. Distinguish the *conditional* gap on a slice G(L|S) (what
tree-infilling / hard-stratum frontiers measure) from the gap of a granularity-k *system*.

**Role of the dense model (important orientation).** The dense model is NOT the approximation-error
bound — it is the empirical estimate of the **Bayes floor**: R_dense = R\* + (dense's own small
approx/est/opt slack). Since R\* ≤ R_dense, gaps measured against the dense ceiling are
**conservative** (lower bounds on true gaps). Ceiling caveats: (i) R\* is relative to x — hidden
rater context inflates apparent noise; (ii) dense can approach R\* via construct-irrelevant
shortcuts (confounds) or sit above it when under-trained → the ceiling needs deconfounding +
saturation checks. "1−C = taste+noise" is the claim dense ≈ Bayes given x; checkable in synthetic
mechanism #4B (one-rater-per-item) where E[y|x] is known in closed form.

## 2. The unifying import: V-usable information

Articulability is **not a new error type**. It is **V-usable information** (Xu et al., ICLR 2020)
with a *communicative* constraint family: V_{L,E} = {E(s) : |s| ≤ L}.

- Articulable information = I_{V_{L,E}}(X→Y).
- Articulability gap = V-information difference between the dense neural family and the
  language+executor family. Tacitness = that difference at L→∞.
- The V/A/T decomposition = a chain of nested V-information gaps
  (code family → judge family → neural family → unconstrained/Bayes). *(But see T4: the first two
  are not literally nested — the chain needs the join family.)*
- Per-item: **pointwise V-information** (Ethayarajh et al. 2022, dataset difficulty) is the formal
  twin of our UQS strata — items with low PVI under the articulation family but high PVI under the
  dense family are "language-hard but not absolutely hard."

What's new in our work: prior V-information uses computational constraints (model class); ours is
communicative (instructions to another agent), and we measure it empirically — feasible only
because LLM judges make the constrained observer cheap to instantiate.

### 2.1 Does V-information generalize across the granularity axis (gap(N) → gap(1))?

Mostly yes, with one required extension at the k=1 end:

- **k = N down to moderate k: native.** A codebook (router + per-cell rules, router length in the
  budget) is still a single function family — V-information with a structured family. Nothing new
  needed.
- **k = 1: NOT native.** Per-case explanation is a different protocol: the describer chooses the
  message *after seeing the instance (and its label)*. Plugged naively into V-information this
  trivializes — the message can leak y, so I_V = H(Y). This is the formal restatement of "gap(1)=0
  without a transfer test." Two principled fixes:
  1. **Local / leave-one-out V-information**: fit the articulation on the case, evaluate on a
     neighborhood distribution that excludes it (CF perturbations + kNN) — gap(1)(x₀) = V-information
     at neighborhood scale. The whole granularity axis then unifies as **cross-validated V-information
     at scale k** (rules fit per cell, evaluated held-out within cell); the estimation term in T3's
     U-shape is exactly the leave-one-out penalty.
  2. **Conditional V-information — already in the literature**: REV (Chen et al., ACL 2023,
     "information-theoretic evaluation of free-text rationales") measures how much *new* usable
     information a rationale adds beyond the input toward the label, constructed precisely to defeat
     label-leaking rationales. Per-case articulation evaluation = REV with executor-constrained V.
- **PVI ≠ gap(1).** PVI is the per-item decomposition of the *global* (k=N) rule — where the one
  articulation fails. gap(1) asks whether *any* local articulation works there. Crossing the two
  gives a per-item 2×2: covered-by-global ∧ locally-articulable = normal; uncovered ∧ locally
  articulable = **exception** (coverage gap — exactly what tree-infilling targets at gap nodes);
  uncovered ∧ not even locally articulable = **taste item** (genuinely tacit case). That 2×2 is a
  measurable refinement of "is the gap from rules covering too much, or do individual explanations
  also fail?"

## 3. Factor → theory map

| factor | formal home | what it gives |
|---|---|---|
| level of tacit knowledge | asymptote τ(E) + knee L\*(E) (sophistication, Vereshchagin–Vitányi 2004) | tacitness is executor-monotone; two-layer measure = τ(code)−τ(judge), τ(judge)−τ(dense) |
| population complexity (raters) | List & Pettit discursive dilemma (judgment aggregation impossibility) | aggregating individually-articulable judges provably breaks articulation structure; empirical = σ²_judge facet |
| population complexity (items) | Zipf tail over latent micro-genres; Michaud et al. 2023 quantization model; Hutter 2021 | power-law frontier (T2); exponent shared with dense data-scaling |
| "right words" existing | IB theory of the lexicon (Zaslavsky/Kemp/Regier/Tishby PNAS 2018); knowledge-compilation succinctness gaps (Darwiche–Marquis) | words exist for communicatively frequent constructs; jargon = minting words; "no right words" = succinctness lower bound in the executor's tractable fragment |
| diffuse signal | ℓ₀/ℓ₁ sparse approximation theory | articulations are k-term descriptions; a coefficient vector with no sparse approximation is formally inarticulable at small L |
| common ground | Clark grounding; Daston cookbook/apprenticeship | thinness is relative to shared background; same words = different specification per executor |
| stationarity | rules presuppose stable targets | drift-induced gap ≠ complexity-induced gap (out of scope, named) |

### 3.1 Contributor audit (2026-06-12): what's missing, what's deliberately excluded

Three contributors not yet in the map, one refinement:

1. **Introspection gap (self- vs. observer-articulability).** Nisbett & Wilson 1977: experts
   confabulate their own criteria. Econ codifiability measures *believed* codifiability
   (surveys); our instrument articulates **from behavior** (reconstruction), measuring
   observer-articulability ≥ self-articulability. Free readout: **seed-to-frontier gap at
   fixed (L, tier)** — community-written seed rubrics (online-rubrics) vs. the GEPA frontier
   = the community's self-articulation deficit, distinct from τ. Named column in E1.
2. **Protocol interactivity.** L is a one-way string; communication complexity separates
   one-round from multi-round protocols. Daston's apprenticeship and Schaekermann's
   deliberation-resolvable split live here. τ(one-way) ≥ τ(interactive); candidate axis 9
   (clarification turns at scoring time, budget-counted). Named, deferred.
3. **Deliberate non-articulation.** Strategic/normative silence (legal exposure,
   gatekeeping, taboo) explains unarticulated-in-the-wild without inarticulability. The
   instrument distinguishes can't-say from won't-say (we attempt the articulation
   ourselves). One scoping sentence; ties to [[project_norm_emergence_future]].
4. **Subculture specificity (refinement, user 2026-06-12): not a gap term, a knee
   modulator.** The dense ceiling absorbs community content, so community-specificity adds
   nothing to τ — correct. But at finite L it shifts L\*: low communicative-frequency
   constructs need more words to point to (IB-lexicon row; Bonus prediction; E4's
   term-frequency discriminant). Predictor in the Cox regression, not a decomposition term.
   Caveat: the dense bound holds only at label-saturation — an under-trained ceiling on a
   small community makes gaps look *smaller* (anti-conservative for articulability claims,
   conservative for tacitness claims); the saturation check carries that load.

Other imports (separations & shapes): rate–distortion (frontier shape); Bottou–Bousquet
approximation/estimation/optimization decomposition (justifies the rule+noise negative control);
MAJ ∉ AC⁰ (heterogeneous-rater synthesis is provably budget-relative); Telgarsky depth separations
(executor-tier gaps can be exponential); sample compression / teaching dimension (few-shots are
compression schemes); Skalse et al. 2022 (specification's Goodhart failure mode as theorem).
Terminology bridge: **underspecification** (D'Amour) = articulation's failure mode (executor DOF;
measured by σ²_judge, panel divergence, words_share); **misspecification** (econometrics, reward
misspecification) = specification's failure mode (determinate-but-wrong; measured by anchor
decoupling, CF failure). MECHANIZE finding in one sentence: *mechanization converts
underspecification into misspecification.*

## 4. Derived predictions and provable statements (T1–T6)

**T1 — Frontier shape, and a free diagnostic.** G_E(L) is nonincreasing in L; convex after
allowing *mixtures* of articulations. Two consequences: (a) any measured non-monotonicity is pure
optimization error — the isotonic-regression residual of the measured frontier is an **empirical
lower bound on GEPA's optimizer slack** (free instrument calibration from data we already log);
(b) a concave bump in the measured curve implies an **ensemble of a short and a long rubric beats
any single rubric at intermediate budgets** — concrete and checkable.

**T2 — Power-law frontier with a shared exponent (the headline prediction).** If norms/criteria
("quanta") have Zipf usage p_k ∝ k^{−(α+1)} and articulating one quantum costs ~c words, then a
budget L covers the top L/c quanta and the residual is Σ_{k>L/c} p_k ∝ L^{−α}:
**G(L) − τ ∝ L^{−α}** — a power law whose exponent is the tail index of the norm-frequency
distribution. The quantization model gives dense data-scaling a related exponent from the *same*
tail. Three-way consistency check, two-thirds of which uses data we already have:
(1) empirical Zipf tail of norm-cluster sizes (norm extraction exists across tasks);
(2) dense data-scaling exponents (per-task sweeps exist);
(3) rubric-budget frontier exponent (the thing the scaling-law experiments will measure).
Qualitative version checkable now: press_release dense saturates fast ⇒ steep norm decay ⇒ rubric
frontier should plateau at small L; creative_writing still climbing ⇒ heavy tail ⇒ slow power-law
frontier that never saturates at practical budgets. Fitting power-law vs exponential decay to the
measured frontier is itself diagnostic of the tail. (Exact exponent mapping depends on the
per-quantum learning-cost model — state qualitatively, cite Hutter/Michaud.)

**T3 — Optimal granularity (provable toy theorem).** Fixed total budget, transfer-tested,
piecewise rules over a partition: G(k) = A(k) + V(k) with bias A nondecreasing in k and estimation
V ~ √(complexity/k) growing as k↓ ⇒ **interior optimum k\***. The "optimal granularity of norms":
doctrines, not constitutions or ad-hocery; k\* shifts with thickness. Pipeline prediction:
DECOMPOSE helps thick metrics up to a point, then hurts (router + estimation dominate) — possibly
already visible in distinctive_voice v004–v006 DECOMPOSE regressions.

**T4 — The layers are not nested; define the B-layer by the join.** Code can compute exact counts,
lengths, hashes that no LLM judge reliably can; the judge reads meaning code can't. So
I_code ≰ I_judge and B−A can be *negative* on mechanical constructs. For a monotone chain, the
articulable layer must be measured as the **join**: best of {code, judge, composites in both
orders} (judge-grounds→code-verifies is v_struct; code-features→judge is the other direction).
Related separation: majority ∈ TC⁰ \ AC⁰ — threshold/counting gates are a qualitatively distinct
resource, and LLM judges are empirically weak counters ⇒ **move tallying out of the judge into the
harness** (DECOMPOSE-with-averaging already does this; theory says that's load-bearing, not
cosmetic).

**T5 — The measured gap is doubly conservative (winner's curse + ceiling direction).** The
frontier is a max over P tried prompts evaluated on finite samples: selection inflates the
frontier estimate by ~σ√(2 ln P), which *shrinks* the measured gap; independently, the dense
ceiling sits at-or-above Bayes, which also shrinks it. Both biases point the same way ⇒
**tacitness claims are safe; articulability claims need fresh-set re-evaluation of the selected
prompt** (acceptance already re-evaluates cross-family; add fresh items to kill the selection
bias).

**T6 — Preprocessing can strictly increase usable information (retro-explains v_struct).**
V-information provably violates the data-processing inequality: transforming the input can
*increase* I_V for a weak family. That is the formal statement of the legal finding — LLM
standardizes thick inputs, arithmetic verifies — and of Daston's "thick rule cleaning up after the
thin rule." Prediction: input-standardization preambles raise the *code*-layer frontier
specifically, with little effect on strong-judge frontiers.

**Bonus (IB lexicon) —** articulability should correlate with the construct's communicative
frequency in the community's meta-discourse. Testable with corpora in hand: norms present in
online-rubrics / editorials (e.g., LeetCode editorial corpus) should show steeper early frontiers
than constructs absent from meta-discourse.

## 5. Honesty caveats

Executor competence is not a clean circuit parameter; natural language is not a formal language.
Posture for the paper: definitions ours; shape results inherited; separations cited as existence
proofs that gaps of this kind are mathematically possible; prove only T3 (and the T2 sum
computation) in an appendix; everything else is measurement.

## 6. The Chinchilla mapping (2026-06-12): what imports, what breaks, the null-model fit

**Precedent.** Scaling laws have been used for theoretical-optimum estimation, not just compute
allocation: the Chinchilla form L(N,D) = E + A/N^α + B/D^β fits an **irreducible term** E read
as the entropy of the data distribution (Hoffmann et al. 2022, E ≈ 1.69 nats); Henighan et al.
2020 made the move explicit — loss = irreducible (true entropy) + reducible (KL to truth) — and
extrapolated asymptotes to *estimate the entropy* of image/video/text distributions. The
cautionary record is equally established: fitted asymptotes are fragile (power-law vs. other
decays indistinguishable over finite budget ranges; refit sensitivity, Besiroglu et al. 2024;
regime breaks, Caballero et al. BNSL). Hence the 06-11 amendment stands and is now a rule:
**fitted τ̂ is a descriptive parameter with a CI, never a defended bound. Bounds are defended
only by the sandwich** (best exhibited articulation ≤ articulability(m;E) ≤ disattenuated
agreement ceiling — see E7 for the operational bracket).

**Where the analogy breaks — and the break is the finding.** Chinchilla's E is meaningful
because data entropy is *model-free*. Articulability has **no executor-free analog of entropy**:
in the double limit (words → ∞, reader capability → ∞) the bound degenerates — a strong enough
reader needs only the construct's *name* (E2's "pointer into the reader's prior" diagnostic
detects exactly this). This is §2's V-information relativity restated: there is no V-free
information to bound. Consequently the "theoretical upper bound of articulability of m" is
**a function, not a number**: τ(E) across the executor ladder, equivalently the floor B\*(m) in
judge-capability units — already the E1 estimand. The paper disavows the single-number version
explicitly; the relativity is what makes the quantity measurable, not a dodge.

**The null-model fit (the real import).** Chinchilla's law is additively separable in its two
resources. The analog, per metric m, over words budget L and reader capability C (per-criterion
*empirical* capability, per the Relative Scaling Laws amendment — never nominal size):

    fidelity\*(m; L, C) = (1 − τ_m) − A_m·L^(−α_m) − B_m·C^(−β_m) − I_m(L, C)

- I_m ≈ 0 (separable; the Chinchilla form holds) ⇔ **thin**: words and reader are substitutable
  resources with a well-defined exchange rate; iso-fidelity contours (design §3) exist.
- I_m large (separability fails) ⇔ **thick**: words only work for capable readers — Daston's
  thick rule written down, and exactly E2's interaction term, now read as a **lack-of-fit
  statistic for the Chinchilla null**. "Connect to Chinchilla" thereby becomes a hypothesis
  test: *thin criteria obey a two-resource scaling law; thick criteria are the residual from it.*
  (§7/U2 sharpens this into a model-*selection* contest — the thick regime is specifically the
  **multiplicative-complementary** form of Montgomery et al. 2510.14919, not merely "a large
  residual".)
- The deep connection remains T2, which is *stronger* than curve-fitting: the quantization/Zipf
  mechanism **predicts** the exponent α (shared with dense data-scaling) rather than fitting it.
  A fitted law describes; a shared-exponent law explains. T2 is the headline; the parametric
  fit is its instrument.

**T7 — Separability as the thin/thick test.** (a) Criteria measured thin by independent
instruments (high words_share from the triad G-study, low floor B\*) fit the separable form
with small I_m; thick criteria show large I_m. (b) I_m predicts B\*(m) at least as well as the
E2 2×3 interaction (same coordinate; the fit pools the whole grid). (c) Fitted α_m agrees with
the parent task's norm-frequency Zipf tail exponent within CI (T2's consistency check executed
at metric level). Failure of (a) kills the thin=law/thick=residual reading (fit becomes
descriptive only); failure of (c) kills the shared-mechanism claim while leaving (a)/(b) intact.

**Decomposition hygiene (fixes to informal statements, so they don't re-tangle):**
1. Words and reader knowledge are *resources*; item difficulty is a *conditioning variable* —
   the frontier is estimated per difficulty stratum (triad layer, design §9), never averaged
   over it. Monotonicity in L holds for the frontier only (T1); realized performance is
   non-monotone (detail hurts weak readers).
2. "y = articulable + inarticulable + noise" is shorthand only: V and A are not additive inputs
   (T4 — layers not nested; the join family defines the articulable layer). The correct
   statement partitions outcome variance by *which observer family can extract it*.
3. Applicability of m (does it bear on x at all — the 0.5/NA channel) is distinct from m's
   score signal; the decomposition is of agreement-with-construct *among applicable items*.
   Keeping the channels separate prevents the relaxed-applicability 0.5 mass from contaminating
   fidelity estimates.
4. Metric-level ceilings exist without outcome anchors, all label-free: anchor scores
   (anchor-relative; E4 robustness swap), inter-anchor/inter-implementation agreement (the
   twin-ceiling / disattenuation denominator), and E0 synthetic constructs (absolute). What is
   missing at metric level is only an *outcome*-anchored ceiling — the preference level's job.

## 7. Updates from observational scaling + the annotation literature (2026-06-13)

Inputs: three Koyejo-line scaling papers — **Ruan/Maddison/Hashimoto, Observational Scaling Laws**
(2405.10938; `E_m≈h·σ(βᵀS_m+α)`, capability `S_m≈θ_f·logC_m+ν_f`, families differ *only* in the
compute→capability efficiency θ_f); **Schaeffer et al., "Why predicting downstream w/ Scale stays
elusive"** (2406.04391; the metric chain `log pⱽᵒᶜᵃᵇ(correct)→pᶜʰᵒⁱᶜᵉˢ→argmax` progressively
decorrelates from capability, incorrect-choice mass is the killer step); **Truong/Tu/Schaeffer/Koyejo,
IRSL** (2606.07616; IRT factorization `p=σ(d_j(θ_i−z_j))`, `θ_i≈a·logFLOP+b`, Beta-IRT for continuous
responses) — the **Montgomery et al. context-aware joint law** (2510.14919) — and a 171-paper sweep of
the LLMs-for-annotation literature (`2026-06-13__llm-annotation-litreview.md`). Six updates, each
tagged with the section it modifies.

**U1 — The capability axis C is empirical/latent, and the instrument now exists** (refines §6, the
Relative-Scaling amendment, E1's x-axis). We already required "per-criterion empirical capability,
never nominal size." Ruan's PCA capability space and IRSL's latent ability θ *are* that axis;
concretely we can fit judge ability θ from our own long table (U4). Importable companions: **Engels
2025** (scalable-oversight scaling makes the judge↔target *capability gap* a first-class, Double-ReLU
axis); **Krumdick 2025** ("No Free Labels": reference-free judge agreement is upper-bounded by judge
competence). Consequence: articulability is reported *as a function of judge capability*, and a written
rubric is a **capability substitute** (Beyer 2025: prompt design > raw judge size). This is the formal
home for the chat's verdict that "compare across families at fixed N" is mis-specified — re-express
family as position on the capability axis.

**U2 — The null is a model-*selection* contest: additive-separable (ours) vs multiplicative-
complementary (Montgomery)** (refines §6 null-model fit + T7a). Our §6 form is additive in two
power-law decays. Montgomery's is **multiplicative**:
`P = [1−e^{−A(L/Lᶜ)^α}]·[1−e^{−B(C/Cᶜ)^β}]·σ(L−L_ctx)` — and "you only benefit from words to the
degree capability lets you use them" *is* our thick-rule claim. Expanding,
`1−(1−e^{−AL^α})(1−e^{−BC^β}) = e^{−AL^α}+e^{−BC^β} − e^{−(AL^α+BC^β)}`: the cross term is a *specific*
coupling. So **T7a sharpens**: thin ⇔ the additive null fits (`I_m≈0`); thick ⇔ the free `I_m`
collapses onto Montgomery's cross-term shape. Thickness = position on the additive↔multiplicative axis,
a named importable competitor rather than an unstructured residual. Empirical precedent that an
*instruction*-budget law is multiplicative-decay: **IFScale (Jaroslawicz 2025)**, prompt-acc
≈ (instr-acc)^n with a per-model instruction budget (~150–200) before collapse. *Decay-shape caveat:*
ours uses power-law `L^{−α}` (T2 Zipf-quanta), Montgomery/IFScale use saturating exponentials;
power-vs-exponential is already T7's multi-form diagnostic (heavy tail = creative-writing never
saturates; fast saturation = press-release plateau).

**U3 — Fidelity must be a *continuous* signal, not the hard 0/0.5/1 label** (new measurement
refinement; prerequisite for the U2/T7 fits). Schaeffer shows the argmax/hard-label is the maximally
*decorrelating* node of the metric chain — exactly what makes downstream scaling look noisy/elusive.
Read fidelity off the judge's **logprob / probability-over-label** (or a Brier-style continuous
agreement), so frontiers are smooth and `α_m` is identifiable. Two clean mappings: our applicability
**0.5/NA mass = Schaeffer's incorrect-choice mass** (keep the channel separate — hygiene #3 now has a
mechanism, not just a hygiene rule); and *nonlinearity-doesn't-average-out* re-justifies estimating
frontiers **per difficulty stratum, never pooled** (§9/triad).

**U4 — IRSL/IRT as the frontier estimator over the long table** (new tool; complements the §9
twin-ceiling). Our `(judge × item × rubric-version × pass)` array is exactly IRT's input. Fit a
**Beta-IRT** model with three factors: judge ability θ (= the C axis, U1), item difficulty z (= the
UQS strata, now *estimated* rather than assumed), and a **rubric-informativeness** parameter. Payoffs:
separates the three confounds we keep fighting (item-hardness / judge-capability / rubric-quality);
natively handles continuous scores (U3); `O(M+N)` + ~50 items/cell relieves the pilot's
n≥60×3×3 cost; θ supplies U1's capability axis from our own data. **Collision (cite + differentiate):**
**Choi 2026** (= arXiv:2602.00521) already does IRT-for-*judge-reliability* (prompt-variants as items)
and IRSL does IRT-for-*scaling* — so our defensible delta is narrowly the **rubric-informativeness
factor + the articulability-frontier framing**, not "IRT for judges" per se. IRSL's documented failure
mode (low item-difficulty variance) = our low-UQS-spread signal.

**⚠ OPEN QUESTION — provenance + identifiability (do not overstate).** The rubric-informativeness
factor is **ours**, in no cited paper. IRSL (2606.07616) is strictly *unidimensional* with no
prompt/rubric/judge parameter; it states only that item difficulties *"may not transfer to different
conditions"* (its words — the "folded into z" reading is our gloss, not a quote). Choi (2602.00521)
treats θ as the *judged item's* quality, fits each judge **independently**, and uses model scale only
as a descriptive correlate of a consistency metric — it does **not** place judges on a common ability
axis, so **judge-articulability-vs-judge-scale is still open after Choi.** Our proposed move: add a rubric/condition factor — either a **threshold shift τ_r**
`σ(d_j·(θ_i−z_j−τ_r))` or a **discrimination scaler κ_r** `σ((d_j·κ_r)(θ_i−z_j))` — *not* IRSL's.
**Provenance, verified 2026-06-13 (drop the earlier "standard explanatory IRT" phrasing — it was an
overclaim):**
- τ_r is **squarely standard** — exactly the rater-severity facet of the **Many-Facet Rasch Model**
  (Linacre 1989), the field's dominant rater model.
- κ_r is **documented-but-niche**: a group-varying-discrimination / **nonuniform-DIF** parameter
  (Swaminathan & Rogers 1990; Berger & Tutz 2016) ≈ rater **centrality/scale-usage** models
  (**Jin & Eckes 2022** "dual DRF facets model" is nearly our exact parameterization; Jin & Wang 2018;
  Myford & Wolfe 2003/04). It is **not** explanatory IRT (LLTM/De Boeck & Wilson put covariates on
  *difficulty*, not the slope; closest slope-covariate work = Embretson 1999 / Petscher et al. 2020).
- **Semantic fork that matters:** articulability ≈ "does the rubric let the judge *resolve* good from
  bad" = discrimination = **κ_r** — but κ_r is the niche, harder, identifiability-fragile one; τ_r is
  safe/standard but only a leniency/calibration shift, not resolving power.
- **Identifiability (textbook, verified):** the `d_j·κ_r` product is the classic **2PL scale/unit
  indeterminacy** (Noventa et al. 2024; San Martín et al. 2015) — identifiable *only* by anchoring a
  reference condition to N(0,1); and varying discrimination is provably harder than additive severity
  (Andersen 1977 sufficiency; Rijmen et al. 2003 product-of-parameters leaves the GLMM family;
  nonuniform DIF is lower-power than uniform).

First test = prototype on the existing long table under an anchoring constraint; **if κ_r does not
separate from z, this estimator is out.** Treat the whole MIRT proposal as a hypothesis, not a result.

**The articulability *ceiling* needs 4PL, and the dense judge is what identifies it** (from the
irreducible-E deep dive, `2026-06-13__irreducible-E-scaling-laws.md` §5.4;
[[project_irreducible_E_estimation_2026_06_13]]). The IRT curve is fidelity-vs-judge-capability θ; its
**upper asymptote** (θ→∞) = the max fidelity any judge can reach with the rubric = `1 − tacit residual`
= our `τ(E)` object. This is distinct from κ_r: **κ_r is the slope (how fast fidelity rises with
capability); the asymptote `d` is where it plateaus** — `d` is the more direct articulability object.
**2PL/Beta-IRT (IRSL, Choi) are built on a plain sigmoid that always →1.0, so they structurally pin the
ceiling at 1.0** — assuming the tacit residual to zero, i.e. assuming away what we want to measure. Only
a **4PL** form `fidelity(θ)=c+(d−c)·σ(d_j·κ_r·(θ−z))` frees the upper asymptote `d<1` to be *estimated*.
But an asymptote describes the *top* of the capability range, so `d` is identifiable only if the judge
panel contains judges near the ceiling — observe a panel that's all on the upslope and you cannot
distinguish "heading to 1.0" from "about to plateau at 0.7" (4PL upper asymptotes are weakly identified
without high-ability respondents — Barton & Lord 1981, helped fit in only 2/4 datasets). **The dense
model is exactly that high-θ anchor**, so it does double duty: Bayes-floor proxy on the target side
(U1/T5) *and* the high-ability respondent that makes the per-item ceiling estimable on the judge side.
2PL-only ⇒ ceiling assumed; 4PL + dense-anchor ⇒ ceiling measured. **Caveat:** κ_r (slope) + 4PL c,d
(two asymptotes) + a judge-θ dimension is parameter-heavy and the identifiability compounds — the
prototype must show the ceiling *separates* from slope and difficulty, not merely that we added params.

**U5 — Inverse-scaling / gap-compression externally confirms the seed-to-frontier + subculture-knee
predictions** (refines §3.1 contributor #4, E1's self-articulation column). Black-box prompt-opt gains
shrink with scale (12%→5.9%→1.1%); PRIME's "effective-parameter-ratio" / ~60% slope-cut. This is the
external twin of our prediction that articulation lift (GEPA frontier − seed) shrinks as judge
capability rises and that subculture-specificity is a *knee modulator* not a τ term. Named calibration
target: our **seed-to-frontier gap vs judge capability should trace the same inverse-scaling curve**.

**U6 — Measure the L budget in tokenizer-invariant units** (refines the §1 frontier definition + §6
L axis). Raw judge-tokenizer tokens are not comparable across tiers (tokenizer-relative — the chat's
sharpest methodological point). Budget in **rules/clauses/"quanta"** (which ties L directly to T2's
Zipf-quanta) or bytes, so `α_m` is comparable across tiers — which T7c's exponent-agreement check
requires.

**The open seam is our seam.** The chat's unanswered question — "is there residual metric/family-
specific structure in the prompt-length exponent after controlling for capability and normalizing
tokens?" — is answered *at the construct level* by our per-metric `α_m` across tiers with capability
controlled (U1) and budget normalized (U6): the **first per-construct, capability-controlled
prompt-budget scaling exponent**. The annotation review's six "nobody-has-done-this" slots map onto
our experiments: slot 1 (controlled instruction-budget articulability curve under no-gold) = E7;
slot 2 (prompt-KIND × LENGTH × capability 3-way) = E7×E2; slot 5 (operator × metric-type lift
attribution) = E3×the GEPA-operator lineage.

**Threat to disarm (Baumann 2025, "LLM hacking").** Across 13M labels, ~31% of downstream conclusions
were wrong and prompt KIND explained <1% of conclusion-correctness variance. So a single-phrasing
articulability number is exploitable. This sharpens **T5** (winner's curse) into a hard requirement:
report articulability as a **distribution/interval over rubric phrasings** (Sclar FormatSpread; Polo
PromptEval), and correct any outcome-anchored estimate with **PPI (Angelopoulos 2023) / DSL (Egami
2023)**. Disagreement-as-signal (Plank 2022; Nie 2020; Sandri 2023's resolvable-vs-irreducible split)
is the formal twin of our articulable-vs-taste partition and predicts the sub-1.0 ceiling is *not*
methodological failure.

## 8. Unifying V and A: V is a special case of articulability (2026-06-13)

The point of this section is **unification, not separation**: V is not a different kind of object from
A — it is *articulability with a formal executor*. (Re-stated for self-containment, in **fidelity** units — higher = better, matching §6/E7: let
`F_E(L)` = best agreement-with-anchor by a ≤L-word description read by E, and `F_C` = the dense/anchor
**ceiling** fidelity. The gap `G_E(L) = F_C − F_E(L) ≥ 0` — how far the best ≤L-word description still
falls short of the ceiling. `τ(E) = lim_{L→∞} G_E(L) = F_C − F_E(∞) = F_C − ceiling(E)`, the tacitness
w.r.t. E (`= 1 − ceiling(E)` only if fidelity is normalized so the dense ceiling `F_C = 1`). §1 writes
the identical gap in *risk* units as `R_E(L) − R_C` — same quantity, opposite sign convention.)

**The genus A.** Articulability — the genus, call it **A** — is `articulability(m; E, x)` (§1) ranged
over the **executor lattice** (compiler → weak LLM → strong LLM → dense/human), partially ordered, with
the articulable layer = the **join** over the lattice (T4: executors are not totally ordered — code
reaches exact counts an LLM can't, and vice versa).

**V ⊆ A — V is a species of the genus.** Set the executor to a **deterministic formal executor**
(compiler/interpreter) and the description to a **formal language** (code): the result is a verifiable
metric. The triple is unchanged — `(rater, description, item)` with rater = compiler, description =
code. So **every V metric is an articulability measurement**, and V is a *subset* of A, not a parallel
track. (The old disjoint "V / A / Taste layers" are simply the pieces of this one object: V, the
LLM-only increment `A∖V`, and the residual `¬A` = Taste.)

**Why the V species looks so different — an IRT degeneracy.** Plug a deterministic formal executor into
the response curve and it collapses from a sigmoid to a step:
- the executor's **ability drops out** — every correct compiler runs identical code, so **θ collapses**;
- **rater noise → 0**, **discrimination d → ∞** (a step at the decision boundary);
- on a V-**amenable** item the **residual is exactly 0** ⇒ `τ(compiler) = 0`, **ceiling = 1**.
The compiler is the **perfect-but-narrow rater corner** of A. These "special properties" are exactly
the boundary case the MIRT estimator (§6 U4) must reduce to — a built-in sanity check.

**V-amenability is per-(construct × item).** `τ(compiler)` is a **step** over constructs (0 where a
finite correct code exists, the whole gap otherwise) where `τ(LLM)` is **graded**; and even for a V
construct it only fires on the items the verifier covers. So one item bundles a V-construct
(correctness), an `A∖V`-construct (readability), and a Taste-construct (elegance) simultaneously.
Coverage = where V fires (ties to [[project_v_conditional_residue]]: V = a thin, sharp, decisive tail).

**V / A / Taste = three regimes of the one `τ(E)` curve** — the **threshold executor at which τ first
reaches 0**: V = the compiler already suffices (`τ(compiler)=0`); `A∖V` = it doesn't, but the join with
an LLM does; Taste = no executor in the lattice closes it (`τ(dense)>0`, i.e. outside A). One curve,
three thresholds — unification, not three mechanisms. T4 is *why V is non-redundant inside A*: the
compiler contributes to the join what the LLM alone can't, so you keep it — `v_struct`
(judge-grounds→code-verifies) is exactly using both arms of the join.

**V-amenable items are the calibration anchors of the whole instrument.** Because the answer is *known*
there (uncorrupted verifier), V items are the **ground-truth island** in the no-ground-truth sea: they
(i) **anchor the latent IRT scale** — known answer, ceiling=1, d→∞ — the constraint that pins the
otherwise-unidentified κ_r and 4PL ceiling (§6 U4); (ii) give **E0 a real-data form** (real
planted-thickness, not synthetic); (iii) let the **V-vs-LLM gap on the same construct measure the
language-tacit increment `A∖V`** directly. Slogan: **V = where you check the ruler against a known
length; A = where you use the ruler on unknown lengths** — the structural answer to "no clean ground
truth": V is the clean corner that validates the instrument we then carry into the messy corner.

**Honesty.** (a) **Corrupted-V = the anchor-is-fallible case** — a wrong test/mislabel makes the
compiler confidently wrong; V coincides with ground truth only if verifier *and* data are clean (§1
anchor note). (b) Real verifiers carry a small **indeterminacy floor** (flaky/timeout/env — the
test-transplant pinned/vacuous/indeterminate buckets), so "deterministic" is an idealization. (c) There
is still a **cost axis inside V** — a one-line `assert` vs a reference implementation; `τ(compiler)=0`
is depth, program length is the knee.

**Positioning.** A is then the **map of verifiable-reward availability** the RLVR literature implicitly
navigates: V = where a free perfect reward exists (RLVR's home), `A∖V` = where you fall back to an
LLM-judge reward model, Taste = where even that fails.

### 8.1 The information-theoretic substrate (corrected, 2026-06-14)

**What is X, what is the family.** The variable is the **datapoint** (the item being judged): `X = datapoint`,
`Y = anchor judgment`. The **rubric is not an input variable — it is the decoder/program** that configures the
predictor. So the constraint family is `V_{L,E} = {E(s) : |s| ≤ L}` (executor `E` reading any description `s`
of length ≤ `L`), exactly as §2 states. Articulability-at-budget-`L` is `I_{V_{L,E}}(X → Y)` — the usable
information the *datapoint* carries about the anchor, **decoded by the best ≤L-word-rubric'd executor** (the
`inf` over rubrics of that length). A **fixed** rubric `R` gives a sub-family `V_{E,R} ⊆ V_{L,E}`; the
budget *curve* is the envelope over rubrics. *(Correction of a chat shorthand that put `X = rubric`: the
rubric indexes the family, the datapoint is X.)*

**The rubric's marginal value is a CONDITIONAL V-information**, not the plain `I_V(X→Y)`. The right object is
`ΔI(R) = H_{V_{∅,E}}(Y | X) − H_{V_{L,E}}(Y | X)` — how much equipping the executor with rubric `R` lowers its
loss in reproducing the anchor *from the datapoint*, relative to the **empty-rubric, datapoint-present**
baseline `H_{V_{∅,E}}(Y | X)` (REV, Chen et al. 2023; defeats label-leaking rubrics). This is why we now log
the empty-rubric baseline as a first-class column (`__vacuous_baseline__`, operator `BASELINE`,
`batch_scoring.vacuous_baseline_artifact`): subtract it from any rubric to get marginal usable info, and split
**substitutive** tacit knowledge (baseline already high — the executor knew it without the rubric, so the
rubric's measured value is small even though fidelity is high) from **complementary** (rubric *unlocks* more
than its standalone value). Note the two distinct baselines: V-info's optional-ignorance term `H_V(Y)` is
*no-datapoint* (the rubric-induced label prior), whereas the actionable one for the tacit split is
*no-rubric-with-datapoint* `H_{V_{∅,E}}(Y|X)` (the executor's holistic tacit prior applied to the item) — the
latter is what the column holds.

**`τ(E)` restated.** `τ(E) = lim_{L→∞} G_E(L)` = the gap from the ceiling that survives unlimited words, for
executor `E`. In information units, `τ(E) = [I(X;Y) − sup_L I_{V_{L,E}}(X→Y)]` netted to the irreducible
anchor-noise floor `H*(Y|X)`. It is tacitness **relative to E**, not "tacit knowledge *of* E": the part of
the anchor an executor of capability/background `E` cannot be made to reproduce via *any* finite written
rubric. Executor-monotone in the limit (`V_{L,E₁} ⊆ V_{L,E₂}` ⇒ more usable info), and **possibly
non-monotone in `E` at finite L** (verbal-overshadowing / CoT inverse-scaling, lit review 2026-06-14) — fit a
spline, test for an inverted-U, do not assume monotone.

**One estimator for V-as-measurement and the IRT (DeCarlo 1998).** The imitation-game / discrimination test —
can a held-out judge tell an anchor verdict from a rubric-pipeline verdict? — fit as a probit GLMM has
slope = `d′`, intercept = `−criterion`, which **is** our 2-parameter probit IRF `Φ(d_j + a_j·θ)`. So the
sociology-of-science tacit-knowledge measurement (Collins) and our judge-IRT are the *same fit*. The logged
empty-rubric baseline supplies the matched null for it: `d′(anchor-vs-rubric) − d′(anchor-vs-vacuous)` isolates
the rubric's contribution, and a second `anchor-vs-anchor` null subtracts label noise (see
[[project_tacit_measurement_litreview_2026_06_14]] for the cross-field menu of how the `τ(E)` ceiling is
quantified elsewhere: G-theory residual / cross-replication reliability, IRL invariance-class diameter, the
rate-distortion distortion-floor, codability name-entropy, the Brunswik configural residual ≈ .08).

### 8.2 How other fields quantify the `τ(E)` ceiling (cross-field menu)

Every mature measurement field has an object that plays our `τ(E)` role: **anchor minus the best
codified-channel reproduction**, an irreducible floor that survives unlimited effort within the channel.
The menu (full BibTeX in the References section):

| Field | The `τ(E)`-analog | Quantified as | Source |
|---|---|---|---|
| Psychometrics | reliability ceiling | `1 − xRR` / generalizability coefficient; MFRM residual variance | Wong, Paritosh & Aroyo 2021; Cronbach et al. 1972; Brennan 2001; Linacre 1989 |
| IRL / reward learning | identifiability limit | diameter of the invariance/equivalence class (**derivable**, not estimated) | Skalse et al. 2023 (partial identifiability; STARC) |
| Pragmatics / info theory | distortion floor | rate-distortion minimum at infinite rate (IB frontier) | Zaslavsky, Kemp, Regier & Tishby 2018 |
| Psycholinguistics | (un)nameability | codability = Shannon entropy of names produced for a stimulus (smell ≫ colour) | Brown & Lenneberg 1954; Majid et al. 2018; van Hoef 2024 |
| Judgment analysis | nonlinear residual | Brunswik lens-model configural `C = 1 − R²` of the linear cue model (avg ≈ **.08**) | Karelaia & Hogarth 2008 |
| Info theory (ours) | usable-info gap | `I(X;Y) − sup_L I_{V_{L,E}}(X→Y)` netted to `H*(Y\|X)` | Xu et al. 2020; Chen et al. 2023 (REV) |

Two of these are **derivable rather than estimated** — the IRL invariance-class diameter (a closed-form
property of the reward equivalence class) and the rate-distortion floor (a property of the source) — which is
the strongest argument that `τ(E)` is a real quantity and not an estimator artifact: when the channel is
specified, the floor exists before any data. The other four are estimated residuals, and all four carry the
same caveat we do — the estimate is only as trustworthy as the anchor's own replication reliability (the
psychometrics row is literally that caveat: `xRR` *is* the cross-replication ceiling). The lens-model `.08`
is the most striking import: across decades of human judgment studies, the *nonlinear* part of expert
judgment that a linear cue model misses averages only ~8% of variance — a strong prior that a fitted rubric
reconstructs most of the anchor linearly and the tacit residual lives in cross-item heterogeneity, not in
the mean response surface.


## References (auto-verified BibTeX, 2026-06-15)

> Citations below were extracted from this document and web-verified by an automated fact-check pass (search → fetch → retrieve resolvable id), with the attributed claim checked against the located paper. 64 entries; 7 also passed an independent second-pass audit (the rest were verified once — the audit pass was cut off by a quota limit, not by a failure). Entries are real located works; do not treat as hand-checked. See "needs manual review" below for 0 citations whose attributed claim the source paper appears to **contradict** and 0 unlocatable shorthands.

```bibtex
@article{andersen1977sufficient,
  author  = {Andersen, Erling B.},
  title   = {Sufficient Statistics and Latent Trait Models},
  journal = {Psychometrika},
  volume  = {42},
  number  = {1},
  pages   = {69--81},
  year    = {1977},
  doi     = {10.1007/BF02293746}
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

@techreport{barton1981upper,
  author      = {Barton, Mark A. and Lord, Frederic M.},
  title       = {An Upper Asymptote for the Three-Parameter Logistic Item-Response Model},
  institution = {Educational Testing Service},
  year        = {1981},
  month       = {July},
  number      = {RR-81-20},
  type        = {ETS Research Report},
  series      = {ETS Research Report Series},
  doi         = {10.1002/j.2333-8504.1981.tb01255.x}
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

@article{berger2016detection,
  author  = {Berger, Moritz and Tutz, Gerhard},
  title   = {Detection of Uniform and Nonuniform Differential Item Functioning by Item-Focused Trees},
  journal = {Journal of Educational and Behavioral Statistics},
  volume  = {41},
  number  = {6},
  pages   = {559--592},
  year    = {2016},
  doi     = {10.3102/1076998616659371}
}

@misc{besiroglu2024chinchilla,
  title={Chinchilla Scaling: A replication attempt},
  author={Besiroglu, Tamay and Erdil, Ege and Barnett, Matthew and You, Josh},
  year={2024},
  eprint={2404.10102},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2404.10102},
  url={https://arxiv.org/abs/2404.10102}
}

@book{brennan2001generalizability,
  author    = {Brennan, Robert L.},
  title     = {Generalizability Theory},
  series    = {Statistics for Social and Behavioral Sciences},
  year      = {2001},
  publisher = {Springer},
  address   = {New York},
  isbn      = {0387952829},
  doi       = {10.1007/978-1-4757-3456-0}
}

@article{brown1954study,
  author  = {Brown, Roger W. and Lenneberg, Eric H.},
  title   = {A study in language and cognition},
  journal = {The Journal of Abnormal and Social Psychology},
  year    = {1954},
  volume  = {49},
  number  = {3},
  pages   = {454--462},
  doi     = {10.1037/h0057814}
}

@inproceedings{caballero2023broken,
  title={Broken Neural Scaling Laws},
  author={Caballero, Ethan and Gupta, Kshitij and Rish, Irina and Krueger, David},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023},
  note={arXiv:2210.14891}
}

@inproceedings{chen2023rev,
    title = "{REV}: Information-Theoretic Evaluation of Free-Text Rationales",
    author = "Chen, Hanjie and Brahman, Faeze and Ren, Xiang and Ji, Yangfeng and Choi, Yejin and Swayamdipta, Swabha",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    pages = "2007--2030",
    doi = "10.18653/v1/2023.acl-long.112",
    url = "https://aclanthology.org/2023.acl-long.112/"
}

@inproceedings{chen2023reva,
    title = "{REV}: Information-Theoretic Evaluation of Free-Text Rationales",
    author = "Chen, Hanjie and Brahman, Faeze and Ren, Xiang and Ji, Yangfeng and Choi, Yejin and Swayamdipta, Swabha",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.112/",
    doi = "10.18653/v1/2023.acl-long.112",
    pages = "2007--2030"
}

@misc{choi2026diagnosing,
  title        = {Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory},
  author       = {Choi, Junhyuk and Park, Sohhyung and Cho, Chanhee and Park, Hyeonchu and Kim, Bugeun},
  year         = {2026},
  eprint       = {2602.00521},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}

@incollection{clark1991grounding,
  author    = {Clark, Herbert H. and Brennan, Susan E.},
  title     = {Grounding in communication},
  booktitle = {Perspectives on Socially Shared Cognition},
  editor    = {Resnick, Lauren B. and Levine, John M. and Teasley, Stephanie D.},
  publisher = {American Psychological Association},
  address   = {Washington, DC},
  pages     = {127--149},
  year      = {1991}
}

@article{collins2014quantifying,
  title={Quantifying the Tacit: The Imitation Game and Social Fluency},
  author={Collins, Harry and Evans, Robert},
  journal={Sociology},
  volume={48},
  number={1},
  pages={3--19},
  year={2014},
  publisher={SAGE Publications},
  doi={10.1177/0038038512455735}
}

@book{cronbach1972dependability,
  title     = {The Dependability of Behavioral Measurements: Theory of Generalizability for Scores and Profiles},
  author    = {Cronbach, Lee J. and Gleser, Goldine C. and Nanda, Harinder and Rajaratnam, Nageswari},
  year      = {1972},
  publisher = {John Wiley \& Sons},
  address   = {New York},
  isbn      = {9780471188506},
  pages     = {410}
}

@article{damour2022underspecification,
  title   = {Underspecification Presents Challenges for Credibility in Modern Machine Learning},
  author  = {D'Amour, Alexander and Heller, Katherine and Moldovan, Dan and Adlam, Ben and Alipanahi, Babak and Beutel, Alex and Chen, Christina and Deaton, Jonathan and Eisenstein, Jacob and Hoffman, Matthew D. and Hormozdiari, Farhad and Houlsby, Neil and Hou, Shaobo and Jerfel, Ghassen and Karthikesalingam, Alan and Lucic, Mario and Ma, Yian and McLean, Cory and Mincu, Diana and Mitani, Akinori and Montanari, Andrea and Nado, Zachary and Natarajan, Vivek and Nielson, Christopher and Osborne, Thomas F. and Raman, Rajiv and Ramasamy, Kim and Sayres, Rory and Schrouff, Jessica and Seneviratne, Martin and Sequeira, Shannon and Suresh, Harini and Veitch, Victor and Vladymyrov, Max and Wang, Xuezhi and Webster, Kellie and Yadlowsky, Steve and Yun, Taedong and Zhai, Xiaohua and Sculley, D.},
  journal = {Journal of Machine Learning Research},
  volume  = {23},
  number  = {226},
  pages   = {1--61},
  year    = {2022},
  note    = {Preprint arXiv:2011.03395 (2020)}
}

@article{darwiche2002knowledge,
  title   = {A Knowledge Compilation Map},
  author  = {Darwiche, Adnan and Marquis, Pierre},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {17},
  pages   = {229--264},
  year    = {2002},
  doi     = {10.1613/jair.989}
}

@book{daston2022rules,
  author    = {Daston, Lorraine},
  title     = {Rules: A Short History of What We Live By},
  series    = {The Lawrence Stone Lectures},
  publisher = {Princeton University Press},
  address   = {Princeton, NJ},
  year      = {2022},
  isbn      = {9780691156989},
  doi       = {10.1515/9780691239187}
}

@book{deboeck2004explanatory,
  title     = {Explanatory Item Response Models: A Generalized Linear and Nonlinear Approach},
  editor    = {De Boeck, Paul and Wilson, Mark},
  year      = {2004},
  publisher = {Springer},
  address   = {New York},
  series    = {Statistics for Social Science and Public Policy},
  isbn      = {978-0-387-40275-8},
  doi       = {10.1007/978-1-4757-3990-9}
}

@article{decarlo1998signal,
  author  = {DeCarlo, Lawrence T.},
  title   = {Signal detection theory and generalized linear models},
  journal = {Psychological Methods},
  year    = {1998},
  volume  = {3},
  number  = {2},
  pages   = {186--205},
  doi     = {10.1037/1082-989X.3.2.186}
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

@article{embretson1999generating,
  author  = {Embretson, Susan E.},
  title   = {Generating Items During Testing: Psychometric Issues and Models},
  journal = {Psychometrika},
  volume  = {64},
  number  = {4},
  pages   = {407--433},
  year    = {1999},
  doi     = {10.1007/BF02294564}
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

@inproceedings{ethayarajh2022understanding,
  title     = {Understanding Dataset Difficulty with {$\mathcal{V}$}-Usable Information},
  author    = {Ethayarajh, Kawin and Choi, Yejin and Swayamdipta, Swabha},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  pages     = {5988--6008},
  year      = {2022},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v162/ethayarajh22a.html},
  note      = {arXiv:2110.08420}
}

@misc{henighan2020scaling,
  title={Scaling Laws for Autoregressive Generative Modeling},
  author={Henighan, Tom and Kaplan, Jared and Katz, Mor and Chen, Mark and Hesse, Christopher and Jackson, Jacob and Jun, Heewoo and Brown, Tom B. and Dhariwal, Prafulla and Gray, Scott and Hallacy, Chris and Mann, Benjamin and Radford, Alec and Ramesh, Aditya and Ryder, Nick and Ziegler, Daniel M. and Schulman, John and Amodei, Dario and McCandlish, Sam},
  year={2020},
  eprint={2010.14701},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@article{hoffmann2022training,
  title={Training Compute-Optimal Large Language Models},
  author={Hoffmann, Jordan and Borgeaud, Sebastian and Mensch, Arthur and Buchatskaya, Elena and Cai, Trevor and Rutherford, Eliza and de Las Casas, Diego and Hendricks, Lisa Anne and Welbl, Johannes and Clark, Aidan and Hennigan, Tom and Noland, Eric and Millican, Katie and van den Driessche, George and Damoc, Bogdan and Guy, Aurelia and Osindero, Simon and Simonyan, Karen and Elsen, Erich and Rae, Jack W. and Vinyals, Oriol and Sifre, Laurent},
  journal={arXiv preprint arXiv:2203.15556},
  year={2022},
  eprint={2203.15556},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@misc{hutter2021learning,
  title         = {Learning Curve Theory},
  author        = {Hutter, Marcus},
  year          = {2021},
  eprint        = {2102.04074},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  howpublished  = {arXiv preprint arXiv:2102.04074}
}

@misc{jaroslawicz2025how,
  title        = {How Many Instructions Can LLMs Follow at Once?},
  author       = {Jaroslawicz, Daniel and Whiting, Brendan and Shah, Parth and Maamari, Karime},
  year         = {2025},
  eprint       = {2507.11538},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2507.11538}
}

@inproceedings{jiang2024rora,
    title = "{RORA}: Robust Free-Text Rationale Evaluation",
    author = "Jiang, Zhengping and Lu, Yining and Chen, Hanjie and Khashabi, Daniel and Van Durme, Benjamin and Liu, Anqi",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    pages = "1070--1087",
    doi = "10.18653/v1/2024.acl-long.60",
    url = "https://aclanthology.org/2024.acl-long.60/"
}

@article{jin2018new,
  author  = {Jin, Kuan-Yu and Wang, Wen-Chung},
  title   = {A New Facets Model for Rater's Centrality/Extremity Response Style},
  journal = {Journal of Educational Measurement},
  volume  = {55},
  number  = {4},
  pages   = {543--563},
  year    = {2018},
  doi     = {10.1111/jedm.12191}
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

@article{karelaia2008determinants,
  author  = {Karelaia, Natalia and Hogarth, Robin M.},
  title   = {Determinants of linear judgment: A meta-analysis of lens model studies},
  journal = {Psychological Bulletin},
  year    = {2008},
  volume  = {134},
  number  = {3},
  pages   = {404--426},
  doi     = {10.1037/0033-2909.134.3.404}
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

@book{linacre1989manya,
  author    = {Linacre, John Michael},
  title     = {Many-Facet Rasch Measurement},
  year      = {1989},
  publisher = {MESA Press},
  address   = {Chicago, IL},
  isbn      = {0-941938-02-6}
}

@article{list2002aggregating,
  title   = {Aggregating Sets of Judgments: An Impossibility Result},
  author  = {List, Christian and Pettit, Philip},
  journal = {Economics \& Philosophy},
  volume  = {18},
  number  = {1},
  pages   = {89--110},
  year    = {2002},
  publisher = {Cambridge University Press},
  doi     = {10.1017/S0266267102001098}
}

@inproceedings{michaud2023quantization,
  title={The Quantization Model of Neural Scaling},
  author={Michaud, Eric J. and Liu, Ziming and Girit, Uzay and Tegmark, Max},
  booktitle={Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year={2023},
  eprint={2303.13506},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2303.13506}
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

@article{myford2003detecting,
  author  = {Myford, Carol M. and Wolfe, Edward W.},
  title   = {Detecting and measuring rater effects using many-facet {Rasch} measurement: {Part} {I}},
  journal = {Journal of Applied Measurement},
  year    = {2003},
  volume  = {4},
  number  = {4},
  pages   = {386--422},
  pmid    = {14523257}
}

@article{myford2004detecting,
  author  = {Myford, Carol M. and Wolfe, Edward W.},
  title   = {Detecting and measuring rater effects using many-facet {Rasch} measurement: {Part} {II}},
  journal = {Journal of Applied Measurement},
  year    = {2004},
  volume  = {5},
  number  = {2},
  pages   = {189--227}
}

@inproceedings{nie2020what,
    title = "What Can We Learn from Collective Human Opinions on Natural Language Inference Data?",
    author = "Nie, Yixin and Zhou, Xiang and Bansal, Mohit",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    pages = "9131--9143",
    doi = "10.18653/v1/2020.emnlp-main.734"
}

@article{nisbett1977telling,
  author  = {Nisbett, Richard E. and Wilson, Timothy D.},
  title   = {Telling more than we can know: Verbal reports on mental processes},
  journal = {Psychological Review},
  year    = {1977},
  volume  = {84},
  number  = {3},
  pages   = {231--259},
  doi     = {10.1037/0033-295X.84.3.231}
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

@article{petscher2020past,
  author  = {Petscher, Yaacov and Compton, Donald L. and Steacy, Laura and Kinnon, Hannah},
  title   = {Past perspectives and new opportunities for the explanatory item response model},
  journal = {Annals of Dyslexia},
  year    = {2020},
  volume  = {70},
  number  = {2},
  pages   = {160--179},
  doi     = {10.1007/s11881-020-00204-y},
  pmid    = {32728972}
}

@inproceedings{plank2022problem,
    title = "The ``Problem'' of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation",
    author = "Plank, Barbara",
    editor = "Goldberg, Yoav and Kozareva, Zornitsa and Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.731/",
    doi = "10.18653/v1/2022.emnlp-main.731",
    pages = "10671--10682"
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

@article{rijmen2003nonlinear,
  author  = {Rijmen, Frank and Tuerlinckx, Francis and De Boeck, Paul and Kuppens, Peter},
  title   = {A nonlinear mixed model framework for item response theory},
  journal = {Psychological Methods},
  year    = {2003},
  volume  = {8},
  number  = {2},
  pages   = {185--205},
  doi     = {10.1037/1082-989X.8.2.185},
  pmid    = {12924814}
}

@inproceedings{ruan2024observational,
  title={Observational Scaling Laws and the Predictability of Language Model Performance},
  author={Ruan, Yangjun and Maddison, Chris J. and Hashimoto, Tatsunori},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  eprint={2405.10938},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{salinas2025tuning,
  title        = {Tuning LLM Judge Design Decisions for 1/1000 of the Cost},
  author       = {Salinas, David and Swelam, Omar and Hutter, Frank},
  year         = {2025},
  eprint       = {2501.17178},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  note         = {Accepted as a poster at ICML 2025},
  url          = {https://arxiv.org/abs/2501.17178}
}

@inproceedings{sandri2023why,
    title = "Why Don't You Do It Right? Analysing Annotators' Disagreement in Subjective Tasks",
    author = "Sandri, Marta and Leonardelli, Elisa and Tonelli, Sara and Jezek, Elisabetta",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    pages = "2428--2441",
    doi = "10.18653/v1/2023.eacl-main.178",
    url = "https://aclanthology.org/2023.eacl-main.178/"
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

@article{schaekermann2018resolvable,
  title     = {Resolvable vs. Irresolvable Disagreement: A Study on Worker Deliberation in Crowd Work},
  author    = {Schaekermann, Mike and Goh, Joslin and Larson, Kate and Law, Edith},
  journal   = {Proceedings of the ACM on Human-Computer Interaction},
  volume    = {2},
  number    = {CSCW},
  articleno = {154},
  pages     = {1--19},
  year      = {2018},
  month     = {nov},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3274423}
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

@inproceedings{skalse2022defining,
  title     = {Defining and Characterizing Reward Hacking},
  author    = {Skalse, Joar and Howe, Nikolaus H. R. and Krasheninnikov, Dmitrii and Krueger, David},
  booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS 2022)},
  year      = {2022},
  eprint    = {2209.13085},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url       = {https://arxiv.org/abs/2209.13085}
}

@inproceedings{skalse2023invariance,
  title     = {Invariance in Policy Optimisation and Partial Identifiability in Reward Learning},
  author    = {Skalse, Joar and Farrugia-Roberts, Matthew and Russell, Stuart and Abate, Alessandro and Gleave, Adam},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML 2023)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  year      = {2023},
  publisher = {PMLR},
  eprint    = {2203.07475},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@misc{skalse2023starc,
  title        = {STARC: A General Framework For Quantifying Differences Between Reward Functions},
  author       = {Joar Skalse and Lucy Farnik and Sumeet Ramesh Motwani and Erik Jenner and Adam Gleave and Alessandro Abate},
  year         = {2023},
  eprint       = {2309.15257},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2309.15257}
}

@article{swaminathan1990detecting,
  author  = {Swaminathan, Hariharan and Rogers, H. Jane},
  title   = {Detecting Differential Item Functioning Using Logistic Regression Procedures},
  journal = {Journal of Educational Measurement},
  year    = {1990},
  volume  = {27},
  number  = {4},
  pages   = {361--370},
  doi     = {10.1111/j.1745-3984.1990.tb00754.x}
}

@misc{truong2026item,
  title={Item Response Scaling Laws: A Measurement Theory Approach for Efficient and Generalizable Neural Scaling Estimation},
  author={Truong, Sang and Tu, Yuheng and Schaeffer, Rylan and Koyejo, Sanmi},
  year={2026},
  eprint={2606.07616},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.07616}
}

@article{vereshchagin2004kolmogorov,
  author  = {Vereshchagin, Nikolai K. and Vit\'{a}nyi, Paul M. B.},
  title   = {Kolmogorov's Structure Functions and Model Selection},
  journal = {IEEE Transactions on Information Theory},
  volume  = {50},
  number  = {12},
  pages   = {3265--3290},
  year    = {2004},
  doi     = {10.1109/TIT.2004.838346}
}

@inproceedings{wong2021cross,
    title = "Cross-replication Reliability - An Empirical Approach to Interpreting Inter-rater Reliability",
    author = "Wong, Ka and Paritosh, Praveen and Aroyo, Lora",
    editor = "Zong, Chengqing and Xia, Fei and Li, Wenjie and Navigli, Roberto",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.548/",
    doi = "10.18653/v1/2021.acl-long.548",
    pages = "7053--7065"
}

@inproceedings{xu2020theory,
  title     = {A Theory of Usable Information under Computational Constraints},
  author    = {Xu, Yilun and Zhao, Shengjia and Song, Jiaming and Stewart, Russell and Ermon, Stefano},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020},
  url       = {https://openreview.net/forum?id=r1eBeyHFDH},
  eprint    = {2002.10689},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}

@inproceedings{xu2020theorya,
  title={A Theory of Usable Information Under Computational Constraints},
  author={Xu, Yilun and Zhao, Shengjia and Song, Jiaming and Stewart, Russell and Ermon, Stefano},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
  url={https://arxiv.org/abs/2002.10689}
}

@article{zaslavsky2018efficient,
  author    = {Zaslavsky, Noga and Kemp, Charles and Regier, Terry and Tishby, Naftali},
  title     = {Efficient compression in color naming and its evolution},
  journal   = {Proceedings of the National Academy of Sciences},
  volume    = {115},
  number    = {31},
  pages     = {7937--7942},
  year      = {2018},
  doi       = {10.1073/pnas.1800521115},
  publisher = {National Academy of Sciences},
  issn      = {0027-8424}
}

@article{zaslavsky2018efficienta,
  title={Efficient compression in color naming and its evolution},
  author={Zaslavsky, Noga and Kemp, Charles and Regier, Terry and Tishby, Naftali},
  journal={Proceedings of the National Academy of Sciences},
  volume={115},
  number={31},
  pages={7937--7942},
  year={2018},
  publisher={National Academy of Sciences},
  doi={10.1073/pnas.1800521115}
}

```

### Citations needing manual review

**Partial claim-match (23)** — paper located, attributed claim only partly supported; spot-check before relying on the exact number/wording:

- `angelopoulos2023prediction`; `barton1981upper`; `berger2016detection`; `choi2026diagnosing`; `clark1991grounding`; `collins2014quantifying`; `damour2022underspecification`; `darwiche2002knowledge`; `decarlo1998signal`; `ethayarajh2022understanding`; `jaroslawicz2025how`; `jin2018new`; `jin2022detecting`; `karelaia2008determinants`; `noventa2024identifiability`; `rijmen2003nonlinear`; `sandri2023why`; `schaekermann2018resolvable`; `skalse2022defining`; `skalse2023starc`; `swaminathan1990detecting`; `vereshchagin2004kolmogorov`; `xu2020theory`
