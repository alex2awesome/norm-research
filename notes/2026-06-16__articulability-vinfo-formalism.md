# Articulability as V-information: a formal sketch

*Working notes, 2026-06-16. Goal: pin down the three quantities and how they compose, for the
articulability-of-evaluation-metrics program.*

## 0. Setup and the one primitive

- **X** — a datapoint (a story, a paper's text, a fact pattern).
- **m(X) ∈ {0,1}** — a *metric*'s verdict on X. A metric is an evaluation criterion (e.g. "the proof
  states a non-trivial result"). We work with binary verdicts, so all quantities are in **bits**.
- **Y ∈ {0,1}** — the *downstream outcome* we ultimately care about (accept/reject, first-draft
  approval, quality label).
- **Predictor family V** — a *set* of functions f: X → Δ({0,1}). This is the modeling choice that
  carries all the content.

The one primitive is **(predictive) V-information** (Xu, Song, Stefano, Ermon, ICLR 2020). For a
target Z and family V,
```
  H_V(Z)    = inf_{f ∈ V} E[ −log f[∅](Z) ]     # best achievable log-loss with NO input
  H_V(Z|X)  = inf_{f ∈ V} E[ −log f[X](Z) ]     # best achievable log-loss given X
  I_V(X → Z) = H_V(Z) − H_V(Z|X)                # usable bits X carries about Z, for family V
```
Properties we lean on: **I_V ≥ 0** (under optional ignorance), **monotone in V** (V ⊆ V′ ⇒ I_V ≤ I_{V′}),
**directed/asymmetric** (hence the arrow), and **I_V → Shannon I(X;Z)** as V → all measurable functions.
Unlike Shannon MI, V-information is *not* invariant under bijections of X — encrypted X has ~0
V-usable information for any realistic V. That non-invariance is exactly what makes "articulable *to a
bounded reader*" a well-posed quantity rather than a triviality.

We parameterize V by three **resource budgets**:

| budget | meaning |
|---|---|
| **L** | articulation length — tokens of the natural-language rule |
| **E** | executor model — the reader/capability that applies the rule |
| **M** | composition / coverage — how many metrics may be combined |

---

## 1. Articulability of a metric — I_{V_{L,E}}(X → m(X))

Predictor family:
```
  V_{L,E} = { x ↦ E(r, x)  :  r a natural-language rule, |r| ≤ L tokens }
```
where `E(r,x)` is executor E applying rule r to x (its P(YES)). Then

> **I_{V_{L,E}}(X → m(X))** = the usable bits X carries about the metric's verdict **when the only
> allowed predictors are length-L rules executed by E**.

This is the **articulability of metric m at budget (L,E)**: how much of m can be *written down in ≤ L
tokens and re-executed by a capability-E reader*. Monotone non-decreasing in L and E.

- **Ceiling.** As L,E → ∞, I_{V_{L,E}}(X→m) ↑ H(m(X)) iff m is a deterministic function of X
  (the metric *is* determined by the text). The shortfall
  `Δ_tacit(L,E) = H(m(X)) − I_{V_{L,E}}(X→m)` is the **tacit residual** of the metric at that budget —
  the part no length-L articulation run by E can recover.
- **Estimator.** Reconstruct the best length-L rule from m's *behavior* on a train split (we use
  **GEPA** — a budget-capped propose→score→select→refine loop — to approximate the `inf`/`sup` over
  V_{L,E}, since the V-entropy is itself an optimization over the family), execute it with a fresh E on
  held-out X to get a recovered verdict m̂, and estimate I via the **transmission** Î = I(m ; m̂)
  (Miller–Madow-debiased, bootstrap CI).
- **Capability-robustness.** Report I(m;m̂), *not* I(X→m̂). The latter (the recovered rule's own
  discrimination) rises with executor strength regardless of fidelity to m; the former only rises if
  the extra capability actually tracks m. (Same instinct as Hewitt–Liang control tasks: subtract what a
  trivial predictor gets — here H(m) is the built-in baseline.)

---

## 2. Predictive sufficiency of articulated metrics — I_{V_{L,E,M}}(X → Y)

Now the target is the **outcome Y**, and the family is *compositions of articulated metrics*:
```
  V_{L,E,M} = { x ↦ g( E(r_1,x), …, E(r_M,x) )  :  M rules each ≤ L tokens, g a simple aggregator/router }
```
Then

> **I_{V_{L,E,M}}(X → Y)** = the usable bits X carries about the **downstream outcome** when X may be
> viewed *only through ≤ M articulated, length-L, executor-E metrics*.

This is **how much of the outcome is captured by articulated metrics** — the verifiable/articulable
portion of Y. Monotone non-decreasing in L, E, and M (M is the *coverage* axis: more metrics =
strictly larger family). The metrics here need not be any pre-specified m's; they are whatever rules
best predict Y — so the Quantity-1 metrics are the *building blocks*, and Quantity 2 asks whether a
*budgeted vocabulary* of them suffices for the outcome.

> **Supervised bridge — not estimated label-free.** Unlike Quantity 1, this object's target is a
> *real* outcome Y, so it **requires ground-truth labels** and is *not* what we measure unsupervised.
> It is the connection to the classical supervised V-information setting (I_V(X→Y)) and the *eventual*
> downstream quantity; on tasks where we lack Y it is aspirational. The label-free quantity we actually
> estimate is Quantity 1, **I_{V_{L,E}}(X → m(X))**, whose target is the metric's own verdict (m is its
> own anchor), estimated via the transmission I(m; m̂). Do not read Quantity 2 as an unsupervised
> measurement.

---

## 3. Putting it together — the outcome decomposition

Let `I_dense(X→Y)` be the V-information of an **unrestricted dense model** (large E, no length limit,
trained end-to-end — no articulation constraint). Three nested quantities partition the outcome's
entropy:

```
  H(Y) =  I_artic^∞(X→Y)                          ARTICULABLE   (captured by articulated metrics, L,E,M→∞)
        + [ I_dense(X→Y) − I_artic^∞(X→Y) ]        TASTE         (predictable from X but NOT via articulated metrics)
        + [ H(Y) − I_dense(X→Y) ]                  NOISE         (not predictable at all — subjectivity / irreducible)
```
where `I_artic^∞ = lim_{L,E,M→∞} I_{V_{L,E,M}}(X→Y)`. The two **gaps** are the measurable objects:

- **Articulation gap (per metric):** `H(m) − I_{V_{L,E}}^∞(X→m)` — fully-tacit metric content.
- **Taste gap (outcome):** `I_dense(X→Y) − I_artic^∞(X→Y)` — learnable-but-inarticulable; the residual
  a dense model exploits that no metric vocabulary expresses.

This is the same `Outcome = Verifiable + Articulable + Taste` decomposition we've been using, now stated
as three V-information levels with explicit predictor families.

---

## 4. Observational scaling laws — how we reach the `∞` ceilings

We can't run L,E,M = ∞, so we **fit I as a function of each budget and extrapolate the asymptote**.
"Observational" (à la Ruan et al. 2024): rather than train a model at each scale, read I off a *ladder*
of existing budgets/models and fit a parametric curve; the **asymptote is the quantity of scientific
interest**, reported with a bootstrap CI.

```
  L-axis:  I(L) = I_∞^{(L)} · (1 − e^{−L / L_c})          # saturating; L_c = describability cost (knee)
  E-axis:  I(E) = I_∞^{(E)} · σ( a·(S_E − b) )            # logistic in a capability scalar S_E (Ruan PCA of benchmarks)
  M-axis:  I(M) = I_∞^{(M)} · (1 − e^{−M / M_c})          # coverage saturation (may be U-shaped: over-fragmentation)
```

- The **L-axis is the cleanest** (one executor, a controllable knob) and is rate–distortion-flavored:
  more description bits → lower recovery distortion → higher I, saturating at the executor's ceiling.
  Monotonicity is guaranteed *only if* the length-L optimizer (GEPA) actually approximates the family
  sup; a single-shot reconstruction is a noisy lower estimate and can look non-monotone.
- The **inferential test**: an asymptote CI that excludes 0 ⇒ the metric/outcome is *articulable in the
  limit*; an asymptote that plateaus well below the dense ceiling ⇒ a real *taste gap*. Poor curve fit
  on the E-axis is itself the honest finding that the model tiers are not a clean capability ladder.

Code: `methods/metric_implementer/scaling.py` (`_sat` for L, `_logistic` for E; `boot_asymptote` for CIs).

> **Caveat (see §6, Prop 5.3).** The map family↦I_V is *not* Lipschitz, so extrapolating a fitted L/E
> curve to an ∞-ceiling is not licensed. Only the **monotone staircase on nested families** is safe.
> Report measured points + running-max; treat the asymptote fit as descriptive within range, never as
> a proven ceiling.

---

## 5. What we've measured so far (anchors)

- **Articulability separates tasks by kind, not just degree** (Llama-8B executor, free-gen lower bound):
  `math_se` articulable (I ≈ 0.22–0.35 bits, CI clear of 0); `reddit_humor` tacit (I ≈ 0, yet metrics
  *discriminate* — tacit ≠ non-discriminating); `creative_writing` tacit-by-authoring (free I ≈ 0.04)
  but **recognizable** (MCQ identification well above chance) — a clean **recognition ≫ recall** gap.
- **Two readouts, one trap:** the recovered rule's discrimination I(X→m̂) is capability-confounded;
  the transmission I(m;m̂) is the capability-robust articulability signal. MCQ *recognition* upper-bounds
  and free-gen *authoring* lower-bounds articulability; together they bracket it.
- **L-axis trial** (powered free-gen L-sweep, **49 metrics** math+CW, R=5, 60 held items,
  E=Llama-8B, forced-detail to bind L; fig `2026-06-16__lsweep_Lcurve.png` is the 6-metric pilot).
  Theory (A.2): I(L) is non-decreasing for the *nested* family V_{≤L}. Observed (at fixed forced
  length): transmission is flat from L=40→100 (paired sign-test p=0.84 math / 1.0 CW — null), then
  decreases when longer rubrics are forced (math L600 median Δ=−0.029, 3/24 positive, p=2e-4; CW
  L≥250 p=0.04). 75–80% of metrics peak at L≤100 tokens. The nested running-max (the V_{≤L} object,
  monotone by construction) rises +0.02–0.03 bits, saturating ~L=100 — partly max-selection bias on
  noisy per-cell estimates. A bare token cap does not bind (the model self-limits to ~40-token rules).
  Methodological notes: a length-≈L *directive* is a non-nested family, so its per-bucket curve can be
  non-monotone — hence the unbiased paired test, not the forced-length mean, is the read; an earlier
  6-metric pilot's apparent "rise to a knee" was max-selection bias. Single executor (8B); the E-axis
  and broader L ranges are not yet tested.

---

## 6. Pathologies of predictive V-information (Koyejo critique) — and how they bear on our use

> **Source (confidential).** *On the Properties and Pathologies of Predictive V-Information*, anonymous,
> under **ICML 2026 review** (shared by Sanmi Koyejo). Double-blind reviewer copy — **do not
> redistribute**; cite as anonymous-under-review. Sanmi's guidance: V-info is *not* an end-all metric;
> use it only where it earns its place.

**Thesis.** I_V looks like Shannon information but lacks its structure: the standard calculus extends
only under strong assumptions, and several pathologies make *raw* I_V unsafe as a generic
data/model-quality scalar. They propose a Kolmogorov-complexity alternative — **predictive complexity**
`C_α(X,Y,ℓ) = inf{ K(f) : f computable, E[ℓ(f(X),Y)] ≤ α }` (min description length to reach loss α) —
which avoids the pathologies.

**What they prove / show.**
- **I_V = Shannon MI ± approximation-error KL** (Thm 4.2): `I_V(X→Y) = I(X;Y) + inf_f D_KL(p_Y‖f[∅]) −
  inf_f E_X D_KL(p_{Y|X}‖f[X])`. The asymmetries of I_V are *approximation error inside V*; equals
  I(X;Y) iff V contains the true conditional + marginal (Cor 4.3).
- **Chain rule is only an inequality** (Lem 4.4): V-entropy is subadditive *only* if the joint family
  contains all factorizations `h[∅](x₁,x₂)=f[∅](x₁)g[x₁](x₂)`; otherwise no incremental decomposition.
- **DPI fails; restored only under Lipschitz log-loss** (Lem 4.5).
- **5.1 Noise can *increase* I_V:** a strictly degraded channel X→Z can give I_V(Z→Y) > I_V(X→Y).
- **5.2 Invertible maps annihilate then resurrect I_V:** I_V is *not* invariant to invertible
  transforms — "information-preserving" preprocessing changes the number.
- **5.3 V ↦ I_V is non-Lipschitz / discontinuous** (Prop 5.3): a sub-family can approximate a
  super-family arbitrarily well in ℓ¹ yet `|I_U − I_V| ≥ ε` for all n. **You cannot extrapolate a
  bigger class's I_V from a well-approximating smaller class.**
- **5.4 Log-loss/0-1 mismatch** (Prop 5.4): the I_V-maximizing predictor can have strictly *worse* 0-1
  loss. Higher I_V ≠ better task accuracy.
- **Empirical:** PVI *sign-flips* for the same datapoint across a model family's scale (Pythia
  14M→12B, Fig 1); PVI oscillates under translation/back-translation (Fig 2).

**How each bears on *our* use** (target = m(X); family = V_{L,E}; headline = **Shannon transmission
I(m;m̂)**; nested families; GEPA selects by agreement-to-m):

| Critique | Hits us? | Why / what we do |
|---|---|---|
| 5.1 noise ↑ I_V (DPI fails) | **No — insulated** | we report **Shannon I(m;m̂)** between two observed verdict vectors, a genuine MI that *obeys* DPI (a noisier executor → *lower* transmission). The pathology is about raw I_V over input reps, which we don't report. Verified: the E0 BSC calibration shows transmission→0 as ε→0.5. |
| 5.2 invertible (de)resurrection | **Partial** | we hold X fixed (no input transforms), so the swap version doesn't apply; but rubric *phrasing*/format (≈invertible rewrites) moves m̂ → moves the number. → **paraphrase-stability check**; never over-read absolute values. |
| 5.3 non-Lipschitz in family ⇒ no extrapolation | **Yes — biggest hit** | directly threatens **§4 scaling laws**. *Survives:* monotonicity on **nested** families (Cor 4.3) — our staircase `I_{V_∅} ≤ I_{V_L} ≤ …` is safe. *Does not:* fitting a smooth curve and extrapolating a **ceiling** past measured points. → **reframe §4 "law" → "measured monotone staircase; ceiling only within range."** |
| 5.4 log-loss vs 0-1 | **No — by construction** | GEPA selects by agreement-to-m and we *report* agreement/MI: same loss for select + report. |
| 4.4 chain-rule inequality | **Yes — M-axis** | `I_{V_{L,E,M}}` over a vocabulary of M metrics is **not additive**. Treat the M-axis as a *monotone family-expansion*, not a sum of per-metric bits (fix §2/§3 wording). |
| 6.1 PVI sign-flips across scale | **Caution — E-axis** | use **aggregate** I_V (non-negative, monotone in family), never per-example PVI; expect E-axis curves erratic, not smooth. |

**Net effect on our design.**
1. **Keep the headline = Shannon transmission I(m;m̂)**, not raw I_V(X→m̂): it is the one quantity here
   that inherits DPI/monotonicity and dodges 5.1/5.4. The formal `I_V(X→m)` stays a *conceptual
   ceiling* that transmission lower-bounds — not a number we trust on its own.
2. **Demote "scaling laws" to "monotone staircases"** (5.3): report measured points + monotone
   running-max; do **not** extrapolate a fitted L/E curve to the ∞-ceiling. (Consistent with the
   already-adopted report-results-not-verdicts stance.)
3. **Add paraphrase-stability** as a standing robustness check (5.2/6.2).
4. **M-axis is monotone, not additive** (4.4): drop "sum of per-metric articulability" phrasing.
5. **Predictive complexity is a flank, not just a threat.** Their `C_α` — *min description length to
   reach loss α* — is essentially our L-axis: *smallest rubric to reach agreement α with m*. Our
   budget-≤L GEPA staircase is a finite-L, learned-executor approximation of `C_α`. If reviewers
   attack I_V, we can reframe the L-axis as **rubric predictive complexity** (pathology-free) and keep
   transmission as the loss target. Cite as the principled version of the L-axis.

---

## 7. Related work (prompt-optimization) — and the scaling-axis decision

*From the 2026-06-18 multi-agent sweep (13 agents: 6 prompt-optimization families, 2
prompting-scaling-law surveys, 5 per-axis dives). This **reframes the earlier L-centric plan**:
rubric length L is one of several candidate scaling axes and the **weakest**; the contribution is the
**recovery y-axis**, not the choice of x-axis. Tables kept as lists for terminal rendering.*

### 7.1 What the field actually optimizes (three-part anatomy)

Prompt-optimization effort concentrates on one narrow cell of each dimension.

**Target — what object is searched:**
- *Instruction wording* (dominant): APE (2211.01910), OPRO (2309.03409), EvoPrompt (2309.08532),
  Promptbreeder (2309.16797), ProTeGi/APO (2305.03495), TextGrad (2406.07496), PromptAgent (2310.16427),
  GEPA (2507.19457). Searched as a blob; sub-factors never isolated.
- *Few-shot demonstrations* (selection/order/bootstrap): DSPy (2310.03714), MIPROv2 (2406.11695). Demos
  drive most of the gain (MIPRO: demos-only ≫ instructions-only), but **count is a fixed
  hyperparameter, never swept** — no accuracy-vs-K curve exists.
- *Soft / continuous prompts*: prompt/prefix/P-tuning (2104.08691, 2101.00190, 2110.07602), BBT
  (2201.03514). Not human-readable.
- *Discrete trigger tokens*: AutoPrompt (2010.15980), RLPrompt (2205.12548), TEMPERA (2211.11890).
  Often gibberish (anti-articulation); FluentPrompt (2212.10539)/TEMPERA deliberately readable.
- *Evaluation criteria / rubrics* (nearest family to us): Auto-Rubric (2510.17314), RRD (2602.05125†),
  RaR (2507.17746), LLM-Rubric (2501.00274), FLASK (2307.10928), Prometheus (2310.08491). Objective is
  label-dependent throughout.
- *The output itself* (not a reusable prompt): Self-Refine (2303.17651), Reflexion (2303.11366).

**Mechanism — how the search is driven:** score+beam (APE/APO); in-context hill-climb over scored
history (OPRO); evolution (EvoPrompt, Promptbreeder); MCTS (PromptAgent); textual-gradient / NL-critique
(ProTeGi, TextGrad); Bayesian/TPE (MIPROv2); RL (RLPrompt, TEMPERA); CMA-ES (BBT); reflective evolution
+ Pareto-merge (GEPA); coding-rate core-set selection (Auto-Rubric).

**Objective — the y-axis people maximize:** task accuracy/F1 dominates; reference metrics (ROUGE) for
generation; multi-objective accuracy + token-cost (InstOptima 2310.17630, CAPO 2504.16005, MOPrompt
2508.01541, CRAFT 2606.04661†); agreement-with-human-grades / preferences (rubric family);
**information-*adjacent* regularizers only** — output-label entropy (GRIPS 2203.07281), prompt
fluency/perplexity (FluentPrompt, InstOptima), criteria coding-rate (Auto-Rubric). **No method
optimizes a label-free information-*recovery* objective.** This is the empty cell.

### 7.2 The scaling-axis map (E / K / N / M / R / L + the y-axis)

Candidate scaling parameters for the recovery instrument, ranked by how clean the axis is and what to
do with it. (E = executor capability, K = demonstrations shown, N = induction-set size the rubric-writer
sees, M = structure/criteria-count, R = recovery passes, L = rubric length.)

- **E — lead axis.** Clean: smooth-metric ⇒ log-linear above an instruction-following threshold. The
  *only axis with a theorem* — Xu monotonicity (2002.10689, Prop 2): more capable family ⇒ ≥ usable
  info from the same X. Empirically Lu in-context PVI sweep 125M→175B (2310.12300); our info-theoretic
  objective sits in the regime where scaling is *predictable* (Schaeffer 2304.15004, Ruan 2405.10938).
  Parameterize E by a capability *score*, not param count (Judge's Verdict 2510.09738); expect
  threshold-then-log-linear, and validate empirically (downstream scaling reliable ~39% of the time,
  Lourie 2507.00885).
- **K — well-posed scaling story; report per-task.** Saturating, with an inverted-U minority
  (Many-Shot ICL 2404.11018). Closed-form saturating curve (Bayesian-ICL, Arora 2410.16531); reduces
  evaluation variance (2212.06713).
- **N — novel x-axis.** Only one prior quality-vs-N curve exists (Prompt-MII 2510.16932: peaks ~N=20,
  then overload). The canonical inducers **Honovich Instruction Induction (2205.10782) and APE
  (2211.01910) hardcoded N=5 and never ablated it** — our exact rubric-induction regime, flagged open.
- **M — covariate, with a confound to guard.** Inverted-U like L, but unlike L admits *derived* optima
  (Chen "Are More LLM Calls All You Need?" 2403.02419; MoP optimal-expert-count 2407.00256, which
  claims expert-count beats length scaling). **Confound:** a holistic judge with the *same* rich rubric
  matches decomposed atomic judges (2603.28005†) — so router-gated must be benchmarked against
  holistic-with-identical-criteria (cf. the apples-to-apples rule).
- **R — pruned.** With no verifier and no Y, extra passes *denoise* a fixed distribution; they shrink
  estimator variance but do not raise true recoverable info (self-consistency cannot select outside the
  modal support; 2506.05295). Fix R≈20–50; do **not** sweep it as a scientific axis. (Becomes a real
  lever only under a coverage/best-of-R metric, which smuggles in an oracle selector we lack.)
- **L — demote to control.** Messy/non-monotone; "length-as-cost" is already taken by the multi-objective
  camp (InstOptima/CAPO/MOPrompt), and CAPO's own ablation shows the length penalty *hurts* accuracy
  (length carries signal). Soft-prompt *embedding* length is the only place length is genuinely ablated
  (Lester plateau ~20; Prefix-tuning U-shape) — and that is capacity, not NL criteria.
- **search/optimization budget — mechanism, not a new dimension.** Coverage is a power law only *under a
  verifier* (Large Language Monkeys 2407.21787); GEPA's "35× fewer rollouts" *shifts the budget curve
  left* rather than adding an axis. Treat as efficiency, not a scaling parameter of recoverable info.
- **y-axis: I(m; m̂) recovery — the contribution.** Never fit as a scaling law. Info-on-y precedents:
  L2M (2503.04725, MI∼L^β on **raw text**, not recovery) and CE-loss/KL (Henighan 2010.14701). V-info
  (2110.08420) and MDL probing (2003.12298) were *measured* but never put on a scaling-law axis.

### 7.3 The reframe — novelty is the y-axis, not the x-axis

Every clean x-axis (E, K/N, search-under-verifier) is well-trodden. The genuinely empty cell is the
**objective**: a label-free *recovery* information quantity I(m; m̂) has never been a fitted scaling-law
target. So **lead with E (and K/N) as borrowed, uncontroversial x-axes; put I(m; m̂) on the y-axis;
demote L and M to covariates we report but don't headline.** The earlier "length-as-family-complexity
axis" framing (References §"Novelty") is superseded by this: the x-axes are borrowed; the
recovery-information y-axis is the open ground.

### 7.4 Show vs Tell — the sharpest articulability framing

Articulability *is* the "tell" channel (the written rubric); demonstrations are the "show" channel (K).
The two are mechanistically distinct (function-vector study 2505.12075); demos under-transmit
input→label *semantics* (Min et al. 2202.12837); and a model's own example-inference beats its
rule-induction by ~16% (MIRAGE 2410.09542) — i.e. for many tasks *show > tell* internally. **Nobody has
cast this as transmitting an evaluation metric and measured it in bits.** A head-to-head **show(K) vs
tell(rubric) transmission under I(m; m̂)** — "how much of a metric can be *said* vs only *shown*" — is a
crisper articulability question than any length/criteria sweep and is confirmed absent.

### 7.5 Where the "consolidate feedback" optimizers sit (GEPA, TextGrad)

GEPA / TextGrad / ProTeGi / PromptAgent / StraGo are **not a scaling axis — they are the articulation
operator**: they turn examples + execution traces + errors into a written rule. In the map they sit at
the **N × search-budget intersection** (naive one-shot instruction induction is the degenerate end;
GEPA the sophisticated, iterative end). They already build our "tell" channel — but **supervised**
(consolidating toward labeled accuracy); we point the *identical* operator at label-free recovery. So
the consolidation machinery is off-the-shelf (GEPA is literally our optimizer) — the novelty is the
target (the judge's own verdict, no Y) and the bits readout, not the mechanism. *Latent sub-axis they
expose:* feedback **richness** per example (bare verdict → +rationale → +contrastive → +iterative) =
bits-consolidated-per-example; never swept, and it interacts with the §7.2-M prompt-richness confound.

### 7.6 Nearest neighbors — cite and distinguish head-on

- **Robertson & Koyejo, *Let's Measure Information Step-by-Step* (2508.05469).** *Same lab as the §6
  critique.* Shares: ground-truth-free + item-level decomposition + information-theoretic. Differs:
  TVD-mutual-information for **adversarial gaming-resistance**, not a recovery scaling curve. **Read in
  full; engage directly** — our differentiator is the granularity/capability scaling of recovered bits.
- **Lu et al., in-context PVI (2310.12300).** Shares: V-info vs capability with fixed prompts. Differs:
  PVI as *instance difficulty*, not recovery of a designated source verdict.
- **Auto-Rubric (2510.17314) / RRD (2602.05125†).** Shares: explicit criteria + an info-theoretic term
  (MCR² coding-rate / correlation-aware weighting). Differs: that term is a redundancy penalty on
  criteria; the learning objective is **label-dependent** preference accuracy. *Do not claim
  "information-theoretic objective" as novel in the abstract — they got there first.*
- **LLM-Rubric (2501.00274).** Shares: predicts a held-out *judge's* rating from rubric outputs.
  Differs: **supervised regression to human labels**, fixed manual 9-question rubric.
- **RaR (2507.17746).** Shares: criteria-count ablation + judge-capability sweep (the only one with
  both). Differs: needs a reference answer; count contrast is binary (essential-vs-all), no sweep.
- **L2M (2503.04725).** Shares: MI fit as a power law. Differs: MI of raw text, not verdict recovery.

Net defensible novelty = the *conjunction* none of these holds: **label-free MI recovery of a held-out
judge verdict as the scaling target, with E (and K/N) as the clean axes, rubric-as-channel.**

### 7.7 Verification status

2025-and-earlier arXiv IDs were fetch-verified. **2026 IDs are unconfirmed (marked †): RRD 2602.05125,
CRAFT 2606.04661, MO-CAPO 2605.18869, atomic-decomposition 2603.28005, and the 2026 rubric-application
cluster (GER-Eval / RubricEval / RULERS).** Verify before citing. **TEMPERA = 2211.11890** (an earlier
draft used a wrong ID, 2211.04719, which is a microfluidics paper).

---

## Appendix A. Articulability **is** supervised V-information with the label swapped

**Claim.** `I_{V_{L,E}}(X → m(X))` is a bona-fide instance of the V-information of Xu et al. (2020),
and therefore inherits *verbatim* every result they (and the Ethayarajh et al. estimation procedure,
and the Hewitt–Liang control-task logic) prove **at the level of an abstract `(X, Z, V)`**. Nothing
about the supervised case used that the target was a human label; we only have to check the
hypotheses for `Z := m(X)` and `V := V_{L,E}`.

**A.0 — The abstract hypotheses (Xu et al.).** A *predictive family* is a set
`V ⊆ { f : 𝒳 ∪ {∅} → 𝒫(𝒵) }` satisfying **optional ignorance**: for every `f ∈ V` and every output
distribution `P ∈ range(f)`, the constant predictor `f′ ≡ P` (returning `P` on every input incl. `∅`)
is also in `V`. Given this, define `H_V(Z) = inf_{f∈V} 𝔼[−log f[∅](Z)]`,
`H_V(Z|X) = inf_{f∈V} 𝔼[−log f[X](Z)]`, and `I_V(X→Z) = H_V(Z) − H_V(Z|X)`. Their theorems —
**(T1)** `I_V ≥ 0`; **(T2)** monotone `V ⊆ V′ ⇒ I_V ≤ I_{V′}`; **(T3)** `→ I(X;Z)` as `V` → all
measurable maps; **(T4)** finite-sample concentration of the plug-in `Î_V` — use *only* these defs.

**A.1 — Proposition 1 (Reduction).** Put `𝒵 = {0,1}`, `Z = m(X)`, and
```
  V_{L,E} = { x ↦ E(r, x) : r ∈ R_L }  ∪  { constant predictors },   R_L = {rules of ≤ L tokens},
```
where `E(r,x) ∈ 𝒫({0,1})` is executor `E`'s P(YES) under rule `r`. Then `(Z, V_{L,E})` satisfies A.0,
so **T1–T4 hold for `I_{V_{L,E}}(X→m(X))`**.
*Proof.* (i) `m(X)` is a measurable `{0,1}`-valued r.v. jointly distributed with `X`; the theory's
hypotheses ask nothing more of the target (in particular it may be a deterministic function of `X` —
then `I(X;Z)=H(Z)` and `I_{V}≤H(Z)`, no axiom is touched). (ii) Each `f_r = E(r,·)` is a conditional
distribution `𝒳∪{∅}→𝒫({0,1})`, so `V_{L,E}` is a set of predictors of the required type. (iii)
Optional ignorance holds **by construction**: we closed `V_{L,E}` under constant predictors (a constant
is a trivial "length-0 rule," a harmless and standard closure). ∎
*Consequence.* `I_{V_{L,E}}(X→m) ≥ 0`, and its empirical estimator is **exactly the Ethayarajh et al.
recipe** — fit the best in-family predictor of the target on train, take the held-out cross-entropy
gap — with the target `Y` replaced by `m(X)` and the predictor class replaced by `V_{L,E}`.

**A.2 — Proposition 2 (Monotone in L; conditional in E).** `L ≤ L′ ⇒ R_L ⊆ R_{L′} ⇒ V_{L,E} ⊆ V_{L′,E}`,
so by **T2** `I_{V_{L,E}}(X→m) ≤ I_{V_{L′,E}}(X→m)` — **the L-curve is provably non-decreasing.** For `E`:
if `E′` *refines* `E` (can emulate its rule-following, `V_{L,E} ⊆ V_{L,E′}`), monotone in `E` likewise;
without an emulation guarantee it need not hold — which is precisely why we flag the capability axis as
provisional rather than a clean ladder.

**A.3 — Proposition 3 (Estimation; L is the capacity knob).** `R_L` is finite with
`log|R_L| ≤ L·log|Σ|` (`Σ` = token vocabulary). Plugging into Xu's finite-class concentration bound
(T4), with clipped log-loss in `[ε,1−ε]`: with probability `≥ 1−δ` over `n` held items,
```
  | Î_{V_{L,E}} − I_{V_{L,E}} |  ≤  c · log(1/ε) · √( (L·log|Σ| + log(1/δ)) / n ).
```
So **the articulation budget `L` is simultaneously (a) the family-capacity that makes the curve rise
(A.2) and (b) the complexity term in the estimation error** — longer rules need more scored examples to
certify. This is the formal content of Hewitt–Liang's warning that high-capacity probes overfit; here
the held-out evaluation plus the explicit `L`-cap is the control, with the data requirement made
quantitative.

**A.4 — Proposition 4 (the reconstruction/GEPA estimator is the empirical V-entropy minimizer, and is a
conservative lower bound).** `H_{V_{L,E}}(m|X) = inf_{r∈R_L} 𝔼[−log E(r,X)(m(X))]` is, by definition,
"the best length-`L` rule for predicting `m`'s verdicts." GEPA minimizes the *empirical* version
(it optimizes a rule against `m`'s train verdicts) — so any rule it returns has loss `≥` the infimum,
hence `Î_{V_{L,E}} ≤ I_{V_{L,E}}`: **search imperfection can only under-state articulability** ("at
least this much is articulable"), the safe direction. The held-out cross-entropy gap is the canonical
`I_V` estimator; the code's transmission `I(m;m̂)` is its binarized plug-in.

**A.5 — Hewitt–Liang correspondence (term-by-term).**

| supervised probing | articulability here |
|---|---|
| probe (bounded-capacity classifier) | length-`L` rule executed by `E` |
| probe accuracy / `−H_V(Y|X)` | `−H_{V_{L,E}}(m|X)` (best rule's fit to `m`) |
| control task / random-label baseline `H_V(Y)` | best **constant** rule `H_{V_{L,E}}(m)` |
| **selectivity** = acc − control | **`I_{V_{L,E}}(X→m)`** = the gap |
| probe-capacity control (cap params) | the `L`-token cap + held-out eval (A.3) |

**A.6 — The single non-trivial modeling commitment (stated honestly).** `E` is a fixed *stochastic*
map, so `V_{L,E}` is the family **actually realizable by prompting `E`** — not all logical length-`L`
predicates. Every theorem above is about *that* realized family, which is exactly the object of
interest (articulability *to this executor*). And `m(X)` is itself an LLM judge's verdict, so "the
target" is the judge's metric *as instantiated* — there is no external ground truth, and none is
needed. That label-free, judge-internal target is the whole point: we imported the supervised
V-information machinery **without importing the supervised label.**

---

## References

**V-information core.**
- Xu, Zhao, Song, Stewart, Ermon. *A Theory of Usable Information under Computational Constraints.* ICLR 2020 (arXiv 2002.10689). The primitive; monotonicity in V (Prop 2.1), optional-ignorance ⇒ nonnegativity, not bijection-invariant.
- Ethayarajh, Choi, Swayamdipta. *Understanding Dataset Difficulty with V-Usable Information.* ICML 2022 (2110.08420). PVI; difficulty relative to V; conditional V-info Eq. 6 conditions on **per-datapoint** side info.
- Hewitt, Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP 2019 (1909.03368). Selectivity = task − control; probe-complexity varied but categorical, y = selectivity not info.
- Voita, Titov. *Information-Theoretic Probing with MDL.* EMNLP 2020 (2003.12298). Codelength as a stable scalar; argues *against* needing a complexity sweep.

**Family-complexity as an axis (the closest precedents to our L-axis).**
- Pimentel, Saphra, Williams, Cotterell. *Pareto Probing: Trading Off Accuracy for Complexity.* EMNLP 2020 (2010.02180). **The one paper that puts predictor-family complexity on the x-axis** — but y = accuracy, not V-info, and complexity = probe rank/arch, not description length.
- Pimentel, Valvoda, Maudslay, Zmigrod, Williams, Cotterell. *Information-Theoretic Probing for Linguistic Structure.* ACL 2020 (2004.03061). Argues the *opposite*: always pick the highest-capacity probe. (Cite to pre-empt the mis-attribution.)

**Conditioning on a written artifact (the steelman for a *conditional* form — on the rubric's per-item outputs, not the string).**
- Hewitt, Ethayarajh, Liang, Manning. *Conditional Probing: Measuring Usable Information Beyond a Baseline.* EMNLP 2021 (2109.09234). Conditions on a per-token baseline RV.
- Jiang et al. *RORA: Robust Free-Text Rationale Evaluation.* ACL 2024 (2402.18678). Conditional V-info on a per-instance *rationale* — nearest published instance of "V-info on a written NL artifact"; no length axis.

**Rubric / instruction length as an objective; info-theoretic rubric handling.**
- Xie et al. *Auto-Rubric: Training-Free Rubric Induction.* arXiv 2510.17314 (2025). Induces rubrics from ~70 pairs, **compresses the criteria pool via a coding-rate objective** — strongest "info-theoretic rubric" neighbor; compression as selection, not a plotted info-vs-length curve. **Candidate for an alternative L-axis (criteria count under compression).**
- Yang, Li. *InstOptima: Evolutionary Multi-objective Instruction Optimization.* EMNLP Findings 2023 (2310.17630). Earliest clean precedent for instruction length as an explicit Pareto objective.
- Zehle et al. *CAPO: Cost-Aware Prompt Optimization.* AutoML 2025 (2504.16005). Length penalty.
- Câmara et al. *MOPrompt.* arXiv 2508.01541 (2025). NSGA-II over accuracy × token count; explicit accuracy-vs-length Pareto front.

**Prompt-optimization context (our GEPA optimizer; note GEPA's "Pareto" is over per-instance scores, NOT length).**
- Agrawal et al. *GEPA: Reflective Prompt Evolution.* arXiv 2507.19457 (ICLR 2026). Shorter prompts are a side-effect, not an optimized axis.
- Honovich et al. *Instruction Induction.* ACL 2023 (2205.10782). Zhou et al. *APE.* ICLR 2023 (2211.01910).

**Scaling laws over length / capability.**
- Montgomery et al. *Predicting Task Performance with Context-aware Scaling Laws.* arXiv 2510.14919 (2025). Chinchilla-style joint fit over compute AND context length — nearest "law over length," but context/examples, not rubric wording.
- Ruan, Maddison, Hashimoto. *Observational Scaling Laws.* NeurIPS 2024. Capability scalar from benchmark PCA (our E-axis).

**Novelty (per 2026-06-17 lit sweep; *reframed in §7.3 after the 2026-06-18 sweep*):** reporting
*V-usable information of a judgment as a function of a rubric/instruction **length budget**, with length
as the family-complexity axis*, is apparently unpublished. Pieces exist separately (V-info, RORA's
V-info-on-an-artifact, InstOptima/CAPO/MOPrompt's length-as-cost, Auto-Rubric's criteria compression,
Context-aware Scaling Laws) but no one assembles them. **→ Superseded framing: §7.3 demotes length and
relocates the novelty to the *recovery y-axis* I(m; m̂) scaled against executor capability E (and
demonstrations K / induction-set N); length is a covariate, not the headline axis.**

---

## Appendix B. BibTeX

```bibtex
@inproceedings{xu2020usable,
  title={A Theory of Usable Information under Computational Constraints},
  author={Xu, Yilun and Zhao, Shengjia and Song, Jiaming and Stewart, Russell and Ermon, Stefano},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}, eprint={2002.10689}, archivePrefix={arXiv}}

@inproceedings{ethayarajh2022vusable,
  title={Understanding Dataset Difficulty with $\mathcal{V}$-Usable Information},
  author={Ethayarajh, Kawin and Choi, Yejin and Swayamdipta, Swabha},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}, eprint={2110.08420}, archivePrefix={arXiv}}

@inproceedings{hewitt2019control,
  title={Designing and Interpreting Probes with Control Tasks},
  author={Hewitt, John and Liang, Percy},
  booktitle={EMNLP}, year={2019}, eprint={1909.03368}, archivePrefix={arXiv}}

@inproceedings{voita2020mdl,
  title={Information-Theoretic Probing with Minimum Description Length},
  author={Voita, Elena and Titov, Ivan},
  booktitle={EMNLP}, year={2020}, eprint={2003.12298}, archivePrefix={arXiv}}

@inproceedings{pimentel2020pareto,
  title={Pareto Probing: Trading Off Accuracy for Complexity},
  author={Pimentel, Tiago and Saphra, Naomi and Williams, Adina and Cotterell, Ryan},
  booktitle={EMNLP}, year={2020}, eprint={2010.02180}, archivePrefix={arXiv}}

@inproceedings{pimentel2020infotheoretic,
  title={Information-Theoretic Probing for Linguistic Structure},
  author={Pimentel, Tiago and Valvoda, Josef and Maudslay, Rowan Hall and Zmigrod, Ran and Williams, Adina and Cotterell, Ryan},
  booktitle={ACL}, year={2020}, eprint={2004.03061}, archivePrefix={arXiv}}

@inproceedings{hewitt2021conditional,
  title={Conditional Probing: Measuring Usable Information Beyond a Baseline},
  author={Hewitt, John and Ethayarajh, Kawin and Liang, Percy and Manning, Christopher D.},
  booktitle={EMNLP}, year={2021}, eprint={2109.09234}, archivePrefix={arXiv}}

@inproceedings{jiang2024rora,
  title={{RORA}: Robust Free-Text Rationale Evaluation},
  author={Jiang, Zhengping and others},
  booktitle={ACL}, year={2024}, eprint={2402.18678}, archivePrefix={arXiv}}

@article{xie2025autorubric,
  title={Auto-Rubric: Training-Free Rubric Induction via Information-Theoretic Compression},
  author={Xie and others},
  journal={arXiv preprint arXiv:2510.17314}, year={2025}}

@inproceedings{yang2023instoptima,
  title={{InstOptima}: Evolutionary Multi-objective Instruction Optimization},
  author={Yang, Heng and Li, Ke},
  booktitle={Findings of EMNLP}, year={2023}, eprint={2310.17630}, archivePrefix={arXiv}}

@inproceedings{zehle2025capo,
  title={{CAPO}: Cost-Aware Prompt Optimization},
  author={Zehle and others},
  booktitle={AutoML}, year={2025}, eprint={2504.16005}, archivePrefix={arXiv}}

@article{camara2025moprompt,
  title={{MOPrompt}: Multi-objective Prompt Optimization},
  author={C{\^a}mara and others},
  journal={arXiv preprint arXiv:2508.01541}, year={2025}}

@inproceedings{agrawal2026gepa,
  title={{GEPA}: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Agrawal and others},
  booktitle={ICLR}, year={2026}, eprint={2507.19457}, archivePrefix={arXiv}}

@inproceedings{honovich2023instruction,
  title={Instruction Induction: From Few Examples to Natural Language Task Descriptions},
  author={Honovich, Or and Shaham, Uri and Bowman, Samuel R. and Levy, Omer},
  booktitle={ACL}, year={2023}, eprint={2205.10782}, archivePrefix={arXiv}}

@inproceedings{zhou2023ape,
  title={Large Language Models Are Human-Level Prompt Engineers},
  author={Zhou, Yongchao and others},
  booktitle={ICLR}, year={2023}, eprint={2211.01910}, archivePrefix={arXiv}}

@article{montgomery2025context,
  title={Predicting Task Performance with Context-aware Scaling Laws},
  author={Montgomery and others},
  journal={arXiv preprint arXiv:2510.14919}, year={2025}}

@inproceedings{ruan2024observational,
  title={Observational Scaling Laws and the Predictability of Language Model Performance},
  author={Ruan, Yangjun and Maddison, Chris J. and Hashimoto, Tatsunori},
  booktitle={NeurIPS}, year={2024}}
```

> BibTeX accuracy note: arXiv IDs and lead authors verified in the 2026-06-17 sweep; full author lists for Auto-Rubric, CAPO, MOPrompt, GEPA, APE, Montgomery use `and others` — fill before any submission.
