# v14 roadmap: the prompt-space autoencoder — tuned decoder, entropy-optimal codes, layered ceilings

**2026-07-13. Supersedes the pool-size design in the two v13.1 recipe notes (S=12 pooling is WITHDRAWN — S is unbounded). Nomenclature: S = teaching pool (large, design-split); b = bootstrap trials (50); k = examples per trial (8).**

## 0. The frame

Encoder = criterion prompt p executed by frozen executor E → code z_T(p) = p's k-bit verdict pattern on trial T's texts. Decoder = reconstructor ρ (MCQ: pick the metric; behavioral: induce M̂ and execute it on held-out H). Value = transmission above blind/shuffled controls. The instrument is a **discrete-bottleneck autoencoder used as a measuring device**, and its capacity is what makes the measurement certifiable:

**I(M̂ ; M_ω) ≤ I(z ; M_ω) ≤ H(z) ≤ k bits.**

Three ceilings, three different meanings (this table goes in every certificate):

| ceiling | moves when we tune the instrument? | role |
|---|---|---|
| trial-table max / free-recombination cap / 1−q0 | rises | OUR RULER — descriptive only, never the headline |
| **I(z ; M_ω)** (DPI, reconstructor-free, executor-side data only) | **no** | what the evidence can possibly convey — the honest ceiling |
| H(M_ω on H) | no | total information in the target |

Tuning the decoder raises the FLOOR (achieved value) and narrows the bracket; it cannot raise `I(z;M_ω)`. Report floor + DPI ceiling as the headline; the instrument's own cap is a diagnostic row.

**The diagnostic that closes the week's confusion:** `I(z;M_ω)` low → the trial's examples didn't capture the metric (fix example selection). `I(z;M_ω)` high, transmission low → the decoder can't read the code (fix the decoder). Both high, achieved value still low → the prompts genuinely cannot articulate the metric — the only claim we are actually in business to make.

## 0b. CANONICAL DEFINITION + ARTIFACT for I(z;M) — and a CORRECTION to the chain in §0

**Artifact (exists, use it — do not reinvent):**
- Script: `sk2:/lfs/skampere2/0/alexspan/cr3-v12/qualification_v1/bottleneck_diag.py`
- Output: `sk2:.../qualification_v1/bottleneck_diag_v1.json`, schema `cr3-bottleneck-diag-v1`
- Run: `ssh sk2 'export HOME=/lfs/skampere2/0/alexspan; /lfs/skampere2/0/alexspan/cr3-v12/envs/ai_usage_v12/bin/python /lfs/skampere2/0/alexspan/cr3-v12/qualification_v1/bottleneck_diag.py'`

**Estimand (exact, population-level — NOT a sample estimate):** let the declared **codebook population** be the task's metric set (humor: the 60 R3 metrics with frozen bootstrap targets), uniform prior over metrics M. For a panel T of k probe indices, metric m's **code** is `z_T(m) = 1[target_m > 0.5]` at those k indices. Then

  **I(z;M) := H(z) = −Σ_c p(c) log2 p(c)**,  p(c) = (# metrics in the codebook whose code is c) / |codebook|.

Because z is deterministic given m, I(z;M) = H(z) exactly; because the codebook IS the population, there is no sampling bias. Ceilings: log2(|codebook|) (= 5.91 bits for 60 metrics) and k bits. The script also reports `H(z | mined prompts)` (how much the mined pool exercises the code space) and `target_unique` (is the target's code distinct from all other codebook metrics' codes).

**⚠ CORRECTION to the chain stated in §0 — do NOT implement I(z;M) as a ceiling on behavioral value.** Two distinct quantities were conflated:

| quantity | question | is it capped by k bits / I(z;M)? |
|---|---|---|
| **Identification**: I(M̂ ; M) over the metric POPULATION | "which metric is this?" | **YES** — DPI: the decoder's answer cannot carry more about M than the code does. `I(M̂;M) ≤ I(z;M) ≤ H(z) ≤ k`. This is the MCQ-relevant ceiling and what `bottleneck_diag.py` measures. |
| **Behavioral transmission**: TR = I(ŷ ; M_ω\|_H), per-probe MI for a FIXED target | "how well does the induced rule reproduce this metric's verdicts?" | **NO.** A single well-chosen M̂ (one code) can in principle reproduce M_ω perfectly. The k-bit code caps the NUMBER of distinct decoder outputs (≤ 2^k per panel), not the LEVEL of transmission. |

**Therefore:**
- The certified ceiling for BEHAVIORAL value remains: (i) `H(M_ω|_H)` — the target's entropy on the held-out set; and (ii) the EXACT enumerated cap `max_c v(c)` per panel. Nothing else.
- `I(z;M)` keeps its role as a **panel-quality DIAGNOSTIC** (low ⇒ the code cannot tell the decoder which metric it is looking at ⇒ fix example selection) and as a **certified ceiling for IDENTIFICATION only**.
- Presenting `I(z;M)` as a bound on behavioral value would be a directionally-wrong ceiling of exactly the kind this project has already retracted once (cf. the Fano-ceiling retraction). Do not.

**Example-selection objective — corrected accordingly.** Raw code-entropy greedy maximizes H(z) over the codebook but can DESTROY target uniqueness (measured: metrics 11 and 34 lost uniqueness under greedy while gaining entropy). The correct objective for identifying a SPECIFIC target is target-vs-rest separation: maximize `margin_T = min over m ≠ target of Hamming(z_T(target), z_T(m))`, tie-broken by H(z). Computable from executor-side signatures only ⇒ certification-safe, design-split-only, frozen before valuation.

## 1. What changed from v13.1 (and why)

- **S is unbounded.** 2^S only appeared in the exact joint cap, which bounds a "fantasy prompt" that may not exist. Withdrawn as a headline object. Pool size is now chosen for evidence quality (use the whole design split).
- **MILP is DEMOTED to appendix.** With large S the trials barely overlap, so text-sharing constraints are thin and the MILP would tighten a cap we no longer headline. Keep the formulation (variables y_{T,s} one-hot per trial + x_i per text, linking constraints, linear objective; any incumbent dual bound is a valid anytime cap) for a future heavily-overlapping design. Do not build it now.
- **Caps that remain:** (i) per-trial exhaustive 2^k tables (cheap, exact — this enumeration stays, it is a small quotient not a search); (ii) free-recombination cap = mean of per-trial maxima, with the recombination slack MEASURED against the mined pool and reported; (iii) DPI bottleneck `I(z;M_ω)`; (iv) reachable-set bound from c/r + DKW (the headline "value-added of more mining").
- **Decoder panel, not a single decoder** (see §3).

## 2. Entropy-optimal example selection (do this first — it is free and it fixes the wasted-code problem)

Evidence: the t8c tables showed 256 states collapsing to 6–41 distinct values — most of the code carried nothing.

Objective: choose each trial's k texts to maximize the behavioral entropy of the resulting k-bit code across the mined prompt population, i.e. maximize H(z_T) (and, as a declared variant, I(z_T ; M_ω)). Greedy submodular selection with the standard 1−1/e guarantee, computed from the executor-side signature matrix ONLY — no labels, no reconstructor, no prompt bodies → certification-safe (design-split-only, frozen before valuation).

Deliverable: `select_trial_examples(signatures, k, b, objective={"code_entropy","target_mi"})` + tests (planted degenerate texts must be excluded; determinism; frozen plan sha).

## 2b. Constructing the b=50 panels: diverse AND target-separating (the panel-builder spec)

Rejected: one repeated best panel (destroys the across-panel variance component — it IS the frozen-T8 instrument-lock we rejected). Rejected: joint global optimization of all 50 panels (coordinated panels are correlated panels ⇒ reduced effective replication, for a more complex optimizer).

**Adopted: per-trial restricted greedy with a hard coverage repair.** For trial t = 1..b:
1. **Diversity engine — subsample the ELIGIBLE PROBE POOL**, stable-hash seeded by (run_sha, t): draw 40–60% of the design-split probes as eligible for this trial. (Competitor-metric bootstrapping may ride along, but it CANNOT be the only diversity source: the same few "good" probes keep winning under most competitor subsamples, yielding 50 heavily-overlapping panels.)
2. **Greedy select k probes** from the eligible subset maximizing
   `score(j) = margin_gain(j) − λ · usage_penalty(j)`,
   where `margin_T = min over m ≠ target of Hamming(z_T(target), z_T(m))` over the codebook (§0b), and `usage_penalty` grows with how many prior panels already used probe j.
3. **Tie-break** by code entropy H(z) over the codebook.
4. **HARD label-balance constraint (correctness, not preference):** the target's verdicts on the panel must contain BOTH classes, target 3–5 YES of 8. An all-YES demo set teaches the inducer NOTHING — no contrast, no rule. The margin objective alone does not guarantee this.
5. **Require target uniqueness** (target's code distinct from every other codebook metric's code); documented exception path recorded in the manifest when infeasible (that is itself a finding — cf. metric10, whose code collides).
6. **Reject duplicate panels** (exact index-set match); resample.

**After all b panels: HARD coverage check + repair pass (§4f).** Every probe in the pool must appear in ≥1 panel (target ≥2, balanced). Repair by force-including unused probes into whichever panel loses the least margin. **Reject the panel family outright if coverage still fails.** Uncovered probes are inert bits that mechanically manufacture tied optima (Wave 1: 30/36 pools covered only 9–11 of 12 units).

**Why the margin objective is right for BOTH channels** (not obvious for the behavioral one): the behavioral decoder does not pick among metrics — it induces a rule. But a demo set whose label pattern is ALSO explainable by another metric in the bank is precisely a demo set that misleads a rule-inducer. That is exactly Wave 1's metric10 failure (a distractor explained the 8 demos better than the target). Maximizing target-vs-rest margin is what prevents it.

**Scope note (prevent over-claiming):** margin-selected panels are NOT random draws from the design distribution, so the generalization claim remains "over the declared panel family, exhaustively evaluated" — as already stated. Selection is certification-safe (executor-side signatures only, design split, frozen in the manifest before any prompt valuation) and strengthens the instrument without touching the caps, which are enumerated ON the chosen panels.

## 3. Decoder panel across b trials

Stratify the 50 trials across **3 frozen decoder families** by a hashed rule fixed in the prereg: one Qwen-lineage, one Llama-lineage, one structurally different (Mistral/OLMo), each passing a mini-qualification (behavioral channel: canonical induction lift > 0 on ≥4/6 sentinels; shuffled control ≈ 0).

Why: (a) de-confounds the instrument (never again an instrument only one model can pass); (b) the across-decoder variance component becomes measurable and reportable; (c) per-trial best-decoder transmission is a strictly higher, still-valid lower bound (the estimand is existential; controls are computed per decoder so nothing is inflated); (d) cost is nil — trials are partitioned, not multiplied.

Decoder identity is part of the frozen instrument; per-decoder value distributions are reported alongside the pooled mean.

## 4. GEPA-tuning the decoder

**Objective:** measured transmission ABOVE the blind and shuffled controls (never raw agreement — this is what makes tuning unable to win by prior-guessing or by ignoring labels), averaged over a frozen **reference prompt set** (stratified low/mid/high-value mined prompts).

**Splits (FOUR, all disjoint, all stable-hash):** DEV METRICS | teaching pool S | decoder-development probes D_dec | certification eval H. GEPA sees only D_dec, the reference prompts, and the DEV METRICS. H is touched ONCE, after the freeze.

**Three overfitting guards, in order of importance:**
(i) **METRIC-LEVEL HOLDOUT (load-bearing — the decoder prompt is a TASK-level parameter, so the holdout must be at the metric level).** GEPA tunes the decoder on a development pool of metrics DISJOINT from every metric we certify, drawn from SEVERAL TASKS so the prompt cannot overfit one bank's idiosyncrasies. Probe-level and prompt-level holdout are NOT sufficient: a decoder tuned on metric50 and then used to value metric50 is contaminated however the probes are split. This is the defense against the leak "GEPA learns to say: choose metric X."
(ii) held-out PROBES (D_dec vs H) — a decoder that memorizes probe quirks shows no lift on H.
(iii) held-out PROMPTS — improvement must transfer to prompts not in the reference set, checked before the freeze.

**Structural anti-leak property (why a tuned decoder cannot win by knowing the answer distribution):** the BLIND no-demo control is computed with the SAME tuned decoder prompt. Any prior the prompt encodes ("criteria in this bank tend to be about wordplay") benefits the blind control equally, and value = transmission − max(blind, shuffled) subtracts it exactly. **Only gains that route THROUGH the demonstration bits survive.** Additionally: the decoder prompt is string-checked to contain no metric names, descriptions, or menu content.

**ONE SHARED DECODER PROMPT across metrics — required, not incidental.** Per-metric tuning IS the leak. The GEPA objective is expected transmission over the POPULATION of dev metrics; a prompt that wins by metric-specific tricks loses on the population average. Sharing is the constraint that enforces generality.

**Report both instruments.** Every certificate is issued under the untuned AND the tuned decoder. Stable conclusions ⇒ the tuning was a free tightening. Flipped conclusions ⇒ a finding about instrument sensitivity that belongs in the paper.

**Articulability vs validity — never let one stand in for the other.** Decoder tuning can raise measured articulability of an arbitrary-but-real behavioral pattern; that is CORRECT (such behavior genuinely is articulable), not a "rescue" of a bad metric. Whether M_ω tracks the human construct its description names is a SEPARATE axis (the gold-fidelity diagnostic). Both numbers are reported side by side for every metric. Tuning cannot manufacture articulability: a degenerate target is capped by H(M_ω), and non-demo-routed success is subtracted by the controls.

**Stopping rule (predeclared):** stop when the dev-transmission gain over the previous round is < 0.01 bits, or when the residual `I(z;M_ω) − I(M̂;M_ω)` on dev is < 0.02 bits (decoder saturated — further tuning is provably wasted), or at 4 rounds, whichever first. Expectation from our GEPA trajectories: gains land in rounds 1–2 and flatten.

**The freeze:** decoder prompt(s), decoder panel, example-selection rule, trials, pools, splits, α — all hashed into the prereg before a single certified draw.

## 4b. GEPA-tuning the BEHAVIORAL decoder — the concrete procedure

**What is being optimized.** The behavioral decoder is an *induction template* t: it takes k labeled demo texts and emits a criterion M̂ ("Below are texts labeled YES/NO by a hidden criterion. State the criterion as a single rubric question that reproduces these labels."). GEPA rewrites t — its instructions, the specificity guidance, whether to hypothesize-then-test, the output format. The executor, probes, pools, trials, and controls are all frozen; t is the ONLY thing that moves.

**Fitness (the objective GEPA maximizes).** For candidate template t:

  fit(t) = mean over dev metrics m, over dev trials T, over reference states s of
           [ TR(M̂) − max(TR_blind, TR_shuffled) ] / H(M_ω^m)

where M̂ = ρ_t(T's texts, labels s), TR = plug-in MI between the executor's verdicts on M̂ and M_ω^m, evaluated on the **decoder-development probe split D_dec** (disjoint from certification set H). **Normalize by H(M_ω^m)** so high-entropy metrics don't dominate the objective. Controls are recomputed with the SAME t (this is what makes any prior baked into t self-subtract).

**Reference states (what evidence GEPA practices on).** A stratified mix per dev metric: (i) the *canonical* state (target's own verdicts on the trial texts) — the best-case "can it read perfect evidence?" test; (ii) a sample of states realized by high-, mid-, and low-value mined prompts — the realistic noisy-evidence test. Roughly 6 states × 4 trials × 8 dev metrics ≈ 200 inductions per candidate template.

**Loop.** Seed = the current induction template. Each round: a strong proposer LLM performs GEPA's reflective mutation — it is shown the induced M̂'s, their scores, and contrast pairs (high- vs low-scoring M̂ for the *same* demos) — and rewrites t. Keep top-k by fit. **≤4 rounds**, stopping when the round's gain < 0.01 bits or when the residual `I(z;M_ω) − I(M̂;M_ω)` on dev falls below 0.02 bits (the decoder has read everything the code contains — further tuning is provably wasted). Freeze the winner: sha the template into the prereg before any certified draw.

**Cost.** ~200 inductions × 8 candidates × 4 rounds ≈ 6K inductions + 6K × |D_dec| executor scorings ≈ 1–2 GPU-hours per channel. Cache aggressively: identical (t, demos) → identical M̂ (temp-0); identical (M̂, probe) → identical verdict.

**THE HAZARD THAT MATTERS: GEPA will find the exemplar dump if we let it.** The degenerate optimum of this objective is a template that makes M̂ restate the demos verbatim ("here are labeled examples; score similar texts alike") — few-shot prompting of the executor, which *does* transfer to held-out probes and *will* score well. That is exemplar-carrying, not verbal articulation (a distinct decompression rung). Mitigation, mandatory: run GEPA under BOTH declared arms —
  (a) unconstrained (headline: honest sup over evidence-constructible prompts),
  (b) no-verbatim-exemplar: M̂ containing demo-text shingles above a declared threshold is rejected and regenerated (string-enforced, during tuning AND during certification).
Report both. **The (a) − (b) gap is the exemplar-vs-rule decomposition, measured directly** — a finding, not a nuisance.

**Guards inherited from §4 (all apply):** dev metrics disjoint from certified metrics and drawn from several tasks; D_dec disjoint from H; H touched once after the freeze; one shared template across metrics (per-metric tuning IS the leak); template string-checked for metric names/descriptions/menu content; certificates reported under both untuned and tuned decoders.

**Decoder-free anchor + distortion diagnostic (why tuning cannot Goodhart the ranking undetected).** The Bayes/oracle decoder (nearest-code over the codebook, computable from executor signatures with NO LLM) attains the DPI ceiling I(z;M) and defines the decoder-free optimum Ω\* — under it, the best prompt is simply the one whose verdicts best reproduce M_ω's. Standing diagnostic, cheap: **rank-correlate (i) each prompt's plain behavioral agreement with M_ω against (ii) its reconstruction value under the LLM decoder.** High ρ ⇒ the decoder is not distorting the ranking (Ω′ ≈ Ω\*). Low ρ ⇒ the value channel is measuring the reader more than the prompt, and no "optimal Ω" may be reported. Run this BEFORE trusting any prompt ranking. (2026-07-13 Wave-1 evidence: near-flat value surfaces, 34/36 behavioral pool optima non-unique, and only 2/6 behavioral winners from their own target's mining family ⇒ rankings are currently noise-dominated. Report tie classes and canonical representatives only.)

**Pooled/specialized decoders (optional, only if the shared template leaves a large residual): K-FOLD CROSS-FITTING at the metric level.** Partition metrics into folds (e.g. 12 folds of 5, optionally clustered on behavioral signatures — executor-side, label-free, certification-safe). For fold f: tune on metrics NOT in f, freeze, value ONLY metrics in f. Every metric is valued by a decoder that never saw it ⇒ leakage structurally impossible while the decoder still specializes. Folds declared in the prereg; report per-fold values and across-fold spread. Shared template FIRST; cross-fitting only if the residual justifies it.

## 4c. Regularizing against the exemplar dump (concrete, layered)

GEPA WILL find the demo-dump optimum unless blocked. Defenses, cheapest first; use all of them:

1. **Shingle/n-gram hard reject.** Any M̂ sharing an n-gram (n=8 words) with any demo text is rejected and regenerated (bounded retries, then fail-closed). Enforced during tuning AND certification, not just at report time.
2. **Length + format cap.** M̂ must be a single rubric question / short criterion (declared token cap). A dump is long by necessity; the cap makes it unrepresentable.
3. **No-copy content rule.** M̂ may contain no proper nouns, quoted spans, or rare tokens lifted from the demos (rare-token overlap check against a corpus frequency list).
4. **Transfer probe — DISSIMILARITY-STRATIFIED WITHIN THE CONSTRUCT'S DOMAIN (near/far), NOT cross-task.**
   **Cross-TASK rotation is REJECTED and would be a broken test:** executing a humor metric's M̂ on code-review texts requires M_ω's verdicts there, which are degenerate (near-constant, ~all-NO) — so MI collapses to ≈0 for genuine rules and exemplar-dumps ALIKE, the ratio is ~0 for everything, and the probe flags every metric. Cross-task transfer destroys the construct's meaning, which is exactly what the probe must preserve.
   **Adopted:** stratify held-out probes by embedding distance from the panel's demo texts; score M̂ on a NEAR stratum and a FAR stratum; the statistic is the **near/far transmission ratio**. The target stays meaningful and non-degenerate on both, so the ratio isolates surface-similarity dependence rather than construct collapse. Where a SECOND CORPUS for the same construct exists (different joke source, different writing-prompt pool), use it — that is the genuinely cross-domain version that keeps the construct intact.
   **What this probe is FOR (do not overclaim it):** it is NOT the primary exemplar defense. Exemplar-carrying M̂ OFTEN GENERALIZES FINE — an executor shown 8 labeled examples does analogical reasoning and may transfer well; that is a different KIND of articulation (the exemplars-vs-rules decompression rung), not a failure. Division of labor: **the rule-vs-exemplar TYPE distinction is enforced structurally by the no-verbatim arm** (defenses 1–3, 6); **the transfer probe's narrower job is catching CORPUS-SURFACE OVERFITTING** — an M̂ tuned to superficial features of this probe corpus that H (drawn from the same corpus) would not expose.
5. **Soft fitness penalty during GEPA** proportional to embedding similarity between M̂ and the demo block (steers the search away from the basin instead of only rejecting at the boundary).
6. **Two declared arms, both reported** (unconstrained vs no-verbatim-exemplar). The (a)−(b) gap IS the exemplar-vs-rule decomposition — a result, not a nuisance.

## 4d. Legibility ≠ fidelity: what the oracle-decoder insight CHANGES in the algorithm

Under the Bayes/oracle decoder the optimal prompt collapses to "the prompt whose verdicts best reproduce M_ω" — i.e. **pure behavioral fidelity, decoder-free**. Therefore the reconstruction channel is NOT measuring fidelity (we can measure that directly, with no reader at all); it is measuring **legibility** — whether a competent reader can recover the criterion from the prompt's behavior. Concrete algorithmic consequences, all mandatory:

1. **Every certificate carries BOTH columns, never one:**
   - **fidelity(p) = MI(σ(p) ; M_ω)** on held-out probes — decoder-free, no reconstructor, cheap.
   - **legibility(p) = reconstruction value** (the v13 value channel) — reader-relative.
   These are the Face-1 / Face-2 pair the project already theorizes: a prompt can reproduce behavior without being legible, and (rarely) vice versa. Reporting one without the other is now a spec violation.
2. **Two different p\*, two different questions.** `argmax fidelity` answers "what is the best prompt?" `argmax legibility` answers "what is the most articulable prompt?" They are NOT interchangeable and must be labeled distinctly. The old "optimal Ω" language conflated them.
3. **Ω\* is defined w.r.t. the ORACLE decoder** (canonical, decoder-free, computable by nearest-code decoding over the codebook with NO LLM). Decoder-specific optima Ω′, Ω″ are reported as *instances*, never as "the" optimum.
4. **Distortion diagnostic gates any ranking claim:** Spearman ρ between fidelity(p) and legibility(p) across the mined pool. High ρ ⇒ the decoder is not distorting (Ω′ ≈ Ω\*). Low ρ ⇒ the value channel is measuring the reader, and NO optimal-Ω may be reported. (Wave-1 2026-07-13: surfaces near-flat, 34/36 behavioral pool optima non-unique, 4/6 behavioral winners from a foreign mining family ⇒ rankings are noise-dominated; tie classes + canonical representatives only.)
5. **The scientific claim is legibility-indexed.** "Articulability" in the paper means: *there exists a prompt whose behavior a competent reader can decode into the criterion* — an inherently reader-relative statement, bounded above by the reader-free ceiling I(z;M_ω). Say it that way; never as an absolute property of the metric.

## 4e. Connecting capture-recapture to value: VALUE-THRESHOLDED missing mass (closes the gap)

> **SUPERSEDED IN PART, 2026-07-13 — see §12.9 of `notes/2026-06-18__prompt-optimality-theory.md`.**
> The theory note derives the identity this section asserts (Lemma V1: improving codes are *necessarily*
> unseen codes, because every seen code has value ≤ A by construction — so `U_A ≤ U_0` is free and exact,
> and the code partition being *value-blind* is what makes Good-Turing on it legitimate).
> It also adds two things this section lacks:
> (1) the slack is a single scalar `theta = P(v(c) > A | c unseen)`, so `U_A = theta · U_0` — the bound below
> silently pins `theta = 1`, and `theta` is directly estimable with NO assumption via split-sample
> freeze-on-discovery (§12.9.3), giving a strictly tighter and equally honest bound;
> (2) **no nontrivial LOWER bound on the improving rate exists** (Theorem V3) — novelty can be 100% with
> value-gain 0, so c/r can certify *when to stop mining* and can never certify that mining *will pay*.
> Implement the `theta` estimator and the three-rung ladder readout per §12.9.7.

Raw behavioral missing mass is NOT a value bound: unseen behaviors mostly collapse to already-seen CODES (the value function cannot distinguish them), and a new code helps only if it is worth more than the current best. The improving event is therefore **"a new code with value above the incumbent"**, and both of its factors are computable:

  **gain(m) ≤ [1 − (1 − U₀(I))^m] · [max_{c ∈ I, UNSEEN} v(c) − A]**,  I = {codes c : v(c) > A}

- **Second factor: EXACT** — we enumerated the code space, so we know v(c) for every code and which codes the pool hit; take the max over the *unseen* ones directly. No "assign leftover mass to the cap."
- **First factor: Good-Turing / CP restricted to I** — the missing mass WITHIN the improving set (singleton rate among draws landing in I, CP upper bound). This is "the chance of finding an unseen value-add", and it is typically ≪ the overall novelty rate because most unseen codes are worthless.
- **This is how flatness enters the bound quantitatively:** a near-flat value surface ⇒ I is tiny ⇒ U₀(I) is tiny ⇒ certified value-added is small — NOT because mining is exhausted (raw novelty may still be 50%), but because the remaining novelty is worthless. The old formulation could not see this distinction.

**THE NOVELTY-COLLAPSE LADDER (measure and report all three — this is the headline diagnostic, and it is computable NOW on harvested data, CPU-only):**

  **P(new behavior) ≥ P(new code) ≥ P(new code with value > A)**

Each step is a coarsening, so each rate is strictly smaller. Mechanism: the executor maps huge numbers of distinct criterion texts onto the SAME verdict pattern on a small panel (the form-quotient invariance the theory already posits), so as the pool grows, new behaviors increasingly land on already-seen codes. The code space is tiny (2^k per panel) and saturates fast, while raw behavioral novelty stays high (replay curves: γ ≈ −0.05, ~10^23 draws to exhaust).

Expected shape (to be confirmed): raw novelty flat and high; code novelty saturating quickly; value-improving novelty pinned near zero. **This ladder is the direct, quantitative answer to "does 50% unrecovered behavior mean unrecovered value?" — no, and here is the decay at each coarsening.** It converts "we found little" into "we certify there was little left to find."

**IMPLEMENTATION (one function, three species maps — TRACK AND PLOT ALL THREE, per metric per family, at every prefix length of the draw sequence ⇒ CURVES, not endpoints):**

| rate | species map | status |
|---|---|---|
| P(new behavior) | full signature σ(p) hash | ESTIMATED (has observations) |
| P(new code) | joint code tuple across panels | ESTIMATED (has observations) |
| P(new code with value > A) | code tuple restricted to I = {c : v(c) > A} | **BOUNDED, zero-observation** (see below) |

Reuse the existing first-contact counter + Clopper-Pearson interval (`cr_audit`); the ONLY parameter that changes is the species map (+ optional restriction set). Evaluate at every prefix length so each rate yields a decay curve against mining draws. These three curves, overlaid, are the headline capture-recapture figure.

**⚠ CORRECTION (2026-07-13, caught in review): the zero-hit CP bound below is INVALID as originally written.** `I = {c : v(c) > A}` is defined FROM the sample (A is the observed max), so "zero hits in I" is guaranteed BY CONSTRUCTION, not observed as evidence. A Clopper-Pearson bound on a set built to be empty carries no information. Two valid replacements — implement BOTH, report both, quote the MIN:

**(A) PRIMARY — the RECORD / RANK bound (exact, distribution-free, no sample splitting, and it is precisely the tool for a data-dependent "better than the best so far" threshold, because it is a statement about RANKS, which are exchangeable).** For an exchangeable value sequence, the overall maximum is equally likely to occupy any position, so the probability that at least one of the next m draws exceeds the max of the first n is exactly

  **P(improvement) ≤ m / (n + m)**  ⇒  **gain(m) ≤ [m / (n + m)] · (V_cap − A)**

with V_cap the EXACT enumerated cap. Ties only make it more conservative. Apply PER FAMILY (exchangeability holds within a family's iid draws); union-bound across families. Worked example (metric 0: n=400, m=100, A=0.0188, cap=0.0357): P ≤ 0.2 ⇒ **gain ≤ 0.0034 bits** — valid, and ~3× tighter than the invalid CP figure previously quoted here.

**(B) COMPLEMENT — split-sample CP ("freeze on discovery").** Define A and I on a frozen DISCOVERY PREFIX; then I is fixed BEFORE the audit stream, so CP applies honestly to audit hits into I and can return a NONZERO estimate (more informative than the generic rank bound when the improving region actually carries mass). Any audit improvement is kept as a SEPARATE achieved lower bound. Costs: sample splitting, and a looser cap term (prefix-A < full-A ⇒ larger I).

The two rest on different premises ⇒ the MIN is the honest headline. (Rejected: a third confirmation stream — Option-1 plus an absorption round, more streams for marginal gain. Rejected: omitting the value-thresholded bound entirely — we can legitimately have it.)

**THE STRUCTURAL FACT that motivated the (invalid) zero-hit reasoning, retained for intuition only: I is UNOBSERVED BY CONSTRUCTION.** A is the max value we have SEEN, so every code with v(c) > A is one we have never drawn. Good-Turing's singleton count within I is therefore ALWAYS zero — there is no data inside I, ever. This is not an estimator bug; it is the shape of the question. Consequences:

1. **The improving mass is bounded by the ZERO-OBSERVATION CP bound: P(next draw ∈ I) ≤ −ln(α)/n.** The SET I is calculated exactly (from the enumerated value table, free); the MASS is bounded empirically, and the bound is driven ENTIRELY by audit sample size n.
2. **Worked example (metric 0, Wave 1):** n=400, α=0.05 ⇒ P(draw ∈ I) ≤ 0.0075. Over m=100 further draws, P(hit anything improving) ≤ 0.53; best unseen code 0.0357, A=0.0188 ⇒ **certified gain ≤ 0.009 bits** (≤ ~50% of current achieved). Real, quotable, assumption-free, available today.
3. **Tightness is bought ONLY with more audit draws** — the bound decays as 1/n. Concrete design target: phantom mass 0.001 needs n ≈ 4,800 draws per family (α=0.05). This replaces vibes with an experimental-design number.
4. **This settles the banding question AGAINST banding in the current regime:** every improving band has zero observations, so B bands each contribute their own phantom −ln(α/B)/n ⇒ total ≈ B·(ln B + ln(1/α))/n, superlinear in B, while the pricing benefit is bounded. **In a zero-observation regime FEWER bands is strictly better.** Single-threshold now; banding only becomes interesting if the improving region ever acquires observations (i.e. A jumps).

**VALUE BANDS — experimental variation, to try AFTER decoder tuning (not now).**

*Bound:* `gain(m) ≤ Σ_bands [1 − (1 − U₀(band)_hi)^m] × (band_max_value − A)`, replacing the single threshold. Each band gets its OWN certified missing-mass bound (Good-Turing/CP restricted to that band's codes), so mass that only barely beats A is charged its own small improvement instead of the top code's big one.

*The tension that sets the band count B:*
- More bands ⇒ tighter pricing (limit: one band per code = tightest possible).
- More bands ⇒ more PHANTOM mass: a CP upper bound on a band with ZERO observed singletons is not zero, it is ≈ −ln(α/B)/n, and the Bonferroni split raises that floor as B grows.

At n ≈ 400 audit draws: B=1 → 0.0075 phantom/band; B=10 → 0.0133; B=100 → 0.0190. Ten mostly-empty bands contribute ≈0.003 bits of fiction (tolerable against a real bound ≈0.013); a hundred bands swamp the signal. **⇒ B ∈ [5, 8], never more than ~10 at our sample sizes.**

*Band edges:* NOT equal-width, and NOT quantiles of the observed prompt values (sample-dependent ⇒ selection). Use **quantiles of the ENUMERATED value table** (v(c) over the whole code space — frozen design data, computed before any audit draw): edges at the 50/75/90/95/99th percentiles plus the top ⇒ six bands, α/6 each. Sample-independent by construction.

*When banding WINS:* the improving set is SKEWED (many codes barely beat A, a few much better) — the single-threshold bound is then badly loose because it charges every hit at the top code's price.
*When banding LOSES:* the improving set is tiny or flat — most bands are empty, phantom mass dominates, and the Bonferroni was paid for nothing. **This is exactly Wave-1's regime** (near-flat surfaces, tiny improving sets, large tie plateaus).

*Recommendation:* single-threshold NOW; banding as a declared experimental variation once decoder tuning produces a value surface with real structure. Both bounds are valid ⇒ report both and quote the min, with a union-bound α/2 split. Which is tighter is then a cheap empirical question, not an argument.

**Two complementary routes; report both, quote the MIN:**
| route | premise | how it prices the unseen |
|---|---|---|
| DKW expected-best (implemented) | future draws iid from the SAME value distribution | at the cap b (loose) |
| c/r value-thresholded (this section) | exchangeability within family; NO distributional assumption on values | EXACTLY, by enumeration |

## 4f. Panel COVERAGE, tie hygiene, and status vocabulary (from Wave-1 tie analysis, 2026-07-13)

**DEFECT (fix, do not document): panel families must COVER their pool.** Wave 1 drew 4 panels × 6 examples per 12-unit pool — 24 slots — yet 30/36 pools covered only 9–11 of the 12 units. An uncovered unit's label bit appears in NO query, so flipping it changes the state but nothing observable: this mechanically manufactures 2^(uncovered) tied optima. The CAP is unaffected (max over 12 bits with u inert = max over the 12−u live bits, exact), but the ARGMAX is degenerate, so "optimal unit set" reporting is meaningless.
**Fix:** require a **covering design** — a balanced incomplete block design over the pool: every unit appears in ≥1 panel (target ≥2, balanced). Trivially satisfiable at 24 slots / 12 units. Make it a HARD, TESTED constraint in the panel builder: reject any panel family that fails to cover its pool. (Also strictly better science: an uncovered unit is evidence we paid for and discarded.)

**Tie hygiene — two more cheap fixes:**
- **Report the RAW (unclipped) lift as a diagnostic column** alongside the certified clipped value. Clipping negative lift to 0 creates huge zero plateaus (most of metric 34's MCQ tie mass). Certified value stays clipped; the raw column preserves ordering and shows HOW FAR below control a state sits (metric 34: −0.0668 is far more informative than "0").
- **Raise |H|.** Plug-in MI on |H|=60 has a coarse finite value set, so ties arise from quantization. Move to |H| ∈ [150, 300] (cached executor scorings — cheap). Refines the value surface and dissolves spurious ties.

**STATUS VOCABULARY — a dead instrument must never print like a resolved metric.** Wave 1's metric-34 MCQ was labeled "RESOLVED" at a zero cap: blind prior 0.317 vs best annotated state 0.250 ⇒ demonstrations uniformly HURT; all 18,432 panel-state cells negative before clipping; all 4,096 pool patterns tied at zero. That is instrument death, the opposite finding from saturation. Add a distinct terminal status — `DEAD_INSTRUMENT` / `ZERO_CAP` — that can never be read as successful resolution, with the cap value and blind-vs-best gap printed alongside. Without this the fan-out yields a table in which the most broken metrics look like the most settled ones.

**Channel decision, reinforced:** on the exact metric where MCQ was stone dead (34), the behavioral channel found 631/1,536 positive cells and a real 0.0072-bit signal (cap 0.0159). Behavioral is primary when the channels disagree.

## 4g. Audit budget (35 certified metrics) and the MCQ decoder objective — decisions

**AUDIT BUDGET: 400 NEW draws per metric, allocated by declared family quotas, ALL never-absorbed (pure audit).**
- **The headline bound is POOLED (process-level), not family-wise.** The frozen proposer mixture (families by fixed quota, iid within family) is itself exchangeable ⇒ the record bound (§4e-A) uses ALL n draws, not n/6. Per-family bounds are a DIAGNOSTIC. This dissolves the "family-wise bounds will be loose" concern and kills the argument for a 6×-cost 400-per-family budget.
- **The existing harvested pools (~400/metric, frozen families, per-draw seeds) ALREADY support the record bound at n≈400** — no new draws are needed for validity. New draws buy exactly two things: (a) a clean never-absorbed audit stream for the split-sample CP complement (§4e-B), and (b) larger n, which tightens m/(n+m).
- Combined: n ≈ 800 pooled for the record bound + a clean 400-draw audit stream for CP. Cost ≈ 14K generations + ~4.2M cached executor scorings ≈ 1–2 GPU-hours across all 35 metrics. (Rejected: "no new draws" — defensible, since the record bound works on existing data, but it forfeits the CP complement for almost no savings.)
- **Caveat that MUST go in the prereg:** the bound scales as m/(n+m), so it is tight only when m ≪ n. "Should we mine 10× more?" will always return a weak bound — and that is FINE, because **the binding constraint is the CAP, not the search**: with enumerated caps of 0.016–0.086 bits, even a GUARANTEED improvement cannot buy much. That is the headline, and it does not depend on the audit budget at all.

**MCQ DECODER OBJECTIVE (demoted diagnostic arm — do not over-invest): normalized choice lift.**
- Optimize the quantity the certificate REPORTS: target-choice lift above max(blind, shuffled), **normalized by 1−blind** (this normalization makes metrics with different blind priors comparable — the same role H(M_ω) plays as the divisor in the behavioral objective, §4b).
- Rejected: full-codebook identification-MI confusion matrices for every metric (theoretically cleaner — it IS the §0b-aligned estimand for MCQ — but real cost for a demoted arm). Rejected: hybrid tie-break (mostly non-binding).
- **BORROW ONE THING from the MI route:** compute identification MI on the **6 sentinels only**, as a diagnostic, and use it for the **stopping rule** — for MCQ (unlike behavioral, cf. §0b correction) the residual `I(z;M) − I(M̂;M)` is the CORRECT chain, so "stop when residual < 0.02 bits" is principled rather than arbitrary. Keep the other stopping conditions: gain < 0.01, transfer failure, or 4 rounds.

## 4h. Three more decisions (holdout-safe stopping, template sharing, what the sentinel may block)

**(1) MCQ stopping rule vs metric-level holdout — compute the residual on DEV METRICS, not sentinels.** My earlier "identification MI on the 6 sentinels only" was a COST heuristic that accidentally broke the holdout: the sentinels are among the 35 CERTIFIED metrics, so any tuning decision made on them is the very leak §4 forbids. The residual `I(z;M) − I(M̂;M)` is computable on ANY metrics — there is nothing special about sentinels for this purpose. Therefore: **stopping decisions use DEV metrics; the sentinel residual is computed ONCE, AFTER the freeze, as a reported diagnostic** (post-freeze evaluation is legitimate). Holdout preserved, stopping rule preserved, nothing lost.

**(2) Decoder templates: ONE per channel/arm, shared across the 3 model families, optimized on POOLED fitness.** Per-family tuning CONFOUNDS model with prompt and destroys the reason the 3-family panel exists (measuring across-decoder variance): an observed family difference could be the model or its template, indistinguishably. Two amendments:
- **Mechanical model-specific FORMATTING is allowed and is not a confound** (chat template, system-prompt conventions — correct usage, not search). Forbid *searched* per-family variation; record the formatting adaptation in the manifest.
- **Pooled MEAN fitness is the objective** (matches the estimand: value is a mean over trials, trials stratified across families), but **report per-family fitness** so across-family variance stays visible. Per-family tuning is a v15 option ONLY if the shared template leaves a large residual (same logic as cross-fitted pooled decoders, §4b).

**(3) What the six-metric sentinel MAY BLOCK before fan-out: structural failures + CONTROL-BASED INSTRUMENT LIVENESS. Never scientific weakness.** Three outcome classes:
- **Structural failure** (crash, leakage, incomplete/non-finite tables, cache disagreement, invalid bounds) ⇒ **BLOCK**.
- **Scientific weakness** (low but real values, wide CIs, small caps) ⇒ **DO NOT BLOCK** — that is a RESULT, and gating on it prejudges the finding. This is exactly where the previous prereg's ≥4/6 headline gate went wrong: it failed on what turned out to be an INSTRUMENT problem and read as a scientific verdict.
- **INSTRUMENT DEATH** ⇒ **BLOCK** (a plain "structural failures only" rule would let this through). Gate on the CONTROLS, not the metrics — this is what makes it a liveness test rather than a scientific one: **the planted-positive control must certify positive value** (a criterion handed to the instrument on a platter — if it cannot detect THAT, nothing it reports means anything); **the degenerate control must not certify positive**; **no control inversion** (blind ≥ annotated on canonical replay); not all caps identically zero.
Rationale: fanning out a dead instrument is precisely how the §4f disaster occurs — 35 certificates in which the MOST BROKEN metrics print as the most "resolved."

## 5. Adaptivity discipline (the two-population design — non-negotiable)

- **Development phase (adaptive, unlimited, ZERO claims):** adaptive mining (MCTS/GEPA over criterion space, seeded by induced M̂'s — they are high-value proposals), decoder tuning, example selection. Nothing here is quotable.
- **FREEZE** (prereg + sha into the launch log).
- **Certification phase (frozen, one pass):** a fresh **i.i.d. audit stream** from the frozen proposer families supplies the c/r missing-mass and CP intervals. Achieved value may include adaptively-mined prompts — **achievement is a lower bound and lower bounds do not need i.i.d.**; only the BOUND comes from the frozen i.i.d. stream. This is the MCTS-efficiency + CP-rigor split.
- Iterate freely BEFORE the freeze; never after. A later insight that demands a better decoder = a new instrument version with its own prereg and its own audit stream. Old certificates stand as certificates for the old instrument. Reporting across versions is honest and monotone: "under v13.1, articulability ≥ A₁; under tuned v14, ≥ A₂ > A₁" — a rising lower bound is a result, not a retraction.

## 6. Implementation order (each step ships tested before the next)

1. `I(z;M_ω)` + `H(z)` bottleneck diagnostics on EXISTING harvested signatures (CPU, hours). This alone tells us whether the week's low values were panel-failure or decoder-failure. **Do this first — it is free and it decides everything downstream.**
2. Entropy-optimal example selection (§2) + tests.
3. Behavioral decoder channel (`behavioral_value_channel.py` per the v13.1-B recipe: induce → execute on H → plug-in MI; anti-degeneracy §5b: held-out execution, k-bit capacity, blind+shuffled controls, two exemplar arms).
4. Decoder mini-qualification (§3) → pick the 3 families.
5. GEPA decoder tuning (§4) on the dev splits → freeze.
6. Prereg (six-metric humor gate + the value-added block) → sentinel → gate evaluation → fan-out.

Steps 1–2 are CPU-only and unblock immediately. Steps 3–5 are ~1 GPU each, parallelizable across the ≤4 authorized GPUs.

## 6b. REUSE LEDGER — what survives, what must be recomputed, and the one rule that decides it

The pipeline has three GPU stages: **(A) mining** (proposer LLMs emit criterion prompts), **(B) executor scoring** (Llama-8B verdicts: σ(p) for mined prompts, σ(M̂) for induced rules, and the target M_ω), **(C) decoder inference** (MCQ choice probabilities, or behavioral inductions). Each v14 change touches a different subset. Nothing needs to start from scratch.

**THE RULE THAT MAXIMIZES REUSE: cache at per-(item, probe) CELL granularity, never at panel/vector granularity.** Signatures are already stored as full prompt×probe matrices (e.g. 400×300), which is why panel redesign is FREE — a new panel is a re-slice of columns we already own, not new GPU work. Apply the same discipline to the rule cache (sha(M̂ text) → verdict PER PROBE, as rows) so that expanding |H| adds rows instead of invalidating vectors. The existing content-addressed evidence store (`cr3_evidence_store.py`, with its excludes-self hashing) is the right vehicle — extend it, don't replace it.

**FREE (CPU only — zero GPU, reuse everything already on disk):**
- All **mined prompt texts** (2,378 unique humor prompts + pools for every other task). Mining is done; never redo it.
- All **executor signatures σ(p)** on the 300-probe panel (Llama-8B, constrained readout v3) and all **bootstrap targets M_ω**.
- **Ledgers** (draw order, per-draw seeds, families, quotas) — needed for capture-recapture; pure metadata.
- ⇒ **Codes for ANY panel design** — including the new covering/BIBD panels (§4f) and entropy-optimal selection (§2): a code is a column-slice of the existing signature matrix. **Panel redesign costs NO GPU.**
- ⇒ **All three c/r curves** (§4e), for any species map, at every prefix length.
- ⇒ **I(z;M_ω), H(z)**, the bottleneck diagnostics (already run: `bottleneck_diag_v1.json`).
- ⇒ **fidelity(p) = MI(σ(p); M_ω)** for every mined prompt — the decoder-free column mandated in §4d. Free, today.
- ⇒ The **oracle/Bayes decoder** and the fidelity-vs-legibility rank correlation (§4d) — nearest-code decoding, no LLM.

**INCREMENTAL GPU (cached, additive — never a full redo):**
- **New panels ⇒ new (panel, state) inductions** (stage C). But the **rule cache** absorbs most of it: distinct (panel, state) cells frequently induce the SAME M̂ text (Wave 1 saw heavy dedup collapse), and any repeated M̂ reuses its cached executor verdicts (stage B) outright.
- **Raising |H| to 150–300** (§4f tie hygiene): score the EXISTING set of induced rules on the NEW probes only. Additive rows; old work untouched.
- **GEPA decoder rounds** (§4b): each candidate template needs fresh inductions (stage C), but stage-B scoring of the resulting M̂ hits the rule cache whenever the induced text recurs — which it does often.
- **Decoder panel of 3 families** (§3): trials are PARTITIONED across families, not multiplied — the total decoder inference count is unchanged.

**FULL RECOMPUTE (only one thing triggers it):**
- **Executor swap (8B → 70B).** Every σ(p), every σ(M̂), every M_ω must be re-scored (stages B, and C's downstream values). Mining texts still survive (stage A never repeats). This is the expensive move and it should be a deliberate, separately-budgeted decision (see the open v14 executor question: metric12's below-chance description-behavior agreement is the argument FOR it).

**EFFECTIVELY DEAD (do not plan around these):**
- The **448,808 Qwen-14B MCQ choice rows** and the **t8c 256-state MCQ tables**: keyed to specific menus + panels + Qwen. Under covering panels and a demoted MCQ channel they are usable only for the MCQ *diagnostic* arm at its original panels. Keep them (never delete), but do not count them as reusable capital.

**Bottom line:** the encoder side (mining + executor signatures) is a durable, reusable asset that the whole v14 redesign sits on top of at zero GPU cost; the recomputation is confined to the decoder side, where it is cached and additive. The only thing that would force a genuine restart is changing the executor — and even then the prompt corpus survives.

## 8. v14.1 EXECUTION PLAN — the ordered set of steps to run. **[2026-07-14; supersedes §6's ordering]**

Written after the first real v13 numbers landed. Three of them are audit failures, one is a genuine finding,
and the ceiling turns out to live somewhere we were not measuring. Run the phases IN ORDER; A and B are
cheap and each one changes what C–E should be.

### The four deliverables this plan must produce

1. A **trustworthy MCQ** test with a prompt-optimized decoder.
2. A **behavioral** test with a prompt-optimized decoder, pushed as high as it will go.
3. Metrics spanning **low / medium / high ceilings** (not a floor-effect-only sample).
4. Enough points for a **scaling law in `|Omega|` = 1, 2, 3, 5, 8**.

D3 and D4 are the same design: compose the target from `|Omega|` criteria and the ceiling falls out as a
gradient BY CONSTRUCTION. Do not stratify post-hoc on measured cap — that is selection on the outcome.

### The numbers this plan is reacting to (v13, 29-35 metrics, untuned decoder)

    target entropy (what is there)          ~1.00 bits   100%    (~= 100% accuracy)
    executor GIVEN the true description     ~0.85 bits    85%    [CIRCULAR - see A1]
      >>> 97% of the information dies at the INDUCTION step <<<
    best possible induced rule (exact cap)   0.0246 bits  2.5%   (~= 59% accuracy)
    mining actually achieved                 0.0088 bits  0.9%   (~= 55.5% accuracy; chance = 50%)

Mining's shortfall is 1.6% of the budget. **We have been optimizing the 1.6% while 97% burned at a step we
never instrumented.** Everything below follows from that.

---

### PHASE A — Fix what is broken. **[blocking; cheap; do first]**

**A1. De-circularize the executor-fidelity gate. [BLOCKING — the top rung of the whole ladder]**
The 2026-07-13 gate scored `AUC_8b = 0.99993` because its reference was `frozen_8b_bootstrap` — the 8B's own
output. It measured "does executor X reproduce what 8B produced," which is rigged for the incumbent by
construction and **cannot adjudicate 8B vs 70B.** Rebuild with a reference no executor wrote:
- **Independent judge reference:** 3 isolated blind Sonnet-class passes over the 300 probes applying the gold
  description; majority-vote. **The inter-pass agreement is not decoration — it is the calibration input for
  the attenuation correction.** Reference noise attenuates MI DOWNWARD, which would INFLATE the apparent
  induction loss (theory note L6: attenuation is provable, deconvolution needs calibration).
- **Mechanical anchors (stronger; do not skip):** planted criteria whose truth is COMPUTABLE ("contains a
  question mark", "mentions a number"). No model writes the answer key. If the 8B executor fails a criterion
  a regex can settle, it is not executing, and we learn that without asking any model's opinion.
- Blinded known-label anchor rows in every judging batch (standing rule).
- Retain the one real finding of the old gate: **executors disagree hugely** (8B vs 70B balanced agreement
  0.711). That is a result. It just does not say who is right.

**A2. Establish the provenance of the 8B-constructor row, or delete it.**
There are **zero 8B-constructor rows in any `results.parquet` under `cr3-v13.1`** (verified 2026-07-14, all
lanes + wave1). The reported "8B constructor: 35/35, MCQ 0.00651, behavioral 0.01602" therefore came from a
different campaign/root/instrument. **The 13.7x MCQ constructor-scaling claim is NOT established** and may be
an instrument change. Either name the source checkpoint + confirm identical panels/probes/code, or strike the
row. Never quote it meanwhile.

**A3. Permutation null on every reported value. [the single most important robustification]**
Values are ~0.009 bits. The plug-in MI estimator on a 2x2 table at N=300 is biased UPWARD by roughly
`1/(2*N*ln2) ~= 0.0024 bits` — **27% of the median signal.** The blind/shuffled subtraction cancels most of it
(same bias both arms) but not exactly, since it depends on marginals. So:
- Shuffle held-out labels B>=200 times, recompute value, report the observed value's percentile in the null.
- Report Miller-Madow-corrected MI alongside the plug-in.
- **If a value sits inside its permutation null, it is not a value.** This gate is not optional.

**A4. Aggregate significance on UNCLIPPED lift.** "24/35 positive" is meaningless while values are clipped at
0 — the clip destroys the sign test. Use raw unclipped lift + Wilcoxon signed-rank against 0.

**A5. Report accuracy alongside bits, always.** `I = 1 - H2(e)` inverted. "0.0088 bits" communicates nothing;
"55.5% vs 50% chance against a 59% ceiling" is legible and devastating. Both columns, every table.

---

### PHASE B — The ceiling ladder. **[decides where the GPUs go; reuses cached cells; no mining]**

Four rungs, same metrics, same executor, same panels. The pattern tells us which bottleneck we are fighting.

| rung | what it is | what it isolates |
|---|---|---|
| **C0** | execute the metric's OWN description (reference per A1) | executor capability |
| **C1** | decoder sees 8 demos + closed menu of task-local descriptions -> **picks** -> **the picked description is EXECUTED** | identification, priced in behavioral bits |
| **C2** | free induction from 8 demos (what v13 measures) | current pipeline = 0.0088 bits |
| **C3** | **2-bit quantized code** (see below) | whether 8 bits was starving the decoder |

**C1 design decisions (frozen):**
- Menu = **full task-local R3 bank** (11-67 options). NOT padded with cross-task descriptions — a humor
  criterion beside legal criteria is separable on domain alone, which inflates C1 and manufactures the
  "identification works" conclusion we are testing.
- **Blind-menu control is MANDATORY:** same menu, no demos, decoder picks from priors, **execute that pick**,
  subtract. This absorbs menu-size differences (chance is 1/11 vs 1/67) and decoder priors at once. The retro
  no-demo audit already caught peer-review "identification" being entirely a menu prior — do not repeat it.
- **Size-matched sensitivity arm:** subsample every menu to the smallest bank (11) and re-run. If the
  conclusion flips, menu size drove it.
- The menu MUST contain **the exact description that generated the frozen target**, not a paraphrase —
  otherwise a perfect pick cannot reproduce `M_omega` and C1 cannot reach C0 even in principle.
- **RANDOM-GOLD validity arm [added 2026-07-14, user request; provenance corrected]:** golds and
  distractors are BOTH R3-bank members with IDENTICAL mining provenance (GEPA-artifact mining ->
  behavioral orthogonalization -> L0->R3 clustering — shared, not an asymmetry source). The asymmetry is
  only in WITHIN-bank selection: golds pass target-entropy-quintile + panel-balance filters that
  distractors never face; distractors are sampled conditional-on-gold (behavioral disagreement,
  non-clone). Add a small arm (~10 trials, 2 tasks) where the gold IS a uniform draw: sample metrics
  uniformly from the task-local R3 bank with NO gold-only filters (or apply identical filters to the whole
  menu pool), score panels for whichever metric is drawn, run C1 blind + canonical. **Prediction: blind
  identification ~= chance (1/menu).** If blind is still >> chance here, an unmodeled tell remains
  (register, length, formatting) and every C1 number needs a bigger haircut than the blind subtraction
  provides. This is the cheap experiment that certifies the control actually closes the confound.
- **CENTRALNESS-BALANCED MENU procedure [added 2026-07-14 — how to run the balancing experiment]:**
  1. **Measure centralness once per bank:** one batched no-demo pass — per task, show the decoder the full
     task-local menu (no demos) and record the constrained posterior over ALL options. Each metric's blind
     posterior mass IS its empirical centralness. One vLLM batch per task; CPU afterward.
  2. **Bin** bank metrics into within-task centralness quartiles.
  3. **Construct menus so the gold's centralness quartile is uniform across trials** (gold in the top
     quartile for exactly 1/4 of trials, etc.), distractors matched to span quartiles. Deterministic
     stable-hash assignment, frozen before any scoring.
  4. **Readout:** C1 value per gold-centralness bin. Value surviving ONLY in the top-quartile bin =
     C1 was prior-driven; flat across bins = the blind subtraction was already sufficient and the standing
     C1 numbers gain a validity certificate. Report alongside the random-gold arm.
- **ELIGIBILITY-CLASS REDESIGN [added 2026-07-14, user decision — makes gold/distractor exchangeability
  hold by construction]:** the v13 filters left almost no distractor-equivalent metrics (a matched
  distractor needs similar entropy AND panel balance). Fix by defining ONE eligibility class applied to
  golds and distractors IDENTICALLY, and drawing both from it:
  1. **Eligibility rule (uniform):** minority-verdict rate >= 0.3 (i.e. base rate p in [0.3, 0.7], which
     implies H2(p) >= ~0.88 bits — this is the operational form of "entropy > .8": it directly guarantees
     panel supply, >= 36 minority texts on a 120-text design split). No gold-only filters of any kind.
  2. **Go to R2 for the pool:** R2 has many more clusters than R3 (known under-merge), so the eligibility
     class is populous enough to sample from. **Same-level menus only** — golds AND distractors both R2,
     never R3-gold-among-R2-distractors (level-mixed descriptions differ in register/length = a fresh
     menu tell). Exclude the gold's own R3-ancestor siblings and behavioral clones (CLONE_CAP) so the menu
     is neither unfairly easy nor unwinnable-by-paraphrase.
  3. **Golds = uniform draws from the class** (this generalizes the random-gold arm from a validity
     check into the default design); optionally entropy-BIN-match distractors to the gold (bins of 0.05)
     for the strictest exchangeability statement.
  4. **What we give up, and where it goes:** dropping target-entropy quintiles loses the deliberate
     low-ceiling coverage — that job MOVES to Phase E, where |Omega| composition manufactures the ceiling
     gradient by construction. C1 gets clean identification measurement; Phase E gets the gradient. Do
     not resurrect low-entropy golds inside C1.
  5. Report the class size per task and the infeasibility rate of the uniform draw (population-selection
     effects stay quotable).

**C3 — the free-ish doubling. [CORRECTED 2026-07-14 — the original "costs no GPU" claim was wrong for the
behavioral channel]:**
The executor already emits a continuous `P(YES)` per probe and **we binarize it at 0.5 before showing the
decoder.** We are discarding gradient information at the very first step of the chain.

    current:   8 examples x 1 bit  =  8 bits  ->  2^8  =    256 states
    C3:        8 examples x 2 bits = 16 bits  ->  2^16 = 65,536 states
    (3 bits/example = 16M states: hopeless. 2 bits is the sweet spot.)

Quantize `P(YES)` into 4 levels; the decoder sees "this one barely fired, that one fired hard" — the
gradient a human would use. **Cost correction:** enumeration-is-cheap holds only where value-per-state is a
table LOOKUP. On the behavioral channel every state costs an induction + |H| executions, and the cached v13
cells cover observed 8-bit states, NOT 16-bit quantized states. Therefore:
- **C3 ACHIEVED value is cheap and runs now** (canonical quantized state + blind + shuffled ~= 3 inductions
  per panel) — and achieved C3-vs-C2 is the question that matters.
- **C3 exact cap is NOT available**: mark `exact_structural_cap: UNAVAILABLE` on every C3 row (never inherit
  C2's cap); ceiling statements limited to the trivial H(z) <= 16-bit budget + the record/rank bound.
- A partial enumeration (2 pools, sentinel metrics only) is priced separately ONLY if achieved C3 beats C2
  decisively.

**Decision rule (predeclared):**

| pattern | meaning | action |
|---|---|---|
| `C0 ~ C1 >> C2` | demos DO identify; decoder cannot ARTICULATE | **Phase D is right** — GEPA the articulation |
| `C0 >> C1 ~ C2` | 8 demos do not identify | decoder tuning is futile — raise `k`, fix panels, C3 |
| `C3 >> C2` | 8 bits was starving the decoder | ship the 2-bit code as the default instrument |
| `C0` itself low | even the TRUE description fails to reproduce the metric | the metric is not a coherent function — a finding about the METRICS, not the instrument |

---

### PHASE C — The decoder-scaling law. **[the experiment we assumed and never ran]**

**Correction of record (2026-07-14):** the roadmap previously reasoned as if "recognition scales with model
size, reconstruction does not." That claim rested entirely on the A2 comparison, which is not controlled. It
is **withdrawn**. Reconstruction plainly needs priors too (interpret the demos, hypothesize a rule, phrase it
so a *different* model executes it correctly), and it should scale.

The asymmetry that IS real, and that generates the prediction:

> **The behavioral channel passes through a second, FROZEN model the decoder does not control (the executor).
> MCQ does not.** A 70B decoder can write a correct rule that the 8B executor fails to execute. Decoder
> scaling is therefore **capped by executor capability** — which is exactly what C0 measures.

So: **run it properly.** Same panels, same executor, same metrics, same templates; both channels. Report
MCQ-lift and behavioral-bits vs decoder scale, with C0 drawn as a horizontal line. **Prediction to test:**
behavioral saturates at C0 while MCQ keeps climbing. If behavioral climbs with scale, the decoder was the
bottleneck and Phase D pays.

**OSL UPGRADE [added 2026-07-14 — the previous panel could not produce a plottable scaling law]:** the old
decoder set {Llama-8B, Mistral-24B, Qwen-32B, Llama-70B} has at most a 2-point within-family leg (Llama
8B->70B) — a difference, not a curve, and cross-family points can never be pooled (standing rule). Fix:
- **Primary staircase: Qwen2.5 {3B, 7B, 14B, 32B, 72B}** — five same-family rungs, and the weights are
  ALREADY cached on sk3 from the OSL executor-sweep work (no downloads). This is the decoder-OSL figure:
  per channel, permutation z (and bits) vs log-params, C0 as the horizontal reference line, per-metric
  spaghetti + median curve.
- **Replication family: Llama {3.1-8B, 3.3-70B}** — 2 points, direction-check only, plotted separately,
  never pooled with Qwen.
- Readout conventions inherit from the OSL program (`osl_battery`/`osl_fit` lineage): same-family
  staircases only, threshold-free y-axis, fit shapes on the declared-primary instrument. Watch for the
  known OSL shapes — a falling limb or inverted-U in the behavioral channel is interpretable here as the
  executor-capability wall (divergence-toward-truth analog), not noise.
- Fast lane runs the whole staircase (5 rungs x 35 metrics is cheap at screening settings); cert lane
  re-measures the 6 sentinels x all rungs for the quoted figure.

**MULTI-DECODER COMMITTEE readout [added 2026-07-14, CERT/slow lane — user request]:** Phase C already
produces per-decoder y-hat tables on IDENTICAL frozen panels, so everything below is a CPU re-slice of Phase C
outputs (no new GPU work). Each decoder induces its OWN rule and scores it; per metric report:
1. **Per-decoder vector (headline robustness statistic):** value_i for each decoder, each against its own
   blind/shuffled controls; summarize as median +- spread. Never pool cross-family into one number for a
   scaling claim (standing rule); the vector itself is the deliverable.
2. **Committee channel (the recommended single extra row):** majority-vote y_maj across the decoders'
   executed verdicts, then `I(M_omega; y_maj) - max(blind_maj, shuffled_maj)` where the controls are the
   majority votes of the per-decoder controls. Stays a single binary channel, so N=240 estimation is
   unchanged. Interpretation: committee >> best single decoder = decoders make COMPLEMENTARY errors
   (articulation is multi-realizable — different induced prompts capture different facets of the criterion);
   committee ~= median = decoders converge on one recoverable core.
3. **Joint MI `I(M_omega; [y_1..y_d])` — diagnostic only, never headline:** upper-bounds what the panel
   collectively received, but the joint alphabet is 2^d states so Miller-Madow correction and small-N bias
   grow with d; the permutation null must permute M_omega against the joint tuple. Report next to the
   committee value as a ceiling check.
4. **Max over decoders is winner's-curse** — max of noisy estimates inflates. Legal ONLY as
   select-on-dev-30 / report-on-held-out-240 (same two-population discipline as fast->cert promotion);
   otherwise do not report a max at all.

---

### PHASE D — GEPA-tuned decoders (v14 proper). **[RUN unconditionally; INTERPRET via B + C]**

**[Re-gated 2026-07-14 per user decision.]** Tuning is cheap (bounded: <=4 rounds x 8 candidates x 3 jobs) and
targets BOTH channels, so it runs regardless of the ladder outcome. What stays gated on Phase B is the
*headline*: `bootstrap_ladder_decision` decides which tuned rung is the paper's number
(ARTICULATION_BOTTLENECK → tuned C2 leads; IDENTIFICATION_BOTTLENECK → tuned C1 leads; MIXED → report the
full tuned ladder, no single headline).

Ship as already implemented (`v14_decoder_tuning.py`: <=4 rounds x 8 candidates, fitness
`[TR - max(blind,shuffled)] / H(M_omega)`, ONE shared template per channel/arm across the 3 families on pooled
mean fitness, copy-penalty inside the objective, both exemplar arms). Additions:
- Tune the **MCQ decoder too** — deliverable D1 requires a prompt-optimized MCQ decoder, not just behavioral.
  MCQ stopping residual on DEV metrics only (§4h); sentinel residual once, post-freeze.
- Run against the **best code from Phase B** (i.e. C3's 2-bit code if C3 > C2).
- Report untuned AND tuned. The delta IS the result.

**Behavioral-signal levers [added 2026-07-14 — the decoder's own words are the degree of freedom]:**
GEPA template tuning is lever #1 (USER MANDATE 2026-07-14: this runs, full stop). Stack on top, each a
declared arm, all capacity-preserving (the decoder still sees ONLY the k demo bits — these improve
DECODING, not the channel, exactly like better error-correction at fixed code length):
1. **Best-of-n induction with demo-fit selection:** sample n=4 candidate M-hats per (panel, state), pick
   the one whose EXECUTED verdicts best fit the k TRAINING demos (never H — selection sees no held-out
   bit). Costs n x induction + n x k executor scorings; the k-scorings are tiny.
2. **Induce-execute-revise (one round):** decoder sees its M-hat's verdicts ON THE DEMOS ONLY, revises
   once. Same capacity argument; list-decoding flavor.
3. **Restate-then-induce:** two-stage prompt — decoder first describes the demo pattern in its own
   vocabulary, then commits to a rule. This is the "say it in the LLM's words" arm made explicit; GEPA can
   tune both stages.
**M-hat ARCHIVAL (required, release-blocking):** every induced rule is stored content-addressed with
(metric, panel, state, arm, decoder family, decoder revision, tuned/untuned) — the rule-dedup store
already keys by content hash; this adds the mandate that the full M-hat text corpus SHIPS with the release
(per-metric reconstructed-prompt gallery, like the notebook #148 gallery). We must be able to diff what
untuned vs tuned decoders actually SAY, not just their scores.

---

### PHASE E — The `|Omega|` scaling law + ceiling stratification. **[D3 + D4; the paper's figure]**

Compose targets from `|Omega|` = **1, 2, 3, 5, 8** criteria (conjunction and weighted-sum compilers, both
declared). This yields the low/med/high ceiling gradient **by construction** rather than by post-hoc selection
on the outcome, which would be selection on the dependent variable.

Per `|Omega|` report: exact cap, achieved (tuned + untuned), C0/C1/C2/C3, permutation-null percentile, and
accuracy. **Predicted:** the cap falls monotonically in `|Omega|` — more criteria = more information to push
through 8 (or 16) bits. If it does NOT fall, the bottleneck is not code capacity and that is a bigger finding
than the scaling law.

Also fit **`gamma_V`** (theory note §12.9.8) per metric on the existing ledgers — CPU-only, no GPU, runs while
everything above is on the accelerators.

### §8.5 STATUS + RANKED NEXT STEPS **[2026-07-14 audit; supersedes any earlier sequencing where they conflict]**

Where things stand on disk (verified 2026-07-14):
- v13 Tier-B COMPLETE: 70 rows = 35 metrics x 2 channels, 70B constructor only
  (`sk3:cr3-v13.1/outputs/tier_b/lanes/*/results.parquet`). Behavioral med 0.00883 / cap 0.02457;
  MCQ med 0.0748 / cap 0.1246 (NOTE: MCQ value = P(gold) lift, NOT bits — never compare across channels).
  Worst-pool median = 0.0 for BOTH channels: teaching-pool sensitivity is a first-class caveat.
- `ceiling_ladder.py` (2,429 lines) + tests built, 6/6 green — **COMMIT IT before any GPU phase**; freeze
  SHAs must bind to a commit.
- **Bias correction of record:** v13 heldout_size = 60 everywhere (manifest-verified), NOT 300. Raw plug-in
  MI bias ~0.012 bits/table — 1.4x the median achieved value. Control subtraction partially cancels it;
  clip-at-zero fights it. Net direction ambiguous → NO quotation of 0.0088 until `audit_native_v13`
  (permutation null) lands. The permutation percentile is the arbiter, not the point estimate.
- Reference pilot: code-review_R3_metric0 → kappa 0.110 with pairwise agreement 0.749 at base rate ~0.89.
  Rule: NEVER quote kappa without agreement + base rate. Predeclare the reference-reliability gate
  (agreement floor vs pooled-marginal chance + planted-anchor hard fail). Free diagnostic: per-metric
  Sonnet-majority base rate vs frozen-8B base rate on the same probes = description-vs-behavior divergence.

Ranked by information-per-GPU-hour:
1. **|H| = 60 → 240** for every rung going forward (pure executor batch scoring — the cheap kind). Cuts MI
   estimator noise ~2x, shrinks raw bias to ~0.003 bits. Standard readout becomes the **permutation z-score**
   `(observed − null_mean)/null_sd` — comparable across metrics and base rates; scaling laws plot in z.
2. **Finish Phase A** (Sonnet reference w/ void-on-failure + `audit_native_v13`, both CPU/API) → **run the
   ladder** (Phase B) on sk3 GPUs 5/7. C1 is the likely hero instrument; C3 is free ceiling doubling.
3. **Phase D tuning runs unconditionally** (see re-gated header above) — both decoders, cheap; the ladder
   picks the headline. Report the unconstrained exemplar-carrying arm as the honest "as high as behavioral
   goes" number; the (a)−(b) gap is a finding.
4. **Phase E composition** — turn one number into a curve; a monotonic cap-vs-|Omega| plot is robust to the
   absolute level being small. gamma_V fits on existing ledgers in parallel (CPU).
5. **Executor decision LAST**, after the reference lands: if C0 is low under the 8B executor, no decoder work
   can beat it — that is the one result that justifies promoting 70B as executor (via the de-circularized
   gate) and re-scoring.

The loop: audit → ladder → tune both decoders → re-measure in z on |H|=240 → extend along |Omega|. Each cycle
either raises achieved value or certifies which wall it hit; both are results.

**Throughput discipline [added 2026-07-14 — the pace target is METRICS PER NIGHT, not certificates per week]:**
the slowness is a DESIGN choice (exhaustive state enumeration for exact caps), not an implementation defect.
Achieved value + permutation null needs only OBSERVED states; exact caps need full enumeration. So split them:
1. **Caps on sentinels only** (6 humor + top-1/task). Everywhere else: level-0 entropy cap + record/rank bound.
   The scaling laws need MANY metrics with achieved values, not exact caps everywhere.
2. **Rule-level dedup before execution** — observed collapse is 6-41 distinct rules per 256 states; scoring
   distinct rules only is a 4-40x cut. Key by content hash (already spec'd in v13.1-B §3; enforce it).
3. **K=6 (64 states) for fan-out**, K=8 only on sentinels. Menu permutations 8 → 4 outside sentinels.
4. **One vLLM batch per resident model per phase** — all metrics' inductions in ONE call, all rule x H
   executions in ONE call (thousands of prompts per call, standing rule). Never per-metric serial loops.
5. **Pipeline the 2 GPUs**: 70B constructor streams completed metrics to the 8B executor lane; no barrier
   between phases.
6. **No re-freezing on fan-out.** Sentinel gates run once; a fan-out lane failure voids that lane only
   (append-only cache), never restarts the campaign.
Target: with 1-6, behavioral Tier-B is ~25-40K queries/metric ~= 15-25 min/metric -> all 35 metrics in one
overnight on 2 GPUs. If a proposed run cannot hit within ~3x of this, the design is wrong — rescope it.

**TWO-LANE STRUCTURE [added 2026-07-14, user decision — formalizes the tiering as two named lanes]:**
- **FAST lane (screening; the default for anything new):** achieved value + permutation z-score ONLY.
  Observed states, K=6, 4 menu permutations, level-0 entropy cap + record/rank bound, frozen executor
  verdicts as the only reference, no exact enumeration, no Sonnet passes, no prereg. ~15-25 min/metric.
  Fan out wide (hundreds of metrics, every task). Output schema carries `lane: "fast"` and is
  **structurally quarantined**: fast-lane rows can never enter a certificate, a figure caption, or a
  headline — they exist to rank what deserves the slow lane.
- **CERT lane (rigorous; everything the paper quotes):** independent reference + chance-bound gate,
  planted + hidden anchors, both controls, exact caps where enumerable (UNAVAILABLE explicitly where not),
  10K permutation null + Miller-Madow sensitivity, bootstrap CIs, prereg freeze. Runs on sentinels + the
  fast lane's top picks + anything a claim depends on.
- **Promotion rule (predeclared):** fast->cert promotion by declared criteria (top-k per task by z, plus
  any metric a planned figure needs). Promotion re-measures FROM SCRATCH in the cert lane on the frozen
  instrument — fast-lane numbers are never "upgraded" in place, so screening selection can never leak into
  certified values (two-population discipline, same §5 logic).

### §8.6 Future work (pointer only)

Channel-battery extension (no-keyword explanation channel / taboo game) and the articulability-routed
acquisition capstone live in `notes/2026-07-14__future-work-channel-battery-and-acquisition.md`. Neither
starts before Phases A–E land.

### GPU discipline (unchanged)

<=4 GPUs total across sk1/sk2/sk3. On sk3 only physical 0/5/6/7 via `scripts/run_v14_campaign_sk3.sh`
(1-4 are someone else's). Kill by specific PID only, parent AND its `VLLM::EngineCore` child. Never
pkill/killall. `value_cells.sqlite` is append-only, so pausing a lane costs only the in-flight batch.

## 7. Related work framing (for the paper)

Nearest neighbors: instruction induction / APE (induce an instruction from examples — that is our decoder, but never treated as the reconstruction half of a capacity-bounded bottleneck); discrete/text autoencoders with continuous latents; prompt compression. **Novel here: the encoder is a prompt executed by a frozen LM, the code is the induced discrete verdict pattern, the decoder is an LM reconstructing the criterion, and the assembly is used to MEASURE the articulability of a human construct — with a capacity chain `I(M̂;M_ω) ≤ I(z;M_ω) ≤ H(z) ≤ k` and coverage certificates over the prompt space.** Frame as: instruction-induction reframed as an information-bottleneck measurement instrument.
