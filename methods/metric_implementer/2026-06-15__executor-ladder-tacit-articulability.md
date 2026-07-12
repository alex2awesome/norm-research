# The executor ladder: tacit knowledge τ(E), the articulability gap a(E), and what is measurable

*(2026-06-15. Companion + notation reconciliation to `2026-06-12__formalization.md`. Settles the
naming so "tacit knowledge as E's resource" and "the articulability residual" are lexically distinct,
formalizes the L↔E substitution measure, and corrects two loose claims: the `L→∞` limit and
`F_E(∞)→C`.)*

## 1. One surface, two axes

`F_E(L)` = best agreement-with-anchor achievable by **executor E reading a ≤L-word rubric**. Two
inputs, both real axes:

- **L** — articulation budget (words / nats of rubric).
- **E** — executor capability. **Tacit knowledge lives here**: a thin rubric ("good stories have an
  inverse-pyramid structure") becomes an accurate judgment only to the degree E supplies the tacit
  knowledge to apply it. A verifiable rule (code) needs *no* interpreter; an articulated rule needs a
  tacit-knowledge-bearing E. The gap between them is the value of the interpreter.

Tacit knowledge enters **through E** (this is the executor-relativity of Chan–Critch–Dragan 2021,
stated as a resource rather than a residual). `E > E' > E''` ⇒ `F_E ≥ F_E' ≥ F_E''` is the
executor-monotonicity assumption; observational scaling laws (Ruan 2024) are the instrument that
places E on a continuous capability axis.

## 2. Naming (locked 2026-06-15)

Read `F_E(L)` up the executor ladder at the L-optimum. Rungs:

| Rung | = | Meaning |
|---|---|---|
| **A** | `F_code` | verifiable — a program, no interpreter |
| **B** | `F_E*` | articulated — best rubric an executor E can apply (see §5: `F_E* = sup_L F_E(L)`, finite `L*`) |
| **C** | `F_dense-direct` | ceiling — dense model predicting Y from X with **no** rubric channel (Bayes-floor proxy) |

Two gaps, opposite-signed, on the same ladder:

| Symbol | Def | Reads as | In E | Old name |
|---|---|---|---|---|
| **τ(E)** | `B − A` | tacit knowledge **contained in** E | ↑ | "language-tacit / LLM-over-code surplus" |
| **a(E)** | `C − B` | the **articulability gap** (Polanyi, "know it when I see it") | ↓ | the old `τ(E)` = `lim_L G_E(L)` |

**Identity:** `C − A = τ(E) + a(E)` — the verifiability gap splits into E's tacit knowledge plus the
articulability residual. Out of scope: `1 − C` = taste + irreducible noise.

⚠️ **The symbol `τ(E)` is REASSIGNED.** The old formalization-doc `τ(E)` (the residual surviving
unlimited words) is now **`a(E)`**. The new `τ(E)` is the `B − A` resource. `a(E)` is the paper's
headline gap.

## 3. Is tacitness measurable at all?

**Only relatively.** There is no executor-free absolute tacitness; you measure `τ(E)`, `a(E)` at a
*specified* E and ceiling C. The intrinsic object is the **`a(E)` curve over the executor lattice**;
its infimum `a(∞)` (the Polanyi floor) is an *extrapolation*, never a direct measurement. ("Upper
bound of articulability of m is a function, not a number" — formalization §7.)

Concrete operationalizations the lit review surfaced, tagged by which gap they hit:

| Method | Targets | Mechanism |
|---|---|---|
| Reber yoked design (1967/89) | `a(E)` | expert verbalizes rules → fresh agent runs only those → perf gap |
| V-info rubric ladder (Hewitt 21; Ethayarajh 22) — *ours* | `a(E)` | fit `I_{V_{L,E}}(X→Y)` vs L; `C −` asymptote |
| RM-overoptimization (Gao 23) | `a(E)` | gold-vs-proxy gap, fitted floor |
| Verbal overshadowing (Schooler/Wilson 90/91) | `a(E)` magnitude | agreement drop when reasons forced (.55→.11) |
| 4PL upper asymptote (Barton-Lord 81; Choi 26) | `a(E)` | sub-1 item ceiling `d<1`, needs high-θ (dense) anchor |
| Imitation game / ID-ratio (Collins-Evans; Arsal 21; DeCarlo=IRT) | `a(E)` | judge tells expert from rubric-pipeline? `d'` CI vs 0 |
| Lens-model residual (Karelaia-Hogarth 08) | `a(E)` magnitude | configural variance a linear cue model misses ≈ .08 |
| G-theory variance decomp (Cronbach 72) | residual share | rater/item/method components |
| IB codability ε/gNID (Zaslavsky 18); naming agreement (Majid 14) | construct articulability | distance-to-optimal-frontier; description length |

All converge on *anchor − best-articulable-reproduction*; each reports it **at an executor**.

### 3.1 Row-by-row (3-sentence summaries)

- **Reber yoked design (1967/89)** — Subjects learn an artificial grammar implicitly, then verbalize
  the rules they think they used; a separate "yoked" group gets *only* those verbalized rules and is
  tested. The yoked group scores above chance but well below the original learners, and that gap is the
  competence that resisted verbalization. Cleanest classical demonstration that `a(E) > 0` — and it is
  exactly our protocol with E = the rule-follower.
- **V-info rubric ladder (Hewitt 21, Ethayarajh 22) — ours** *(extra detail).* Predictive (V-usable)
  information generalizes mutual information by restricting the decoder to a family V:
  `H_V(Y|X) = inf_{f∈V} E[−log f[x](y)]` and `I_V(X→Y) = H_V(Y) − H_V(Y|X)` = the bits of the target a
  V-decoder can actually extract. We set `V = V_{L,E} = {E reading a ≤L-word rubric}`, so
  `I_{V_{L,E}}(X→Y)` is literally "how much of the anchor a ≤L-word rubric applied by judge E recovers";
  sweeping L traces the fidelity ladder, the dense/unconstrained family gives ceiling C, and
  `a(E) = C − sup_L I_{V_{L,E}}` is read straight off the gap between the ladder's plateau and C.
  Ethayarajh's **pointwise** V-info (PVI) gives the per-datapoint version — an item with low PVI under
  the rubric family but high PVI under the dense family is "language-hard but not absolutely hard,"
  localizing *where* the residual lives. Caveat that drives the unsupervised question: V-info is
  **supervised** — the `inf` is over a loss that needs a target `Y` (or a target law `p(y|x)`, §7).
- **RM-overoptimization (Gao 23)** — A "gold" RM stands in for true preference; a "proxy" RM trained on
  its labels is then optimized against, and the gold-vs-proxy gap is tracked vs optimization pressure
  (KL). The gap follows a clean parametric law with a nonzero floor — proxy reproduction saturates
  below the gold ceiling. The empirical precedent that a fidelity-vs-budget curve with a positive
  residual exists and is fittable; the template for reading `a(E)` as our `fidelity(L)` floor.
- **Verbal overshadowing (Schooler/Wilson 90/91)** — Participants judge a stimulus (face, wine, jam) and
  half first describe it in words; describers' agreement with the criterion judgment drops sharply
  (≈.55→.11). Verbalizing *degrades* a judgment that was accurate when left tacit, so the drop
  lower-bounds how much the judgment rode on non-verbalizable processing. Proves `a(E)` can be large and
  that forcing articulation is not free — predicting non-monotonicity of fidelity in L.
- **4PL upper asymptote (Barton-Lord 81; Choi 26)** — A 4-parameter-logistic item curve adds an upper
  asymptote `d < 1`: even an infinitely-able respondent succeeds with prob `< 1` on the item. That `d`
  is a per-item measurement of the irreducible residual, identifiable only if the sample has
  near-ceiling (high-θ) respondents. The dense model is that high-θ anchor, so `a(E)` is *measured*
  under 4PL rather than *assumed* away (2PL/GRM fix the ceiling at 1).
- **Imitation game / ID-ratio (Collins-Evans; Arsal 21; DeCarlo=IRT)** — A judge sees paired responses,
  one from a genuine in-group expert and one from the candidate (rubric+executor) pretending, and tries
  to tell them apart. Above-chance discrimination (identification ratio > 1, or `d'` with CI excluding
  0) means the pretender has not closed the gap, i.e. `a(E) > 0`. Via DeCarlo's SDT=IRT identity this is
  the *same* estimator as our judge-IRT, giving a principled "residual is real / stop" criterion.
- **Lens-model residual (Karelaia-Hogarth 08)** — Regress both expert judgments and the true criterion
  on a shared set of articulated cues; the linear additive-cue model captures the bulk of expert
  judgment. The configural (nonlinear/interaction) residual a linear model misses averages only ~8% of
  variance across ~250 studies. A decades-deep meta-analytic measurement of `a(E)`'s *magnitude* and the
  basis for the optimistic "thin tail" expectation.
- **G-theory variance decomposition (Cronbach 72)** — Generalizability theory partitions measurement
  variance into object/rater/item/occasion components plus interactions and residual. The
  residual / unmodeled-interaction component is the share of judgment variance no articulated facet
  explains. Operationalizes the tacit residual as a *variance share* (and a 4-coordinate reliability
  profile) rather than a regression gap.
- **IB codability ε/gNID (Zaslavsky 18); naming agreement (Majid 14)** — Codability = how reliably and
  economically a population puts a stimulus into words (color: high; most smells in English: low).
  Zaslavsky's Information-Bottleneck treats a naming system as a soft encoder and scores its distance
  ε/gNID from the optimal complexity-accuracy frontier. Together a *derivable* (not merely fitted)
  articulability measure of a construct × population — the human-language analog of `a(E)`.

## 4. The L↔E substitution measure (describability cost of E's tacit advantage)

Use the budget-optimal **conditional V-entropy** `H_{E,L}(Y|X) = inf_{|s|≤L} E[−log E(s)[x](y)]`
(Hewitt et al. 2021), NOT a single rubric's cross-entropy `H_{f_{E,r_L}}` (which is an upper bound on
it). Iso-fidelity substitution:

> `H_{E,L}(Y|X) = H_{E',L'}(Y|X)`, with `E > E'`, `L < L'`.

`L' − L` is the **local exchange rate** `∂L/∂E |_{H const}` (finite-differenced): a *describability-cost*
(words/nats) measure of E's tacit advantage over E′ for **this construct at this fidelity**. It is the
**differential** version of the global `τ(E) = B − A` (the integrated vertical gain). Three caveats:

1. **Local, not global** — varies over the (L,E) surface; a derivative, not a constant.
2. **Units = rubric-words ⇒ confounded by instruction-following capacity** (IFScale). A bigger E may
   need fewer words because it *knows* more or *reads* better. Control IF-capacity to isolate tacit
   knowledge.
3. **Defined only in the substitutable region.** If E′ can't reach E's fidelity at any L, no finite
   `L'` exists — a *non-substitutable* gap = an `a(E')` residual E′ never closes with words. Word-
   substitution measures only the verbalizable part of the tacit difference.

Cross-thread: `L*(E)` (knee) = minimum description length to exhaust E's reach = **sophistication**
(Vereshchagin–Vitányi); `L'−L` = the contract-theory **describability cost** (Anderlini–Felli). The
substitution idea unifies the MDL, contract, and scaling-law framings of one quantity.

## 5. The two infinities (scaling-law language) — `F_E(∞) ↛ C` in general

There are **two distinct bottlenecks**, and infinite capacity removes only one:

- **Capacity (`E → ∞`):** scaling N, D. Chinchilla `L(N,D)=E+A/N^α+B/D^β → E > 0`. **Infinite capacity
  gives the Bayes floor, not omniscience.** For *direct* prediction, `F_E(∞capacity) → C`.
- **Channel (`L`):** articulation budget + instruction-following + verbalizability. A *separate* axis.
  IFScale (Jaroslawicz 25): following-acc ≈ `(per-instr acc)^n`, **degrades** in n; per-instr acc `< 1`
  ⇒ `(acc)^n → 0`. So `F_E(L)` is single-peaked with a **knee `L*`** beyond which words hurt
  (rubric-overshadowing). The right object is `F_E* = sup_L F_E(L)` at **finite `L*`**, NOT `lim_{L→∞}`.

Therefore `a(E) = C − F_E* ≥ 0`, and **`a(E) → 0` as `E → ∞` is an empirical extrapolation, not a
theorem.** Two independent reasons to expect `a(∞) > 0`: (i) Chinchilla — a positive floor survives
infinite capacity; (ii) Polanyi — a non-verbal core may exist that no rubric encodes. The clean
experimental signature of a *real* fully-tacit residual: **`a(E)` plateaus above zero** as E climbs the
observational-scaling axis, rather than trending to 0. Extrapolation is as fraught as all
scaling-law extrapolation (Schaeffer 24: ceilings are metric-manufacturable; irreducible terms only
jointly identified). We report `a(E)` on our ladder + a model-dependent `a(∞)`, with honest CIs — we
never "see" `F_E(∞)=C`.

## 6. Measuring C with one model class — split `a` into two objects

Llama-8B-for-everything is a real limitation on both sides: as a Bayes proxy 8B is likely
under-trained ⇒ `C` too low ⇒ `a` under-estimated (can yield `B>C`); as a single class it conflates
capability with articulability whenever the judge E ≠ that 8B. Decompose:

- **`a_self(E) = C_{E-class} − F_E(rubric)`** — same model, direct vs rubric-mediated. Capability
  fixed ⇒ the **pure channel/serialization gap** ("what E knows but can't put into words and reapply").
  Use **same-class (ideally same checkpoint) C** here.
- **`a_ceiling(E) = C_best − F_E(rubric)`** — distance to the strongest predictor; conflates.
- **`a_ceiling = a_self + (C_best − C_{E-class})`** = articulability + capability-gap.

Operating rule: same-class C for `a_self`; a larger model / **ensemble / class-sweep** for the
Bayes-floor C; saturation checks (does C still improve with data/params? then not a ceiling) +
deconfounding on de-leaked/CF data. Report `a_self` and the capability-gap separately.

## 7. V-information is supervised — how the unsupervised metrics merge

`I_{V_{L,E}}(X→Y)` needs sampled `Y` (its estimator minimizes `E_{(x,y)}[−log f[x](y)]`). With `Y`
latent there is no loss ⇒ no V-information. So **unsupervised metric learning does not yield an
articulability gap by itself.** Relate them by **composition, not unification**:

- Unsupervised metric learning (metric-tree / autometrics) = a **proposal distribution over rubrics
  `s`**, optimized by an internal criterion (coverage/consistency/held-out metric prediction). Upper
  bound on **metric-space** articulability.
- Downstream articulability = `sup_{|s|≤L} I_{V_{L,E}}(X→Y; s)`; a rubric's marginal value is the
  **conditional V-information** `ΔI(s) = H_V(Y|X) − H_{V∪s}(Y|X)` (§7.1). The sup is defined **only
  relative to `Y`**.

So unsupervised = *propose* `s`; supervised V-info = *score* `s` against `Y`. The "one step away" is
exactly that grounding step: an internally-excellent metric can still have large `a(E)` to `Y`. Without
an observed `Y` (or validated proxy), `a(E)`/`τ(E)` for the downstream variable are **undefined**.

**No truly anchor-free articulability — but the anchor can be a *distribution*, not labels.** `H(Y|X)`
and `I_V(X→Y)` are **functionals of a target law `p(y|x)`**: given `p(y|x)` you compute
`H(Y|X) = E_x[H(p(·|x))]` and `H_V(Y|X) = inf_{f∈V} E_x[CE(p(·|x), f[x])]` with **no realized labels**.
So the requirement is a target *law*, not a labeled dataset — and `p(y|x)` **is** a (soft) anchor;
"information about Y" is undefined without some Y. For *articulability* specifically you need `p(y|x)`
scored against **two** families (rubric vs dense); `a(E)` = their cross-entropy difference. `H(Y|X)`
alone is just the noise floor, not articulability.

Legitimate non-judge sources of `p(y|x)`: (i) **empirical human-rating distributions** (perspectivist /
ChaosNLI — intersubjective, not one model); (ii) **known synthetic generative mechanisms** (E0,
`p(y|x)` closed-form); (iii) **multi-rater latent consensus** (IRT posterior over the shared signal).
A *single dense judge's softmax* is the degenerate case → reduces to "explain this model's decision" =
XAI; **rejected, not the direction.** Truly label-free (no `p(y|x)` of any kind) you can only measure
**internal metric-space structure** — predict one metric from others `I_V(m_i→m_j)`, reconstruction,
paraphrase-stability — which scores the metric system's *coherence/compressibility*, NOT downstream
articulability. That is the honest scope of the "one step away": unsupervised buys metric-space
structure; converting to construct articulability requires grounding ≥1 metric in a real (possibly
soft/distributional) target.

## 9. Anchor-free articulability: codability, the communication game, synthetic calibration

Goal: characterize articulability with **no external Y**, using only the unsupervised metrics we have
(consistency, reconstruction, synthetic). Frame = Cronbach–Meehl **construct validity** + Campbell–Fiske
**MTMM**. (A teacher testing a no-ground-truth concept uses consistency / generalization / combinatorial
checks — the same triangulation.)

### 9.1 The codability/relevance factoring — where anchor-free can fully reach
Anchored articulability of metric m (rubric s, executor E, budget L) against Y runs along the chain
`s → m_recovered → Y`:
- **codability** `= I(s; m_recovered)` — can the words transmit the concept to a naive reader. **ANCHOR-FREE.**
  A generalization of V-information that overlaps validity theory (Cronbach): the "target" is the
  recovered concept, not an external label.
- **relevance** `= I_E(f(m); Y)` — do the learned metrics, **composed into a rubric `f(m)` and read by
  E**, recover the target. **needs Y (we have it for preference).**
DPI: if the rubric reaches Y *only* via the concept it transmits (`s → m_recovered → Y` Markov), then
**`I(s; Y) ≤ I(s; m_recovered)`** — **codability is a data-processing UPPER BOUND on anchored
articulability.** So anchor-free codability gives a real *ceiling* on `a(E)` with zero labels; only the
relevance factor needs Y. That is how far anchor-free reaches — and it is a lot.

**Relevance is the metrics composed in a PROMPT and decoded by E** (`f(m)` read by E = the actual
`V_{L,E}` family) — not the legacy linear-regression-over-metric-scores (a convenience from the
Autometrics paper; dropped). Decompose the anchored side: `I_E(f(m); Y)` = **achieved**;
`sup_{|s|≤L} I_E(s; Y)` = budget-L **frontier**; `C` = **ceiling**; `a(E) = C − frontier`;
`ε = frontier − I_E(f(m); Y)` = quality of our discovery+composition (the achieved-encoder gap).
**This is the final step — not yet done in earnest; we have ground truth Y for preference.**

### 9.2 Codability operationalized (psycholinguistics → metrics)
Lit definitions (Brown & Lenneberg 1954; Majid et al. 2018; Zaslavsky 2018):
- **agreement**: modal-response share / Simpson diversity (P two speakers agree) / entropy of the name dist.
- **length**: name/description length (= our L budget; MDL).
- **latency**: production speed (LLM analog: scoring logprob / confidence / score-entropy).
- **communication (strongest)**: reference-game accuracy `= I(stimulus; listener-recovery)` through the
  naming channel; Zaslavsky's IB = complexity `I(W;M)` vs accuracy `I(W;U)`, distance ε/gNID to frontier.

### 9.2b Description-space (concept) codability — analyzing the metric library
A *distinct* question from §9.3: not "re-score X," but **codability of the concept itself in
description-space** — "how many *truly different* ways is 'inverse pyramid' encoded in our library?" on
the verbalizations, no data X. Raw verbalization dispersion (what `embedding_diversity_per_task.parquet`
already gives: `mean_pairwise_cossim`, `participation_ratio`) is **insufficient** — it cannot separate
*benign synonymy* (many wordings, all recovering the same concept = HIGH codability) from *ambiguity /
fragmentation* (many wordings, recovered as different concepts = LOW codability).

**Key realization: we already compute both halves separately and never join them.**
- surface distance `S` = cosine of **frozen** `emb_rubric_cluster_<task>.npy` (OpenAI text-embedding-3-small,
  exogenous to clustering); cross-check with frozen BGE-large (`emb_bge_rubric_cluster`) / MiniLM.
  **NOT** `emb_lora_*` (LoRA-BGE *trained on the verdicts*) and **NOT** cluster co-membership (the
  production clustering distance `1−(0.5·CE+0.5·cos)` uses a verdict-trained ModernBERT CE **and** a
  verdict-trained LoRA-BGE cos, so cluster labels leak S).
- recoverability `R` = the **raw 434K v6 LLM pair verdicts** (score 0/1/2; sk3
  `norm_embed/all_verdicts.jsonl`), binarized same-vs-different — an **X-free** "same concept?" judgment.
- **Provenance audit (2026-06-15, workflow w7lorpo1n):** `S⊥R` HOLDS iff `S` = a frozen exogenous encoder
  (OpenAI/MiniLM/frozen-BGE — all cached, zero verdict exposure) and `R` = the raw verdicts. It is
  VIOLATED if you reuse the production cluster labels or `emb_lora_*` (both verdict-trained). So Analysis A
  is valid — on the frozen embeddings, against raw verdicts, **not** piggybacked on the existing clusters.

Three analyses, inline with this infra, beyond dispersion:
- **A — codability curve `P(same-concept | S)`**: bin rubric pairs by surface distance `S`, plot
  `P(verdict ∈ {dup,paraphrase})`. A curve that stays **high at low `S`** = highly codable (truly
  different wordings still recovered as one concept); a curve that **drops fast as `S` falls** = low
  codable (only near-copies recognized as equivalent → the concept doesn't transmit / fragments).
  Codability index = recoverability at the low-`S` decile (the *residual* same-concept mass that surface
  similarity does NOT explain). Per task and per concept-cluster.
- **B — synonymy multiplier**: raw effective rank (`participation_ratio`, have it) ÷ concept-collapsed
  effective rank (after merging paraphrase-linked rubrics) = avg number of *distinct* encodings per
  concept. The codability-adjusted dimensionality of the library.
- **C — K-way concept-discrimination game** (the genuinely new, X-free, beyond-pairwise upgrade): given a
  verbalization, a listener LLM picks its concept from K prototypes (cluster medoids); accuracy =
  codability, and the **confusion structure** localizes which encodings are ambiguous. Zaslavsky's
  reference game on our actual library.

**Subtask stratification (metrics carry a `subtask` field).** Metrics were discovered on different
subtasks, so every pair is *within-subtask* or *cross-subtask*. Compute the codability curve per stratum:
- **cross-subtask same-concept recovery at low `S`** = the CLEAN codability signal — the concept
  transmits across contexts, not bound to one subtask's local jargon (MTMM convergent validity: trait
  surviving across methods=subtasks).
- **`within − cross` gap** = shared-subtask-vocabulary inflation (a shared-method confound that makes
  within-subtask pairs look more codable than the concept is).
- Per concept, report **codability × generality** (how many subtasks it spans): universal+codable =
  robust shared norm; codable-but-subtask-local = dialect; cross-subtask-but-low-codability = same word,
  different meaning per subtask (genuine ambiguity). Ties to the generic-vs-partition-specific question.
This also partly hardens against the `S⊥R` independence risk: cross-subtask pairs are a held-out-context
transfer test even if the encoder saw within-/pooled pairs.

Deeper options (later): **MDL** (min description length to specify a concept unambiguously) and
**iterated-learning / transmission chains** (Kirby; Zaslavsky–Imel — codable concepts *stabilize* under
describe→re-describe, tacit ones drift). A+B are runnable from artifacts on disk; C needs one cheap
listener pass.

### 9.3 The metric communication game (beyond variance, for L0/R1/R2)
Our variance tests are the **agreement-only** special case (listener = identity). Upgrade to *transmission*:
1. Define m by a seed **independent of the speaker** (contrast exemplars, or SYNTHETIC by construction).
2. SPEAKER `E_s` verbalizes m at budget L (sample → distribution over wordings = the L0/R1/R2 levels).
3. LISTENER `E_l` (**different** executor, blind to the seed) reads s and re-scores held-out `X_test`.
4. `codability(m; L) = agreement(listener scores, m-reference) = I_V(s → m_recovered)`.
5. Trace over L → **codability-vs-budget ladder**: asymptote = anchor-free articulability ceiling of m,
   knee = description length to transmit it.
Circularity guards: `E_s ≠ E_l`, disjoint `X_test`, m-reference NOT derived from the speaker's own
wording. This is the **same ladder as `I_{V_{L,E}}`** with target = *transmitted concept* instead of an
external Y — i.e. the anchor-free articulability instrument, and the operationalization of "communicable
community-agreeable metrics" (Noah framing).

### 9.4 Reliability → validity (the honest limit)
Consistency / reconstruction alone = **reliability**; inflatable by shared-method (same-E) variance.
Upgrade to **validity**: (a) **MTMM** — vary executor AND verbalization; articulability = trait variance
surviving across methods (convergent) and distinct from other metrics (discriminant); (b) **synthetic
calibration** — on constructs known by construction, both `a(E)` and codability are computable, so verify
codability tracks true `a(E)` BEFORE deploying it anchor-free (synthetic is the one judge-free real
anchor); (c) **latent-Z V-info** — `z(x)` = shared factor across `{E(s_i)}`, articulability `= I_{V_{L,E}}(X→z)`,
but `z` must be identified from a *different* method-set than the one evaluated, else circular.

## 8b. Open

- Does `a(E)` plateau or → 0 along the observational-scaling ladder? (the central empirical test of
  whether a fully-tacit residual exists).
- IF-capacity control to de-confound the `L'−L` describability measure.
- Same-class vs ensemble C: how loose is the 8B Bayes proxy? (saturation sweep).
- Calibrate codability (§9.3) against synthetic `a(E)`: does anchor-free transmission fidelity track the
  true articulability ceiling, and how tight is the DPI bound (§9.1) empirically?
