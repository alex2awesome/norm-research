# Metric seam — certified prompt↔code↔workflow evolution of evaluation metrics

*2026-07-01. Concrete proposal for `methods/metric_seam/` (working name; user's original:
`metrics_prompt_code_tradeoff`). Graduates ideas-backlog **§B** ("Prompt specificity trajectory & the
prompt↔code (V/A) seam") from design-sketch to buildable spec. Companion theory:
`2026-06-18__prompt-optimality-theory.md` (cited as **PO §n** throughout); related-work sweep:
deep-research run 2026-07-01 (§9). Antecedents: `project_refactoring_algorithm_idea` (library=norms,
main()=taste), `project_thin_thick_rules_philosophy` (rule- vs input-thickness 2×2, v_struct),
the manual patents §102 retrieval workflow.*

---

## 0. Summary

Take ONE explicit metric. Run the existing fidelity-only GEPA refinement (unsupervised reconstruction
objective `R`, no label `Y`) — but make the unit of evolution the **typed criterion channel**
(extractor ∘ predicate) instead of a monolithic prompt, and add two gated structural operators:
**MIGRATE** (channel predicate/extractor prompt→code, accepted only on a faithfulness gate) and
**WORKFLOW** (attach an evidence or computation tool-op to a starving channel). The evolved artifact's
fixed point — (code library, residual LLM rubric, workflow graph) — *is the V/A frontier of that metric,
made physical*, and the trajectory to it measures how the frontier is reached. Everything is certified
with the machinery we already have: recovery's global ceiling `headroom = T(m_ω) − R` survives verbatim
(DPI is executor-agnostic, PO §11.1), the within-class bound extends to implementation choice via a
partition matroid, and migration monotonically *tightens* the certificate (§5.3). Positioning after the
2026-07-01 related-work sweep: automatic agentic-workflow optimization is crowded (AFlow, ADAS, MASS,
GEPA's own Full Program Adapter), but every published system optimizes **task accuracy** and none
carries an **optimality bound**; our cell — unsupervised fidelity objective + certificates + the target
being a *measurement instrument* whose evolution measures articulability — is empty (§9).

**What this is not:** a task solver, an agent framework, or a claim that reflective workflow evolution
is novel per se. The workflow evolution is apparatus; the *measurement* (per-task, per-metric V/A
frontier + its certificate) is the contribution.

*Navigation (2026-07-02):* the full certificate math is **§5bis** (implemented + tested + empirically
validated at 160-metric scale); the per-channel code-vs-prompt-vs-tool decision recipe is **§5bis S6**;
the explicit shared/modified/new map against the prompt-optimality theory is **§5ter**.

---

## 1. Research questions

- **RQ1 (seam).** For a given metric, which criteria migrate to code (V), which resist codegen but
  survive LLM judging (A), and which need channel augmentation (evidence beyond `X`)? What fraction of
  each — the 2D frontier of §3.2 — and how does it differ across tasks (creative-writing vs math vs
  patents)?
- **RQ2 (trajectory).** How does prompt specificity evolve across GEPA rounds, and does its
  *composition* shift toward the A-layer residual as code coverage grows (the backlog-§B prediction)?
  Does specificity saturate (magic words found) or keep climbing, and where does saturation differ by
  task?
- **RQ3 (bounds).** Can the prompt-certification ladder (PO §3) be extended to the enlarged search
  space (content × implementation × form × workflow-ops) — and does migration provably concentrate the
  uncertified residual on the shrinking LLM part?
- **RQ4 (workflow).** Can the manual patents-§102 move (add retrieval when the signal isn't in the
  document) be automated by a per-channel diagnostic + typed op library — and certified?

**Headline claims we aim to support:**
1. The V/A boundary is measurable *constructively*: a criterion is V iff an explicit program witness
   reproduces its LLM channel at its **attenuation ceiling** held-out — κ→√rel_K, not κ→1; the judge
   channel's own noise caps every implementation (§5bis S1). (Stronger than the `R=T` saturation
   witness, PO §3.4.)
2. The V/A frontier is also the **certifiability frontier**: the certified optimality gap of the whole
   metric shrinks monotonically as channels migrate (§5.3).
3. Workflow ops split into **evidence** and **computation** types with different formal treatment
   (§3.3), both slotting into existing machinery — and the split is *empirically decidable* per channel.
4. "Should this metric be prompt or code?" is **one-sided decidable** (§3.4): codability is positively
   certified by a program witness, negatively only bounded by search saturation — so the "fuzzy
   boundary" resolves into five measured regimes (δ_e, adjudicated disagreement direction, κ_e(L)
   shape) plus a Kaplow cost tiebreak at deployment frequency.

---

## 2. Relation to existing assets (build little, reuse much)

| exists | where | role here |
|---|---|---|
| GEPA loop over prompt OR code artifacts, fidelity-only objective | `metric_implementer/optimizer.py::improve` | outer loop, unchanged invariant |
| `MetricArtifact(kind="prompt"\|"code")` + complexity measures | `artifact.py` | extend with `kind="hybrid"` (§4.1) |
| operators incl. MECHANIZE, DECOMPOSE; failure-attributed choice | `optimizer.py::_OPERATORS` | MIGRATE = gated MECHANIZE-per-channel; DECOMPOSE = channel splitter |
| immutable version registry + lineage (operator/round/parent/tokens) | `registry.py`, `gepa_lineage.py` | unit-trajectory graph for RQ2, free |
| recovery channel `R` (behavioral/semantic/tvd_mi), cross-family acceptance | `measures.py`, `recon_channel.py` | objective, unchanged |
| per-aspect codegen programs, `score(text)->float` contract | `runs/validity_full/v2/*/codegen_claude/` (654–1184/task) | MIGRATE's codegen template + warm-start pool |
| code↔judge disagreement queues | `registry.py` `disagreements/` | faithfulness-gate bookkeeping |
| channel-cleanliness instruments (adversarial saturation, PRUNE-help, CF validity) | `orthogonalize.py`, `measures.py` | migration validity gate (§6) |
| `U₂` within-class bound, exact small-Ω brute force | `experiments/large_omega.py`, `omega_certificate.py` | extend to matroid ground set (§5.2) |
| TOOLS_NEEDED taxonomy (citation-verify 29%, claim-check 29%, repro 21%, novelty 12% on peer-review) | `verification_library` | seed of the typed op library (§4.3) |
| v_struct (LLM grounds → code verifies) | `datasets/legal-outcome-prediction` | the thin-rule/thick-input quadrant, already validated |

Prerequisite that stops being a separate backlog item: the **criteria-based parseable GEPA pivot**
(backlog §2) — typed channels give the stable unit IDs that prose GEPA lacks; it is Phase 0 here.

---

## 3. Formal objects

### 3.1 The hybrid artifact

```
HybridMetric := {
  channels: [ Channel ],          # the unit of evolution, selection, and certification
  aggregator: fixed               # linear weights or fixed template — FROZEN in v1 (see note)
}
Channel := {
  id: stable across rounds (operators preserve; splits/merges recorded in lineage),
  extractor:  { type: regex | program | llm | tool_op,  spec },   # produces features/evidence
  predicate:  { type: code | llm,                        spec },   # maps features -> channel verdict
}
```

A channel's verdict process is `v_e(x) = pred(feat(x))`. `kind="prompt"` and `kind="code"` are the
degenerate one-channel cases, so the registry/scorecard plumbing applies unchanged.

*Aggregator note:* aggregation is **frozen** in v1 (linear or fixed template). Learning the aggregator
is Idea A's question (flexible vs additive heads); freezing it here keeps the seam study unconfounded,
and the per-channel score matrix this method caches is exactly Idea A's input — run B, get A's decisive
probe nearly free.

*Why typed channels and not a free code string (GEPA Full Program Adapter) or an explicit graph
(AFlow/MaAS):* the representation is chosen for **measurement**, not optimization power. Free code
strings have no unit identity (can't track a criterion across rounds); bare graphs lose text semantics
(can't measure specificity). Typed channels support both, plus per-unit certificates. No prior system
makes this argument (§9).

### 3.2 The seam is 2D (rule-thickness × input-thickness)

Each channel lands in a quadrant, and MIGRATE moves the two components independently:

| | thin extractor (regex/program) | thick extractor (LLM / tool) |
|---|---|---|
| **thin predicate (code)** | fully V | v_struct quadrant: LLM/tool grounds → code verifies |
| **thick predicate (LLM)** | rare (structured input, holistic call) | fully A |

Per-task readouts: fraction of predicates that compile (rule-side V share), fraction of extraction that
stays LLM (input-side thickness), and the migration *trajectory* between quadrants. Daston's "behind
every thin rule is a thick rule cleaning up after it" is operationalized as thick extraction feeding
thin predicates — and we measure how much of the metric lives there.

### 3.3 Workflow ops: evidence vs computation. **[the X→(X,Z) taxonomy — do not conflate]**

Any tool call yields some `Z`; its formal status splits on whether `Z` touches state outside the
document:

- **Evidence op** — `Z = g(q(X), W)`, `W` external world state (retrieval of prior art, citation DB,
  web search, answer keys, metadata). The **channel itself changes**: `I(M; X, Z) ≥ I(M; X)` (chain
  rule; free monotone lemma). The outer ceiling moves. This is the patents-§102 case; also the case
  where the operationalized `m_ω` without the op was measuring a *different construct* than the ideal
  `M*` (PO §12.4 — construct-fidelity work, not just executor work).
- **Computation op** — `Z = f(X)` deterministic in the document (execute embedded code, sympy-check a
  derivation, recompute a statistic, compile, parse). **By DPI it adds ZERO Shannon information**:
  `I(M; X, f(X)) = I(M; X)`. It helps because the executor is computationally bounded and cannot
  compute `f` reliably — the tool **widens `B_E`**; it is part of the *executor*, not the channel.
  This is exactly the "processing creates usable information" phenomenon V-information was built for,
  and PO §12.7 already prescribes the handling: executor-indexing, no functional. `E+tool` is a rung on
  the PO §5.5 ladder; the E-vs-E+tool recovery gap is measured exactly like compiler-vs-LLM in C4.

**Consequence 1 — the theory needed is smaller than feared.** The "workflow axis" decomposes into
*widen X* (one trivial chain-rule lemma) and *widen E* (existing executor-ladder machinery). No new
formalism.

**Consequence 2 — the split is empirically decidable, giving the repair decision tree:**

```
channel fails (low R) ──► T(channel) healthy? ── yes ──► articulation problem: keep GEPA-ing prompt
                                   │
                                   no (verdicts carry ~nothing from X)
                                   │
                    stronger executor (no tool) raises T? ── yes ──► computation-limited: COMPUTATION op
                                   │
                                   no ──► signal is external: EVIDENCE op  (the automated patents move)
```

(Also catches the degenerate-criterion case: if no op ever helps and CF validity fails, the channel is
noise — PRUNE.)

*Boundary cases:* an LLM sub-call as extractor = executor widening (computation-type); agentic
retrieval `Z = lookup(q(X))` touches `W` ⇒ evidence-type even though the query is document-derived.

### 3.4 The SHOULD decision rule — what the seam answers that type-classification can't

*The standing question ("why is 'indentation = 4 spaces' a top A metric when it looks like it should be
code?") has always been answered "the boundary is fuzzy." The seam machinery replaces that answer. The
question is ill-posed as a CLASSIFICATION of the criterion's description and well-posed as a COMPARISON
of its two best implementations — which is exactly what the MIGRATE gate measures.*

**Where the fuzz lives.** A human-enforced criterion is `thin core ∧ applicability guard`: reviewers
waive "indent==4" for generated code, continuation alignment, embedded YAML. The LLM channel is not
computing the core; it is computing `core ∧ guard`. The boundary question is never "does the happy path
compile" — it is "how much mass does the guard carry, and is the guard enumerable?" Both measurable
(Daston's thick-rule-cleaning-up, operationalized).

**Three numbers per criterion replace "it's fuzzy":**
1. **Residual disagreement mass** `δ_e = 1 − κ(best code, LLM channel)`, held-out — how often the
   guard binds. `δ_e ≈ 0` ⇒ code, done.
2. **Adjudicated direction of disagreements** (the registry `disagreements/` queues): if the CODE is
   right on the disagreement cells, the LLM channel was noise and code SHOULD win despite moderate κ —
   the case type-intuition systematically misses. Only LLM-right-with-CF-validity marks real exceptions.
3. **The codability profile** `κ_e(L)` — best faithfulness by programs of complexity ≤ L
   (`artifact.py` LOC/AST/cyclomatic; "maybe longer code would cover it" becomes this curve's shape):

| `κ_e(L)` shape | diagnosis | SHOULD |
|---|---|---|
| saturates ≈1 at small L | thin | code (witness in hand) |
| saturates ≈1 only at large L | thick-but-enumerable | Kaplow: code iff evaluation volume amortizes writing + maintaining the exception list; else LLM |
| plateaus < 1, LLM right on residual | open-textured guard | hybrid: code core + LLM applicability guard (v_struct sandwich) |
| plateaus < 1, code right on residual | LLM channel noisy | code |
| LLM channel's own `R` low | below the articulation floor | neither — criterion isn't stably articulated; prompt-vs-code is moot |

**The one-sidedness (the honest form of "fuzzy").** Codability is **positively certifiable, only
negatively boundable**: "should be code" is PROVED by exhibiting a gate-passing program (explicit
witness — the strongest certificate in the framework); "cannot be code" is never proved — only
"resisted a saturating multi-family codegen search up to complexity L" (discovery-curve epistemics,
PO §6.9). The fuzz is entirely on the not-codable side, shrinks monotonically with search budget, and
its position is a measured number with a CI. **Kaplow supplies the normative tiebreak in the middle
band** (code = high fixed / near-zero marginal cost; LLM judge = the reverse): SHOULD =
`argmin total cost at the metric's evaluation frequency, s.t. κ ≥ fidelity requirement`. The seam
supplies the fidelity frontier; deployment supplies the frequency — which is why the same criterion
legitimately gets different answers in different settings, i.e. why the question ever felt unstable.

**Construct-bundling detection (dissolves the indentation puzzle specifically).** When a code-looking
criterion sits in the A layer, a live hypothesis is that the judged construct ≠ the named construct —
the LLM "indentation" channel may score a bundled care/conformity signal that indentation proxies for.
CF validity decides: perturb ONLY indentation; if the verdict tracks the bundle rather than the
perturbation, prompt-metric and would-be code-metric are *different constructs sharing a name*, and
migration would silently swap constructs (the MECHANIZE drift, §4.2.3, now with a mechanism). The
honest output is then two metrics: the coded pure core + the residual LLM bundle, each certified
separately. Sometimes "why isn't this coded?" = "the coded version is a purer but different metric" —
demonstrated, not asserted.

---

## 4. The method — loop, operators, gates

Staged local→global, per metric (external validation for staging over joint search: MASS,
arXiv 2502.02533, finds staged prompt→topology→global beats joint combinatorial; our PO §6.8
block-coordinate structure is the *certified* version of the same design):

```
Phase 0  DECOMPOSE seed metric into typed channels (parseable-criteria GEPA pivot).
Phase 1  Per-channel + whole-prompt GEPA (existing improve(); fidelity-only objective unchanged).
Phase 2  MIGRATE pass: per channel, attempt codegen of predicate (and thinning of extractor);
         accept only through the gate (§4.2). Pass → V; fail → A-with-witness-of-failure.
Phase 3  WORKFLOW pass: for channels failing the §3.3 decision tree into op territory,
         propose from the typed op library (§4.3); accept through the same gate + ΔT readout.
Phase 4  Re-GEPA the residual LLM part (budget freed); loop to Phase 2 until no gated move fires.
Readout  Fixed point = (code library, residual rubric, workflow graph) + full lineage trajectory.
```

### 4.2 The MIGRATE gate (all four must hold — migration is gated, never score-chased)

1. **Faithfulness:** per-channel agreement κ(code, LLM verdicts) on held-out items above threshold,
   with CI (the constructive witness of claim H1).
2. **Non-inferiority:** metric-level ΔR ≥ −ε on held-out (bootstrap).
3. **Cleanliness (PO §6.10):** adversarial saturation ≈ 0, no PRUNE-help, and **counterfactual
   validity** — the code must track the planted direction, not a length/format proxy that happens to
   correlate with the LLM verdicts. This is the guard against the measured MECHANIZE failure mode
   (*mechanization converts underspecification into misspecification*: agreement up, anchor decoupled —
   the construct-drift finding). A migration that raises κ but fails CF is REJECTED and logged as a
   Goodhart instance (itself a useful readout).
4. **Determinism audit (PO §5.5 C2):** migrated channel must pin `T_norm = 1` under fixed code
   (violations = data bug, not articulation).

Failures are kept: a channel that *fails* the gate carries the failed program + failure mode as
evidence for "articulable-but-not-codable" — a much stronger A-classification than never trying.

### 4.3 The WORKFLOW operator

Typed op library, seeded from the verification_library TOOLS_NEEDED taxonomy + the manual patents
recipe: `{retrieve_prior_art, citation_lookup, web_evidence, answer_key}` (evidence) ×
`{execute_code, sympy_check, recompute_stat, compile, parse_struct}` (computation). GEPA's reflective
proposer picks ops from failure traces exactly as it currently picks operators from failure
attribution; ops are per-channel attachments to the extractor. Acceptance: same gate as §4.2, plus the
type-specific readout — evidence op must raise the measured channel ceiling `T` on the augmented input;
computation op must close the strong-executor gap it was prescribed for. Op *discovery* beyond the
library (novel tools) is out of scope v1 (see honest-scope, §5.4).

---

## 5. Certificates (RQ3) — the ladder, ported

### 5.1 Global rung — unchanged, free

`headroom = T(m_ω) − R` (PO §11.1) is executor-agnostic: DPI holds whatever hybrid `E` executes the
channels. After an evidence op the ceiling must be **re-measured on the augmented channel**
`T(m_ω; X, Z)`; after a computation op the outer `I(M;X)` is *unchanged by construction* and only the
achieved/achievable `T(m_ω)` moves. Reporting both is itself a finding (how much frontier is executor
vs evidence).

### 5.2 Within-class rung — partition-matroid extension **[derived + implemented 2026-07-02 → §5bis S3; formal PO-appendix write-up still pending]**

Ground set `Ω × {code, llm}` (optionally × op-config), constraint "≤1 implementation per criterion" =
partition matroid. The Minoux/`1/γ` instance bound `U₂` (PO §3.2) goes through with top-k restricted to
*feasible* marginals; worst-case multiplicative guarantees for weakly-submodular + matroid exist in the
literature to cite. Deliverable: one-page derivation appended to PO, plus `U2_bound` accepting a
feasibility mask.

### 5.3 The tightening lemma **[derived pilot-form + implemented 2026-07-02 → §5bis S4 — headline claim H2; formal PO-appendix write-up still pending]**

Decompose the certificate residual over channels. For a migrated channel: tacitness (a)=0,
executor-limitation (c)=0, `T=1`, `A = 1−R` = pure learnability (PO §5.5) — its contribution to the
uncertified gap collapses to a measured learnability term with CI. Therefore the whole-metric
uncertified residual is confined to the LLM-judged share and shrinks monotonically with each accepted
migration. *The seam is where the certificate is tight.*

### 5.4 Honest scope

- Form axis Φ and workflow-op set W are **materialized finite sets** → Rung-3 best-in-set with CIs
  (PO §6.8 verbatim). No global bound over "all workflows" exists for the same reason none exists over
  all strings: op-universe ignorance is `B_E`-ignorance redux. (If pushed: PO §11.3a capture–recapture
  applies to op discovery verbatim; hold for v2.)
- GEPA remains a generator, not a coverage certifier (PO §4.4a): the measured frontier is the
  **achieved** frontier — a *lower bound on the V share*; ALPHA-PROBE remains the companion coverage
  instrument.
- `R` is label-free: the seam classifies articulability/verifiability, **not correctness** (PO §4.3).
  A codable wrong-attribute criterion is still codable. Correctness stays bracketed with `Y`-work.

---

## 5bis. Seam certificates — full design. **[added 2026-07-02; implemented + tested]**

*Implementation: `methods/metric_seam/certificates.py`; known-answer battery:
`tests_certificates.py` (14 planted/dummy tests, all green 2026-07-02). The PO ladder ports
rung-by-rung with the class swapped from criterion-subsets to PROGRAMS — and the seam setting is
MORE certifiable than the prompt setting: (i) witnesses are constructive (a program can be
exhibited and verified, a prompt only scored); (ii) `B_code,L` is a FORMAL class (VC dim, exact
enumeration at small L) where `B_E` was sampling-oracle-only; (iii) code channels pin `T=1`.*

*Empirical status (2026-07-02 overnight, `notes/2026-07-01__metric-seam-pilot-results.md` +
report notebook `notebooks/2026-07-02__metric-seam-certificates-and-overnight-report.ipynb`):
S1 verified at scale — 0 ceiling violations across 116 code-rung metrics on 4 tasks; the S5 op
taxonomy validated cross-task (PR=computation/mixed, math=A-dominant pending AST ops,
code_review=evidence-starved, patents=evidence-dominant); Rung-3 gates certified 20 evolved PR
hybrids (7 at P(gate)≥.95, 16/20 beat baselines at P≥.96, 1 honest null, 1 flagged artifact).*

### S1 — Attenuation ceiling (seam Rung 1). **[derived; tight; tested T2–T6]**

**Model.** Judge verdict on item `i`, pass `p`: `M_p(i) = τ(i) + ε_p(i)`, with `ε` uncorrelated
across passes, with `τ`, and with any deterministic `f(X)`. `τ` is the judge's *stable score* —
its own channel's true score, NOT the construct (correctness stays bracketed, PO §4.3).

**Lemma S1.** For the K-pass mean `M̄_K` and any implementation `f` with reliability `rel_f`:
> `corr(f, M̄_K) = corr(f, τ)·√(rel_K · rel_f) ≤ √(rel_K · rel_f)`,
> `rel_K = K·rel₁/(1+(K−1)rel₁)` (Spearman–Brown), `rel₁ = corr(M₁, M₂)`.

*Proof sketch.* `Cov(f, M̄) = Cov(f, τ)` since `Cov(f, ε̄)=0`; `Var(M̄) = Var(τ)/rel_K`;
Cauchy–Schwarz gives the bound — **no homoscedasticity needed** (tested under item-dependent
noise, T4). Deterministic code: `rel_f = 1`. ∎

**Consequences.** (1) **Ceiling-normalized fidelity** `ρ̃ = ρ̂/√rel_K` estimates `corr(f, τ)` —
report THIS, not raw ρ; the per-channel headroom analog is `√rel_K − ρ̂`. (2) **Gates must be
ceiling-relative**: an absolute 0.60 gate misclassifies judge noise (a118, rel₁=.59 ⇒ ceiling
≈.86) as A-layer thickness. (3) Our two passes use different templates, so measured `rel₁` folds
form-variance into noise ⇒ the ceiling is *conservative* (still valid). (4) Rank-correlation
version is approximate (exact for Pearson under CTT); planted check T5 bounds the slack at ~0.01–
0.03 — report Pearson alongside Spearman when near the ceiling.

### S2 — Codability bracket (seam Rung 2). **[design + implemented; tested T7–T8]**

Define `κ*(C)` = best achievable judge-fidelity by programs in class `C` (with ops `O`). The
certified object is the **two-sided bracket**:

> `κ*(C) ∈ [ ρ̂_witness − CI_boot , min( √rel_K , U_enum + ε_n ) ]`

- **Lower edge (constructive witness):** the exhibited program's held-out fidelity, bootstrap CI.
  PAC form: population fidelity ≥ ρ̂ − O(√((d_VC log n + log 1/δ)/n)) (`hoeffding_term`).
- **Upper edge (exact, small classes):** brute-force over a materialized class (stumps/thresholds
  over a feature dictionary — `enumerate_stump_class`) certifies "no member exceeds `U_enum`" on
  the sample, + uniform-convergence `ε_n`. This is PO's `|Ω|≤15` exact regime, ported. Planted
  check T7b: stump class certified ≤ .53 on planted XOR (truth .5).
- **Open-ended class (LLM codegen): NO upper edge.** Search saturation (multi-family codegen as
  capture–recapture lists, PO §11.3 verbatim on *program* space) is EVIDENCE, never certificate —
  absolute non-codability = circuit lower bounds. This is §3.4's one-sidedness, now with the
  attenuation ceiling closing part of the top.

The **V/A frontier = the vector of brackets** across a task's metrics; "the seam moved" =
bracket's lower edge crossed the migration threshold (a witness event, hence certified).

### S3 — Matroid-U₂ (within-class rung over criteria × implementations). **[derived; tested T10]**

Ground set `Ω × {code, llm}` (× op-configs), partition matroid `≤1` implementation per criterion.
PO §3.2's Minoux/`1/γ` derivation goes through: monotonicity step unchanged; the top-k sum ranges
over ALL marginals (a superset of the feasible ones) ⇒ **valid, slightly loose**:
`OPT_matroid ≤ R(S_g) + (1/γ)·Σ_{top-k, all e} δ(e|S_g)` (`u2_matroid_bound`). Same monotonized-R
caveat as PO §3.2.

### S4 — Tightening decomposition. **[derived, pilot-form; tested T11]**

With a FIXED linear aggregator, per-channel residuals add (`tightening_decomposition`):
- **code channel** (`rel_f=1`, realizability by construction): residual `w·(1−κ̂)` = pure
  learnability, **CI-only** — certified class;
- **LLM channel**: residual `w·(1 − ρ̂/√rel_K)` = articulation headroom — the uncertified class.

⇒ the metric's uncertified residual is **confined to the LLM channels and shrinks monotonically
with each accepted migration** ("the seam is where the certificate is tight"). Learned aggregators
void the additivity (that's Idea A's interaction term — keep the aggregator frozen in v1).

### S5 — Op-value certificates. **[derived; tested T13]**

- **Evidence op** (`Z` touches world state): `I(M; X, Z) ≥ I(M; X)` — chain-rule monotone lemma;
  the OUTER ceiling moves and must be re-measured on the augmented channel.
- **Computation op** (`Z=f(X)`): outer ceiling fixed (DPI); the op enlarges the program class ⇒
  `κ*(C; O)` **monotone in O** (T13a). The op's value `Δ(o|O) = κ*(C;O∪{o}) − κ*(C;O)` is exactly
  what the ablation Shapley estimates (`shapley_2` exact for the 2-component lattice, T12).
- **Op-subset certificate:** measure the empirical weak-submodularity ratio `γ̂` over the op
  lattice (`op_submodularity_ratio`); `γ̂>0` ⇒ `U₂` applies to op subsets ⇒ certified best-op-set
  and **certified prunes** (an op whose bounded marginal is below threshold is deleted with a
  guarantee, not a hunch — retro-applies to the a110 retrieval op).
- Caveat: Shapley-on-ρ is attribution, not variance decomposition; the clean version uses
  per-channel TVD-MI as the value function (upgrade path).

### Rung 3 — statistical gate. **[implemented; tested T9]**

`bootstrap_gate`: item-bootstrap `P(ρ_hyb ≥ max(ρ_base + margin, floor))` + `P(beats baseline)`,
reported at stated `B`, `δ`. Planted checks: fires (P=1.0) on a dominant hybrid, silent (P=0.0) on
a null one. Gate margins must exceed 2·SE or `n_test` scales up — the 2026-07-02 bootstrap
correction (a110 "PASS" → unresolved) is the case study.

**Constant-baseline caveat [added 2026-07-02, a119 case].** When every description-compiled
baseline is broken (recorded as failed rungs, never silently skipped), the gate runs against a
constant — its `P(gate)`/`P(beats)` are then ARTIFACTS, not certificates. The certificate degrades
gracefully to the hybrid's own bootstrap CI vs the floor (a119: CI [.754,.892] clears .60
decisively). Always label which readout is in force.

### S6 — The placement procedure: certifying code vs prompt vs tool, per channel. **[added 2026-07-02 — §3.4's SHOULD rule made operational; this is THE recipe]**

Inputs per channel `e`: 2-pass judge scores (→ `rel₁`, ceiling `√rel_K`), the scope channel, the
code-rung programs + any hybrid witness, the typed op library. Output: a **placement with a named
certificate class** — never a vibe. Each step states what kind of guarantee it yields.

0. **Measurability gate (precondition — no placement claim without it).** Degenerate judge passes
   (constant pass, <30 paired items) or `rel₁ ≤ .05` ⇒ flag `unmeasurable` and stop; report, don't
   drop (patents had 2 such channels). All fidelities below are read **ceiling-relative**
   (`ρ̃ = ρ̂/√rel_K`, S1) and **scoped** (guard variance removed — unscoped, soft criteria borrow
   "is this item in scope at all" variance and the seam table lies; pilot v1 finding).
1. **CODE — certified POSITIVELY, by witness (S2 lower edge + full MIGRATE gate).** Exhibit a
   program whose held-out bracket lower edge clears the ceiling-relative threshold AND that passes
   §4.2 in full: κ + ΔR non-inferiority + **CF validity + adjudicated disagreement direction** +
   `T=1` determinism audit. Agreement alone is NOT a placement certificate: a86's presence-proxy
   had high κ and was refuted 10/0 on blinded adjudication — CF perturbation is what separates
   construct from proxy. Certificate class: *constructive witness* (the strongest in the framework).
2. **PROMPT — certified only RELATIVE TO A CLASS; evidenced beyond it (the §3.4 one-sidedness).**
   "Stays LLM" is *certified* against a formal class `C` by the S2 upper edge (exact enumeration +
   `ε_n` sitting below the LLM channel's `ρ̃`); against open-ended codegen it is only
   *search-saturation evidence* (multi-family discovery curve, capture–recapture epistemics).
   State which of the two is meant, every time. The five `κ_e(L)` regimes (§3.4) name the
   diagnosis; the **Kaplow frequency tiebreak** resolves the middle band — a normative cost
   argument, reported separately from the statistics, never blended into them.
3. **TOOL — routed by the §3.3 decision tree, certified by the type-specific readout (S5).**
   - **Evidence op**: accept only if measured `T` on the augmented channel `(X, Z)` rises with CI —
     the claim is `I(M;X,Z) > I(M;X)`, i.e. the ceiling itself moved.
   - **Computation op**: outer ceiling unchanged by DPI; accept only if it closes the
     strong-executor gap it was prescribed for; its value is the ablation marginal `Δ(o|O)`.
   - **Op-set selection**: `γ̂ > 0` on the op lattice ⇒ `U₂` yields a certified best-op-set and
     *certified prunes* (bounded marginal below threshold ⇒ delete with a guarantee, not a hunch).
4. **Mixedness readout — WHERE in the flow each medium sits.** The 2×2 ablation lattice (LLM
   on/off × ops real/null) + per-item touch shares give the exact Shapley split (L vs T) and the
   interleaving order; the code-only core `ρ` is the floor against which the LLM/tool components
   are certified. (Empirical shape so far: every gated hybrid is a **C→L→C sandwich** — code frames
   the extraction, the LLM reads the thick input, code validates and fuses. Daston's
   thick-rule-cleaning-up, measured.)
5. **Every verdict ships with its Rung-3 gate probabilities** (`P(gate)`, `P(beats baseline)` at
   stated `B`, margins > 2·SE) — point estimates overclaim (a110). No working baseline ⇒ gate
   probabilities are artifacts; the CI-vs-floor readout is the certificate (a119, Rung-3 caveat).

*Summary of certificate classes by placement:* CODE = constructive witness (positive certificate);
PROMPT = class-relative certificate (formal `C`) or saturation evidence (open class) — never a
positive proof of non-codability; TOOL = type-specific measured readout (ceiling moved vs executor
gap closed) + lattice-`U₂` for the op set. The asymmetry is the honest content of "the boundary is
fuzzy" — the fuzz is one-sided, quantified, and shrinks with search budget.

### What remains impossible (inherited PO scope, correctly stated)
Absolute non-codability (complexity theory); construct correctness without `Y`/human anchor
(everything certifies reproduction of `m_ω`); executor-relativity (the seam is indexed to the
judge family — cross-family replication or §12.6.5 flatness required for "the criterion" claims).

---

## 5ter. Correspondence with the prompt-optimality theory — shared, modified, new. **[added 2026-07-02]**

*One table, so the mapping doesn't have to be reverse-engineered from scattered PO §n citations.
The single structural change everything follows from: the class under certification swaps from
PROMPT-SUBSETS (criteria in `Ω`, executed by one fixed `E`) to PROGRAMS (implementations of one
criterion, executable by anything). The objective (unsupervised fidelity, no labels) and the
epistemics (witnesses up, saturation evidence sideways, impossibilities stated) are unchanged.*

| PO object | seam counterpart | status |
|---|---|---|
| global headroom `T(m_ω) − R`, DPI (PO §2.2, §11.1) | identical, executor-agnostic; re-measure `T` only after an EVIDENCE op (§5.1) | **shared verbatim** |
| Rung-3 best-in-set bootstrap (PO §3.3) | `bootstrap_gate` over the materialized implementation/op set | **shared verbatim** (+ constant-baseline caveat) |
| cleanliness gates: adversarial saturation, PRUNE-help, CF validity (PO §6.10) | MIGRATE gate conditions 2–3 (§4.2) | **shared verbatim** |
| capture–recapture / discovery-curve saturation epistemics (PO §6.9, §11.3) | codegen-search saturation over PROGRAM space = the evidence (non-certificate) side of the S2 bracket | **shared, re-targeted** (criteria → programs) |
| GEPA = generator, not coverage certifier (PO §4.4a) | achieved frontier = lower bound on the V share (§5.4) | **shared verbatim** |
| block-coordinate staged search (PO §6.8) | Phases 0–4 (§4); MASS is the external validation | **shared design** |
| executor-relativity caveat (PO §12.6.5, §12.7) | seam is judge-family-indexed; cross-family replication required for "the criterion" claims | **shared limitation** |
| Rung-2 `U₂` on subset ground set (PO §3.2) | ground set `Ω × {code,llm} × op-configs` under a partition matroid; top-k over the superset ⇒ valid, slightly loose (S3) | **ported, modified** |
| executor class `B_E`: sampling-oracle only, never enumerable (PO §11.3, §12.7) | program class `B_code,L`: FORMAL — VC dimension, exact enumeration at small `L` ⇒ upper bracket edges exist (S2) | **ported, strictly stronger** |
| optimality witness = `R=T` saturation (PO §3.4-adjacent) | witness = an EXHIBITED gate-passing program — an inspectable, re-runnable object | **ported, strictly stronger** |
| `T_norm=1` determinism corner C2 (PO §5.5) | code channels pin `T=1` by construction → powers S4's certified/uncertified residual split | **ported, becomes load-bearing** |
| *(no analog — PO's target `m_ω` is noiseless once executed)* | **S1 attenuation ceiling**: the seam's target is a NOISY judge channel, so reliability enters; ceilings and ceiling-relative gates throughout | **new to seam** |
| *(no analog)* | **S4 tightening decomposition** — uncertified residual confined to the LLM share, shrinks monotonically with accepted migrations | **new to seam** (built from PO §5.5 parts) |
| *(no analog)* | **S5 evidence-vs-computation op taxonomy** (§3.3) + op-lattice `γ̂`-`U₂` with certified prunes | **new to seam** (each lemma is one line: chain rule / DPI) |
| *(no analog)* | **§3.4 SHOULD rule + S6 placement procedure**, incl. the Kaplow frequency tiebreak | **new to seam** (the tiebreak is normative — deliberately outside the statistics) |

**Two direction-of-bounds cautions, carried over unchanged (standing corrections):**
1. `T` is a **lower**-bound-side object on the ideal `M*` (a ceiling on the operationalized `m_ω`
   is a floor on `M*`); `B_E`-style class estimation is the **upper**-bound side (PO §12.4). The
   seam's `τ` is the *judge channel's stable score* — S1 ceilings are ceilings on
   judge-reproduction, never on the construct. Do not collapse these.
2. **Objective mismatch, pilot-tier (the one real gap):** PO's `R` is TVD-MI reconstruction with
   cross-family acceptance; the pilot's fidelity is Spearman-vs-2-pass-judge-mean. The bridge —
   per-channel TVD-MI as S5's value function, `T`/headroom measured on hybrid channels — is
   designed but not yet run. Until it runs, seam numbers are *calibrated instrument readings*
   (ceilinged, gated, CI'd) and PO numbers are the *theory-native measurements*; comparisons across
   the two stacks are directional only.

---

## 6. Validity guards (standing, from memory + PO)

1. MECHANIZE construct drift → CF-validity in the gate (§4.2.3). 2. Side-channel code (length/format
proxies) → cleanliness gate. 3. Judge score-distribution collapse check on every new channel
(feedback_check_judge_score_distribution). 4. Same split + same input for any before/after comparison
(feedback_apples_to_apples). 5. No "expect empty" rhetoric in channel prompts. 6. Per-metric, never
task-level, for α/coverage readouts (feedback_alpha_probe_is_metric_level).

*Added from the pilot (2026-07-02):*
7. **One canonical text per item** (head 5,000 + tail 2,500 chars), byte-identical for the judge and
   every implementation — the concrete instantiation of guard 4 (v0 shipped the bug this guards
   against: code saw full text, LLM saw a truncation; all v0 seam numbers were voided).
8. **Scope channel for every metric; seam tables read scoped.** Unscoped, soft criteria borrow
   "is this item even the right kind of document" variance and look codable when only the
   applicability guard is (pilot v1: structural metrics ↑ to .76–.83 scoped, soft lede fell to ≈0).
9. **Degenerate/unreliable judge channels flagged, never dropped** (constant pass, <30 paired items,
   `rel₁ ≤ .05`): they enter the counts as `unmeasurable` — silently dropping them inflates a task's
   apparent codability (code_review: 20/40; patents: 17/40 flagged).
10. **Broken codegen recorded as failed rungs**, never skipped — the survey's code rung is a LOWER
   bound on codability and must not silently condition on "program happened to run."
11. **Held-out gating on every reflective-improver round** — round-1 refinement of a80 improved all
   12/12 train cells and dropped test ρ .709→.579; the train/test split is what caught it.

---

## 7. Experimental plan

| # | experiment | cost | gate/decides |
|---|---|---|---|
| **E-S0** | **Longitudinal mining of EXISTING registry lineages** (`outputs/metric_implementer/*/registry/`): specificity trajectories per version (tokens, rare-term density, if-then concreteness, embedding drift, criterion rarefaction), MECHANIZE/DECOMPOSE operator frequency, saturation shape by task. RQ2 answered on existing data. | free (no new runs) | validates unit-alignment tooling before any build |
| **E-S1** | **Planted-seam calibration (the E0-style kill-switch).** Plant a hybrid metric with known seam: `AND(word-count threshold [codable], "argument is charitable" [LLM-only])`. Pipeline must migrate the first and refuse the second. Extend with two planted op-recoveries: a criterion whose signal is in an external answer key (must select an EVIDENCE op) and an arithmetic-correctness criterion (must select sympy = COMPUTATION op). | small (CPU + few LLM calls) | HARD GATE: fail ⇒ no real-task frontier is trustworthy |
| **E-S2** | One real metric end-to-end on **peer_review** (654 codegen programs + aspects.json exist; TOOLS_NEEDED taxonomy known). Full loop Phases 0–4, all gates, both certificates. | moderate | first real seam + tightening curve; template for scaling |
| **E-S3** | Cross-task frontier: repeat on math (sympy-rich, expect high V), creative_writing (expect low V), patents (expect evidence-op-dominant — the §102 validation of RQ4). | larger | the per-task V/A frontier figure; RQ1 |
| **E-S4** | *(optional ablation, reviewer-friendly)* reflective (GEPA) vs numeric (MCTS à la AFlow) proposer for Phase-3 op selection under the SAME fidelity objective — the head-to-head the literature lacks (open question #2 of the 2026-07-01 sweep), cheap here because the objective is pluggable. | moderate | mechanism claim; drop if budget-tight |

Models/budget: in-loop roles as configured (Llama-8B judge / Llama-70B reviser / Qwen-72B cross-family
acceptance via OpenRouter); codegen via Sonnet on Max-plan subagents (never the USC key); GLM-5 (z.ai)
only for acceptance-time reconstruction, small dev sets (quota memory). No local GPU serving needed for
E-S0–E-S2.

---

## 8. Falsifiable predictions

1. **Composition shift (backlog-§B):** after each accepted migration wave, residual-prompt specificity
   composition shifts toward A-layer criteria (measured on the E-S0 axes). *Falsified if* composition
   is static while code coverage grows.
2. **Task ordering:** V share math > code_review > peer_review > creative_writing; evidence-op share
   maximal on patents. *Falsified by* any inversion — which would itself be a finding about where
   verifiability lives.
3. **Tightening:** certified gap `U − R̂` decreases with migration count, slope ∝ migrated channels'
   marginal weight (§5.3). *Falsified if* gap is flat while migrations accept (would indicate the
   residual was never the binding term — diagnose against the T decomposition).
4. **Goodhart incidence:** a non-trivial fraction of κ-passing migrations FAIL CF validity (the
   MECHANIZE drift, now counted per task). If ≈0, the gate was cheap; if large, the seam without the
   gate is an illusion — either way, reportable.

---

## 9. Related work & positioning (verified sweep, 2026-07-01; 25/25 claims confirmed 3-0)

| system | optimizes | mechanism | objective | certificate |
|---|---|---|---|---|
| GEPA (2507.19457) + Full Program/MCP adapters (gepa-ai/gepa) | prompts; code-string incl. control flow; tool descriptions | reflective genetic-Pareto | task accuracy | none |
| ADAS / Meta Agent Search (2408.08435) | whole agent in code | reflective code-gen + archive | task accuracy | none |
| AFlow (2410.10762) | workflow topology (code-repr.) | MCTS | task accuracy | none |
| MASS (2502.02533) | prompts + topology, staged | MIPRO + influence pruning | task accuracy | none |
| DAAO (2509.11079) | per-query topology/depth/LLM | learned policy (VAE+MoE) | accuracy/cost | none |
| MASFly (2602.13671) | prompts + topology, test-time | Watcher reflection → SOPs | task success | none |
| AgentCo-Op (2605.20425) | roles/tools/graph | retrieval synthesis + local repair | task success | none |
| Semantic-backprop/GASO (2412.03624), Trace/OptoPrime (2406.16218), TextGrad (2406.07496) | node prompts in fixed graph | textual gradients | task accuracy | none |
| DGM (2505.22954) | own codebase | reflective gen + Darwinian archive | benchmark score | none |
| **this proposal** | **metric channels: impl + prompts + ops** | **reflective GEPA, gated structural moves** | **unsupervised fidelity `R`** | **headroom + matroid-U₂ + tightening lemma + program witnesses** |

Unclaimed cell confirmed by the sweep: GEPA-style reflective evolution over an explicit tool-topology
space; and NO system optimizes an unsupervised instrument-fidelity objective or carries a bound. Our
novelty rests on objective + certificate + measurement purpose, NOT on "reflective workflow evolution"
(occupied). Watchlist: a published end-to-end "agentic GEPA" joint topology+tools+prompts result, and
any reflective-over-explicit-topology paper. Also not in the verified set (mechanism unconfirmed):
ScoreFlow (2502.04306), EvoAgentX (2507.03616), GPTSwarm, AgentSquare, MaAS. Survey anchor for the
landscape: EMNLP 2025 compound-AI optimizer survey (2025.emnlp-main.1463), whose "Flexible Structure"
class is the right citation frame.

Also adjacent-but-different: LILO/STITCH (library compression for program synthesis) and FunSearch/
AlphaEvolve (evolutionary code search) — see `project_refactoring_algorithm_idea` for the fuller list;
none evaluates text, none measures articulability.

---

## 10. Risks

| risk | mitigation |
|---|---|
| planted-seam calibration fails (pipeline can't recover a KNOWN seam) | E-S1 is a hard gate before any real-task claim; debug learnability vs gate thresholds separately (PO §5.5 C1–C6) |
| few migrations fire on real tasks (V share ≈ 0 everywhere) | itself a publishable negative given the constructive-witness framing; check math first (sympy-rich, should migrate) |
| channel decomposition unstable across rounds (unit identity breaks) | Phase-0 parseable pivot + lineage recording of splits/merges; E-S0 validates alignment tooling on existing lineages first |
| op library too small to matter | seed from TOOLS_NEEDED empirical percentages; patents §102 is the known-positive control |
| scooped on "reflective workflow evolution" | headline never rests there (§9); certificate + fidelity objective + frontier measurement are the moat |
| GLM quota / cost creep | E-S0 free; E-S1 CPU-scale; role models per config; Sonnet via Max subagents |

---

## 11. Immediate next actions **[statuses updated 2026-07-02]**

1. E-S0 mining script over existing registries (no new compute) → RQ2 first read + tooling shakeout.
   *(still free & unblocked)*
2. ~~Derive the two small lemmas~~ → **partially done**: matroid-`U₂` and tightening are derived
   pilot-form, implemented, and tested (§5bis S3/S4); remaining = the formal one-page PO-appendix
   write-up.
3. `artifact.py` `kind="hybrid"` + channel schema; MIGRATE gate as a scorecard extension.
   *(the pilot's `LLM_FIELDS` + `score(text, extracted, ops)` contract in
   `methods/metric_seam/hybrids/` is the working prototype of this schema)*
4. E-S1 planted-seam battery. *(the certificate-level planted battery exists —
   `tests_certificates.py`, 14 green; the PIPELINE-level planted seam of §7 E-S1 does not yet)*
5. E-S2 on one peer-review metric.

**Done 2026-07-02 (see §5bis empirical-status block):** certificates library + tests; 160-metric
4-task certified survey; 20 gated PR hybrids; report notebook. **Queued pending sign-off:**
wave-2 mixedness ablation (2×2 lattice on all 20); hybrid evolution for math (ready — AST/sympy
ops), code_review (blocked on diff-enrichment of `X`), patents (blocked on prior-art retrieval
corpus); TVD-MI bridge (§5ter caution 2); cross-family judge replication.
